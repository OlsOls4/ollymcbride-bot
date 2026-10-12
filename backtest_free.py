"""FREE_MODE backtest: score free model chains on RESOLVED questions, at zero cost.

Run by ``.github/workflows/backtest_free.yaml`` (workflow_dispatch only). Never publishes.

Design (ollymcbride-bot fork, 2026-10-12):

* Zero cost and quota protection. Requires ``FREE_MODE=true``. Before anything imports the bot,
  every OpenRouter / paid-provider key is removed from the environment, so the live bot's
  OpenRouter ``:free`` budget (50 requests/day, account-wide) is never touched and no paid model
  can be reached. Members are NVIDIA NIM and Gemini Flash-Lite links by default; Gemini 3.x Flash
  (about 20 requests/day, shared with the live bot) is opt-in and capped by ``--flash-questions``.
* Leakage protections. Resolved questions go through the repo's backtest prep
  (``backtest.question_prep``: resolution fields and background stripped, state set OPEN), bots
  run with ``is_benchmarking=True``, the research phase is replaced by a fixed "no research"
  string (live research would return post-resolution news), and the prompts' "Today:" lines are
  anchored to each question's mid-window (the ablation harness's convention) instead of now.
* One bot per member model (a single-model chain, 2 tries). Ensemble variants are then scored
  OFFLINE from the members' own forecasts (median, as the live aggregator), so comparing several
  ensembles costs no extra calls.

Outputs ``backtests/free_backtest_results.json`` and ``backtests/free_backtest_report.md``, and
appends the report to ``$GITHUB_STEP_SUMMARY`` when set.
"""

from __future__ import annotations

import os
import sys

# --- environment guard: runs BEFORE any metaculus_bot import (llm_configs builds chains at import) ---
_REMOVED_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_DONATED",
    "METACULUS_OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "TAVILY_API_KEY",
    "EXA_FREE_API_KEY",
    "SERPER_API_KEY",
)


def _sanitize_environment() -> list[str]:
    if os.getenv("FREE_MODE", "").strip().lower() not in ("true", "1", "yes"):
        raise SystemExit("backtest_free.py refuses to run unless FREE_MODE=true")
    removed = [k for k in list(os.environ) if k in _REMOVED_KEYS or k.startswith("OPENROUTER")]
    for k in removed:
        os.environ.pop(k, None)
    os.environ["FREE_RESEARCH_ENABLED"] = "false"
    for flag in ("PREDICTION_MARKETS_ENABLED", "RESOLUTION_SOURCE_ENABLED", "GEMINI_SEARCH_ENABLED"):
        os.environ[flag] = "false"
    return removed


# Imported by tests: no env mutation. Run as a script: sanitize before the bot imports below.
_REMOVED_AT_START: list[str] = _sanitize_environment() if __name__ == "__main__" else []

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterable, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from forecasting_tools import GeneralLlm, MetaculusApi  # noqa: E402
from forecasting_tools.data_models.binary_report import BinaryReport  # noqa: E402
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport  # noqa: E402
from forecasting_tools.data_models.numeric_report import NumericReport  # noqa: E402

from metaculus_bot import prompts as prompts_module  # noqa: E402
from metaculus_bot.ablation.window_patch import compute_mid_window_today  # noqa: E402
from metaculus_bot.api_preflight import verify_metaculus_api_identity  # noqa: E402
from metaculus_bot.backtest import question_prep  # noqa: E402
from metaculus_bot.backtest.scoring import (  # noqa: E402
    mc_log_score_from_report,
    numeric_crps_from_report,
    numeric_log_score_from_report,
)
from metaculus_bot.benchmark.bot_factory import (  # noqa: E402
    BENCHMARK_BOT_CONFIG,
    DEFAULT_HELPER_LLMS,
    create_individual_bots,
)
from metaculus_bot.benchmark.logging_setup import configure_benchmark_logging  # noqa: E402
from metaculus_bot.constants import FREE_MODE_FORECASTER_LINK_TIMEOUT_S, FREE_MODE_FORECASTER_MAX_TOKENS  # noqa: E402
from metaculus_bot.free_mode import build_free_chain  # noqa: E402
from metaculus_bot.scoring_common import binary_log_score, brier_score  # noqa: E402
from metaculus_bot.scoring_patches import apply_scoring_patches  # noqa: E402

logger: logging.Logger = logging.getLogger("backtest_free")

NO_RESEARCH_TEXT = (
    "No research report is available for this question. Forecast from the question text, the resolution "
    "criteria and your own background knowledge, anchored on base rates."
)

DEFAULT_MEMBERS: tuple[str, ...] = (
    "nvidia_nim/moonshotai/kimi-k3",
    "nvidia_nim/z-ai/glm-5.3",
    "nvidia_nim/deepseek-ai/deepseek-v4.1-flash",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
)
FLASH_MODEL = "gemini/gemini-3.8-flash"


def short_name(model: str) -> str:
    return model.split("/")[-1]


# ---------------------------------------------------------------------------------------------
# Offline aggregation + scoring (pure; unit-tested)
# ---------------------------------------------------------------------------------------------


@dataclass
class MemberForecast:
    """One member's forecast on one question, reduced to what scoring needs."""

    qtype: str  # binary | multiple_choice | numeric
    binary: float | None = None
    mc: dict[str, float] | None = None
    cdf_x: list[float] | None = None
    cdf_p: list[float] | None = None
    question: Any = None


@dataclass
class Truth:
    qtype: str
    resolution: Any  # bool | str | float | OutOfBoundsResolution


def aggregate(forecasts: Sequence[MemberForecast]) -> MemberForecast:
    """Median across members (binary / per-option MC renormalised / pointwise numeric CDF)."""
    first = forecasts[0]
    if first.qtype == "binary":
        return MemberForecast("binary", binary=float(np.median([f.binary for f in forecasts])), question=first.question)
    if first.qtype == "multiple_choice":
        options = list(first.mc or {})
        med = {o: float(np.median([(f.mc or {}).get(o, 0.0) for f in forecasts])) for o in options}
        total = sum(med.values()) or 1.0
        return MemberForecast("multiple_choice", mc={o: v / total for o, v in med.items()}, question=first.question)
    stacked = np.array([f.cdf_p for f in forecasts], dtype=float)
    return MemberForecast(
        "numeric", cdf_x=list(first.cdf_x or []), cdf_p=list(np.median(stacked, axis=0)), question=first.question
    )


def score(forecast: MemberForecast, truth: Truth) -> dict[str, float]:
    """Metric -> value. ``baseline`` is the Metaculus-style log score used as the headline
    (binary log, MC log, numeric PMF log; higher is better, uniform = 0)."""

    if forecast.qtype == "binary":
        p = 0.5 if forecast.binary is None else float(forecast.binary)
        outcome = bool(truth.resolution)
        log = binary_log_score(p, outcome)
        return {"baseline": log, "binary_log": log, "brier": brier_score(p, outcome)}
    if forecast.qtype == "multiple_choice":
        report = SimpleNamespace(
            question=forecast.question,
            prediction=SimpleNamespace(
                predicted_options=[
                    SimpleNamespace(option_name=o, probability=p) for o, p in (forecast.mc or {}).items()
                ]
            ),
        )
        log = mc_log_score_from_report(report, str(truth.resolution))  # type: ignore[arg-type]
        return {} if log is None else {"baseline": log, "mc_log": log}
    report = SimpleNamespace(
        question=forecast.question,
        prediction=SimpleNamespace(
            cdf=[
                SimpleNamespace(value=x, percentile=p)
                for x, p in zip(forecast.cdf_x or [], forecast.cdf_p or [], strict=False)
            ]
        ),
    )
    log = numeric_log_score_from_report(report, truth.resolution)  # type: ignore[arg-type]
    crps = numeric_crps_from_report(report, truth.resolution)  # type: ignore[arg-type]
    out: dict[str, float] = {}
    if log is not None:
        out["baseline"] = log
        out["numeric_log"] = log
    if crps is not None:
        out["crps"] = crps
    return out


@dataclass
class VariantResult:
    name: str
    members: list[str]
    n: int
    metrics: dict[str, float]
    per_question: dict[int, float] = field(default_factory=dict)  # qid -> baseline


def evaluate_variant(
    name: str,
    members: Sequence[str],
    forecasts: dict[str, dict[int, MemberForecast]],
    truths: dict[int, Truth],
    *,
    qids: Iterable[int] | None = None,
) -> VariantResult:
    """Score the median of ``members`` on the questions where every member answered."""
    pool = set(qids) if qids is not None else set(truths)
    common = sorted(q for q in pool if all(q in forecasts.get(m, {}) for m in members))
    sums: dict[str, list[float]] = {}
    per_q: dict[int, float] = {}
    for qid in common:
        combined = aggregate([forecasts[m][qid] for m in members])
        for metric, value in score(combined, truths[qid]).items():
            sums.setdefault(metric, []).append(value)
            if metric == "baseline":
                per_q[qid] = value
    metrics = {k: float(statistics.fmean(v)) for k, v in sums.items()}
    metrics.update({f"n_{k}": float(len(v)) for k, v in sums.items() if k != "baseline"})
    return VariantResult(name, list(members), len(common), metrics, per_q)


def paired_bootstrap(
    a: dict[int, float], b: dict[int, float], *, iters: int = 4000, seed: int = 0
) -> tuple[float, float, float, int]:
    """Mean of (a - b) over shared questions and its 90% bootstrap interval."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return float("nan"), float("nan"), float("nan"), len(shared)
    diffs = np.array([a[q] - b[q] for q in shared])
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(iters, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boots, 5)), float(np.percentile(boots, 95)), len(shared)


def variant_definitions(members: Sequence[str], member_rank: Sequence[str]) -> list[tuple[str, list[str]]]:
    """The ensembles compared offline. ``member_rank`` is best-first by mean baseline score."""
    names = list(members)
    has = set(names)
    out: list[tuple[str, list[str]]] = [(f"single:{short_name(m)}", [m]) for m in names]
    kimi, glm, dsk = (
        "nvidia_nim/moonshotai/kimi-k3",
        "nvidia_nim/z-ai/glm-5.3",
        "nvidia_nim/deepseek-ai/deepseek-v4.1-flash",
    )
    if {kimi, glm} <= has:
        # Live default leads (kimi, inkling, 3.8-flash, glm) with OpenRouter and 3.x Flash
        # unavailable to the backtest: slot 2 falls to kimi and slot 3 to glm.
        out.append(("live_proxy(kimi,kimi,glm,glm)", [kimi, kimi, glm, glm]))
    if {kimi, glm, dsk} <= has:
        out.append(("nim3(kimi,glm,deepseek)", [kimi, glm, dsk]))
    if len(names) >= 3:
        out.append(("all_members", names))
        weakest = member_rank[-1]
        out.append((f"all_minus_weakest(-{short_name(weakest)})", [m for m in names if m != weakest]))
        out.append(("top3_members", list(member_rank[:3])))
    return out


def render_report(
    *,
    variants: list[VariantResult],
    reference: str,
    deltas: dict[str, tuple[float, float, float, int]],
    meta: dict[str, Any],
) -> str:
    lines = [
        "## FREE_MODE backtest (resolved questions, no research, mid-window date)",
        "",
        f"Questions: {meta['n_questions']} ({meta['type_counts']}) from {meta['tournaments']}; "
        f"resolved after {meta['resolved_after']}. Members: {', '.join(meta['members'])}.",
        "Headline = mean Metaculus-style log score (binary log / MC log / numeric PMF log; higher is better, "
        "uniform = 0). Brier and CRPS: lower is better. Community prediction is not available from the "
        "Metaculus list API any more, so it is not reported.",
        "",
        "| Variant | n | Baseline log | Binary log | Brier | MC log | Numeric log | CRPS | Δ vs ref (90% CI) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v: float | None, nd: int = 3) -> str:
        return "-" if v is None or v != v else f"{v:.{nd}f}"

    for v in variants:
        m = v.metrics
        d = deltas.get(v.name)
        delta = (
            "ref"
            if v.name == reference
            else ("-" if d is None or d[0] != d[0] else f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] (n={d[3]})")
        )
        lines.append(
            f"| {v.name} | {v.n} | {fmt(m.get('baseline'), 1)} | {fmt(m.get('binary_log'), 1)} | {fmt(m.get('brier'))} | "
            f"{fmt(m.get('mc_log'), 1)} | {fmt(m.get('numeric_log'), 1)} | {fmt(m.get('crps'), 4)} | {delta} |"
        )
    lines += [
        "",
        f"Coverage (forecasts produced / questions): {meta['coverage']}",
        f"Reference for Δ: `{reference}`. Δ is the paired mean difference in baseline log score; an interval "
        "that spans 0 means the difference is not distinguishable from noise at this sample size.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Live part (network: Metaculus resolved questions + free LLM calls)
# ---------------------------------------------------------------------------------------------


def _member_forecast_from_report(report: Any) -> MemberForecast | None:

    if isinstance(report, BinaryReport):
        return MemberForecast("binary", binary=float(report.prediction), question=report.question)
    if isinstance(report, MultipleChoiceReport):
        mc = {o.option_name: float(o.probability) for o in report.prediction.predicted_options}
        return MemberForecast("multiple_choice", mc=mc, question=report.question)
    if isinstance(report, NumericReport):
        cdf = report.prediction.cdf
        return MemberForecast(
            "numeric",
            cdf_x=[float(p.value) for p in cdf],
            cdf_p=[float(p.percentile) for p in cdf],
            question=report.question,
        )
    return None


def _install_mid_window_dates(questions: Sequence[Any]) -> None:
    """Anchor the prompts' "Today:" (window block) and "as of <today>" lines to each question's
    mid-window. The window block is rendered just before the status-quo line inside one f-string
    with no await in between, so the window patch records the date the ``_today_str`` patch reads."""

    mids = {q.id_of_question: compute_mid_window_today(q) for q in questions}
    original_window = prompts_module._forecasting_window_str
    last: list[datetime] = [min(mids.values())]

    def _window(q: Any) -> str:
        mid = mids.get(getattr(q, "id_of_question", None))
        if mid is None:
            return original_window(q)
        last[0] = mid
        return prompts_module.forecasting_window_at(q, mid)

    def _today() -> str:
        return last[0].astimezone(UTC).strftime("%Y-%m-%d")

    prompts_module._forecasting_window_str = _window
    prompts_module._today_str = _today


def _type_name(q: Any) -> str:
    name = type(q).__name__
    if "Binary" in name:
        return "binary"
    if "MultipleChoice" in name:
        return "multiple_choice"
    return "numeric"


def _resolution_fields(q: Any) -> dict[str, Any]:
    """Resolution-related raw API fields of a RESOLVED question (fetch diagnostics only)."""
    api = getattr(q, "api_json", None) or {}
    qobj = api.get("question") if isinstance(api.get("question"), dict) else api
    if not isinstance(qobj, dict):
        return {}
    picked = {k: qobj.get(k) for k in qobj if "resol" in k.lower() or k in ("status", "type")}
    return {"post_id": getattr(q, "id_of_post", None), "question_keys": sorted(qobj)[:60], **picked}


async def _usable_from_tournament(
    slug: str, args: argparse.Namespace, diag: dict[str, Any]
) -> tuple[list[Any], dict[int, Any]]:
    """Resolved questions with ground truth for one tournament.

    When the list endpoint returns resolved questions without a ``resolution`` value (seen on
    2026-10-12 for every question of a bot tournament), each post is re-read from the individual
    question endpoint, which carries the resolution.
    """
    resolved_after_dt, resolved_before_dt = question_prep._parse_resolution_window(args.resolved_after, None)
    raw = await question_prep._fetch_with_retries(
        tournament=slug, count=args.per_tournament_cap * 3, min_forecasters=args.min_forecasters
    )
    kept, gts, skips = question_prep._filter_and_extract(raw, resolved_after_dt, resolved_before_dt)
    entry: dict[str, Any] = {"raw": len(raw), "kept_from_list": len(kept), "skips": skips}
    entry["samples"] = [_resolution_fields(q) for q in raw[:2]]
    if not kept and raw:
        refetched: list[Any] = []
        for q in raw[: args.per_tournament_cap * 2]:
            post_id = getattr(q, "id_of_post", None)
            if post_id is None:
                continue
            try:
                full = await asyncio.to_thread(MetaculusApi.get_question_by_post_id, post_id)
            except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # one unreadable post must not abort the fetch
                entry.setdefault("refetch_errors", []).append(f"{post_id}: {str(exc)[:120]}")
                continue
            if not isinstance(full, list):
                refetched.append(full)
        kept, gts, skips = question_prep._filter_and_extract(refetched, resolved_after_dt, resolved_before_dt)
        entry.update({"refetched": len(refetched), "kept_after_refetch": len(kept), "skips_after_refetch": skips})
        entry["refetch_samples"] = [_resolution_fields(q) for q in refetched[:2]]
    diag[slug] = entry
    clean = [question_prep._prepare_question_for_backtest(q) for q in kept]
    return clean, gts


async def _fetch_questions(args: argparse.Namespace) -> tuple[list[Any], dict[int, Any], list[str]]:
    questions: list[Any] = []
    truths: dict[int, Any] = {}
    used: list[str] = []
    diag: dict[str, Any] = {}
    for raw_slug in [t.strip() for t in args.tournaments.split(",") if t.strip()]:
        slug = str(MetaculusApi.CURRENT_MINIBENCH_ID) if raw_slug == "CURRENT_MINIBENCH" else raw_slug
        try:
            qs, gts = await _usable_from_tournament(slug, args, diag)
        except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # an unknown/empty tournament slug must not abort the run
            logger.warning("Tournament %s: fetch failed (%s)", slug, str(exc)[:200])
            diag[slug] = {"error": str(exc)[:300]}
            continue
        added = 0
        for q in qs:
            if q.id_of_question in truths:
                continue
            questions.append(q)
            truths[q.id_of_question] = gts[q.id_of_question]
            added += 1
        logger.info("Tournament %s: %d usable resolved questions", slug, added)
        if added:
            used.append(slug)
    _write_json("backtests/fetch_diagnostics.json", diag)
    # Stratified sample: ~half binary, quarter MC, quarter numeric (fill from what exists).
    rng = random.Random(args.seed)  # noqa: S311  # sampling, not crypto
    by_type: dict[str, list[Any]] = {}
    for q in questions:
        by_type.setdefault(_type_name(q), []).append(q)
    for lst in by_type.values():
        rng.shuffle(lst)
    n = args.num_questions
    quota = {"binary": n // 2, "multiple_choice": n // 4, "numeric": n - n // 2 - n // 4}
    chosen: list[Any] = []
    for t, k in quota.items():
        chosen += by_type.get(t, [])[:k]
    leftovers = [q for t in by_type for q in by_type[t] if q not in chosen]
    chosen += leftovers[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen, {q.id_of_question: truths[q.id_of_question] for q in chosen}, used


async def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0914, PLR0915  # one linear script

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    if any(m.startswith("openrouter/") for m in members):
        raise SystemExit("backtest_free.py never uses OpenRouter (protects the live bot's 50/day budget)")

    questions, gts, used_tournaments = await _fetch_questions(args)
    if len(questions) < 5:
        raise SystemExit(f"Only {len(questions)} resolved questions found; widen --tournaments/--resolved-after")
    truths = {qid: Truth(gt.question_type, gt.resolution) for qid, gt in gts.items()}
    type_counts: dict[str, int] = {}
    for q in questions:
        type_counts[_type_name(q)] = type_counts.get(_type_name(q), 0) + 1
    logger.info("Backtest questions: %d %s", len(questions), type_counts)

    _install_mid_window_dates(questions)
    apply_scoring_patches()
    research_cache = {q.id_of_question: NO_RESEARCH_TEXT for q in questions}

    # Optional Gemini 3.x Flash member on a capped subset (shares ~20/day with the live bot).
    member_questions: dict[str, list[Any]] = dict.fromkeys(members, questions)
    if args.flash_questions > 0:
        members.append(FLASH_MODEL)
        member_questions[FLASH_MODEL] = questions[: args.flash_questions]

    specs: list[dict[str, Any]] = []
    for m in members:
        chain: GeneralLlm = build_free_chain(
            (m, m),  # the same model twice: one retry on a transient failure, no fallback to others
            role=f"forecaster:backtest:{short_name(m)}",
            timeout=FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
            max_tokens=FREE_MODE_FORECASTER_MAX_TOKENS,
        )
        specs.append({"name": m, "forecaster": chain})
    config = dict(BENCHMARK_BOT_CONFIG)
    config["publish_reports_to_metaculus"] = False  # belt and braces: the benchmark config is already False
    config["is_benchmarking"] = True
    bots = create_individual_bots(specs, DEFAULT_HELPER_LLMS, config, batch_size=4, research_cache=research_cache)
    for bot in bots:
        assert bot.publish_reports_to_metaculus is False

    started = time.monotonic()

    async def _one(bot: Any) -> tuple[str, list[Any]]:
        qs = member_questions[bot.name]
        try:
            reports = await asyncio.wait_for(
                bot.forecast_questions(qs, return_exceptions=True), timeout=args.member_timeout_min * 60
            )
        except TimeoutError:
            logger.warning("Member %s hit its %d-minute cap", bot.name, args.member_timeout_min)
            reports = []
        return bot.name, list(reports)

    results = await asyncio.gather(*(_one(b) for b in bots))
    forecasts: dict[str, dict[int, MemberForecast]] = {}
    coverage: dict[str, str] = {}
    errors: dict[str, list[str]] = {}
    for name, reports in results:
        got: dict[int, MemberForecast] = {}
        for r in reports:
            if isinstance(r, BaseException):
                errors.setdefault(name, []).append(f"{type(r).__name__}: {str(r)[:160]}")
                continue
            mf = _member_forecast_from_report(r)
            if mf is not None:
                got[r.question.id_of_question] = mf
        forecasts[name] = got
        coverage[short_name(name)] = f"{len(got)}/{len(member_questions[name])}"

    singles = [evaluate_variant(f"single:{short_name(m)}", [m], forecasts, truths) for m in members]
    rank = [
        v.members[0]
        for v in sorted(
            (s for s in singles if s.members[0] != FLASH_MODEL and s.n), key=lambda s: -s.metrics.get("baseline", -1e9)
        )
    ]
    defs = variant_definitions([m for m in members if m != FLASH_MODEL], rank or members)
    variants = [evaluate_variant(n, ms, forecasts, truths) for n, ms in defs]
    if FLASH_MODEL in members:
        variants.append(next(s for s in singles if s.members[0] == FLASH_MODEL))
    reference = next((v.name for v in variants if v.name.startswith("live_proxy")), variants[0].name)
    ref = next(v for v in variants if v.name == reference)
    deltas = {v.name: paired_bootstrap(v.per_question, ref.per_question, seed=args.seed) for v in variants}

    meta = {
        "n_questions": len(questions),
        "type_counts": type_counts,
        "tournaments": used_tournaments,
        "resolved_after": args.resolved_after,
        "members": [short_name(m) for m in members],
        "coverage": coverage,
        "minutes": round((time.monotonic() - started) / 60, 1),
        "removed_env_keys": _REMOVED_AT_START,
    }
    report = render_report(variants=variants, reference=reference, deltas=deltas, meta=meta)
    out = {
        "meta": meta,
        "reference": reference,
        "variants": [
            {"name": v.name, "members": v.members, "n": v.n, "metrics": v.metrics, "delta_vs_ref": deltas.get(v.name)}
            for v in variants
        ],
        "errors": {k: v[:10] for k, v in errors.items()},
        "per_question": {
            str(qid): {
                "type": truths[qid].qtype,
                "baseline_by_variant": {v.name: v.per_question.get(qid) for v in variants},
            }
            for qid in truths
        },
    }
    _write_outputs(out, report)
    max(variants, key=lambda v: v.metrics.get("baseline", -1e9))
    for v in variants:
        deltas.get(v.name) or (float("nan"),) * 4
    return out


def _write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def _write_outputs(out: dict[str, Any], report: str) -> None:
    Path("backtests").mkdir(exist_ok=True)
    Path("backtests/free_backtest_results.json").write_text(json.dumps(out, indent=2, default=str))
    Path("backtests/free_backtest_report.md").write_text(report + "\n")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as fh:
            fh.write(report + "\n")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-questions", type=int, default=36)
    p.add_argument(
        "--tournaments", default="fall-futureeval-2026,summer-futureeval-2026,CURRENT_MINIBENCH,metaculus-cup-fall-2026"
    )
    p.add_argument("--resolved-after", default="2026-06-01")
    p.add_argument("--per-tournament-cap", type=int, default=60)
    p.add_argument("--min-forecasters", type=int, default=5)
    p.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    p.add_argument("--flash-questions", type=int, default=0, help="Gemini 3.8 Flash member on the first K questions")
    p.add_argument("--member-timeout-min", type=int, default=240)
    p.add_argument("--seed", type=int, default=20261012)
    return p


def main() -> None:

    configure_benchmark_logging(log_dir="backtests")
    args = _parser().parse_args()
    if args.flash_questions > 25:
        raise SystemExit("--flash-questions is capped at 25 to protect the live bot's Gemini Flash quota")
    logger.info("Removed from env for this run (names only): %s", _REMOVED_AT_START)
    verify_metaculus_api_identity()
    asyncio.run(run(args))


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.exit(main())
