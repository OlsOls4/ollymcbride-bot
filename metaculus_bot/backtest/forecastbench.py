"""ForecastBench data for the zero-cost FREE_MODE backtest (``backtest_free.py --source forecastbench``).

Data: ForecastBench datasets, https://github.com/forecastingresearch/forecastbench-datasets
(Forecasting Research Institute; Karger et al., ICLR 2025), licensed CC BY-SA 4.0. The raw JSON
is fetched keylessly from raw.githubusercontent.com at run time; nothing from it is committed.

Two kinds of binary question, chosen with ``kind``:

* ``market``: MARKET-source questions (Metaculus, Manifold, Polymarket, INFER/RFI).
* ``dataset``: DATASET-source questions (ACLED, DBnomics, FRED, Wikipedia, yfinance, and any other
  non-market source ForecastBench adds). These carry several ``resolution_dates``; exactly one is
  used per question, deterministically the FIRST that is on or after the floor date and after the
  due date (:func:`pick_resolution_date`). If that date has not resolved yet the question is
  skipped (no fall-through to a later date, which would bias toward whatever resolves first).
  The ``{forecast_due_date}`` / ``{resolution_date}`` placeholders are filled in, and the
  question's own freeze value (``freeze_datetime_value`` + explanation) is what the model sees
  as research, as ForecastBench supplies it (:func:`freeze_context_text`).

Only questions that actually resolved (``resolved`` true, ``resolved_to`` exactly 0 or 1) on or
after a floor date are used, so free models cannot already know the outcome. Each question set
``<due>-llm.json`` is joined to its own resolution set ``<due>_resolution_set.json`` on
ForecastBench's question id, and each resolved row is keyed by (question id, resolution date).

Cheap non-LLM baselines for dataset questions (no model calls): the source's base rate and the
same series' base rate from resolutions already known on the due date (:func:`prior_rate`), and a
naive status-quo / persistence forecast from the question text and freeze value
(:func:`persistence_probability`).

Pure stdlib (no bot imports): unit-tested offline in ``tests/test_forecastbench_backtest.py``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

FB_RAW_BASE: str = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
FB_ATTRIBUTION: str = (
    "Questions and resolutions: ForecastBench (Forecasting Research Institute, "
    "https://github.com/forecastingresearch/forecastbench-datasets; Karger et al., ICLR 2025), "
    "CC BY-SA 4.0."
)
MARKET_SOURCES: frozenset[str] = frozenset({"metaculus", "manifold", "polymarket", "infer"})
# Question sets are released every 14 days on a Sunday; this is one known release date.
FB_ANCHOR_DUE_DATE: date = date(2026, 10, 11)


@dataclass(frozen=True)
class ModelCutoff:
    """A model's training-data cutoff as used for leakage filtering."""

    cutoff: date
    kind: str  # "stated" (provider's own figure) | "release-date proxy" (no cutoff published)
    source: str


# Effective cutoff per backtest member (checked 2026-10-12). Where the provider publishes no
# cutoff, the RELEASE date is used as a conservative proxy (the true cutoff is earlier, so this
# only excludes more). Questions resolved before a member's effective cutoff are excluded for
# that member.
MODEL_CUTOFFS: dict[str, ModelCutoff] = {
    "gemini/gemini-3.8-flash": ModelCutoff(
        date(2026, 3, 31), "stated", "https://deepmind.google/models/model-cards/gemini-3-8-flash/ (March 2026)"
    ),
    "gemini/gemini-3.1-flash-lite": ModelCutoff(
        date(2025, 1, 31), "stated", "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite (Jan 2025)"
    ),
    "gemini/gemini-3.5-flash-lite": ModelCutoff(
        date(2026, 7, 21),
        "release-date proxy",
        "Google's model page states none (third parties say Jan 2025); GA 2026-07-21 per "
        "https://ai.google.dev/gemini-api/docs/deprecations",
    ),
    "nvidia_nim/moonshotai/kimi-k3": ModelCutoff(
        date(2026, 7, 16), "release-date proxy", "none published; released 2026-07-16 (github.com/MoonshotAI/Kimi-K3)"
    ),
    "nvidia_nim/z-ai/glm-5.3": ModelCutoff(
        date(2026, 8, 14), "release-date proxy", "none published; released 2026-08-14 (https://z.ai/blog/glm-5.3)"
    ),
    "nvidia_nim/deepseek-ai/deepseek-v4.1-flash": ModelCutoff(
        date(2026, 9, 10),
        "release-date proxy",
        "none published; released 2026-09-10 (https://api-docs.deepseek.com/news/news260910)",
    ),
    "mistral/mistral-medium-latest": ModelCutoff(
        date(2026, 4, 29),
        "release-date proxy",
        "Medium 3.5 (the -latest alias); none published; released 2026-04-29 (huggingface.co/mistralai)",
    ),
    "groq/openai/gpt-oss-120b": ModelCutoff(
        date(2024, 6, 30), "stated", "OpenAI gpt-oss model card, arXiv 2508.10925 (June 2024)"
    ),
    "groq/openai/gpt-oss-20b": ModelCutoff(
        date(2024, 6, 30), "stated", "OpenAI gpt-oss model card, arXiv 2508.10925 (June 2024)"
    ),
    "cloudflare/@cf/openai/gpt-oss-120b": ModelCutoff(
        date(2024, 6, 30), "stated", "OpenAI gpt-oss model card, arXiv 2508.10925 (June 2024)"
    ),
    "nvidia_nim/z-ai/glm-5.3-flash": ModelCutoff(
        date(2026, 8, 26), "release-date proxy", "none published; released 2026-08-26 (https://z.ai/blog/glm-5.3-flash)"
    ),
    "nvidia_nim/nvidia/nemotron-3-super-120b-a12b": ModelCutoff(
        date(2026, 3, 11),
        "release-date proxy",
        "released 2026-03-11 (https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard)",
    ),
    "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b": ModelCutoff(
        date(2026, 6, 4),
        "release-date proxy",
        "released 2026-06-04 (https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard)",
    ),
    "groq/qwen/qwen3.8-27b": ModelCutoff(
        date(2026, 8, 15), "release-date proxy", "none published; released 2026-08-15 (huggingface.co/Qwen)"
    ),
}


def effective_cutoff(model: str) -> ModelCutoff | None:
    return MODEL_CUTOFFS.get(model)


@dataclass
class FBQuestion:
    """One resolved market-source binary ForecastBench question, as one due-date instance."""

    fb_id: str
    source: str
    due_date: date  # the question set's forecast due date: the backtest's "today"
    resolution_date: date
    outcome: bool
    question: str
    background: str
    resolution_criteria: str
    source_intro: str = ""
    url: str = ""
    open_date: date | None = None
    close_date: date | None = None
    market_value: float | None = None  # freeze_datetime_value (market / crowd probability at freeze)
    market_value_explanation: str = ""
    freeze_date: date | None = None
    kind: str = "market"  # "market" | "dataset"
    freeze_value: str = ""  # raw freeze_datetime_value (dataset questions: a number or a label)
    extra: dict[str, Any] = field(default_factory=dict)


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_fb_date(value: Any) -> date | None:
    """First YYYY-MM-DD in a ForecastBench date field ('2025-12-11', 'Thu 2026-06-11 12:59 AM BST ...')."""
    if value is None:
        return None
    m = _DATE_RE.search(str(value))
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.upper() == "N/A":
        return ""
    return _TAG_RE.sub("", s).replace("&nbsp;", " ").strip()


def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def candidate_due_dates(since: date, until: date, anchor: date = FB_ANCHOR_DUE_DATE) -> list[date]:
    """Every 14-day release date in [since, until] on the anchor's cadence (missing files are skipped)."""
    out: list[date] = []
    offset = (since - anchor).days
    k = -(-offset // 14)  # ceil
    d = anchor + timedelta(days=14 * k)
    while d <= until:
        if d >= since:
            out.append(d)
        d += timedelta(days=14)
    return out


def fetch_json(url: str, *, tries: int = 3, timeout: float = 60.0) -> Any | None:
    """GET a JSON file; None on 404 (a release that does not exist). Keyless."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ollymcbride-bot-backtest"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # fixed https host
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ForecastBench fetch failed for {url}: {last}")


def join_market_binary(
    question_set: dict[str, Any], resolution_set: dict[str, Any], *, resolved_after: date
) -> list[FBQuestion]:
    """Resolved market-source binary questions of one due date, joined on (id, resolution date)."""
    due = parse_fb_date(question_set.get("forecast_due_date"))
    if due is None:
        return []
    by_id = {str(q.get("id")): q for q in question_set.get("questions", []) if q.get("source") in MARKET_SOURCES}
    out: list[FBQuestion] = []
    seen: set[tuple[str, date]] = set()
    for r in resolution_set.get("resolutions", []):
        if r.get("source") not in MARKET_SOURCES or r.get("resolved") is not True:
            continue
        rt = r.get("resolved_to")
        if rt not in (0, 1, 0.0, 1.0) or isinstance(rt, bool):
            continue
        fb_id = str(r.get("id"))
        rdate = parse_fb_date(r.get("resolution_date"))
        q = by_id.get(fb_id)
        if q is None or rdate is None or rdate < resolved_after or rdate <= due:
            continue
        if (fb_id, rdate) in seen:
            continue
        seen.add((fb_id, rdate))
        crit = _text(q.get("resolution_criteria"))
        extra_crit = _text(q.get("market_info_resolution_criteria"))
        if extra_crit:
            crit = f"{crit}\n\n{extra_crit}" if crit else extra_crit
        out.append(
            FBQuestion(
                fb_id=fb_id,
                source=str(q.get("source")),
                due_date=due,
                resolution_date=rdate,
                outcome=float(rt) == 1.0,
                question=_text(q.get("question")),
                background=_text(q.get("background")),
                resolution_criteria=crit,
                source_intro=_text(q.get("source_intro")),
                url=str(q.get("url") or ""),
                open_date=parse_fb_date(q.get("market_info_open_datetime")),
                close_date=parse_fb_date(q.get("market_info_close_datetime")),
                market_value=_float(q.get("freeze_datetime_value")),
                market_value_explanation=_text(q.get("freeze_datetime_value_explanation")),
                freeze_date=parse_fb_date(q.get("freeze_datetime")),
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# Dataset-source questions (ACLED, DBnomics, FRED, Wikipedia, yfinance, ...)
# ---------------------------------------------------------------------------------------------

KNOWN_DATASET_SOURCES: frozenset[str] = frozenset({"acled", "dbnomics", "fred", "wikipedia", "yfinance"})


def is_dataset_source(source: Any) -> bool:
    """Any non-market ForecastBench source (new dataset sources are picked up automatically)."""
    return bool(source) and str(source) not in MARKET_SOURCES


def fill_question_dates(text: str, *, due: date, resolution: date) -> str:
    return text.replace("{forecast_due_date}", due.isoformat()).replace("{resolution_date}", resolution.isoformat())


def pick_resolution_date(dates: Iterable[Any], *, due: date, floor: date) -> date | None:
    """The first listed resolution date on/after ``floor`` and strictly after ``due`` (deterministic)."""
    parsed = sorted(d for d in (parse_fb_date(x) for x in dates or ()) if d is not None)
    return next((d for d in parsed if d >= floor and d > due), None)


def _binary_outcome(row: dict[str, Any]) -> bool | None:
    if row.get("resolved") is not True:
        return None
    rt = row.get("resolved_to")
    if isinstance(rt, bool) or rt not in (0, 1, 0.0, 1.0):
        return None
    return float(rt) == 1.0


def join_dataset_binary(
    question_set: dict[str, Any], resolution_set: dict[str, Any], *, resolved_after: date
) -> list[FBQuestion]:
    """Resolved dataset-source questions of one due date, one resolution date each (see module doc)."""
    due = parse_fb_date(question_set.get("forecast_due_date"))
    if due is None:
        return []
    res: dict[tuple[str, date], bool] = {}
    for r in resolution_set.get("resolutions", []):
        if not is_dataset_source(r.get("source")) or not isinstance(r.get("id"), str):
            continue
        rdate = parse_fb_date(r.get("resolution_date"))
        o = _binary_outcome(r)
        if rdate is not None and o is not None:
            res[(str(r["id"]), rdate)] = o
    out: list[FBQuestion] = []
    seen: set[str] = set()
    for q in question_set.get("questions", []):
        if not is_dataset_source(q.get("source")) or not isinstance(q.get("id"), str):
            continue  # combination questions carry a list id
        fb_id = str(q["id"])
        if fb_id in seen:
            continue
        rdate = pick_resolution_date(q.get("resolution_dates") or (), due=due, floor=resolved_after)
        if rdate is None or (fb_id, rdate) not in res:
            continue
        seen.add(fb_id)
        out.append(
            FBQuestion(
                fb_id=fb_id,
                source=str(q.get("source")),
                due_date=due,
                resolution_date=rdate,
                outcome=res[(fb_id, rdate)],
                question=fill_question_dates(_text(q.get("question")), due=due, resolution=rdate),
                background=_text(q.get("background")),
                resolution_criteria=fill_question_dates(_text(q.get("resolution_criteria")), due=due, resolution=rdate),
                source_intro=_text(q.get("source_intro")),
                url=str(q.get("url") or ""),
                market_value=None,  # a dataset freeze value is a level, never a probability
                market_value_explanation="",
                freeze_date=parse_fb_date(q.get("freeze_datetime")),
                kind="dataset",
                freeze_value=_text(q.get("freeze_datetime_value")),
                extra={"freeze_value_explanation": _text(q.get("freeze_datetime_value_explanation"))},
            )
        )
    return out


@dataclass(frozen=True)
class FBResolved:
    """One resolved dataset-source row (any due date): the history behind the prior-rate baselines."""

    fb_id: str
    source: str
    due_date: date
    resolution_date: date
    outcome: bool


def resolved_dataset_rows(resolution_set: dict[str, Any]) -> list[FBResolved]:
    due = parse_fb_date(resolution_set.get("forecast_due_date"))
    if due is None:
        return []
    out: list[FBResolved] = []
    for r in resolution_set.get("resolutions", []):
        if not is_dataset_source(r.get("source")) or not isinstance(r.get("id"), str):
            continue
        rdate, o = parse_fb_date(r.get("resolution_date")), _binary_outcome(r)
        if rdate is not None and o is not None:
            out.append(FBResolved(str(r["id"]), str(r["source"]), due, rdate, o))
    return out


def freeze_context_text(item: FBQuestion) -> str | None:
    """Research text carrying ONLY the question's own freeze value, as ForecastBench supplies it."""
    if item.kind != "dataset" or not item.freeze_value:
        return None
    when = (item.freeze_date or item.due_date).isoformat()
    expl = item.extra.get("freeze_value_explanation") or "Latest value of the series."
    return (
        "## Data snapshot\n"
        f"The question's own reference value, as of {when}:\n"
        f"- {expl} Value: {item.freeze_value}\n\n"
        "No other research report is available. Forecast from the question text, the resolution "
        "criteria, this value and your own background knowledge, anchored on base rates."
    )


def prior_rate(
    history: Iterable[FBResolved], *, before: date, source: str | None = None, fb_id: str | None = None
) -> float | None:
    """Laplace-smoothed YES rate, (k + 1) / (n + 2), over rows resolved strictly before ``before``
    (so known on the forecast due date), optionally restricted to one source and/or one series id.
    None when no such row exists."""
    k = n = 0
    for h in history:
        if h.resolution_date >= before:
            continue
        if source is not None and h.source != source:
            continue
        if fb_id is not None and h.fb_id != fb_id:
            continue
        n += 1
        k += int(h.outcome)
    return None if n == 0 else (k + 1) / (n + 2)


PERSISTENCE_YES: float = 0.9  # fixed, not tuned: "nothing changes" resolves YES
PERSISTENCE_NO: float = 0.1  # fixed, not tuned: "nothing changes" resolves NO
_ACLED_TENX = re.compile(r"more than ten times as many", re.I)


def _num(value: str) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def poisson_sf(threshold: float, lam: float) -> float:
    """P(N > threshold) for N ~ Poisson(lam), N an integer count."""
    if lam <= 0:
        return 0.0 if threshold >= 0 else 1.0
    k_max = math.floor(threshold)
    if k_max < 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for k in range(1, k_max + 1):
        term *= lam / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def status_quo_direction(item: FBQuestion) -> int:
    """+1 when an unchanged world resolves the question YES, -1 when NO, 0 when it says nothing
    (a continuous level that will almost surely move, e.g. a stock close)."""
    q = item.question.lower()
    if item.source == "wikipedia":
        if "as high or higher" in q or "still hold" in q:
            return 1
        if "at least 1% higher" in q:
            return -1
        if "vaccine have been developed" in q:
            return -1 if item.freeze_value.strip().lower() in ("no", "0", "false") else 1
    return 0


def persistence_probability(item: FBQuestion) -> float:
    """Naive persistence / status-quo forecast from the question text and freeze value only.

    ACLED: the 30-day count is Poisson with the freeze value's rate held constant (``ten times
    ... one plus`` questions: rate = freeze - 1, threshold = 10 * freeze). Other sources: 0.9 / 0.1
    when an unchanged value resolves YES / NO (:func:`status_quo_direction`), else 0.5 (a level
    that moves either way: price, yield, temperature).
    """
    if item.source == "acled":
        v = _num(item.freeze_value)
        if v is not None and v >= 0:
            if _ACLED_TENX.search(item.question):
                return min(0.99, max(0.01, poisson_sf(10.0 * v, max(0.0, v - 1.0))))
            return min(0.99, max(0.01, poisson_sf(v, v)))
        return 0.5
    d = status_quo_direction(item)
    return PERSISTENCE_YES if d > 0 else PERSISTENCE_NO if d < 0 else 0.5


def load_forecastbench(
    *,
    since: date,
    until: date,
    resolved_after: date,
    fetch: Callable[[str], Any | None] = fetch_json,
    kind: str = "market",
    history: list[FBResolved] | None = None,
) -> tuple[list[FBQuestion], dict[str, Any]]:
    """Fetch every question set + resolution set pair with due date in [since, until] and join them.

    ``kind``: "market" or "dataset". With ``history`` (a list), every resolved dataset row of every
    fetched resolution set is appended to it (the prior-rate baselines' input).
    """
    if kind not in ("market", "dataset"):
        raise ValueError(f"kind must be 'market' or 'dataset', got {kind!r}")
    items: list[FBQuestion] = []
    diag: dict[str, Any] = {"due_dates": {}, "kind": kind}
    for due in candidate_due_dates(since, until):
        d = due.isoformat()
        qset = fetch(f"{FB_RAW_BASE}/question_sets/{d}-llm.json")
        rset = fetch(f"{FB_RAW_BASE}/resolution_sets/{d}_resolution_set.json") if qset is not None else None
        if qset is None or rset is None:
            diag["due_dates"][d] = "missing"
            continue
        if kind == "dataset":
            joined = join_dataset_binary(qset, rset, resolved_after=resolved_after)
            if history is not None:
                history.extend(resolved_dataset_rows(rset))
        else:
            joined = join_market_binary(qset, rset, resolved_after=resolved_after)
        diag["due_dates"][d] = len(joined)
        items.extend(joined)
    return items, diag


def dedupe_by_question(items: Iterable[FBQuestion], *, seed: int) -> list[FBQuestion]:
    """One due-date instance per ForecastBench question id (a seeded random pick among its sets)."""
    groups: dict[str, list[FBQuestion]] = {}
    for it in items:
        groups.setdefault(it.fb_id, []).append(it)
    rng = random.Random(seed)  # noqa: S311  # sampling, not crypto
    out = []
    for fb_id in sorted(groups):
        options = sorted(groups[fb_id], key=lambda q: q.due_date)
        out.append(options[rng.randrange(len(options))])
    return out


def stratified_sample(items: Sequence[FBQuestion], n: int, *, seed: int) -> list[FBQuestion]:
    """About ``n`` questions, an equal share per source (shortfalls refilled from the other sources)."""
    rng = random.Random(seed)  # noqa: S311  # sampling, not crypto
    by_src: dict[str, list[FBQuestion]] = {}
    for it in sorted(items, key=lambda q: (q.source, q.fb_id)):
        by_src.setdefault(it.source, []).append(it)
    for lst in by_src.values():
        rng.shuffle(lst)
    chosen: list[FBQuestion] = []
    remaining = dict(by_src)
    budget = n
    while budget > 0 and remaining:
        share = max(1, budget // len(remaining))
        nxt: dict[str, list[FBQuestion]] = {}
        for src in sorted(remaining):
            take = remaining[src][:share]
            chosen.extend(take)
            rest = remaining[src][share:]
            if rest:
                nxt[src] = rest
        budget = n - len(chosen)
        remaining = nxt
    chosen = chosen[:n]
    rng.shuffle(chosen)
    return chosen


_MARKET_LABEL: dict[str, str] = {
    "polymarket": "Polymarket price",
    "manifold": "Manifold (play-money) price",
    "metaculus": "Metaculus community prediction",
    "infer": "INFER crowd forecast",
}


def market_context_text(item: FBQuestion) -> str | None:
    """Research text carrying ONLY the question's own market/crowd probability at freeze time.

    The live bot's prediction-market research shows a same-question market price; this mirrors
    that for the backtest (no URL, no other research). None when the item has no market value.
    """
    if item.market_value is None:
        return None
    when = (item.freeze_date or item.due_date).isoformat()
    label = _MARKET_LABEL.get(item.source, f"{item.source} crowd forecast")
    return (
        "## Prediction Market Snapshot\n"
        f"One market/crowd forecast on this same question, as of {when}:\n"
        f"- {label}: {item.market_value:.1%} probability of YES.\n\n"
        "No other research report is available. Forecast from the question text, the resolution "
        "criteria, this market price and your own background knowledge, anchored on base rates."
    )


def parse_fb_ids(spec: str) -> list[tuple[str, date | None]]:
    """``"id_prefix@YYYY-MM-DD,..."`` (the date part optional) -> [(prefix, due date or None)]."""
    out: list[tuple[str, date | None]] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        prefix, _, due = tok.partition("@")
        out.append((prefix.strip(), parse_fb_date(due) if due else None))
    return out


def select_by_ids(items: Sequence[FBQuestion], spec: str) -> list[FBQuestion]:
    """The items named by ``spec`` (see :func:`parse_fb_ids`), in spec order, one per entry.

    Pins a rerun to an earlier run's exact sample, which a re-fetch would not reproduce once
    ForecastBench publishes more resolutions. An entry with no match is skipped.
    """
    chosen: list[FBQuestion] = []
    used: set[tuple[str, date]] = set()
    for prefix, due in parse_fb_ids(spec):
        cands = sorted(
            (it for it in items if it.fb_id.startswith(prefix) and (due is None or it.due_date == due)),
            key=lambda q: (q.due_date, q.resolution_date),
        )
        pick = next((it for it in cands if (it.fb_id, it.due_date) not in used), None)
        if pick is not None:
            used.add((pick.fb_id, pick.due_date))
            chosen.append(pick)
    return chosen


def eligible_for(model: str, item: FBQuestion) -> bool:
    """False when the question resolved before the model's effective training cutoff (or the cutoff is unknown)."""
    c = effective_cutoff(model)
    return c is not None and item.resolution_date >= c.cutoff


def as_datetime(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------
# Binary scoring, chain simulation and bootstrap (pure)
# ---------------------------------------------------------------------------------------------

LOG_CLIP: float = 0.001  # log score clips p to [0.001, 0.999] so a 0/1 market price is finite


def brier(p: float, outcome: bool) -> float:
    return (float(p) - (1.0 if outcome else 0.0)) ** 2


def log_score(p: float, outcome: bool) -> float:
    """ln(probability given to what happened): 0 is perfect, ln(0.5) = -0.693 for a coin flip; higher is better."""
    q = min(1.0 - LOG_CLIP, max(LOG_CLIP, float(p)))
    return math.log(q if outcome else 1.0 - q)


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def trimmed_mean(xs: Sequence[float]) -> float:
    """Mean after dropping the single lowest and highest value (plain mean below 3 values)."""
    s = sorted(xs)
    return _mean(s[1:-1]) if len(s) >= 3 else _mean(s)


def shrink(p: float, target: float, weight: float) -> float:
    """``(1 - weight) * p + weight * target``."""
    return (1.0 - weight) * float(p) + weight * float(target)


def extremize(p: float, alpha: float) -> float:
    """Log-odds scaled by ``alpha`` (> 1 extremizes, < 1 de-extremizes); p clipped to [0.001, 0.999]."""
    q = min(1.0 - LOG_CLIP, max(LOG_CLIP, float(p)))
    z = alpha * math.log(q / (1.0 - q))
    return 1.0 / (1.0 + math.exp(-z))


AGGREGATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "median": _median,
    "mean": _mean,
    "trimmed_mean": trimmed_mean,
}


def simulate_chains(
    chains: Sequence[Sequence[str]],
    forecasts: dict[str, dict[str, float]],
    qid: str,
    *,
    drop: frozenset[str] = frozenset(),
) -> list[float] | None:
    """Each slot's forecast on ``qid``: its first link that produced one (live fallback order).

    Links with no forecast (not in the backtest, failed, quota-capped, or excluded by the cutoff
    filter) fall through to the next link, exactly as the live chain falls through. None when any
    slot has no link left.
    """
    out: list[float] = []
    for chain in chains:
        got = next((forecasts[m][qid] for m in chain if m not in drop and qid in forecasts.get(m, {})), None)
        if got is None:
            return None
        out.append(got)
    return out


def bootstrap_mean_ci(
    values: Sequence[float], *, iters: int = 5000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI."""
    vals = list(values)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)  # noqa: S311  # resampling, not crypto
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = means[int((1 - level) / 2 * iters)]
    hi = means[min(iters - 1, int((1 + level) / 2 * iters))]
    return sum(vals) / n, lo, hi


@dataclass
class BinaryVariant:
    name: str
    preds: dict[str, float]  # qid -> probability

    def scores(self, outcomes: dict[str, bool], metric: Callable[[float, bool], float]) -> dict[str, float]:
        return {q: metric(p, outcomes[q]) for q, p in self.preds.items() if q in outcomes}


def summarize_variants(
    variants: Sequence[BinaryVariant],
    outcomes: dict[str, bool],
    *,
    reference: str,
    seed: int,
    iters: int = 5000,
) -> list[dict[str, Any]]:
    """Per variant: n, Brier and log score with 95% bootstrap CIs, and the paired difference vs
    ``reference`` on shared questions (Brier: negative = better than reference; log: positive = better)."""
    ref = next((v for v in variants if v.name == reference), None)
    rows: list[dict[str, Any]] = []
    for v in variants:
        b = v.scores(outcomes, brier)
        lg = v.scores(outcomes, log_score)
        row: dict[str, Any] = {"name": v.name, "n": len(b)}
        row["brier"], row["brier_lo"], row["brier_hi"] = bootstrap_mean_ci(list(b.values()), iters=iters, seed=seed)
        row["log"], row["log_lo"], row["log_hi"] = bootstrap_mean_ci(list(lg.values()), iters=iters, seed=seed)
        if ref is not None and v.name != reference:
            rb = ref.scores(outcomes, brier)
            rl = ref.scores(outcomes, log_score)
            shared = sorted(set(b) & set(rb))
            row["n_shared"] = len(shared)
            if len(shared) >= 3:
                row["d_brier"], row["d_brier_lo"], row["d_brier_hi"] = bootstrap_mean_ci(
                    [b[q] - rb[q] for q in shared], iters=iters, seed=seed
                )
                row["d_log"], row["d_log_lo"], row["d_log_hi"] = bootstrap_mean_ci(
                    [lg[q] - rl[q] for q in shared], iters=iters, seed=seed
                )
        rows.append(row)
    return rows
