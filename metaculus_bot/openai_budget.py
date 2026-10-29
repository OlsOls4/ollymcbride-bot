"""Hard-capped, opt-in OpenAI ensemble link for FREE_MODE (ollymcbride-bot fork, 2026-10-29).

The operator has a small pool of FREE OpenAI API credits (no auto-recharge, dashboard budget limit
set). This module lets ONE ensemble slot try ``openai/gpt-6.1-sol`` first while that pool lasts,
and otherwise leaves the free chain exactly as it was. It must never cause a card charge, so it is
built to fail closed:

* Off unless ``OPENAI_CAPPED_ENABLED=true`` AND ``OPENAI_CAPPED_API_KEY`` is set (fed from the repo secret OPENAI_API_KEY) AND the workflow says
  the spend ledger was loaded (``OPENAI_LEDGER_OK=true``) AND a known tier is set.
* One model only (:data:`OPENAI_CAPPED_MODEL`); priced from OpenAI's published standard rates.
* Every call reserves its WORST-CASE cost (estimated input + the full output cap) before it is
  sent. A call that fails or is cancelled keeps its full reservation (we cannot know what it cost).
* Three guards per call: a per-run cap for the workflow's tier, a lifetime cap (code ceiling
  :data:`OPENAI_LIFETIME_CAP_CEILING_USD`; the env var can only LOWER it), and a pacing curve so
  the pool lasts to the end of the Fall 2026 season. Lower-priority tiers (MiniBench, Cup, Market
  Pulse) may only use a fraction of the pacing curve, which leaves headroom for the main tournament.
* Spend is persisted across GitHub Actions runs as one JSON file per run in the ``openai-ledger``
  branch (append-only, so concurrent workflows never conflict). This run's file is rewritten on
  every reservation, so even a killed run leaves its reservations on disk for the push step.
* No retries (``allowed_tries=1``, ``max_retries=0``). A quota/auth/429 error from OpenAI disables
  the link for the rest of the run. The free chain behind it answers instead.

Batch API (50% off) is not used: its turnaround is up to 24 h, but a run must forecast and post
its comment before it ends. Prompt caching is automatic on OpenAI for repeated prefixes; cached
tokens are still charged at the full input rate here, which only over-counts.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from metaculus_bot.credit_telemetry import DIRECT_KEY_ALIAS, llm_call_metadata
from metaculus_bot.llm_retry import llm_status_code

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "OPENAI_CAPPED_MODEL",
    "CappedOpenAILink",
    "OpenAIBudget",
    "OpenAIBudgetDeclinedError",
    "build_capped_openai_link",
    "get_budget",
    "reset_openai_budget",
]

OPENAI_CAPPED_ENABLED_ENV = "OPENAI_CAPPED_ENABLED"
OPENAI_API_KEY_ENV = "OPENAI_CAPPED_API_KEY"  # NOT OPENAI_API_KEY: litellm and SDKs auto-read that one
OPENAI_BUDGET_TIER_ENV = "OPENAI_BUDGET_TIER"
OPENAI_LEDGER_DIR_ENV = "OPENAI_LEDGER_DIR"
OPENAI_LEDGER_OK_ENV = "OPENAI_LEDGER_OK"
OPENAI_LIFETIME_CAP_ENV = "OPENAI_LIFETIME_CAP_USD"

OPENAI_CAPPED_MODEL = "openai/gpt-6.1-sol"
# OpenAI standard pricing for gpt-6.1-sol (developers.openai.com/api/docs/pricing, 2026-10-29).
OPENAI_INPUT_USD_PER_TOKEN = 2.00 / 1_000_000
OPENAI_OUTPUT_USD_PER_TOKEN = 10.00 / 1_000_000
OPENAI_REASONING_EFFORT = "medium"  # never "max"
# Output cap per call, reasoning tokens included. Bounds the worst case at $0.06 of output.
OPENAI_MAX_COMPLETION_TOKENS = 6_000
# A prompt estimated above this many tokens is not sent to OpenAI at all (free chain takes it).
OPENAI_MAX_INPUT_TOKENS_EST = 40_000
OPENAI_CHARS_PER_TOKEN_EST = 3.0  # deliberately low, so the input estimate runs high
OPENAI_INPUT_ESTIMATE_MARGIN = 1.25
OPENAI_LINK_TIMEOUT_S = 300.0

# The bot's share of the credit pool is $20 to 2027-01-06. The internal cap sits below it so that
# runs of different workflows overlapping (each sees the ledger as it was when it started, at most
# one per-run cap each) still cannot reach $20.
OPENAI_LIFETIME_CAP_CEILING_USD = 17.50
OPENAI_POOL_START = dt.datetime(2026, 10, 29, tzinfo=dt.UTC)
OPENAI_POOL_END = dt.datetime(2027, 1, 7, tzinfo=dt.UTC)  # end of 6 Jan 2027 UTC; season close
OPENAI_PACING_HEADSTART_DAYS = 7.0  # spending may run this many days ahead of a straight line


@dataclass(frozen=True)
class TierPolicy:
    per_run_cap_usd: float
    pacing_fraction: float  # share of the pacing curve this tier may spend up to


# Main tournament first. Market Pulse re-forecasts every 6 h, so it gets the least.
OPENAI_TIERS: dict[str, TierPolicy] = {
    "tournament": TierPolicy(per_run_cap_usd=0.80, pacing_fraction=1.00),
    "minibench": TierPolicy(per_run_cap_usd=0.60, pacing_fraction=0.85),
    "metaculus_cup": TierPolicy(per_run_cap_usd=0.30, pacing_fraction=0.70),
    "market_pulse": TierPolicy(per_run_cap_usd=0.30, pacing_fraction=0.60),
    "test": TierPolicy(per_run_cap_usd=0.25, pacing_fraction=1.00),
}

LEDGER_FILE_PREFIX = "run-"


class OpenAIBudgetDeclinedError(RuntimeError):
    """The capped OpenAI link declined a call (cap, pacing, disabled). The free chain takes over."""


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("true", "1", "yes")


def _run_id() -> str:
    run = os.getenv("GITHUB_RUN_ID") or f"local-{os.getpid()}"
    attempt = os.getenv("GITHUB_RUN_ATTEMPT") or "1"
    return f"{run}-{attempt}"


def pacing_allowance_usd(cap: float, now: dt.datetime) -> float:
    """Cumulative spend allowed by ``now``: a straight line to ``cap`` with a head start."""
    total_days = (OPENAI_POOL_END - OPENAI_POOL_START).total_seconds() / 86400.0
    elapsed = (now - OPENAI_POOL_START).total_seconds() / 86400.0
    if elapsed < 0:
        return 0.0
    if now >= OPENAI_POOL_END:
        return 0.0  # season over: nothing left to forecast for
    return cap * min(1.0, (elapsed + OPENAI_PACING_HEADSTART_DAYS) / total_days)


class OpenAIBudget:
    """Process-wide reservation ledger for the capped OpenAI link."""

    def __init__(
        self,
        *,
        enabled: bool,
        tier: str | None,
        ledger_dir: Path | None,
        lifetime_cap_usd: float,
        now_fn: Any = None,
    ) -> None:
        self._lock = threading.Lock()
        self.tier = tier
        self.policy = OPENAI_TIERS.get(tier or "")
        self.ledger_dir = ledger_dir
        self.lifetime_cap_usd = min(lifetime_cap_usd, OPENAI_LIFETIME_CAP_CEILING_USD)
        self._now = now_fn or (lambda: dt.datetime.now(dt.UTC))
        self.run_id = _run_id()
        self.settled_usd = 0.0
        self.reserved: dict[int, float] = {}
        self._next_id = 0
        self.calls = 0
        self.declined = 0
        self.disabled_reason: str | None = None
        self.prior_usd = 0.0
        if not enabled:
            self.disabled_reason = "disabled"
        elif self.policy is None:
            self.disabled_reason = f"unknown tier {tier!r}"
        elif ledger_dir is None:
            self.disabled_reason = "no ledger"
        else:
            try:
                self.prior_usd = self._load_prior()
            except (OSError, ValueError) as exc:
                self.disabled_reason = f"ledger unreadable ({type(exc).__name__})"

    # ---- ledger ----
    def _own_file(self) -> Path:
        assert self.ledger_dir is not None
        return self.ledger_dir / f"{LEDGER_FILE_PREFIX}{self.run_id}.json"

    def _load_prior(self) -> float:
        assert self.ledger_dir is not None
        total = 0.0
        own = self._own_file().name
        for path in sorted(self.ledger_dir.glob(f"{LEDGER_FILE_PREFIX}*.json")):
            if path.name == own:
                continue
            data = json.loads(path.read_text())
            total += float(data["spent_usd"])
        return total

    def _write_own(self) -> None:
        if self.ledger_dir is None:
            return
        payload = {
            "run_id": self.run_id,
            "tier": self.tier,
            "model": OPENAI_CAPPED_MODEL,
            # Outstanding reservations count as spent: a killed run must never under-report.
            "spent_usd": round(self.settled_usd + sum(self.reserved.values()), 6),
            "calls": self.calls,
            "updated": self._now().isoformat(),
        }
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._own_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        tmp.replace(self._own_file())

    # ---- reservations ----
    @property
    def run_spent_usd(self) -> float:
        return self.settled_usd + sum(self.reserved.values())

    def limit_usd(self) -> float:
        """The cumulative (all runs) ceiling this tier may reach right now."""
        assert self.policy is not None
        return min(
            self.lifetime_cap_usd,
            pacing_allowance_usd(self.lifetime_cap_usd, self._now()) * self.policy.pacing_fraction,
        )

    def try_reserve(self, worst_case_usd: float) -> int | None:
        with self._lock:
            if self.disabled_reason is not None or self.policy is None:
                self.declined += 1
                return None
            run_after = self.run_spent_usd + worst_case_usd
            total_after = self.prior_usd + run_after
            if run_after > self.policy.per_run_cap_usd or total_after > self.limit_usd():
                self.declined += 1
                logger.info(
                    "OPENAI_BUDGET: declined tier=%s need=$%.4f run=$%.4f/%.2f total=$%.4f/%.4f",
                    self.tier,
                    worst_case_usd,
                    self.run_spent_usd,
                    self.policy.per_run_cap_usd,
                    self.prior_usd + self.run_spent_usd,
                    self.limit_usd(),
                )
                return None
            rid = self._next_id
            self._next_id += 1
            self.reserved[rid] = worst_case_usd
            self.calls += 1
            try:
                self._write_own()
            except OSError:
                # Cannot persist the reservation: refuse the call rather than spend unrecorded.
                del self.reserved[rid]
                self.calls -= 1
                self.disabled_reason = "ledger write failed"
                return None
            return rid

    def settle(self, rid: int, actual_usd: float | None) -> None:
        """Turn a reservation into spend. ``None`` (unknown cost) keeps the full reservation."""
        with self._lock:
            reserved = self.reserved.pop(rid, None)
            if reserved is None:
                return
            cost = reserved if actual_usd is None else min(max(actual_usd, 0.0), reserved * 4)
            self.settled_usd += cost
            try:
                self._write_own()
            except OSError:
                self.disabled_reason = "ledger write failed"
            logger.info(
                "OPENAI_BUDGET: settled tier=%s cost=$%.4f%s run=$%.4f total=$%.4f limit=$%.4f",
                self.tier,
                cost,
                " (unknown, full reservation)" if actual_usd is None else "",
                self.run_spent_usd,
                self.prior_usd + self.run_spent_usd,
                self.limit_usd() if self.policy else 0.0,
            )

    def disable(self, reason: str) -> None:
        with self._lock:
            if self.disabled_reason is None:
                self.disabled_reason = reason
                logger.warning("OPENAI_BUDGET: disabled for this run (%s); free chain only", reason)


_STATE: dict[str, OpenAIBudget | None] = {"budget": None}
_BUDGET_LOCK = threading.Lock()


def _lifetime_cap_from_env() -> float:
    raw = os.getenv(OPENAI_LIFETIME_CAP_ENV, "").strip()
    try:
        return float(raw) if raw else OPENAI_LIFETIME_CAP_CEILING_USD
    except ValueError:
        return 0.0  # unparsable: fail closed


def get_budget() -> OpenAIBudget:
    with _BUDGET_LOCK:
        if _STATE["budget"] is None:
            enabled = (
                _env_true(OPENAI_CAPPED_ENABLED_ENV)
                and bool(os.getenv(OPENAI_API_KEY_ENV))
                and _env_true(OPENAI_LEDGER_OK_ENV)
            )
            ledger = os.getenv(OPENAI_LEDGER_DIR_ENV, "").strip()
            budget = OpenAIBudget(
                enabled=enabled,
                tier=os.getenv(OPENAI_BUDGET_TIER_ENV, "").strip() or None,
                ledger_dir=Path(ledger) if ledger else None,
                lifetime_cap_usd=_lifetime_cap_from_env(),
            )
            logger.info(
                "OPENAI_BUDGET: init tier=%s prior=$%.4f cap=$%.2f state=%s",
                budget.tier,
                budget.prior_usd,
                budget.lifetime_cap_usd,
                budget.disabled_reason or "armed",
            )
            _STATE["budget"] = budget
        return _STATE["budget"]  # type: ignore[return-value]


def reset_openai_budget(budget: OpenAIBudget | None = None) -> None:
    """Tests: drop (or replace) the process-wide budget."""
    with _BUDGET_LOCK:
        _STATE["budget"] = budget


def capped_openai_configured() -> bool:
    """Cheap import-time check: is it even worth building the link?"""
    return _env_true(OPENAI_CAPPED_ENABLED_ENV) and bool(os.getenv(OPENAI_API_KEY_ENV))


def _prompt_chars(prompt: Any, system_prompt: str | None) -> int:
    if isinstance(prompt, str):
        n = len(prompt)
    elif isinstance(prompt, list):
        n = len(json.dumps(prompt, default=str))
    else:
        n = len(str(prompt))
    return n + len(system_prompt or "")


def worst_case_cost_usd(prompt_chars: int) -> tuple[float, int]:
    est_in = int(prompt_chars / OPENAI_CHARS_PER_TOKEN_EST * OPENAI_INPUT_ESTIMATE_MARGIN) + 50
    cost = est_in * OPENAI_INPUT_USD_PER_TOKEN + OPENAI_MAX_COMPLETION_TOKENS * OPENAI_OUTPUT_USD_PER_TOKEN
    return cost, est_in


_CURRENT_RESERVATION: contextvars.ContextVar[int | None] = contextvars.ContextVar("openai_reservation", default=None)

_DISABLING_CUES: tuple[str, ...] = ("insufficient_quota", "billing", "exceeded your current quota", "invalid_api_key")


class CappedOpenAILink(GeneralLlm):
    """A single OpenAI link that asks :class:`OpenAIBudget` before every call."""

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        budget = get_budget()
        worst, est_in = worst_case_cost_usd(_prompt_chars(prompt, system_prompt))
        if est_in > OPENAI_MAX_INPUT_TOKENS_EST:
            raise OpenAIBudgetDeclinedError(f"prompt too large for the capped link (~{est_in} tokens)")
        rid = budget.try_reserve(worst)
        if rid is None:
            raise OpenAIBudgetDeclinedError(f"capped link declined ({budget.disabled_reason or 'cap or pacing'})")
        token = _CURRENT_RESERVATION.set(rid)
        try:
            return await super().invoke(prompt, system_prompt)
        except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # classify, then re-raise
            status = llm_status_code(exc)
            msg = str(exc).lower()
            if status in (400, 401, 402, 403, 404, 429) or any(cue in msg for cue in _DISABLING_CUES):
                budget.disable(f"provider refused (status={status})")
            raise
        finally:
            _CURRENT_RESERVATION.reset(token)
            budget.settle(rid, None)  # no-op when the call already settled with its real cost

    async def _mockable_direct_call_to_model(self, prompt: Any) -> TextTokenCostResponse:  # type: ignore[override]
        response = await super()._mockable_direct_call_to_model(prompt)
        rid = _CURRENT_RESERVATION.get()
        if rid is not None:
            p = int(response.prompt_tokens_used or 0)
            c = int(response.completion_tokens_used or 0)
            actual = p * OPENAI_INPUT_USD_PER_TOKEN + c * OPENAI_OUTPUT_USD_PER_TOKEN if (p or c) else None
            get_budget().settle(rid, actual)
        return response


def build_capped_openai_link(*, role: str) -> CappedOpenAILink:
    return CappedOpenAILink(
        model=OPENAI_CAPPED_MODEL,
        api_key=os.getenv(OPENAI_API_KEY_ENV),
        timeout=OPENAI_LINK_TIMEOUT_S,
        allowed_tries=1,
        max_retries=0,
        stream=False,
        max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
        reasoning_effort=OPENAI_REASONING_EFFORT,
        metadata=llm_call_metadata(role, DIRECT_KEY_ALIAS),
    )
