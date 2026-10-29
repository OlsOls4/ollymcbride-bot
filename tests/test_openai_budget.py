"""Budget guard for the opt-in, hard-capped OpenAI forecaster link (metaculus_bot/openai_budget.py)."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from metaculus_bot import free_mode, openai_budget
from metaculus_bot.openai_budget import (
    OPENAI_LIFETIME_CAP_CEILING_USD,
    CappedOpenAILink,
    OpenAIBudget,
    OpenAIBudgetDeclinedError,
    pacing_allowance_usd,
    worst_case_cost_usd,
)

FAKE_KEY = "sk-test-DO-NOT-LEAK-1234567890"
ENV_NAMES = (
    "OPENAI_CAPPED_ENABLED",
    "OPENAI_CAPPED_API_KEY",
    "OPENAI_BUDGET_TIER",
    "OPENAI_LEDGER_DIR",
    "OPENAI_LEDGER_OK",
    "OPENAI_LIFETIME_CAP_USD",
)


def _at(month: int, day: int) -> Any:
    year = 2026 if month >= 10 else 2027
    return lambda: dt.datetime(year, month, day, 12, 0, tzinfo=dt.UTC)


def _budget(
    tmp_path: Path, *, tier: str = "tournament", now=None, cap: float = 17.5, enabled: bool = True
) -> OpenAIBudget:
    return OpenAIBudget(enabled=enabled, tier=tier, ledger_dir=tmp_path, lifetime_cap_usd=cap, now_fn=now or _at(12, 1))


def _write_prior(tmp_path: Path, run: str, spent: float) -> None:
    (tmp_path / f"run-{run}.json").write_text(json.dumps({"run_id": run, "spent_usd": spent}))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    free_mode.reset_free_mode_state()
    openai_budget.reset_openai_budget()
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    yield
    openai_budget.reset_openai_budget()


# ---- wiring ----
def test_free_chain_unchanged_when_not_opted_in(monkeypatch):
    monkeypatch.setenv("OPENAI_CAPPED_API_KEY", FAKE_KEY)  # key alone is not enough
    llms = free_mode.build_free_forecaster_llms()
    assert all(not isinstance(link, CappedOpenAILink) for llm in llms for link in llm.links)  # type: ignore[attr-defined]


def test_link_leads_exactly_one_slot_when_opted_in(monkeypatch):
    monkeypatch.setenv("OPENAI_CAPPED_ENABLED", "true")
    monkeypatch.setenv("OPENAI_CAPPED_API_KEY", FAKE_KEY)
    monkeypatch.setenv("FREE_MODE_ENSEMBLE_SIZE", "4")
    llms = free_mode.build_free_forecaster_llms()
    hits = [i for i, llm in enumerate(llms) if isinstance(llm.links[0], CappedOpenAILink)]  # type: ignore[attr-defined]
    assert hits == [3]
    assert llms[3].model == "openai/gpt-6.1-sol"
    # the free links behind it are the slot's normal chain, in order
    monkeypatch.setenv("OPENAI_CAPPED_ENABLED", "false")
    plain = free_mode.build_free_forecaster_llms()
    assert [link.model for link in llms[3].links[1:]] == [link.model for link in plain[3].links]  # type: ignore[attr-defined]
    for i in (0, 1, 2):
        assert [link.model for link in llms[i].links] == [link.model for link in plain[i].links]  # type: ignore[attr-defined]


def test_link_has_no_retries_and_capped_output(monkeypatch):
    monkeypatch.setenv("OPENAI_CAPPED_API_KEY", FAKE_KEY)
    link = openai_budget.build_capped_openai_link(role="forecaster:free4")
    kw = link.litellm_kwargs
    assert kw["max_retries"] == 0
    assert kw["max_completion_tokens"] == openai_budget.OPENAI_MAX_COMPLETION_TOKENS
    assert kw["reasoning_effort"] == "medium"
    assert link.allowed_tries == 1


# ---- gate ----
@pytest.mark.parametrize(
    "env",
    [
        {},
        {"OPENAI_CAPPED_ENABLED": "true"},
        {"OPENAI_CAPPED_ENABLED": "true", "OPENAI_CAPPED_API_KEY": FAKE_KEY},  # ledger not loaded
        {"OPENAI_CAPPED_ENABLED": "true", "OPENAI_CAPPED_API_KEY": FAKE_KEY, "OPENAI_LEDGER_OK": "true"},  # no tier
    ],
)
def test_fails_closed(monkeypatch, tmp_path, env):
    monkeypatch.setenv("OPENAI_LEDGER_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    b = openai_budget.get_budget()
    assert b.disabled_reason is not None
    assert b.try_reserve(0.01) is None


def test_armed_with_everything(monkeypatch, tmp_path):
    for k, v in {
        "OPENAI_CAPPED_ENABLED": "true",
        "OPENAI_CAPPED_API_KEY": FAKE_KEY,
        "OPENAI_LEDGER_OK": "true",
        "OPENAI_BUDGET_TIER": "tournament",
        "OPENAI_LEDGER_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(k, v)
    assert openai_budget.get_budget().disabled_reason is None


def test_env_cap_can_only_lower(monkeypatch, tmp_path):
    assert _budget(tmp_path, cap=500).lifetime_cap_usd == OPENAI_LIFETIME_CAP_CEILING_USD
    monkeypatch.setenv("OPENAI_LIFETIME_CAP_USD", "not-a-number")
    assert openai_budget._lifetime_cap_from_env() == 0.0


def test_ceiling_is_under_the_twenty_dollar_share():
    worst_overlap = sum(p.per_run_cap_usd for p in openai_budget.OPENAI_TIERS.values())
    assert OPENAI_LIFETIME_CAP_CEILING_USD + worst_overlap <= 20.0


# ---- caps ----
def test_per_run_cap(tmp_path):
    b = _budget(tmp_path, tier="market_pulse")
    n = 0
    while b.try_reserve(0.07) is not None:
        n += 1
    assert n == int(0.30 // 0.07)


def test_lifetime_cap_with_prior_runs(tmp_path):
    _write_prior(tmp_path, "1-1", 17.45)
    b = _budget(tmp_path, now=_at(1, 5))
    assert b.prior_usd == pytest.approx(17.45)
    assert b.try_reserve(0.07) is None
    assert b.try_reserve(0.04) is not None


def test_pacing_curve_and_tier_priority(tmp_path):
    early = _at(11, 1)
    allowed = pacing_allowance_usd(17.5, early())
    assert 0 < allowed < 17.5 * 0.25
    assert pacing_allowance_usd(17.5, dt.datetime(2027, 1, 8, tzinfo=dt.UTC)) == 0.0
    _write_prior(tmp_path, "1-1", allowed * 0.7)
    tournament = _budget(tmp_path, tier="tournament", now=early)
    pulse = _budget(tmp_path, tier="market_pulse", now=early)
    assert pulse.try_reserve(0.05) is None  # Market Pulse is held to 60% of the curve
    assert tournament.try_reserve(0.05) is not None


def test_unknown_cost_keeps_full_reservation_and_is_persisted(tmp_path):
    b = _budget(tmp_path)
    rid = b.try_reserve(0.10)
    assert rid is not None
    on_disk = json.loads((tmp_path / "run-999-1.json").read_text())
    assert on_disk["spent_usd"] == pytest.approx(0.10)  # reservation recorded before the call
    b.settle(rid, None)
    assert b.settled_usd == pytest.approx(0.10)
    b.settle(rid, 0.01)  # double settle is a no-op
    assert b.settled_usd == pytest.approx(0.10)


def test_own_file_not_counted_as_prior(tmp_path):
    _write_prior(tmp_path, "999-1", 5.0)
    assert _budget(tmp_path).prior_usd == 0.0


def test_corrupt_ledger_fails_closed(tmp_path):
    (tmp_path / "run-1-1.json").write_text("{not json")
    b = _budget(tmp_path)
    assert b.disabled_reason
    assert b.try_reserve(0.01) is None


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    b = _budget(tmp_path, tier="test")  # $0.25 a run

    async def one() -> bool:
        await asyncio.sleep(0)
        return b.try_reserve(0.06) is not None

    async def many() -> list[bool]:
        return await asyncio.gather(*(one() for _ in range(20)))

    granted = sum(asyncio.run(many()))
    assert granted == 4
    assert b.run_spent_usd <= 0.25


# ---- the link ----
def _armed(monkeypatch, tmp_path, tier="test") -> OpenAIBudget:
    monkeypatch.setenv("OPENAI_CAPPED_API_KEY", FAKE_KEY)
    b = _budget(tmp_path, tier=tier)
    openai_budget.reset_openai_budget(b)
    return b


def test_link_settles_real_cost(monkeypatch, tmp_path):
    b = _armed(monkeypatch, tmp_path)

    async def fake_direct(self, prompt):  # stands in for the litellm call
        return TextTokenCostResponse(
            data="Probability: 42%",
            prompt_tokens_used=1000,
            completion_tokens_used=500,
            total_tokens_used=1500,
            model="m",
            cost=0.0,
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_direct)
    link = openai_budget.build_capped_openai_link(role="forecaster:free4")
    assert asyncio.run(link.invoke("hello")) == "Probability: 42%"
    assert b.settled_usd == pytest.approx(1000 * 2e-6 + 500 * 10e-6)
    assert not b.reserved


def test_link_declines_without_calling_when_capped(monkeypatch, tmp_path):
    b = _armed(monkeypatch, tmp_path)
    b.disable("test")
    called = []

    async def fake_direct(self, prompt):
        called.append(1)
        raise AssertionError("must not be called")

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_direct)
    link = openai_budget.build_capped_openai_link(role="forecaster:free4")
    with pytest.raises(OpenAIBudgetDeclinedError):
        asyncio.run(link.invoke("hello"))
    assert not called


def test_quota_error_disables_and_chain_falls_back(monkeypatch, tmp_path, caplog):
    b = _armed(monkeypatch, tmp_path)

    class QuotaErr(Exception):
        status_code = 429

    async def fake_direct(self, prompt):
        if self.model.startswith("openai/"):
            raise QuotaErr(f"insufficient_quota: You exceeded your current quota (key {FAKE_KEY[:7]}...)")
        return TextTokenCostResponse(
            data="free answer", prompt_tokens_used=0, completion_tokens_used=0, total_tokens_used=0, model="m", cost=0.0
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_direct)
    free_link = GeneralLlm(model="gemini/gemini-3.1-flash-lite", allowed_tries=1)
    chain = free_mode.FreeModelChainLlm(
        [openai_budget.build_capped_openai_link(role="forecaster:free4"), free_link],
        role="forecaster:free4",
        annotate_fallback=True,
    )
    with caplog.at_level(logging.DEBUG):
        out = asyncio.run(chain.invoke("q"))
    assert out.endswith("free answer")
    assert "gemini-3.1-flash-lite" in out
    assert b.disabled_reason is not None
    assert b.settled_usd > 0  # unknown cost is charged at the full reservation
    assert FAKE_KEY not in caplog.text
    # a second call never reaches OpenAI
    out2 = asyncio.run(chain.invoke("q2"))
    assert out2.endswith("free answer")


def test_oversized_prompt_declined(monkeypatch, tmp_path):
    _armed(monkeypatch, tmp_path)
    link = openai_budget.build_capped_openai_link(role="forecaster:free4")
    with pytest.raises(OpenAIBudgetDeclinedError, match="too large"):
        asyncio.run(link.invoke("x" * 200_000))


def test_worst_case_estimate_is_conservative():
    cost, est_in = worst_case_cost_usd(30_000)
    assert est_in >= 30_000 / 4  # typical English is ~4 chars/token
    assert cost >= openai_budget.OPENAI_MAX_COMPLETION_TOKENS * 10e-6
