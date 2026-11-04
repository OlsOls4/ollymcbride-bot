"""FREE_MODE forecaster wall timeout (4 Nov 2026 review fix 1).

Before the fix every forecaster runner handed ``invoke_with_broad_retry`` the paid-path
``FORECASTER_SOFT_DEADLINE`` (600 s), which is shorter than the free chain budget (780 s) and
the free soft deadline (900 s): a free chain whose Flash-Lite floor answered after ~660 s was
killed at 600 s and the whole slot was dropped. These tests pin the new per-mode cap and run a
binary forecast whose chain "answers at 700 s" on a scaled clock (1 simulated s = 1 ms).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm

from metaculus_bot import constants as c
from metaculus_bot import forecaster_runners
from metaculus_bot.forecaster_runners import run_binary_forecast
from metaculus_bot.llm_retry import invoke_with_broad_retry
from metaculus_bot.value_extraction import ExtractionOutcome

_SCALE = 1000.0  # simulated seconds per real second


def test_free_mode_wall_timeout_sits_between_chain_budget_and_soft_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREE_MODE", "true")
    cap = c.forecaster_wall_timeout()
    assert cap >= c.FREE_MODE_FORECASTER_CHAIN_BUDGET_S
    assert cap < c.FREE_MODE_FORECASTER_SOFT_DEADLINE_S
    assert cap > c.FORECASTER_SOFT_DEADLINE


def test_paid_path_wall_timeout_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREE_MODE", raising=False)
    assert c.forecaster_wall_timeout() == c.FORECASTER_SOFT_DEADLINE


def _binary_question() -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 4242
    q.page_url = "https://metaculus.com/questions/4242"
    q.question_text = "Will X happen?"
    return q


async def _scaled_broad_retry(make_awaitable, *, wall_timeout: float, label: str) -> str:
    # The real retry loop, with the cap it was handed converted to the scaled clock.
    return await invoke_with_broad_retry(make_awaitable, wall_timeout=wall_timeout / _SCALE, label=label)


async def _answer_at_700s(*_args, **_kwargs) -> str:
    await asyncio.sleep(700.0 / _SCALE)
    return "Floor link answered.\n\nProbability: 40%"


@pytest.mark.asyncio
async def test_free_chain_answering_at_700s_is_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREE_MODE", "true")
    forecaster_llm = GeneralLlm(model="gemini/gemini-3.5-flash-lite")
    with (
        patch.object(forecaster_runners, "invoke_with_broad_retry", new=_scaled_broad_retry),
        patch.object(forecaster_runners, "binary_prompt", return_value="prompt"),
        patch.object(forecaster_llm, "invoke", new=_answer_at_700s),
        patch.object(
            forecaster_runners,
            "extract_binary",
            new=AsyncMock(return_value=ExtractionOutcome(value=0.4, rung="block", block_present=True)),
        ),
    ):
        result = await run_binary_forecast(_binary_question(), "research", forecaster_llm, GeneralLlm(model="p"))
    assert result.prediction_value == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_paid_path_still_cuts_a_700s_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREE_MODE", raising=False)
    forecaster_llm = GeneralLlm(model="test-forecaster")
    with (
        patch.object(forecaster_runners, "invoke_with_broad_retry", new=_scaled_broad_retry),
        patch.object(forecaster_runners, "binary_prompt", return_value="prompt"),
        patch.object(forecaster_llm, "invoke", new=_answer_at_700s),
        pytest.raises(TimeoutError),
    ):
        await run_binary_forecast(_binary_question(), "research", forecaster_llm, GeneralLlm(model="p"))
