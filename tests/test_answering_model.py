"""Truthful attribution when a FREE_MODE fallback link answers (4 Nov 2026 review fix 4).

Seen live on MiniBench Q45950: slot 2's summary bullet and ``Model:`` line named the slot's first
choice while the text had been written by a fallback model. The comment, the summary bullet and
the MEMBER_FORECAST log line must all name the model that answered.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm

from metaculus_bot import forecaster_runners
from metaculus_bot.answering_model import answering_model, fallback_models, format_fallback_note
from metaculus_bot.comment.formatting import format_research_summary_with_models
from metaculus_bot.performance_analysis.parsing import parse_per_model_forecasts
from metaculus_bot.value_extraction import ExtractionOutcome

_PRIMARY = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
_ANSWERED = "cloudflare/@cf/nvidia/nemotron-3-120b-a12b"


def _fallback_reply(text: str = "Probability: 40%") -> str:
    return f"{format_fallback_note(_ANSWERED, _PRIMARY)}\n\n{text}"


def test_answering_model_reads_the_note_else_the_slot() -> None:
    assert answering_model(_fallback_reply(), _PRIMARY) == _ANSWERED
    assert answering_model("Probability: 40%", _PRIMARY) == _PRIMARY
    assert fallback_models(f"Model: {_ANSWERED}\n\n{_fallback_reply()}") == (_ANSWERED, _PRIMARY)


def test_pre_fix_note_wording_is_still_recognised() -> None:
    old = "_(Free-mode fallback: answered by `nvidia_nim/moonshotai/kimi-k3`; slot primary `x/inkling:free` was unavailable.)_"
    assert fallback_models(old + "\n\nbody") == ("nvidia_nim/moonshotai/kimi-k3", "x/inkling:free")


def test_summary_bullet_names_the_answering_model_and_its_slot() -> None:
    base_text = (
        "## Report 1 Summary\n"
        "### Forecasts\n"
        "*Forecaster 1*: 72.0%\n"
        "*Forecaster 2*: 40.0%\n"
        "### Research Summary\nSome research.\n"
    )
    predictions = [
        MagicMock(reasoning="Model: gemini/gemini-3.8-flash\n\nanalysis"),
        MagicMock(reasoning=f"Model: {_ANSWERED}\n\n{_fallback_reply()}"),
    ]
    result = format_research_summary_with_models(base_text, predictions, report_number=1)
    assert "*Forecaster 1 (gemini-3.8-flash)*: 72.0%" in result
    assert (
        "*Forecaster 2 (nemotron-3-120b-a12b [fallback; first choice nemotron-3-ultra-550b-a55b:free])*: 40.0%"
        in result
    )
    # The analysis parsers still read both bullets, each under its own key.
    parsed = parse_per_model_forecasts(result)
    assert len(parsed) == 2


@pytest.mark.asyncio
async def test_member_forecast_line_names_the_answering_model(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
    question = MagicMock(spec=BinaryQuestion)
    question.id_of_question = 777
    question.page_url = "https://metaculus.com/questions/777"
    slot = GeneralLlm(model=_PRIMARY)
    with (
        patch.object(forecaster_runners, "binary_prompt", return_value="prompt"),
        patch.object(slot, "invoke", new=AsyncMock(return_value=_fallback_reply())),
        patch.object(
            forecaster_runners,
            "extract_binary",
            new=AsyncMock(return_value=ExtractionOutcome(value=0.4, rung="block", block_present=True)),
        ),
    ):
        await forecaster_runners.run_binary_forecast(question, "research", slot, GeneralLlm(model="parser"))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]
    assert lines == [f"MEMBER_FORECAST: question=777 model={_ANSWERED} role=member qtype=binary raw=0.4 published=0.4"]
