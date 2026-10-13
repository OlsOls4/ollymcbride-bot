"""Market Pulse numeric groups through the bot's numeric path in FREE_MODE: CDF shape and bounds.

Every subquestion of the fixture's two numeric group posts (tests/fixtures/market_pulse_group_posts.json)
is parsed by forecasting-tools' real group unpacking, then forecast through
``TemplateForecaster._run_forecast_on_numeric`` with the LLM and the parser mocked (the same seam
tests/test_main_numeric_routing_integration.py uses). The fixture covers all four bound shapes
(closed/closed, open/open, closed-lower/open-upper, open-lower/closed-upper) and a log-scaled group
(``zero_point`` set). The CDF that would be published must satisfy what Metaculus accepts:
``cdf_size`` points, strictly increasing by at least the minimum step, no step above the max PMF,
exactly 0 / 1 at a closed bound and at least 0.001 / at most 0.999 at an open one.
"""

from __future__ import annotations

import asyncio
import json
from itertools import pairwise
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import GeneralLlm, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericDefaults, NumericDistribution, Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from tests.test_market_pulse import _fetch_open_questions

_PCTS = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99]
# Metaculus' minimum CDF step for a continuous question (0.01 / inbound_outcome_count).
_MIN_STEP = 0.01 / 200


def _bot() -> TemplateForecaster:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": [test_llm],
        "stacker": test_llm,
        "analyzer": test_llm,
        "default": test_llm,
        "parser": test_llm,
        "researcher": test_llm,
        "summarizer": test_llm,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        llms=llms,
        is_benchmarking=True,
    )


def _values_for(question: NumericQuestion, *, spill: bool) -> list[float]:
    """Plausible increasing percentile values inside the range; ``spill`` pushes both tails past the bounds."""
    lo, hi = float(question.lower_bound), float(question.upper_bound)
    if question.zero_point is not None:
        # Log-scaled group: spread geometrically, centred on the geometric middle.
        mid = (lo * hi) ** 0.5
        factors = [0.45, 0.52, 0.6, 0.7, 0.82, 0.95, 1.0, 1.05, 1.2, 1.4, 1.6, 1.8, 2.0]
        values = [mid * f for f in factors]
    else:
        span = hi - lo
        values = [lo + span * f for f in [0.18, 0.22, 0.27, 0.33, 0.4, 0.47, 0.5, 0.53, 0.6, 0.67, 0.73, 0.78, 0.82]]
    if spill:
        # Past an OPEN bound any distance is legal; past a CLOSED bound the bot clamps a value
        # within its tolerance (bounds_clamping: 0.5% of the range here) and rejects one beyond it.
        span = hi - lo
        values[0] = lo - (0.05 if question.open_lower_bound else 0.002) * span
        values[-1] = hi + (0.05 if question.open_upper_bound else 0.002) * span
    return values


def _rationale(values: list[float]) -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in zip(_PCTS, values, strict=True)},
    }
    lines = "\n".join(f"Percentile {round(p * 100, 1)}: {v}" for p, v in zip(_PCTS, values, strict=True))
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n\n{lines}"


def _parser(values: list[float]) -> Any:
    calls = {"n": 0}

    async def _impl(_reasoning: str, _return_type: Any, *_a: Any, **_k: Any) -> Any:
        await asyncio.sleep(0)
        calls["n"] += 1
        if calls["n"] == 1:
            return OutcomeTypeResult(is_discrete_integer=False)
        return [Percentile(percentile=p, value=v) for p, v in zip(_PCTS, values, strict=True)]

    return _impl


async def _forecast(question: NumericQuestion, values: list[float]) -> NumericDistribution:
    bot = _bot()
    with (
        patch("metaculus_bot.forecaster_runners.parse_structured", new=_parser(values)),
        patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=_rationale(values))),
    ):
        pred = await bot._run_forecast_on_numeric(question, research="r", llm_to_use=GeneralLlm(model="test-model"))
    assert isinstance(pred.prediction_value, NumericDistribution)
    return pred.prediction_value


def _subquestions() -> list[NumericQuestion]:
    questions = [q for q in _fetch_open_questions()[0] if isinstance(q, NumericQuestion)]
    assert len(questions) == 6
    return questions


def _assert_publishable(question: NumericQuestion, dist: NumericDistribution) -> None:
    cdf = [p.percentile for p in dist.get_cdf()]
    assert len(cdf) == question.cdf_size == 201
    steps = [b - a for a, b in pairwise(cdf)]
    assert min(steps) >= _MIN_STEP - 1e-12, f"q{question.id_of_question}: step {min(steps)} under the minimum"
    assert max(steps) <= NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False) + 1e-9
    if question.open_lower_bound:
        assert cdf[0] >= 0.001 - 1e-9
    else:
        assert cdf[0] == pytest.approx(0.0, abs=1e-9)
    if question.open_upper_bound:
        assert cdf[-1] <= 0.999 + 1e-9
    else:
        assert cdf[-1] == pytest.approx(1.0, abs=1e-9)


def test_fixture_covers_every_bound_shape_and_log_scaling() -> None:
    questions = _subquestions()
    shapes = {(q.open_lower_bound, q.open_upper_bound) for q in questions}
    assert shapes == {(False, False), (True, True), (False, True), (True, False)}
    assert any(q.zero_point is not None for q in questions)
    for q in questions:
        assert q.lower_bound < q.upper_bound
        assert q.cdf_size == 201


@pytest.mark.asyncio
@pytest.mark.parametrize("spill", [False, True], ids=["inside-range", "tails-past-bounds"])
@pytest.mark.parametrize("index", range(6))
async def test_every_subquestion_yields_a_publishable_cdf(
    index: int, spill: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FREE_MODE", "true")
    question = _subquestions()[index]
    dist = await _forecast(question, _values_for(question, spill=spill))
    _assert_publishable(question, dist)


@pytest.mark.asyncio
async def test_publish_posts_the_cdf_to_the_subquestion_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumericReport publish seam posts a 201-point CDF to the SUBquestion id, then a comment on the post."""
    from forecasting_tools import NumericReport

    monkeypatch.setenv("FREE_MODE", "true")
    question = next(q for q in _subquestions() if q.id_of_question == 60003)
    dist = await _forecast(question, _values_for(question, spill=False))
    report = NumericReport(question=question, prediction=dist, explanation="# Summary\nreasoning text\n")
    client = MagicMock()
    await report.publish_report_to_metaculus(client)
    q_id, cdf = client.post_numeric_question_prediction.call_args.args[:2]
    assert q_id == 60003
    assert len(cdf) == 201
    post_id, comment = client.post_question_comment.call_args.args[:2]
    assert post_id == 50001
    assert "reasoning text" in comment
