"""Market Pulse mode: group expansion, the rules-based re-forecast clock, and the cli wiring.

Fixture: tests/fixtures/market_pulse_group_posts.json, two numeric group posts shaped like the
Metaculus ``/api/posts/`` list payload, with timestamps relative to the fixture's pinned ``now``.
Expected classification at that instant:

- 60001 never forecast                                     -> new
- 60002 closes in 20 h, last forecast 8 h ago              -> closing
- 60003 last forecast 80 h ago                             -> stale
- 60004 last forecast 2 h ago                              -> fresh
- 60005 subquestion status "upcoming"                      -> filtered out by forecasting-tools
- 60011 standing forecast withdrawn (end_time passed)      -> new
- 60012 closes in 30 h, last forecast 5 h ago (< 6 h)      -> fresh
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import MetaculusQuestion, NumericQuestion
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot import cli
from metaculus_bot.constants import MARKET_PULSE_TOURNAMENT_ID, MARKET_PULSE_TOURNAMENT_SLUG
from metaculus_bot.market_pulse import (
    MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN,
    MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV,
    classify_for_refresh,
    last_forecast_time,
    market_pulse_max_questions_per_run,
    recover_unreadable_forecast_history,
    select_questions_for_refresh,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "market_pulse_group_posts.json"


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def _now() -> datetime:
    return datetime.fromisoformat(_fixture()["now"].replace("Z", "+00:00"))


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = json.dumps(payload).encode()
        self.status_code = 200
        self.text = self.content.decode()

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return json.loads(self.content)


def _fetch_open_questions(posts: list[dict[str, Any]] | None = None) -> tuple[list[MetaculusQuestion], MagicMock]:
    """Run forecasting-tools' REAL tournament fetch over the fixture payload (HTTP stubbed)."""
    payload = {
        "count": 2,
        "next": None,
        "previous": None,
        "results": posts if posts is not None else _fixture()["posts"],
    }
    client = MetaculusClient(token="test-token", sleep_seconds_between_requests=0, sleep_jitter_seconds=0)
    fake_get = MagicMock(return_value=_FakeResponse(payload))
    with patch("forecasting_tools.helpers.metaculus_client.requests.get", fake_get):
        questions = client.get_all_open_questions_from_tournament(MARKET_PULSE_TOURNAMENT_ID)
    return questions, fake_get


def _by_qid(questions: list[MetaculusQuestion]) -> dict[int, MetaculusQuestion]:
    return {q.id_of_question: q for q in questions if q.id_of_question is not None}


class TestFrameworkExpandsGroups:
    def test_groups_unpack_into_open_numeric_subquestions(self) -> None:
        questions, _fake_get = _fetch_open_questions()
        by_qid = _by_qid(questions)
        # 60005 is "upcoming" inside an open post: forecasting-tools filters subquestion status locally.
        assert sorted(by_qid) == [60001, 60002, 60003, 60004, 60011, 60012]
        assert all(isinstance(q, NumericQuestion) for q in questions)
        assert {q.id_of_post for q in questions} == {50001, 50002}
        # Group-level text is copied onto each subquestion, and the label names the subquestion.
        assert by_qid[60001].resolution_criteria == "Resolves to the official close."
        assert by_qid[60001].group_question_option == "Nov 14"
        assert by_qid[60001].question_ids_of_group == [60001, 60002, 60003, 60004, 60005]

    def test_fetch_asks_for_the_tournament_with_my_forecasts(self) -> None:
        _, fake_get = _fetch_open_questions()
        params = fake_get.call_args.kwargs["params"]
        assert params["tournaments"] == [MARKET_PULSE_TOURNAMENT_ID]
        assert params["statuses"] == ["open"]
        # with_cp=true is what puts the token's own my_forecasts on the payload.
        assert params["with_cp"] == "true"
        assert fake_get.call_args.kwargs["headers"]["Authorization"] == "Token test-token"

    def test_each_subquestion_carries_its_own_forecast_history(self) -> None:
        by_qid = _by_qid(_fetch_open_questions()[0])
        now = _now()
        assert last_forecast_time(by_qid[60001], now) is None
        assert by_qid[60002].already_forecasted is True
        assert now - last_forecast_time(by_qid[60003], now) == timedelta(hours=80)


class TestRefreshClock:
    @pytest.mark.parametrize(
        ("qid", "expected"),
        [(60001, "new"), (60002, "closing"), (60003, "stale"), (60004, "fresh"), (60011, "new"), (60012, "fresh")],
    )
    def test_classification(self, qid: int, expected: str) -> None:
        by_qid = _by_qid(_fetch_open_questions()[0])
        assert classify_for_refresh(by_qid[qid], _now())[0] == expected

    def test_thresholds_are_about_72h_and_6h(self) -> None:
        question = _by_qid(_fetch_open_questions()[0])[60003]
        last = last_forecast_time(question, _now())
        assert last is not None
        # 60003 closes in 26 days: only the 72 h rule applies.
        assert classify_for_refresh(question, last + timedelta(hours=70))[0] == "fresh"
        assert classify_for_refresh(question, last + timedelta(hours=72))[0] == "stale"
        closing = _by_qid(_fetch_open_questions()[0])[60002]
        closing_last = last_forecast_time(closing, _now())
        assert closing_last is not None
        assert classify_for_refresh(closing, closing_last + timedelta(hours=4))[0] == "fresh"
        assert classify_for_refresh(closing, closing_last + timedelta(hours=6))[0] == "closing"

    def test_past_close_is_never_selected(self) -> None:
        question = _by_qid(_fetch_open_questions()[0])[60002]
        assert classify_for_refresh(question, _now() + timedelta(hours=21))[0] == "closed"

    def test_decision_ignores_forecast_values(self) -> None:
        # The clock reads timestamps only: changing the stored forecast values changes nothing.
        posts = _fixture()["posts"]
        for post in posts:
            for sub in post["group_of_questions"]["questions"]:
                for entry in [sub["my_forecasts"].get("latest"), *sub["my_forecasts"]["history"]]:
                    if entry:
                        entry["forecast_values"] = [0.0] * 201
        a = select_questions_for_refresh(_fetch_open_questions()[0], now=_now(), max_questions=50)
        b = select_questions_for_refresh(_fetch_open_questions(posts)[0], now=_now(), max_questions=50)
        assert [q.id_of_question for q in a.selected] == [q.id_of_question for q in b.selected]

    def test_selection_priority_and_cap(self) -> None:
        questions = _fetch_open_questions()[0]
        selection = select_questions_for_refresh(questions, now=_now(), max_questions=50)
        # closing first, then new by earliest close, then stale oldest-first.
        assert [q.id_of_question for q in selection.selected] == [60002, 60001, 60011, 60003]
        capped = select_questions_for_refresh(questions, now=_now(), max_questions=2)
        assert [q.id_of_question for q in capped.selected] == [60002, 60001]
        assert [q.id_of_question for q in capped.capped] == [60011, 60003]
        assert capped.marker(6) == (
            "MARKET_PULSE: seen=6 new=2 closing=1 stale=1 fresh=2 closed=0 unreadable=0 selected=2 capped=2"
        )


class TestUnreadableHistoryFailsShut:
    def _strip_my_forecasts(self) -> list[dict[str, Any]]:
        posts = _fixture()["posts"]
        for sub in posts[0]["group_of_questions"]["questions"]:
            sub.pop("my_forecasts")
        return posts

    def test_missing_my_forecasts_is_unreadable_not_new(self) -> None:
        questions = _fetch_open_questions(self._strip_my_forecasts())[0]
        selection = select_questions_for_refresh(questions, now=_now(), max_questions=50)
        assert selection.counts["unreadable"] == 4
        assert {q.id_of_question for q in selection.selected} == {60011}

    def test_recovery_refetches_each_post_once(self) -> None:
        stripped = _fetch_open_questions(self._strip_my_forecasts())[0]
        unreadable = [q for q in stripped if q.id_of_post == 50001]
        full = [q for q in _fetch_open_questions()[0] if q.id_of_post == 50001]
        refetch = MagicMock(return_value=full)
        recovered = recover_unreadable_forecast_history(unreadable, refetch)
        refetch.assert_called_once_with(50001)
        assert sorted(q.id_of_question for q in recovered) == [60001, 60002, 60003, 60004]

    def test_recovery_survives_a_failed_refetch(self) -> None:
        stripped = _fetch_open_questions(self._strip_my_forecasts())[0]
        refetch = MagicMock(side_effect=RuntimeError("boom"))
        assert recover_unreadable_forecast_history(stripped[:1], refetch) == []


class TestCap:
    def test_default_and_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, raising=False)
        assert market_pulse_max_questions_per_run() == MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN
        monkeypatch.setenv(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, "7")
        assert market_pulse_max_questions_per_run() == 7
        monkeypatch.setenv(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, "-1")
        assert market_pulse_max_questions_per_run() == MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN
        monkeypatch.setenv(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, "x")
        assert market_pulse_max_questions_per_run() == MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN


class TestCliWiring:
    def _bot(self, questions: list[MetaculusQuestion]) -> MagicMock:
        bot = MagicMock()
        bot.skip_previously_forecasted_questions = True
        bot.max_questions_per_run = 10
        bot.metaculus_client.get_all_open_questions_from_tournament.return_value = questions
        return bot

    def test_market_pulse_is_a_run_mode(self) -> None:
        assert "market_pulse" in cli.get_args(cli.RunMode)
        assert cli.persisted_tournament_id("market_pulse") == MARKET_PULSE_TOURNAMENT_SLUG
        assert cli.persisted_platform("market_pulse") == "metaculus"
        assert cli._mode_forecaster_overrides("market_pulse") == {"max_concurrent_research": 2}
        assert cli._mode_forecaster_overrides("tournament") == {}

    def test_source_updates_instead_of_skipping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, raising=False)
        bot = self._bot(_fetch_open_questions()[0])
        factory = cli._market_pulse_source(bot, None, now=_now())
        bot.metaculus_client.get_all_open_questions_from_tournament.assert_called_once_with(MARKET_PULSE_TOURNAMENT_ID)
        assert bot.skip_previously_forecasted_questions is False
        factory()
        forecast_args = bot.forecast_questions.call_args
        assert [q.id_of_question for q in forecast_args.args[0]] == [60002, 60001, 60011, 60003]
        assert forecast_args.kwargs == {"return_exceptions": True}
        assert bot.max_questions_per_run == 4

    def test_question_source_routes_market_pulse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bot = self._bot([])
        factory = cli._question_source(bot, "market_pulse", only_posts=None)
        assert bot.skip_previously_forecasted_questions is False
        factory()
        # Nothing due: the framework still runs (empty), so the per-run summary lines are emitted.
        assert bot.forecast_questions.call_args.args[0] == []

    def test_only_posts_narrows(self) -> None:
        bot = self._bot(_fetch_open_questions()[0])
        factory = cli._market_pulse_source(bot, frozenset({50002}), now=_now())
        factory()
        assert [q.id_of_question for q in bot.forecast_questions.call_args.args[0]] == [60011]

    def test_unreadable_payload_triggers_one_refetch_per_post(self) -> None:
        posts = _fixture()["posts"]
        for sub in posts[0]["group_of_questions"]["questions"]:
            sub.pop("my_forecasts")
        bot = self._bot(_fetch_open_questions(posts)[0])
        full = [q for q in _fetch_open_questions()[0] if q.id_of_post == 50001]
        bot.metaculus_client.get_question_by_post_id.return_value = full
        factory = cli._market_pulse_source(bot, None, now=_now())
        bot.metaculus_client.get_question_by_post_id.assert_called_once_with(50001, "unpack_subquestions")
        factory()
        assert [q.id_of_question for q in bot.forecast_questions.call_args.args[0]] == [60002, 60001, 60011, 60003]

    def test_tournament_dates_check_is_advisory(self) -> None:
        with patch.object(cli, "check_tournament_dates", return_value=True) as check:
            assert cli._check_tournament_dates("market_pulse") is False
        assert check.call_args.kwargs["tournament_id"] == MARKET_PULSE_TOURNAMENT_SLUG


def test_fixture_now_is_utc() -> None:
    assert _now().tzinfo is not None
    assert _now().utcoffset() == timedelta(0)
    assert _now() == datetime(2026, 11, 2, 12, tzinfo=UTC)
