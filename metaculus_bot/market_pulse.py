"""Market Pulse Challenge (26Q4) re-forecast selection: which open questions to (re)forecast this run.

Market Pulse is spot-scored: only the forecast standing at each subquestion's close counts, and
the tournament rules require bots to keep updating during a question's lifetime. The other
tournament modes pin ``skip_previously_forecasted_questions`` on and never update; this mode
instead re-forecasts on a fixed, rules-based clock read from the API payload's ``my_forecasts``:

- never forecast (or the standing forecast was withdrawn)          -> forecast now
- closes within ~24 h and the last forecast is more than ~6 h old  -> re-forecast
- the last forecast is more than ~72 h old                         -> re-forecast (fallback)
- otherwise                                                         -> leave it standing

The decision reads ONLY timestamps (forecast age, close time). It never looks at a forecast's
value or comment, so an update is never triggered by judging the bot's own content.

Because only the forecast standing at close is scored, routine refreshes are spaced out (72 h) to
save free-tier quota, and the updates are concentrated in the final 24 h before close.

Both age thresholds sit ``MARKET_PULSE_SCHEDULE_SLACK`` under the nominal 72 h / 6 h: the
workflow fires every 6 h and a forecast is published some minutes after its run starts, so a
strict 6 h threshold would be missed by the next run and slip to 12 h. The slack absorbs that
offset (the workflow's backup cron entry covers a run that published late).

Group questions arrive already unpacked into subquestions by forecasting-tools
(``get_all_open_questions_from_tournament`` defaults to ``unpack_subquestions`` and filters the
subquestion status locally), and each subquestion carries its own ``my_forecasts``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from forecasting_tools import MetaculusQuestion
from forecasting_tools.data_models.questions import QuestionState

logger = logging.getLogger(__name__)

# Nominal thresholds from the Market Pulse update rules (docs/market-pulse.md).
MARKET_PULSE_STALE_AFTER: timedelta = timedelta(hours=72)
MARKET_PULSE_CLOSING_WINDOW: timedelta = timedelta(hours=24)
MARKET_PULSE_CLOSING_STALE_AFTER: timedelta = timedelta(hours=6)
# Subtracted from both age thresholds so a 6-hourly schedule actually meets them (module docstring).
MARKET_PULSE_SCHEDULE_SLACK: timedelta = timedelta(minutes=30)

# Per-run cap, so one run stays inside free-tier rate limits and the job's time cap. 65
# subquestions at a 72 h refresh need ~6 per 6-hourly run; 20 leaves room for new and closing updates.
MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV: str = "MARKET_PULSE_MAX_QUESTIONS_PER_RUN"
MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN: int = 20
# Research fan-out for this mode: lower than the default 6, to spread free-tier calls out.
MARKET_PULSE_MAX_CONCURRENT_RESEARCH: int = 2

RefreshReason = Literal["new", "closing", "stale", "fresh", "closed", "unreadable"]
# Lower sorts first: a closing question's update is worth most under spot scoring.
_PRIORITY: dict[str, int] = {"closing": 0, "new": 1, "stale": 2}


def market_pulse_max_questions_per_run() -> int:
    """The per-run cap, from ``MARKET_PULSE_MAX_QUESTIONS_PER_RUN`` when it is a positive int."""
    raw = os.environ.get(MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV, "").strip()
    if not raw:
        return MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV,
            raw,
            MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN,
        )
        return MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN
    if value <= 0:
        logger.warning(
            "%s=%d is not positive; using %d",
            MARKET_PULSE_MAX_QUESTIONS_PER_RUN_ENV,
            value,
            MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN,
        )
        return MARKET_PULSE_DEFAULT_MAX_QUESTIONS_PER_RUN
    return value


def _to_utc(value: Any) -> datetime | None:
    """An API timestamp (epoch seconds, ISO string or datetime) as an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def my_forecasts_payload(question: MetaculusQuestion) -> dict[str, Any] | None:
    """The question's ``my_forecasts`` block, or None when the payload does not carry it."""
    question_json = question.api_json.get("question") if isinstance(question.api_json, dict) else None
    if not isinstance(question_json, dict):
        return None
    block = question_json.get("my_forecasts")
    return block if isinstance(block, dict) else None


def last_forecast_time(question: MetaculusQuestion, now: datetime) -> datetime | None:
    """Start time of the bot's STANDING forecast, read from ``my_forecasts``; None if there is none.

    ``latest`` is preferred; ``history`` is the fallback. A forecast whose ``end_time`` has passed
    was withdrawn or superseded with nothing standing, so it does not count as standing.
    """
    block = my_forecasts_payload(question)
    if block is None:
        return None
    entries: list[dict[str, Any]] = []
    latest = block.get("latest")
    if isinstance(latest, dict):
        entries.append(latest)
    history = block.get("history")
    if isinstance(history, list):
        entries.extend(entry for entry in history if isinstance(entry, dict))
    best: datetime | None = None
    for entry in entries:
        start = _to_utc(entry.get("start_time"))
        if start is None:
            continue
        end = _to_utc(entry.get("end_time"))
        if end is not None and end <= now:
            continue
        if best is None or start > best:
            best = start
    return best


def classify_for_refresh(question: MetaculusQuestion, now: datetime) -> tuple[RefreshReason, timedelta | None]:
    """The rules-based refresh decision for one question, and the age of its standing forecast."""
    if my_forecasts_payload(question) is None:
        return "unreadable", None
    close_time = _to_utc(question.close_time)
    if close_time is not None and close_time <= now:
        return "closed", None
    if question.state is not None and question.state != QuestionState.OPEN:
        return "closed", None
    last = last_forecast_time(question, now)
    if last is None:
        return "new", None
    age = now - last
    if (
        close_time is not None
        and close_time - now <= MARKET_PULSE_CLOSING_WINDOW
        and age > MARKET_PULSE_CLOSING_STALE_AFTER - MARKET_PULSE_SCHEDULE_SLACK
    ):
        return "closing", age
    if age > MARKET_PULSE_STALE_AFTER - MARKET_PULSE_SCHEDULE_SLACK:
        return "stale", age
    return "fresh", age


@dataclass
class RefreshSelection:
    """What one run will forecast, and the counts the ``MARKET_PULSE`` log marker reports."""

    selected: list[MetaculusQuestion] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    capped: list[MetaculusQuestion] = field(default_factory=list)
    unreadable: list[MetaculusQuestion] = field(default_factory=list)

    def marker(self, seen: int) -> str:
        parts = [f"seen={seen}"]
        for key in ("new", "closing", "stale", "fresh", "closed", "unreadable"):
            parts.append(f"{key}={self.counts.get(key, 0)}")
        parts.append(f"selected={len(self.selected)}")
        parts.append(f"capped={len(self.capped)}")
        return "MARKET_PULSE: " + " ".join(parts)


def select_questions_for_refresh(
    questions: Iterable[MetaculusQuestion], *, now: datetime, max_questions: int
) -> RefreshSelection:
    """Classify every question and keep the due ones, most urgent first, up to ``max_questions``.

    Order: closing-soon updates (earliest close first), then never-forecast questions (earliest
    close first), then stale ones (oldest forecast first). Questions over the cap wait for the
    next run; they stay due, so nothing is lost but time.
    """
    selection = RefreshSelection()
    due: list[tuple[int, float, MetaculusQuestion]] = []
    for question in questions:
        reason, age = classify_for_refresh(question, now)
        selection.counts[reason] = selection.counts.get(reason, 0) + 1
        if reason == "unreadable":
            selection.unreadable.append(question)
            continue
        if reason not in _PRIORITY:
            continue
        close_time = _to_utc(question.close_time)
        if reason == "stale" and age is not None:
            secondary = -age.total_seconds()
        else:
            secondary = close_time.timestamp() if close_time is not None else float("inf")
        due.append((_PRIORITY[reason], secondary, question))
    due.sort(key=lambda item: (item[0], item[1]))
    ordered = [question for _, _, question in due]
    selection.selected = ordered[:max_questions]
    selection.capped = ordered[max_questions:]
    return selection


def recover_unreadable_forecast_history(
    unreadable: Sequence[MetaculusQuestion],
    refetch_post: Callable[[int], MetaculusQuestion | list[MetaculusQuestion]],
) -> list[MetaculusQuestion]:
    """Re-read posts whose list payload lacked ``my_forecasts``, one detail GET per post.

    Returns the subquestions that came back readable (matched on question id, so only the ones
    asked about); anything still unreadable is dropped by the caller's fail-shut guard, so a
    payload change can never turn into re-publishing every question on every run.
    """
    wanted: dict[int, set[int]] = {}
    for question in unreadable:
        if question.id_of_post is None or question.id_of_question is None:
            continue
        wanted.setdefault(question.id_of_post, set()).add(question.id_of_question)
    recovered: list[MetaculusQuestion] = []
    for post_id, question_ids in sorted(wanted.items()):
        try:
            fetched = refetch_post(post_id)
        # Boundary: one post's detail GET failing must not cost the rest of the run.
        except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # logged, post skipped
            logger.warning("MARKET_PULSE_REFETCH_FAILED: post_id=%s error=%s", post_id, type(exc).__name__)
            continue
        fetched_list = fetched if isinstance(fetched, list) else [fetched]
        for question in fetched_list:
            if question.id_of_question in question_ids and my_forecasts_payload(question) is not None:
                recovered.append(question)
    return recovered
