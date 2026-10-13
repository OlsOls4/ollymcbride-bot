# Market Pulse Challenge 26Q4

`--mode market_pulse` forecasts the Metaculus **Market Pulse Challenge 26Q4** (tournament **33131**,
slug `market-pulse-26q4`) in FREE_MODE, at zero cost. Workflow: `.github/workflows/run_bot_on_market_pulse.yaml`.

## The tournament

- Runs 7 Oct to 31 Dec 2026. **21 Dec 2026 is the last day for prize participation.**
- Bots may enter, but only once per person, as a bot *or* as a user.
- About 12 numeric question groups, ~65 subquestions in total, opening on a rolling basis.
- **Spot scoring**: only the forecast standing at each subquestion's close counts. Bots must keep
  updating their forecasts during each question's lifetime.

## What the mode does

Unlike the other tournament modes (which pin `skip_previously_forecasted_questions=True` and never
update), Market Pulse re-forecasts on an automated, **rules-based clock**:

| Situation (per subquestion) | Action |
| --- | --- |
| never forecast, or the standing forecast was withdrawn | forecast |
| closes within ~36 h and the last forecast is more than ~6 h old | re-forecast |
| last forecast more than ~24 h old | re-forecast |
| anything else | leave it standing |

- The age comes from `my_forecasts` in the Metaculus API payload (`latest.start_time`, falling back
  to `history`). The decision reads only timestamps and close times. It **never** looks at a
  forecast's value or comment.
- Both age thresholds carry a 30-minute schedule slack (23.5 h and 5.5 h), because a forecast
  is published some minutes after its run starts. Without the slack the next 6-hourly run would
  just miss a strict threshold.
- Group questions are unpacked into subquestions by forecasting-tools
  (`get_all_open_questions_from_tournament`, `unpack_subquestions`, with a local status filter so
  upcoming or closed subquestions inside an open group are dropped). Each subquestion carries its
  own `my_forecasts`.
- **Fail-shut**: if a list payload lacks `my_forecasts`, the post is re-read once through the
  detail endpoint. A subquestion that is still unreadable is skipped (`SKIP_GUARD_UNREADABLE`),
  so an API change can never turn into re-publishing everything on every run.
- **Priority and cap**: updates for questions closing soon come first, then never-forecast
  questions (earliest close first), then stale ones (oldest first). A run takes at most
  `MARKET_PULSE_MAX_QUESTIONS_PER_RUN` questions (default 20; at a 24 h refresh, ~65 subquestions
  need ~17 per run). The rest stay due for the next run (`MARKET_PULSE_CAPPED`).
- **Rate limits**: research fan-out is 2 (the default is 6), and the FREE_MODE chains keep their
  own pacing and fallbacks.
- Every forecast posts a fresh comment with that run's reasoning, the same publish path as every
  other mode.
- Each run logs one line: `MARKET_PULSE: seen=… new=… closing=… stale=… fresh=… closed=… unreadable=… selected=… capped=…`.

## Schedule

`13 */6 * * *` (00:13, 06:13, 12:13 and 18:13 UTC), plus a backup entry at `47 */6 * * *`, in case
GitHub drops a scheduled run. A run that finds nothing due costs only the question fetch. It can
also be started by hand (`workflow_dispatch`). The workflow uses the same secrets and free env as
`run_bot_on_tournament.yaml`.

## Code and tests

- `metaculus_bot/market_pulse.py` holds the clock, the selection and the unreadable-history recovery.
- `metaculus_bot/cli.py` has `_market_pulse_source` and the `market_pulse` run mode.
- `metaculus_bot/constants.py` holds `MARKET_PULSE_TOURNAMENT_ID`, `MARKET_PULSE_TOURNAMENT_SLUG` and `MARKET_PULSE_END_DATE`.
- `tests/test_market_pulse.py` runs the framework's real group fetch over a fixture payload, then checks the clock, the priority and cap, the fail-shut path and the cli wiring.
- `tests/test_market_pulse_numeric.py` takes every fixture subquestion (closed/closed, open/open, closed/open,
  open/closed and a log-scaled group) through the numeric path. It checks the 201-point CDF (minimum step,
  maximum PMF, 0 and 1 at closed bounds, 0.001 and 0.999 at open bounds), and that publishing posts the CDF to
  the subquestion id and the comment to the post.
- Fixture: `tests/fixtures/market_pulse_group_posts.json`.
