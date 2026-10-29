# Hard-capped OpenAI link (added 2026-10-29)

Olly has a pool of **free** OpenAI API credits (auto-recharge off, dashboard budget limit set).
The bot's share is **$20 to 6 Jan 2027**. Zero cost to his card is a hard rule: never top up.

## What it does

- `openai/gpt-6.1-sol` (reasoning effort `medium`, never `max`) leads FREE_MODE ensemble **slot 4**,
  in front of that slot's normal free chain. Slot 4's free primary (NIM glm-5.3) was the one
  that timed out most.
- When the budget declines, the prompt is too large, or OpenAI errors, the free chain answers
  exactly as before. The comment says which model answered.
- Code: `metaculus_bot/openai_budget.py`; wiring: `build_free_forecaster_llms` in
  `metaculus_bot/free_mode.py`; tests: `tests/test_openai_budget.py`.

## Guards (all in code; env vars can only switch it off or lower the cap)

| Guard | Value |
| --- | --- |
| Lifetime cap | $17.50 (ceiling in code, `OPENAI_LIFETIME_CAP_USD` can only lower it). Leaves room for overlapping runs under $20. |
| Per-call worst case | output capped at 6,000 tokens (reasoning included) = $0.06, plus estimated input at $2/M. Prompts over ~40k tokens are not sent. |
| Per run | tournament $0.80, MiniBench $0.60, Metaculus Cup $0.30, Market Pulse $0.30, Test Bot $0.25 |
| Pacing | cumulative spend may not exceed a straight line from 29 Oct 2026 to 7 Jan 2027 (7-day head start). MiniBench may use 85% of that line, Cup 70%, Market Pulse 60%, so the main tournament always has headroom. |
| Retries | none (`allowed_tries=1`, `max_retries=0`). A 401/402/403/429 or quota error switches the link off for the rest of the run. |
| Failed or cancelled call | charged at its full worst-case reservation. |

Prices used: $2/M input, $10/M output (OpenAI standard pricing, 29 Oct 2026). Cached input is
counted at the full rate, which only over-counts. The Batch API (50% off) is not used: it can take
up to 24 h, and each run has to forecast and comment before it ends.

## Spend ledger

Orphan branch `openai-ledger`, one `run-<run_id>-<attempt>.json` per run. `scripts/openai_ledger.sh
load` runs before the bot (anonymous read; on any failure the link stays off for that run) and
`save` runs with `if: always()` (pushes with the job's `GITHUB_TOKEN`, `contents: write`). The bot
rewrites its run file on every reservation, so even a killed run records what it reserved.
Total so far = sum of `spent_usd` over the branch's files.

## Secrets and switches

- Repo secret **`OPENAI_API_KEY`** (Olly pastes it: Settings → Secrets and variables → Actions).
  Workflows pass it to the bot as `OPENAI_CAPPED_API_KEY`, so litellm and other SDKs never pick it
  up implicitly.
- `OPENAI_CAPPED_ENABLED: 'true'` and `OPENAI_BUDGET_TIER` in the tournament, MiniBench, Cup,
  Market Pulse and Test Bot workflows. Set `'false'` to switch it off.
- Log lines: `OPENAI_BUDGET: init|declined|settled|disabled`.
