# FREE_MODE (ollymcbride-bot fork)

Zero-cost operation: every LLM call goes to a free tier, paid research is forced off in code, and
optional free keys add capacity. With `FREE_MODE` unset the code path is upstream's.

## What runs

- **Model chains** (`metaculus_bot/constants.py: FREE_MODE_FORECASTER_CHAINS`). Each ensemble
  slot is a fallback chain, strongest first, ending on a Gemini Flash-Lite floor. Links whose
  key is missing are dropped. 4 slots by default (`FREE_MODE_ENSEMBLE_SIZE`, max 5), median
  aggregation, stacking off.
- **Free-only guard** (`free_mode.assert_free_model`, at build time): OpenRouter slugs must end in
  `:free`; Gemini, Groq and Mistral ids must be on their `FREE_MODE_*_FREE_TIER_MODELS`
  allow-lists; NVIDIA NIM is allowed; every other provider is refused.
- **Pacing and quota handling**: calls are spaced per quota bucket (`FREE_MODE_MIN_INTERVAL_S`);
  a daily (or monthly) cap 429 marks that bucket exhausted for the rest of the run.
- **Free research add-on** (`metaculus_bot/research/free_sources.py`, provider `free_sources`,
  on by default under FREE_MODE, `FREE_RESEARCH_ENABLED=false` turns it off). Each source fails
  soft and the provider never raises, so it cannot redden a run:
  - GDELT DOC 2.0 news headlines (no key), gated to one request per 5.5 s process-wide (GDELT
    asks for at most one every 5 s); a question that would queue for over 90 s skips GDELT.
  - Wikipedia lead sections for background (no key; descriptive User-Agent per Wikimedia policy).
  - Optional keyed web search, first key present wins: `TAVILY_API_KEY`, `EXA_FREE_API_KEY`,
    `SERPER_API_KEY`; capped at `FREE_WEB_SEARCH_MAX_PER_RUN` (default 40) searches per run; a
    401/402/403/429/432/433 retires that backend for the run.
  - Hard-disabled under `is_benchmarking` (live content can reveal a resolved outcome).
- Other free research as before: prediction markets, resolution-source free rungs,
  yfinance/FRED, time-series anchor. AskNews only with its (sponsored) secrets.

## Chains (live, 12 Oct 2026)

| Slot | 1st | 2nd | 3rd | 4th (optional key) | Floor |
|---|---|---|---|---|---|
| 1 | nvidia_nim/moonshotai/kimi-k3 | openrouter/thinkingmachines/inkling:free | nvidia_nim/z-ai/glm-5.3 | mistral/mistral-medium-latest | gemini/gemini-3.5-flash-lite |
| 2 | openrouter/thinkingmachines/inkling:free | nvidia_nim/moonshotai/kimi-k3 | openrouter/nvidia/nemotron-3-ultra-550b-a55b:free | groq/openai/gpt-oss-120b | gemini/gemini-3.1-flash-lite |
| 3 | gemini/gemini-3.8-flash | nvidia_nim/z-ai/glm-5.3 | openrouter/nvidia/nemotron-3-ultra-550b-a55b:free | mistral/mistral-medium-latest | gemini/gemini-3.5-flash-lite |
| 4 | nvidia_nim/z-ai/glm-5.3 | openrouter/nvidia/nemotron-3-ultra-550b-a55b:free | nvidia_nim/deepseek-ai/deepseek-v4.1-flash | groq/openai/gpt-oss-120b | gemini/gemini-3.1-flash-lite |
| 5 (opt) | openrouter/apodex/apodex-1.1-mini:free | nvidia_nim/deepseek-ai/deepseek-v4.1-flash | openrouter/dots-studio/dots-3-note-preview:free | groq/qwen/qwen3.8-27b | gemini/gemini-3.5-flash-lite |

Utility chain: gemini-3.5-flash-lite, gemini-3.1-flash-lite, nvidia_nim/google/gemma-4-31b-it,
groq/openai/gpt-oss-20b (optional), openrouter/google/gemma-4-31b-it:free.

The Groq and Mistral links only serve after the stronger links fail, and drop out while their
keys are unset, so adding them changed nothing live until those secrets exist.

## Free services checked on 12 Oct 2026

| Service | Status | Terms / limits (source) |
|---|---|---|
| GitHub Models | **Not usable** | Retired 30 Jul 2026; inference API gone (docs.github.com/en/github-models). |
| GDELT DOC 2.0 | Integrated, no key | "unlimited and unrestricted use for any academic, commercial, or governmental use of any kind without fee", cite GDELT (gdeltproject.org/about.html#termsofuse); max 1 request / 5 s. |
| Wikipedia API | Integrated, no key | Descriptive User-Agent required (foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy). |
| Google News RSS | **Skipped** | Feed is "solely for ... a personal feed reader for personal, non-commercial use. Any other use ... is expressly prohibited"; Google News terms (google.com/intl/en_us/terms_google_news.html) forbid robots. |
| Groq Free | Optional key `GROQ_API_KEY` | No card; gpt-oss-120b / gpt-oss-20b / qwen3.8-27b at 30 RPM, 1K RPD, 8K TPM, 200K TPD (console.groq.com/docs/rate-limits). 8K TPM rejects long prompts, so it is a late fallback. |
| Mistral Free mode | Optional key `MISTRAL_API_KEY` | No card (phone verification); lowest limits plus a monthly allowance; "intended for evaluation and prototyping"; free-mode traffic may be used for training unless opted out (help.mistral.ai). |
| Tavily | Optional key `TAVILY_API_KEY` | 1,000 free credits/month, no card (docs.tavily.com/documentation/api-credits). |
| Exa | Optional key `EXA_FREE_API_KEY` | Free tier $10 credits/month (+$20 signup), no payment method; requests stop at zero balance unless auto-recharge is enabled (exa.ai/docs/reference/billing). Separate name from the paid path's `EXA_API_KEY`, which bot workflows may not wire. |
| Serper | Optional key `SERPER_API_KEY` | 2,500 one-time free queries, no card (serper.dev). |
| Cerebras | **Skipped** | No permanent free tier; trial needs a payment method (inference-docs.cerebras.ai/support/rate-limits). |
| Cloudflare Workers AI | Skipped | 10,000 neurons/day free, but its strong models need a paid plan and the free ones are weak or overlap Groq (developers.cloudflare.com/workers-ai/platform/pricing). |
| Cohere trial | **Skipped** | Trial keys "are not permitted to be used for production or commercial purposes" (cohere.com/pricing). |
| Brave Search API | **Skipped** | Prepaid plans with $5/month credit; payment method expected. |
| NewsAPI | Skipped | Developer plan is development-only. |
| Jina Reader | Not added | Keyless 20 RPM reader; the repo's fetch ladder already reads pages. |

## Backtest (`backtest_free.py`, `.github/workflows/backtest_free.yaml`)

Dispatch-only. Never publishes, never uses OpenRouter (the secret is not wired and the script
strips any `OPENROUTER*` env), keeps Gemini 3.x Flash off unless `flash_questions` > 0 (capped
at 25). Leakage protections: resolved questions only, the repo's backtest prep (resolution and
background stripped, state OPEN), `is_benchmarking=True`, no live research (fixed "no research"
text), and the prompts' dates anchored to each question's mid-window. One single-model bot per
member; ensembles are scored offline from the members' forecasts (median), so variants cost
nothing extra. Results land in the step summary, `::notice` annotations and the `free-backtest`
artifact (`backtests/free_backtest_results.json`, `free_backtest_report.md`,
`fetch_diagnostics.json`, `free_research_smoke.txt`).

### First runs (12 Oct 2026): no scores yet, live config unchanged

- Runs 38302483704 and 38302957396 (dispatch, about 1 and 10 minutes) ended before any LLM call:
  **the Metaculus API returned `"resolution": null` (and `"resolution_criteria": null`) for every
  resolved question** reachable with the bot's `METACULUS_TOKEN`. That held for 180 resolved
  questions in `summer-futureeval-2026` and 2 in `metaculus-cup-fall-2026`, from both the list
  endpoint and the per-post endpoint (`backtests/fetch_diagnostics.json` in the artifact). The
  `fall-futureeval-2026` and current MiniBench slugs returned 0 resolved questions. With no
  ground truth there is nothing to score, so the chain order and members were left as they were.
- The script stops with "Only 0 resolved questions found" in that case, and no model call is made.
- (Superseded, see "Resolution access" below: the cause is Metaculus's API access policy, not a
  bot-account redaction.)
- Smoke step: Wikipedia answered from the runner. GDELT returned HTTP 429 ("one request every 5
  seconds") on its first request, because runners share IPs. GDELT now retries once after a slot;
  it stays best-effort and fails soft.

### Resolution access (cause found 12 Oct 2026, late evening)

- Metaculus changed API data access in March 2026 (github.com/Metaculus/metaculus/issues/4466,
  public post metaculus.com/notebooks/42554): an account gets question text and the
  **resolution value only for questions that account has itself predicted** (detected through
  `my_forecasts`, which needs `with_cp=true`; forecasting-tools' list call already sends it).
  Separately, a "bot benchmarking" access tier gives resolutions for about 250 resolved
  questions in project **32979**. The public website still shows "Resolved Yes/No".
- So a new personal account with no forecasts gets `resolution: null` exactly like the bot. Run
  38312527791 confirmed this with `METACULUS_BACKTEST_TOKEN` (token in use, still 180/180 null in
  summer-futureeval-2026, 2/2 in metaculus-cup-fall-2026).
- Rerun 38314063448 (same token, tournaments `32979,bot-benchmarking,summer-futureeval-2026,
  metaculus-cup-fall-2026`, resolved after 2026-01-01): project 32979 is listable (180 resolved
  posts) but every resolution is null too, so the personal account does not have the
  bot-benchmarking tier; the `bot-benchmarking` slug returns an HTTP error. No model was called,
  no scores, live chains/ensemble unchanged. The backtest was not retried again.
- Ways to get ground truth, all free: (a) the bot's own questions once they resolve (MiniBench
  questions resolve within days to weeks, so the bot token will start to see resolutions of
  questions it forecast; still a valid backtest of *other* free models on resolved questions);
  (b) ask Metaculus for the bot-benchmarking tier (project 32979) or via the non-commercial Data
  Needs Form on metaculus.com/api; (c) questions Olly's personal account has forecast, once
  resolved. Scraping the website was not used (bot protection, and terms).
- Token wiring: `backtest_free.yaml` passes `METACULUS_BACKTEST_TOKEN` (personal, non-bot,
  read-only) only to the backtest step; `backtest_free.py` swaps it into `METACULUS_TOKEN` for
  question/resolution fetches (falls back to the bot token) and never publishes. A "Require the
  read-only backtest token" step fails fast with an `::error` when the secret is empty. A test
  pins that no other workflow references the secret.

### Free key health (`free_keys_health.yaml`, dispatch-only, infra only)

One tiny call per optional provider through the bot's own code paths (Tavily/Exa/Serper
backends in `research/free_sources.py`; Groq/Mistral `/v1/models` plus one "Reply OK" chat via
litellm). Prints statuses only, touches no Metaculus question. Run 38313534110 (12 Oct 2026):

| Provider | Result |
|---|---|
| Tavily | OK (5 results) |
| Exa (free key) | OK (5 results). Exa usage draws account credits; auto-recharge settings were not touched. |
| Serper | OK (10 results) |
| Groq | OK: all 3 allow-listed ids on its model list; chat on gpt-oss-20b answered |
| Mistral | Key valid, but `mistral-large-latest` is not on its model list, and a chat on mistral-small-latest got HTTP 429 (Free-mode limits) |

Fix pushed: `mistral-large-latest` removed from the allow-list; chain slot 1's Mistral link is now
`mistral-medium-latest`. Mistral links sit just before the Flash-Lite floor, so a 429 there only
costs one fallback hop. Live tournament/MiniBench runs since the keys were added (to 23:15 BST)
had no new questions ("No forecasts attempted"), so the keyed sources have not yet been used in a
live forecast; the health run is the evidence they work from a runner. Search order in live
runs is Tavily first, then Exa, then Serper (first backend with hits wins, capped at 40
searches per run).

## ForecastBench backtest (13 Oct 2026)
Backtest data: ForecastBench (forecastbench.org), CC BY-SA 4.0

`backtest_free.py --source forecastbench` scores resolved ForecastBench market questions (2026-resolved, pool after 2026-08-15, to limit training leakage). Never publishes, zero cost.

- Run 38322653424 (first try): poor coverage from 600 s timeouts and 429s. Chain median Brier 0.224 (n=57), base rate 0.228, always-0.5 0.250, market at freeze 0.154. kimi-k3 0.085, but only on n=25.
- Run 38324248999 (rerun, same seed, 3 questions per model at a time, NVIDIA retries; Mistral and gemini-3.8-flash excluded): 60 questions (20 Polymarket, 20 Metaculus, 20 Manifold; 21 yes / 39 no), 200 min.

Brier (lower is better) and log score (higher is better), 95% bootstrap CI on Brier. dB = Brier minus live chain median.

| Variant | n | Brier [95% CI] | Log | dB vs chain median [CI] |
|---|---|---|---|---|
| Market/crowd at freeze | 60 | 0.154 [0.098, 0.216] | -0.452 | -0.072 [-0.144, -0.007] |
| Live chain mean | 60 | 0.223 [0.151, 0.301] | -0.723 | -0.003 [-0.013, +0.007] |
| gemini-3.5-flash-lite | 60 | 0.224 [0.151, 0.302] | -0.740 | -0.001 [-0.041, +0.039] |
| Live chain median | 60 | 0.225 [0.150, 0.307] | -0.752 | — |
| All members mean (>=3) | 60 | 0.227 [0.156, 0.305] | -0.734 | +0.002 [-0.015, +0.018] |
| Base rate (0.35, in-sample) | 60 | 0.228 [0.193, 0.268] | -0.647 | +0.002 [-0.058, +0.056] |
| All members median (>=3) | 60 | 0.230 [0.154, 0.316] | -0.773 | +0.005 [-0.010, +0.020] |
| gpt-oss-120b (Groq) | 20 | 0.230 [0.102, 0.381] | -0.753 | -0.014 [-0.034, +0.007] |
| kimi-k3 | 60 | 0.234 [0.157, 0.317] | -0.774 | +0.008 [-0.007, +0.027] |
| Always 0.5 | 60 | 0.250 | -0.693 | +0.025 [-0.058, +0.098] |
| gemini-3.1-flash-lite | 60 | 0.261 [0.181, 0.346] | -0.869 | +0.036 [-0.001, +0.075] |
| qwen3.8-27b (Groq) | 16 | 0.344 [0.197, 0.509] | -1.030 | +0.095 [-0.006, +0.222] |
| glm-5.3 (NIM) | 0/60 | — | — | every call hit the soft deadline |
| deepseek-v4.1-flash (NIM) | 0/28 | — | — | every call hit the soft deadline |

Findings:
- No model or ensemble variant beats the live chain median beyond its CI, and none beats the in-sample base rate. kimi-k3's 0.085 on n=25 was small-sample noise (0.234 on n=60).
- The market at freeze beats the bot clearly (CI excludes 0).
- gemini-3.1-flash-lite is the weakest member: its log score is clearly worse than the chain median (dL -0.117 [-0.233, -0.012]); its Brier CI just touches 0. Dropping it from the chain shows no measurable gain (n=20).
- Mistral returns 429 on every call with this free key.
- Groq's free 8K tokens/min cap likely makes live Groq links fail (prompt ~4.5K + 4,096 output tokens). In the backtest, qwen3.8-27b on Groq was also rejected by a 1,000 output-tokens/min (OTPM) limit ("Requested 3000").
- NIM glm-5.3 and deepseek-v4.1-flash timed out on every question, even with batching and retries.

Decision: live ordering and members unchanged (no difference is clear beyond its CI). No open-question forecasts touched.

Proposed fixes (not applied; need clearer evidence or are config-only):
1. Use the market price as a prior/anchor where the question links a market (largest, clearest gap in this backtest). Needs a design that stays within the no-human-in-the-loop rule; backtest it first.
2. Cap Groq requests at max_tokens <= 1,000 and a shorter prompt, or remove the Groq links from live chains.
3. Remove the Mistral links until the free key stops returning 429.
4. Move gemini-3.1-flash-lite below gemini-3.5-flash-lite as the floor, once a larger backtest confirms it.
5. Raise the NIM soft deadline or drop glm-5.3 / deepseek-v4.1-flash from backtests; check whether they serve live at all.

## 13 Oct changes

Code and workflow changes, pushed 13 Oct 2026:
- Inkling is deny-listed. OpenRouter returns 403 "agentic harnesses only" for it.
- No chain uses Mistral any more. The free key gets 0 RPM on medium and small.
- Groq: 2,500 output tokens and 55 s pacing, so a call fits its 8K TPM limit.
- `max_retries=0` on every link, a 900 s free-mode deadline, and a 780 s chain budget that keeps 120 s back for the Flash-Lite floors. NIM 429s back off and retry (10 s, then 30 s). NIM glm-5.3 and deepseek-v4.1-flash stream.
- New optional keyed providers: Z.ai GLM Flash (`ZAI_API_KEY`) and Cloudflare Workers AI (`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`, Free plan, which fails rather than bills). Gemma 4 runs on `GEMINI_API_KEY`.

Health run [38381214605](https://github.com/OlsOls4/ollymcbride-bot/actions/runs/38381214605) (`free_keys_health.yaml`, chain=true):

| Model | Result |
|---|---|
| Z.ai glm-4.7-flash | FAIL: 429, code 1305 "temporarily overloaded" |
| Z.ai glm-4.5-flash | not probed (the script probes 4.7-flash only) |
| Cloudflare @cf/openai/gpt-oss-120b | OK (HTTP 200) |
| Other Cloudflare models | not probed |
| AI Studio gemma-4-31b-it | FAIL: 500 InternalServerError (model list OK) |
| NIM glm-5.3-flash | OK, 51 s |
| NIM kimi-k2.6 | FAIL: 404 |
| NIM kimi-k3 | OK, 51 s |
| NIM nemotron-3-ultra | OK, 15 s |
| NIM nemotron-3-super | OK, 6 s |
| NIM glm-5.3 | OK, 26 s (bot chain path: OK, 3 s) |
| NIM deepseek-v4.1-flash | OK, 135 s (bot chain path: OK, 134 s) |
| Groq gpt-oss-120b | OK, 1 s, 3.1K prompt tokens |
| Mistral | open-mistral-nemo, ministral-8b and ministral-14b OK; small, medium and magistral-small 429 (0 RPM). None of them is in a chain. |

Swaps made after this run (one fix-up push):
- Chain 1: gemma-4-31b-it was replaced by NIM nemotron-3-super, and Z.ai glm-4.7-flash by Cloudflare gpt-oss-120b.
- Chain 2: NIM kimi-k2.6 was replaced by NIM nemotron-3-super.
- Chain 3: gemma-4-31b-it was replaced by Groq gpt-oss-120b.

Z.ai, Gemma and kimi-k2.6 are still allow-listed and can go back into a chain once a health run shows them OK.

## 13 Oct (afternoon): health re-probe, tournament run, market anchor

### Health run [38390386053](https://github.com/OlsOls4/ollymcbride-bot/actions/runs/38390386053) (`free_keys_health.yaml`, chain=true)

`scripts/free_keys_health.py` now probes every allow-listed Z.ai and Cloudflare id, both Gemma
ids, and runs every non-floor live-chain link (plus the swapped-out candidates) end to end
through `build_free_chain` with a realistic-size prompt (commit 676449d).

| Model | Tiny probe | Bot chain path (full prompt) |
|---|---|---|
| NIM kimi-k3 | OK 10 s | OK 13 s |
| NIM glm-5.3-flash | OK 25 s | OK 21 s |
| NIM nemotron-3-super | OK 10 s | OK 18 s |
| NIM nemotron-3-ultra | OK 9 s | OK 13 s |
| NIM glm-5.3 | OK 6 s | OK 6 s |
| NIM deepseek-v4.1-flash | raw stream timed out at 180 s | OK 284 s |
| Cloudflare @cf/openai/gpt-oss-120b | HTTP 200 | OK 2 s |
| Cloudflare glm-4.7-flash, nemotron-3-120b, gemma-4-26b | HTTP 200 | not in a chain |
| Groq gpt-oss-120b | OK 1 s | OK 2 s |
| Z.ai glm-4.7-flash | 429 code 1305 (overloaded) | OK 48 s |
| Z.ai glm-4.5-flash | HTTP 200 | OK 19 s |
| AI Studio gemma-4-31b-it | OK 51 s | FAIL: 500 |
| AI Studio gemma-4-26b-a4b-it | FAIL: 500 | not probed |
| NIM kimi-k2.6 | 404 | FAIL: model_unavailable |

Every link in the live chains passed end to end, so no chain was swapped. Z.ai glm-4.5-flash
passed both probes and can be considered for a tail slot; glm-4.7-flash is intermittent
(overloaded on the tiny call, OK on the chain call); Gemma 4 and kimi-k2.6 stay out.

### Tournament run 38390400790 (manual dispatch, a normal autonomous run)

Job succeeded. 0 questions returned for `fall-futureeval-2026` (all open ones already
forecast), so 0 forecasts attempted, 0 published; RUN_STATUS models attempted 0, timed out 0,
dropped 0; no model-slot errors.

### Market anchor (live from commit be081a8)
Backtest data: ForecastBench (forecastbench.org), CC BY-SA 4.0

Two runs on the same 60 resolved questions (20 Polymarket, 20 Manifold, 20 Metaculus; pinned
with `--fb-ids`): run 38324248999 (no research) and run
[38393603023](https://github.com/OlsOls4/ollymcbride-bot/actions/runs/38393603023)
(`--market-context`: each model's only research is the question's own market/crowd price at
freeze, as the live market research gives it; members kimi-k3, gemini-3.5/3.1-flash-lite, Groq
gpt-oss-120b on 20; 36 min, zero cost, no Flash, no OpenRouter). Chains simulated with the
current live chains. Paired bootstrap 95% CI on Brier (10,000 resamples); dB < 0 is better.

| Variant (n=60) | Brier | dB vs live chain, no research |
|---|---|---|
| Live chain median, no research | 0.224 | — |
| Market price alone | 0.154 | -0.071 [-0.150, +0.001] |
| Chain median, market in prompt | 0.199 | -0.025 [-0.091, +0.035] |
| Blend w=0.3, no-research model | 0.182 | -0.043 [-0.075, -0.015] |
| Blend w=0.5, no-research model | 0.163 | -0.061 [-0.110, -0.018] |
| Blend w=0.7, no-research model | 0.153 | -0.071 [-0.135, -0.015] |
| Blend w=0.3, market-in-prompt model | 0.175 | -0.049 [-0.115, +0.012] |
| Blend w=0.5, market-in-prompt model | 0.164 | -0.060 [-0.130, +0.004] |
| Blend w=0.7, market-in-prompt model | 0.157 | -0.068 [-0.142, +0.001] |

The live bot already sees the market in its research, so the closest reference is the
market-in-prompt chain. Against it the blend wins beyond noise at every weight: w=0.3 -0.024
[-0.051, -0.003], w=0.5 -0.035 [-0.075, -0.003], w=0.7 -0.042 [-0.092, -0.002] (log score
+0.128 [+0.008, +0.294] at 0.7). On Polymarket+Manifold only (the venues the live bot reads,
n=40): w=0.7 -0.065 [-0.134, -0.006]. Leave-one-out choice of w on the no-research run also
picks 0.7 (-0.069 [-0.127, -0.017]).

By venue (n=20 each, so wide CIs): the gain is mostly Manifold (blend 0.7 on the no-research
model -0.202 [-0.344, -0.079]); Polymarket shows none (+0.010 [-0.051, +0.075]; the Polymarket
price alone was worse than the chain here); Metaculus community prediction -0.021 [-0.103,
+0.038] (the live bot does not see Metaculus CP). Model-alone note: with the market in the
prompt, kimi-k3 scored 0.160 and the chain mean 0.179 vs the chain median 0.199 (dB -0.020
[-0.042, -0.002]); not acted on.

Live change (`metaculus_bot/market_anchor.py`, wired in `research/prediction_market.py` and
`forecaster.py`, tests `tests/test_market_anchor.py`):
- When the prediction-market research finds a market the ranker graded
  `same_quantity_same_date` that is open, unresolved, a plain binary price (no children, no
  scalar, no withheld price, not staleness-capped), liquid (Polymarket/Kalshi volume or open
  interest >= $10k; Manifold >= 20 bettors and non-zero volume; PredictIt never) and with a
  spread <= 10 points when bid/ask are known, the binary aggregate becomes
  `0.7 * market + 0.3 * ensemble`, clamped to 2-98%. Several qualifying markets are averaged
  if they agree within 15 points; otherwise no anchor.
- No qualifying market, a non-binary question, `is_benchmarking`, or FREE_MODE off: unchanged.
  `MARKET_ANCHOR_ENABLED=false` turns it off. Selection errors fail soft (no anchor).
- The comment gets a line under its header naming the venue, its price and link, the weight,
  the ensemble value before blending and the published value. Log marker `MARKET_ANCHOR:`.
- Caveats: n=60 and the CI upper bounds sit just below 0; the gain comes mostly from Manifold
  questions. Watch resolved MiniBench scores for anchored vs unanchored questions.

Also fixed: training-cutoff entries for the swapped-in chain links (glm-5.3-flash,
nemotron-3-super/ultra, Cloudflare gpt-oss-120b, Groq gpt-oss-20b), which
`test_live_chain_members_all_have_cutoffs` needed.

## 13 Oct (evening): no-market questions (ForecastBench dataset sources)
Backtest data: ForecastBench (forecastbench.org), CC BY-SA 4.0

The market anchor cannot help a question with no market, so the ForecastBench source now has
`--fb-kind dataset` (workflow input `fb_kind`; commits 907bc52, 06712f8, b20eba1, a99c2ae). It
covers every non-market source in forecastbench-datasets (today: acled, dbnomics, fred,
wikipedia, yfinance; new sources are picked up automatically). Same rules as the market run:
keyless raw JSON, resolved questions only, resolution on/after 2026-06-01, sample pool resolved
on/after 2026-08-15, per-member training-cutoff filter, no Metaculus token, no OpenRouter.
Each dataset question has several resolution dates; the backtest uses only the FIRST one on/after
2026-06-01 and after the due date, and skips the question if that date has not resolved (no
fall-through). The `{forecast_due_date}`/`{resolution_date}` placeholders are filled in, and
the model's only research is the question's own freeze value plus its explanation, as
ForecastBench supplies them. Cheap non-LLM baselines (no model calls): the source's YES rate and
the same series' YES rate from resolutions known on the due date (Laplace-smoothed), and naive
persistence (ACLED: Poisson with the freeze-value rate held; Wikipedia: 0.9/0.1 when an unchanged
value resolves YES/NO; prices, yields and temperatures: 0.5).

Run [38414094348](https://github.com/OlsOls4/ollymcbride-bot/actions/runs/38414094348) (42 min,
zero cost, no Flash): 60 questions, 12 from each source, seed 20261012, from a pool of 772; 23 YES
and 37 NO. Horizons are 7 days (47 questions, due 16 Aug-27 Sep) and 180 days (13 questions due
1 Mar, resolved 28 Aug). Members: kimi-k3 60/60, gemini-3.5-flash-lite 53/60,
gemini-3.1-flash-lite 60/60, Groq gpt-oss-120b 6/20. Groq hit its daily token quota, which is
shared with the live bot's Groq link. Chains were simulated with the current live chains.

Paired bootstrap 95% CI on Brier (5,000 resamples; dB < 0 is better than the live chain median):

| Variant (n=60 unless noted) | Brier [95% CI] | dB vs chain median |
|---|---|---|
| Live chain median | 0.168 [0.134, 0.203] | — |
| Chain mean | 0.163 [0.131, 0.197] | -0.004 [-0.010, +0.001] |
| Chain trimmed mean (= median with 4 slots) | 0.168 | 0.000 |
| Chain median without gpt-oss-120b | 0.157 [0.126, 0.189] | -0.010 [-0.026, +0.001] |
| Chain median without gemini-3.5-flash-lite | 0.171 | +0.003 [-0.007, +0.013] |
| All members mean (>=3, n=54) | 0.154 | -0.010 [-0.023, +0.001] |
| Shrink toward 0.5, w chosen leave-one-out | 0.168 (LOO picks w=0) | 0.000 |
| Extremize, alpha chosen leave-one-out | 0.170 (LOO picks 0.8-1.0) | +0.002 [+0.000, +0.004] |
| Shrink toward persistence, LOO w | 0.171 | +0.004 [-0.007, +0.018] |
| Shrink toward source rate, LOO w | 0.169 | +0.001 [-0.009, +0.011] |
| Shrink toward series rate, LOO w | 0.166 | -0.002 [-0.012, +0.008] |
| Baseline: persistence (text + freeze value) | 0.171 [0.136, 0.207] | +0.003 [-0.021, +0.031] |
| Baseline: source rate (prior) | 0.185 [0.146, 0.225] | +0.017 [-0.022, +0.057] |
| Baseline: series rate (prior, source-rate fallback) | 0.152 [0.115, 0.193] | -0.015 [-0.055, +0.023] |
| Baseline: base rate 0.383 (in-sample) | 0.236 [0.209, 0.268] | +0.069 [+0.034, +0.103] |
| Baseline: always 0.5 | 0.250 | +0.082 [+0.048, +0.117] |

Per source (chain median vs persistence): acled 0.028 vs 0.029 (0 of 12 YES), wikipedia 0.049
vs 0.077, dbnomics 0.246 vs 0.250, yfinance 0.242 vs 0.250, fred 0.274 vs 0.250 (the series rate
does 0.182 on FRED).

Findings:
- With the question's own data point, the chain beats the base rate and always-0.5 beyond
  noise. It ties naive persistence and the prior series rate. Where the right answer is "no
  change" (ACLED, Wikipedia) it is already near-perfect; on coin-flip series (prices,
  temperatures) it is near 0.25, as expected.
- No candidate beats the current setup beyond noise. LOO chooses no shrinkage (w=0) and
  near-zero extremizing (alpha 0.8-1.0). Mean vs median, the trimmed mean and dropping a member
  all have CIs that cross 0. A status-quo prompt nudge was not dispatched: persistence ties the
  chain, and shrinking toward it does not help (+0.004), so there was no single candidate worth
  the one extra run.
- Re-check on the earlier 60 market-source questions (current live chains): the transforms
  chosen above change nothing there either. Shrink w=0 is the identity. Extremize 0.9 gives
  -0.002 [-0.006, +0.001] with no research and -0.004 [-0.009, -0.000] with the market in the
  prompt.
- Chain mean is the most consistent direction across the three runs: dataset -0.004 [-0.010,
  +0.001], market no-research -0.004 [-0.017, +0.007], market-in-prompt -0.020 [-0.041, -0.003].
  It is not beyond noise on the no-market questions, so it is not shipped. Caveat: with only 4
  backtest members, the simulated chains repeat one member in several slots (gpt-oss-120b or
  gemini-3.5-flash-lite), so median vs mean is measured on an ensemble less diverse than the
  live one.

Live change: none. Questions without a market anchor are forecast exactly as before (chain
median, no transform). Analysis script: `nomarket_analysis.py` in the Hark project workspace (not
committed).

## 18 Oct: Timestral prior (time-series model blend)

Mistral's Timestral-28M (Apache-2.0, released 15 Oct 2026; CPU, no API key, zero quota) runs in
the GitHub runner. Code: `metaculus_bot/research/timestral_prior.py`; switch: `TIMESTRAL_PRIOR_ENABLED`
(`'true'` in the tournament, MiniBench, Metaculus Cup, Market Pulse and Test Bot workflows; off when unset).
torch + timestral are installed by a continue-on-error workflow step (not in the lockfile).

Evidence (resolved questions only; Backtest data: ForecastBench (forecastbench.org), CC BY-SA 4.0):
- Offline study, 2,892 resolved "will the series rise" questions (FRED / yfinance / dbnomics), history cut
  before each due date: Timestral 0.207 Brier vs always-50% 0.250 vs empirical rise rate 0.243.
  Strong on temperatures (0.143), small on FRED (0.227), none on stocks (0.248). Caveat: Timestral's
  post-training may overlap 2026 data, so treat the size of the edge as optimistic.
- Head-to-head, run 39127193293 (60 resolved questions, 20 per source, `--fb-extra-preds`): chain median
  0.235, Timestral 0.215, 0.7 Timestral blend 0.214, 0.3 Timestral blend 0.222. Only the 0.3 blend beat
  the chain beyond its paired 95% CI: dBrier -0.013 [-0.024, -0.002]. Hence weight 0.3.

Scope and guards (all fail open to the unchanged ensemble): numeric questions only; routed by the
time-series anchor to one series (no spreads), derivation `level`, scale 1, not a forward-max question,
and only after the anchor section passed its band and bounds guards. Horizon = native steps from the last
observation to the scheduled resolution date. Published CDF = 0.7 x ensemble + 0.3 x Timestral
(renormalised onto closed bounds, ensemble's open-tail mass kept), then the min-step sweep; a mixture that
breaks the server's step rules, or a Timestral forecast with <5% mass inside the bounds, drops the blend.
The comment states the Timestral median and 10-90% range and the 70/30 weights.

Not yet measured: the numeric (CDF) blend on resolved Metaculus questions, which the API does not expose
to our tokens. Re-check once the bot's own series-tied MiniBench questions resolve.
