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
