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
| 1 | nvidia_nim/moonshotai/kimi-k3 | openrouter/thinkingmachines/inkling:free | nvidia_nim/z-ai/glm-5.3 | mistral/mistral-large-latest | gemini/gemini-3.5-flash-lite |
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
- Likely next step: try a token from a non-bot Metaculus account (an assumption that the API
  redacts resolutions for bot accounts; not confirmed), or a tournament whose resolutions the
  token can read. Then dispatch `backtest_free.yaml` again (defaults: 36 questions, 5 members).
- Smoke step: Wikipedia answered from the runner. GDELT returned HTTP 429 ("one request every 5
  seconds") on its first request, because runners share IPs. GDELT now retries once after a slot;
  it stays best-effort and fails soft.
