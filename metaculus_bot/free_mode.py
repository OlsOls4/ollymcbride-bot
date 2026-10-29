"""FREE_MODE: zero-cost LLM fallback chains for the ollymcbride-bot fork.

With ``FREE_MODE=true`` every LLM the bot builds is a :class:`FreeModelChainLlm`: an ordered list
of free-tier models (OpenRouter ``:free`` variants, Google AI Studio free-tier Gemini, optional
NVIDIA NIM free endpoints, and optional Groq Free-plan / Mistral Free-mode models). A call tries the first link and falls through to the next on ANY
failure (429 / daily quota / 5xx / timeout / empty or unparseable reply / missing model), so a
question still gets a forecast when one provider is out of quota. Model ids live in
``constants.py`` (``FREE_MODE_*``); this module holds no ids.

Safety and serving properties, each pinned by ``tests/test_free_mode.py``:

1. ``assert_free_model`` refuses an OpenRouter slug without ``:free``, any Gemini / Groq /
   Mistral / Z.ai / Cloudflare id outside its ``FREE_MODE_*_FREE_TIER_MODELS`` allow-list, every
   model on ``FREE_MODE_DENIED_MODELS``, and every other provider, at BUILD time, so a typo
   cannot ship a paid model.
2. Calls are paced per quota bucket (``FREE_MODE_MIN_INTERVAL_S``) so a burst stays under the
   free RPM limits instead of burning requests on 429s.
3. A daily-quota 429 (OpenRouter ``free-models-per-day``, Gemini ``...PerDay...``) marks that
   bucket exhausted for the rest of the process, so later calls skip straight to the next
   provider instead of spending time (and requests) on a cap that will not reset mid-run.
4. A forecaster chain has a total time BUDGET (``FREE_MODE_FORECASTER_CHAIN_BUDGET_S``) that
   always keeps ``FREE_MODE_FLOOR_RESERVE_S`` for its last link, every link is built with
   ``max_retries=0`` (no hidden SDK retries), slow NIM reasoning models are streamed, and a plain
   NIM 429 is retried on the same link with backoff while the budget allows.

Nothing here runs unless ``constants.free_mode_enabled()`` is true; the paid path never imports
this module's builders.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import litellm
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from metaculus_bot.constants import (
    CLOUDFLARE_API_TOKEN_ENV,
    FREE_MODE_CLOUDFLARE_FREE_TIER_MODELS,
    FREE_MODE_DENIED_MODELS,
    FREE_MODE_FLOOR_RESERVE_S,
    FREE_MODE_FORECASTER_CHAIN_BUDGET_S,
    FREE_MODE_FORECASTER_CHAINS,
    FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
    FREE_MODE_FORECASTER_MAX_TOKENS,
    FREE_MODE_GEMINI_FREE_TIER_MODELS,
    FREE_MODE_GROQ_EXTRA_PARAMS,
    FREE_MODE_GROQ_FREE_TIER_MODELS,
    FREE_MODE_GROQ_MAX_TOKENS,
    FREE_MODE_GROQ_REASONING_MODELS,
    FREE_MODE_MIN_INTERVAL_S,
    FREE_MODE_MIN_LINK_S,
    FREE_MODE_MISTRAL_FREE_TIER_MODELS,
    FREE_MODE_NIM_MAX_TOKENS,
    FREE_MODE_NIM_SLOW_LINK_TIMEOUT_S,
    FREE_MODE_NIM_SLOW_MODELS,
    FREE_MODE_RATE_LIMIT_BACKOFF_S,
    FREE_MODE_RATE_LIMIT_RETRY_PROVIDERS,
    FREE_MODE_STREAM_IDLE_TIMEOUT_S,
    FREE_MODE_UTILITY_CHAIN,
    FREE_MODE_UTILITY_LINK_TIMEOUT_S,
    FREE_MODE_UTILITY_MAX_TOKENS,
    FREE_MODE_ZAI_FREE_TIER_MODELS,
    GOOGLE_API_KEY_ENV,
    GROQ_API_KEY_ENV,
    MISTRAL_API_KEY_ENV,
    NVIDIA_API_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    ZAI_API_KEY_ENV,
    free_mode_ensemble_size,
    free_openai_compat_endpoint,
)
from metaculus_bot.credit_telemetry import DIRECT_KEY_ALIAS, PERSONAL_KEY_ALIAS, llm_call_metadata
from metaculus_bot.llm_retry import llm_status_code
from metaculus_bot.openai_budget import build_capped_openai_link, capped_openai_configured

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "FreeChainExhaustedError",
    "FreeModelChainLlm",
    "NotFreeModelError",
    "assert_free_model",
    "build_free_chain",
    "build_free_forecaster_llms",
    "build_free_utility_llm",
    "quota_bucket",
    "reset_free_mode_state",
]

_OPENROUTER = "openrouter"
_GEMINI = "gemini"
_NIM = "nvidia_nim"
_GROQ = "groq"
_MISTRAL = "mistral"
_ZAI = "zai"
_CLOUDFLARE = "cloudflare"

_PROVIDER_KEY_ENV: dict[str, str] = {
    _OPENROUTER: OPENROUTER_API_KEY_ENV,
    _GEMINI: GOOGLE_API_KEY_ENV,
    _NIM: NVIDIA_API_KEY_ENV,
    _GROQ: GROQ_API_KEY_ENV,
    _MISTRAL: MISTRAL_API_KEY_ENV,
    _ZAI: ZAI_API_KEY_ENV,
    _CLOUDFLARE: CLOUDFLARE_API_TOKEN_ENV,
}

# Providers whose free tier is an allow-list of model ids (anything else is refused at build time).
_ALLOW_LISTED: dict[str, frozenset[str]] = {
    _GEMINI: FREE_MODE_GEMINI_FREE_TIER_MODELS,
    _GROQ: FREE_MODE_GROQ_FREE_TIER_MODELS,
    _MISTRAL: FREE_MODE_MISTRAL_FREE_TIER_MODELS,
    _ZAI: FREE_MODE_ZAI_FREE_TIER_MODELS,
    _CLOUDFLARE: FREE_MODE_CLOUDFLARE_FREE_TIER_MODELS,
}
# OpenAI-compatible providers litellm reaches as ``openai/<id>`` at an explicit api_base.
_OPENAI_COMPAT: frozenset[str] = frozenset({_ZAI, _CLOUDFLARE})

# English cues of a DAILY cap in the provider's 429 text (OpenRouter "free-models-per-day",
# Gemini "GenerateRequestsPerDayPerProjectPerModel-FreeTier", Groq "requests per day (RPD)" /
# "tokens per day (TPD)", Mistral's monthly allowance).
_DAILY_QUOTA_CUES: tuple[str, ...] = ("per-day", "per day", "perday", "daily", "(rpd)", "(tpd)", "per month", "monthly")
# A model the key cannot reach at all (Gemini 2.5 "no longer available to new users", NIM
# "Not found for account"): skip it for the rest of the run.
_MODEL_GONE_CUES: tuple[str, ...] = ("no longer available", "not found for account", "model not found")


class NotFreeModelError(ValueError):
    """A FREE_MODE chain was asked to build a model that is not on a free tier."""


class FreeChainExhaustedError(RuntimeError):
    """Every link of a FREE_MODE chain failed for this call."""


def _provider(model: str) -> str:
    return model.split("/", 1)[0]


def assert_free_model(model: str) -> None:
    """Raise :class:`NotFreeModelError` unless ``model`` is a free-tier slug this mode allows."""
    provider = _provider(model)
    if model in FREE_MODE_DENIED_MODELS:
        raise NotFreeModelError(f"FREE_MODE refuses {model!r}: {FREE_MODE_DENIED_MODELS[model]}")
    if provider == _OPENROUTER:
        if not model.endswith(":free"):
            raise NotFreeModelError(f"FREE_MODE refuses paid OpenRouter slug {model!r} (no ':free' suffix)")
        return
    if provider in _ALLOW_LISTED:
        model_id = model.split("/", 1)[1] if "/" in model else ""
        if model_id not in _ALLOW_LISTED[provider]:
            raise NotFreeModelError(f"FREE_MODE refuses {provider} model {model!r}: not on the free-tier list")
        return
    if provider == _NIM:
        return  # NVIDIA's hosted developer endpoints have no paid tier on the same key
    raise NotFreeModelError(f"FREE_MODE refuses {model!r}: provider {provider!r} has no free tier wired")


def quota_bucket(model: str) -> str:
    """The rate-limit bucket a call to ``model`` draws on (account-wide, per key, or per model)."""
    provider = _provider(model)
    if provider == _GEMINI:
        return model  # Gemini quotas are per project PER MODEL
    return provider  # OpenRouter :free is account-wide; NIM is per key


def _min_interval_s(model: str) -> float:
    provider = _provider(model)
    if provider == _GEMINI:
        key = "gemma" if "/gemma-" in model else ("gemini-flash-lite" if "lite" in model else "gemini-flash")
        return FREE_MODE_MIN_INTERVAL_S.get(key, 0.0)
    return FREE_MODE_MIN_INTERVAL_S.get(provider, 0.0)


def _max_tokens_for(provider: str, max_tokens: int) -> int:
    if provider == _NIM:
        return min(max_tokens, FREE_MODE_NIM_MAX_TOKENS)
    if provider == _GROQ:
        return min(max_tokens, FREE_MODE_GROQ_MAX_TOKENS)
    return max_tokens


# Process-wide pacing clock and the buckets/models found exhausted this run.
_NEXT_START: dict[str, float] = {}
_EXHAUSTED: set[str] = set()


def reset_free_mode_state() -> None:
    """Clear the pacing clock and the exhausted set. Used by tests."""
    _NEXT_START.clear()
    _EXHAUSTED.clear()


async def _pace(bucket: str, interval_s: float, max_wait_s: float | None = None) -> bool:
    """Reserve the next start slot for ``bucket`` and sleep until it.

    Lock-free on purpose: the read and the write of ``_NEXT_START`` happen with no await in
    between, so concurrent coroutines on one event loop each get a distinct slot. When the slot is
    further away than ``max_wait_s`` nothing is reserved and False is returned, so the caller can
    move on to the next link instead of sleeping through its budget.
    """
    if interval_s <= 0:
        return True
    now = time.monotonic()
    start = max(now, _NEXT_START.get(bucket, 0.0))
    if max_wait_s is not None and start - now > max_wait_s:
        return False
    _NEXT_START[bucket] = start + interval_s
    if start > now:
        await asyncio.sleep(start - now)
    return True


def _classify_and_mark(model: str, exc: Exception) -> str:
    """Mark a daily-capped bucket or an unreachable model as exhausted; return a short reason."""
    status = llm_status_code(exc)
    msg = str(exc).lower()
    if status == 429 or "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
        if any(cue in msg for cue in _DAILY_QUOTA_CUES):
            _EXHAUSTED.add(quota_bucket(model))
            return "daily_quota"
        return "rate_limited"
    if status == 404 or any(cue in msg for cue in _MODEL_GONE_CUES):
        _EXHAUSTED.add(model)
        return "model_unavailable"
    if status is not None and status >= 500:
        return "server_error"
    return type(exc).__name__


class _StreamedLink(GeneralLlm):
    """A ``GeneralLlm`` whose completion is STREAMED and re-assembled into one string.

    For slow NIM reasoning models: a non-streamed call sends no byte until the whole answer is
    ready, so its only protection is a whole-response timeout that kills a model that is still
    producing tokens. Streamed, the litellm ``timeout`` becomes an IDLE read timeout
    (``FREE_MODE_STREAM_IDLE_TIMEOUT_S``: a dead connection fails fast) and the link's total cap
    is enforced here with ``asyncio.wait_for``. Only ``content`` deltas are kept (reasoning deltas
    are dropped), so the forecaster parses exactly what a non-streamed call would have returned.
    """

    def __init__(self, *args: Any, total_timeout_s: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.total_timeout_s: float = total_timeout_s

    async def _mockable_direct_call_to_model(self, prompt: Any) -> TextTokenCostResponse:  # type: ignore[override]
        self._everything_special_to_call_before_direct_call()
        litellm.drop_params = True
        kwargs = dict(self.litellm_kwargs)
        kwargs["stream"] = True
        kwargs["timeout"] = FREE_MODE_STREAM_IDLE_TIMEOUT_S
        messages = self.model_input_to_message(prompt)

        async def _collect() -> str:
            stream = await litellm.acompletion(messages=messages, **kwargs)
            parts: list[str] = []
            async for chunk in stream:  # type: ignore[union-attr]
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    piece = getattr(delta, "content", None) if delta is not None else None
                    if piece:
                        parts.append(piece)
            return "".join(parts)

        answer = await asyncio.wait_for(_collect(), timeout=self.total_timeout_s)
        if not answer.strip():
            raise RuntimeError(f"streamed answer from {self.model} was empty")
        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=0,
            completion_tokens_used=0,
            total_tokens_used=0,
            model=self.model,
            cost=0.0,
        )


def _link_timeout_s(link: GeneralLlm) -> float:
    if isinstance(link, _StreamedLink):
        return link.total_timeout_s
    return float(link.litellm_kwargs.get("timeout") or FREE_MODE_FORECASTER_LINK_TIMEOUT_S)


class FreeModelChainLlm(GeneralLlm):
    """A ``GeneralLlm`` that tries each free link in order until one answers.

    ``model`` reports the slot's PRIMARY link so roster names, the comment's ``Model:`` line and
    the telemetry stay stable per slot. When a fallback link answers a forecaster slot, the reply
    is prefixed with one line naming the model that actually answered, so the published comment
    never misattributes a forecast.

    With ``budget_s`` set (forecaster slots) the whole chain finishes inside that many seconds:
    each non-final link may use at most ``remaining - FREE_MODE_FLOOR_RESERVE_S`` (and is skipped
    when that is under ``FREE_MODE_MIN_LINK_S``), so the last link, the Flash-Lite floor, always
    keeps its reserve. A link whose pacing slot lies beyond its allowance is skipped, not waited
    for. A plain (non-daily) 429 from a provider in ``FREE_MODE_RATE_LIMIT_RETRY_PROVIDERS`` is
    retried on the same link after ``FREE_MODE_RATE_LIMIT_BACKOFF_S`` while the budget allows.
    """

    def __init__(
        self,
        links: list[GeneralLlm],
        *,
        role: str,
        annotate_fallback: bool = False,
        budget_s: float | None = None,
    ) -> None:
        if not links:
            raise ValueError("FreeModelChainLlm needs at least one link")
        primary = links[0]
        super().__init__(
            model=primary.model,
            allowed_tries=1,
            timeout=primary.litellm_kwargs.get("timeout"),
            metadata=llm_call_metadata(role, DIRECT_KEY_ALIAS),
        )
        self.links: list[GeneralLlm] = list(links)
        self.role: str = role
        self.annotate_fallback: bool = annotate_fallback
        self.budget_s: float | None = budget_s

    def _allowance(self, deadline: float | None, reserve: float) -> float | None:
        return None if deadline is None else deadline - time.monotonic() - reserve

    async def _try_link(
        self, *, idx: int, link: GeneralLlm, prompt: Any, system_prompt: str | None, deadline: float | None
    ) -> tuple[str | None, str]:
        """One link, with same-link 429 retries. Returns (text, "") or (None, failure reason)."""
        n_links = len(self.links)
        bucket = quota_bucket(link.model)
        reserve = FREE_MODE_FLOOR_RESERVE_S if idx < n_links - 1 else 0.0
        retryable = _provider(link.model) in FREE_MODE_RATE_LIMIT_RETRY_PROVIDERS
        retries_left = len(FREE_MODE_RATE_LIMIT_BACKOFF_S) if retryable else 0
        attempt = 0
        while True:
            allowance = self._allowance(deadline, reserve)
            if allowance is not None and allowance < FREE_MODE_MIN_LINK_S:
                return None, "skipped(budget)"
            started = time.monotonic()
            max_wait = None if allowance is None else allowance - FREE_MODE_MIN_LINK_S
            if not await _pace(bucket, _min_interval_s(link.model), max_wait):
                return None, "skipped(pacing)"
            cap = _link_timeout_s(link)
            if allowance is not None:
                cap = min(cap, allowance - (time.monotonic() - started))
            try:
                text = await asyncio.wait_for(link.invoke(prompt, system_prompt), timeout=cap)
            # Boundary: a fallback ladder exists to survive ANY provider failure; CancelledError
            # (the caller's soft deadline) is a BaseException and still propagates.
            except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # free-tier fallback ladder
                reason = "timeout" if isinstance(exc, TimeoutError) else _classify_and_mark(link.model, exc)
                if reason == "rate_limited" and attempt < retries_left:
                    backoff = FREE_MODE_RATE_LIMIT_BACKOFF_S[attempt]
                    attempt += 1
                    left = self._allowance(deadline, reserve)
                    if left is None or left - backoff >= FREE_MODE_MIN_LINK_S:
                        logger.warning(
                            "FREE_MODE_RETRY: role=%s model=%s 429; retry %d/%d in %.0fs",
                            self.role,
                            link.model,
                            attempt,
                            retries_left,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                logger.warning(
                    "FREE_MODE_FALLBACK: role=%s link=%d/%d model=%s failed (%s: %s); trying next link",
                    self.role,
                    idx + 1,
                    n_links,
                    link.model,
                    reason,
                    str(exc)[:300],
                )
                return None, reason
            if not isinstance(text, str) or not text.strip():
                logger.warning(
                    "FREE_MODE_FALLBACK: role=%s link=%d/%d model=%s returned empty text; trying next link",
                    self.role,
                    idx + 1,
                    n_links,
                    link.model,
                )
                return None, "empty"
            logger.info(
                "FREE_MODE_LINK: role=%s served_by=%s link=%d/%d elapsed=%.0fs",
                self.role,
                link.model,
                idx + 1,
                n_links,
                time.monotonic() - started,
            )
            return text, ""

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        failures: list[str] = []
        deadline = time.monotonic() + self.budget_s if self.budget_s else None
        for idx, link in enumerate(self.links):
            if quota_bucket(link.model) in _EXHAUSTED or link.model in _EXHAUSTED:
                failures.append(f"{link.model}: skipped(exhausted)")
                continue
            text, reason = await self._try_link(
                idx=idx, link=link, prompt=prompt, system_prompt=system_prompt, deadline=deadline
            )
            if text is None:
                failures.append(f"{link.model}: {reason}")
                continue
            if idx > 0 and self.annotate_fallback:
                text = (
                    f"_(Free-mode fallback: answered by `{link.model}`; slot primary `{self.model}` was unavailable.)_"
                    f"\n\n{text}"
                )
            return text
        raise FreeChainExhaustedError(f"FREE_MODE chain role={self.role} exhausted: " + "; ".join(failures))


def _build_link(
    model: str,
    *,
    api_key: str | None,
    role: str,
    timeout: float,
    max_tokens: int,
    response_format: Any | None,
) -> GeneralLlm:
    provider = _provider(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": None,
        "timeout": timeout,
        "max_tokens": _max_tokens_for(provider, max_tokens),
        "stream": False,
        "allowed_tries": 1,  # the chain is the retry layer; no un-gated tenacity sleeps per link
        "max_retries": 0,  # ...and no hidden OpenAI-SDK retries (they tripled a 240 s timeout)
        "metadata": llm_call_metadata(role, PERSONAL_KEY_ALIAS if provider == _OPENROUTER else DIRECT_KEY_ALIAS),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if response_format is not None:
        kwargs["response_format"] = response_format
    if provider == _GROQ and model in FREE_MODE_GROQ_REASONING_MODELS:
        kwargs.update(FREE_MODE_GROQ_EXTRA_PARAMS)
    if model in FREE_MODE_NIM_SLOW_MODELS and response_format is None:
        link: GeneralLlm = _StreamedLink(total_timeout_s=max(timeout, FREE_MODE_NIM_SLOW_LINK_TIMEOUT_S), **kwargs)
    else:
        link = GeneralLlm(**kwargs)
    if provider in _OPENAI_COMPAT:
        # litellm has no stable provider prefix for these; reach them as OpenAI-compatible while
        # ``link.model`` keeps the bot's slug for pacing, quota buckets and attribution.
        compat = "openai/" + model.split("/", 1)[1]
        link._litellm_model = compat
        link.litellm_kwargs["model"] = compat
        api_base, _ = free_openai_compat_endpoint(provider)
        if api_base:
            link.litellm_kwargs["api_base"] = api_base
    return link


def build_free_chain(
    models: tuple[str, ...] | list[str],
    *,
    role: str,
    timeout: float,
    max_tokens: int,
    response_format: Any | None = None,
    annotate_fallback: bool = False,
    budget_s: float | None = None,
) -> FreeModelChainLlm:
    """Build a chain over ``models``, dropping links whose provider key is not set.

    Every model is checked by :func:`assert_free_model` first, keyed or not. With no provider key
    at all in the process (unit tests, a mis-wired workflow) the chain keeps every link keyless,
    so import still succeeds and a call fails loudly at invoke time instead of silently.
    """
    for model in models:
        assert_free_model(model)
    links: list[GeneralLlm] = []
    for model in models:
        provider = _provider(model)
        if provider in _OPENAI_COMPAT:
            api_key = free_openai_compat_endpoint(provider)[1]
        else:
            api_key = os.getenv(_PROVIDER_KEY_ENV[provider])
        if not api_key:
            continue
        links.append(
            _build_link(
                model,
                api_key=api_key,
                role=role,
                timeout=timeout,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        )
    if not links:
        logger.warning("FREE_MODE: no free-provider key set for role=%s; building keyless links", role)
        links = [
            _build_link(
                model,
                api_key=None,
                role=role,
                timeout=timeout,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            for model in models
        ]
    return FreeModelChainLlm(links, role=role, annotate_fallback=annotate_fallback, budget_s=budget_s)


# Slot (0-based) whose chain gets the hard-capped OpenAI link in front when it is opted in
# (metaculus_bot/openai_budget.py). Slot 4's primary, NIM glm-5.3, is the one that times out most.
FREE_MODE_OPENAI_SLOT: int = 3


def build_free_forecaster_llms() -> list[GeneralLlm]:
    """The FREE_MODE ensemble: ``free_mode_ensemble_size()`` slots, one fallback chain each.

    When the capped OpenAI link is opted in (Olly's free OpenAI credits, approved 2026-10-29), it
    leads slot ``FREE_MODE_OPENAI_SLOT``; the rest of that chain is unchanged and answers whenever
    the budget declines or OpenAI fails.
    """
    slots = FREE_MODE_FORECASTER_CHAINS[: free_mode_ensemble_size()]
    openai_slot = min(FREE_MODE_OPENAI_SLOT, len(slots) - 1) if capped_openai_configured() else -1
    llms: list[GeneralLlm] = []
    for slot, chain in enumerate(slots):
        role = f"forecaster:free{slot + 1}"
        llm = build_free_chain(
            chain,
            role=role,
            timeout=FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
            max_tokens=FREE_MODE_FORECASTER_MAX_TOKENS,
            annotate_fallback=True,
            budget_s=FREE_MODE_FORECASTER_CHAIN_BUDGET_S,
        )
        if slot == openai_slot:
            llm = FreeModelChainLlm(
                [build_capped_openai_link(role=role), *llm.links],
                role=role,
                annotate_fallback=True,
                budget_s=FREE_MODE_FORECASTER_CHAIN_BUDGET_S,
            )
            logger.info("OPENAI_BUDGET: capped OpenAI link leads forecaster slot %d", slot + 1)
        llms.append(llm)
    return llms


def build_free_utility_llm(
    *, role: str | None, response_format: Any | None = None, timeout: float | None = None
) -> GeneralLlm:
    """The FREE_MODE stand-in for every support-role LLM (parser, summarizer, ranker, ...).

    Called by ``build_llm_with_openrouter_fallback`` in place of the paid model it was asked for.
    Only the caller's ``response_format`` and ``timeout`` carry over: reasoning-effort, verbosity
    and OpenRouter ``extra_body`` knobs are paid-model specific and can 400 a free provider.
    """
    return build_free_chain(
        FREE_MODE_UTILITY_CHAIN,
        role=role or "untagged",
        timeout=float(timeout) if timeout else FREE_MODE_UTILITY_LINK_TIMEOUT_S,
        max_tokens=FREE_MODE_UTILITY_MAX_TOKENS,
        response_format=response_format,
    )
