"""FREE_MODE: zero-cost LLM fallback chains for the ollymcbride-bot fork.

With ``FREE_MODE=true`` every LLM the bot builds is a :class:`FreeModelChainLlm`: an ordered list
of free-tier models (OpenRouter ``:free`` variants, Google AI Studio free-tier Gemini, optional
NVIDIA NIM free endpoints). A call tries the first link and falls through to the next on ANY
failure (429 / daily quota / 5xx / timeout / empty or unparseable reply / missing model), so a
question still gets a forecast when one provider is out of quota. Model ids live in
``constants.py`` (``FREE_MODE_*``); this module holds no ids.

Three safety properties, each pinned by ``tests/test_free_mode.py``:

1. ``assert_free_model`` refuses an OpenRouter slug without ``:free`` and any Gemini id outside
   ``FREE_MODE_GEMINI_FREE_TIER_MODELS``, at BUILD time, so a typo cannot ship a paid model.
2. Calls are paced per quota bucket (``FREE_MODE_MIN_INTERVAL_S``) so a burst stays under the
   free RPM limits instead of burning requests on 429s.
3. A daily-quota 429 (OpenRouter ``free-models-per-day``, Gemini ``...PerDay...``) marks that
   bucket exhausted for the rest of the process, so later calls skip straight to the next
   provider instead of spending time (and requests) on a cap that will not reset mid-run.

Nothing here runs unless ``constants.free_mode_enabled()`` is true; the paid path never imports
this module's builders.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.constants import (
    FREE_MODE_FORECASTER_CHAINS,
    FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
    FREE_MODE_FORECASTER_MAX_TOKENS,
    FREE_MODE_GEMINI_FREE_TIER_MODELS,
    FREE_MODE_MIN_INTERVAL_S,
    FREE_MODE_NIM_MAX_TOKENS,
    FREE_MODE_UTILITY_CHAIN,
    FREE_MODE_UTILITY_LINK_TIMEOUT_S,
    FREE_MODE_UTILITY_MAX_TOKENS,
    GOOGLE_API_KEY_ENV,
    NVIDIA_API_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    free_mode_ensemble_size,
)
from metaculus_bot.credit_telemetry import DIRECT_KEY_ALIAS, PERSONAL_KEY_ALIAS, llm_call_metadata
from metaculus_bot.llm_retry import llm_status_code

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

_PROVIDER_KEY_ENV: dict[str, str] = {
    _OPENROUTER: OPENROUTER_API_KEY_ENV,
    _GEMINI: GOOGLE_API_KEY_ENV,
    _NIM: NVIDIA_API_KEY_ENV,
}

# English cues of a DAILY cap in the provider's 429 text (OpenRouter "free-models-per-day",
# Gemini "GenerateRequestsPerDayPerProjectPerModel-FreeTier").
_DAILY_QUOTA_CUES: tuple[str, ...] = ("per-day", "per day", "perday", "daily")
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
    if provider == _OPENROUTER:
        if not model.endswith(":free"):
            raise NotFreeModelError(f"FREE_MODE refuses paid OpenRouter slug {model!r} (no ':free' suffix)")
        return
    if provider == _GEMINI:
        model_id = model.split("/", 1)[1] if "/" in model else ""
        if model_id not in FREE_MODE_GEMINI_FREE_TIER_MODELS:
            raise NotFreeModelError(f"FREE_MODE refuses Gemini model {model!r}: not on the free-tier list")
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
        key = "gemini-flash-lite" if "lite" in model else "gemini-flash"
        return FREE_MODE_MIN_INTERVAL_S[key]
    return FREE_MODE_MIN_INTERVAL_S.get(provider, 0.0)


# Process-wide pacing clock and the buckets/models found exhausted this run.
_NEXT_START: dict[str, float] = {}
_EXHAUSTED: set[str] = set()


def reset_free_mode_state() -> None:
    """Clear the pacing clock and the exhausted set. Used by tests."""
    _NEXT_START.clear()
    _EXHAUSTED.clear()


async def _pace(bucket: str, interval_s: float) -> None:
    """Reserve the next start slot for ``bucket`` and sleep until it.

    Lock-free on purpose: the read and the write of ``_NEXT_START`` happen with no await in
    between, so concurrent coroutines on one event loop each get a distinct slot.
    """
    if interval_s <= 0:
        return
    now = time.monotonic()
    start = max(now, _NEXT_START.get(bucket, 0.0))
    _NEXT_START[bucket] = start + interval_s
    if start > now:
        await asyncio.sleep(start - now)


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


class FreeModelChainLlm(GeneralLlm):
    """A ``GeneralLlm`` that tries each free link in order until one answers.

    ``model`` reports the slot's PRIMARY link so roster names, the comment's ``Model:`` line and
    the telemetry stay stable per slot. When a fallback link answers a forecaster slot, the reply
    is prefixed with one line naming the model that actually answered, so the published comment
    never misattributes a forecast.
    """

    def __init__(self, links: list[GeneralLlm], *, role: str, annotate_fallback: bool = False) -> None:
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

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        failures: list[str] = []
        n_links = len(self.links)
        for idx, link in enumerate(self.links):
            bucket = quota_bucket(link.model)
            if bucket in _EXHAUSTED or link.model in _EXHAUSTED:
                failures.append(f"{link.model}: skipped(exhausted)")
                continue
            await _pace(bucket, _min_interval_s(link.model))
            try:
                text = await link.invoke(prompt, system_prompt)
            # Boundary: a fallback ladder exists to survive ANY provider failure; CancelledError
            # (the caller's soft deadline) is a BaseException and still propagates.
            except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # free-tier fallback ladder
                reason = _classify_and_mark(link.model, exc)
                failures.append(f"{link.model}: {reason}")
                logger.warning(
                    "FREE_MODE_FALLBACK: role=%s link=%d/%d model=%s failed (%s: %s); trying next link",
                    self.role,
                    idx + 1,
                    n_links,
                    link.model,
                    reason,
                    str(exc)[:300],
                )
                continue
            if not isinstance(text, str) or not text.strip():
                failures.append(f"{link.model}: empty")
                logger.warning(
                    "FREE_MODE_FALLBACK: role=%s link=%d/%d model=%s returned empty text; trying next link",
                    self.role,
                    idx + 1,
                    n_links,
                    link.model,
                )
                continue
            logger.info("FREE_MODE_LINK: role=%s served_by=%s link=%d/%d", self.role, link.model, idx + 1, n_links)
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
        "max_tokens": min(max_tokens, FREE_MODE_NIM_MAX_TOKENS) if provider == _NIM else max_tokens,
        "stream": False,
        "allowed_tries": 1,  # the chain is the retry layer; no un-gated tenacity sleeps per link
        "metadata": llm_call_metadata(role, PERSONAL_KEY_ALIAS if provider == _OPENROUTER else DIRECT_KEY_ALIAS),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if response_format is not None:
        kwargs["response_format"] = response_format
    return GeneralLlm(**kwargs)


def build_free_chain(
    models: tuple[str, ...] | list[str],
    *,
    role: str,
    timeout: float,
    max_tokens: int,
    response_format: Any | None = None,
    annotate_fallback: bool = False,
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
        api_key = os.getenv(_PROVIDER_KEY_ENV[_provider(model)])
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
    return FreeModelChainLlm(links, role=role, annotate_fallback=annotate_fallback)


def build_free_forecaster_llms() -> list[GeneralLlm]:
    """The FREE_MODE ensemble: ``free_mode_ensemble_size()`` slots, one fallback chain each."""
    return [
        build_free_chain(
            chain,
            role=f"forecaster:free{slot + 1}",
            timeout=FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
            max_tokens=FREE_MODE_FORECASTER_MAX_TOKENS,
            annotate_fallback=True,
        )
        for slot, chain in enumerate(FREE_MODE_FORECASTER_CHAINS[: free_mode_ensemble_size()])
    ]


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
