"""FREE_MODE (ollymcbride-bot fork): zero-cost config, the free fallback chain, and the paid-path guards.

Everything here is offline: chain links are stubbed at ``invoke`` and no provider is called.
"""

from __future__ import annotations

import litellm
import pytest
from pydantic import BaseModel

from metaculus_bot import free_mode
from metaculus_bot.constants import (
    BINARY_STACKING_ENABLED_ENV,
    FREE_MODE_FORECASTER_CHAINS,
    FREE_MODE_GEMINI_GROUNDING_MODEL,
    FREE_MODE_UTILITY_CHAIN,
    GAP_FILL_ENABLED_ENV,
    GAP_FILL_V2_ENABLED_ENV,
    GEMINI_SEARCH_ENABLED_ENV,
    NATIVE_SEARCH_ENABLED_ENV,
    PREDICTION_MARKETS_ENABLED_ENV,
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV,
    env_flag_enabled,
    free_mode_ensemble_size,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.free_mode import (
    FreeChainExhaustedError,
    FreeModelChainLlm,
    NotFreeModelError,
    assert_free_model,
    build_free_chain,
    build_free_forecaster_llms,
)

_PAID_FLAGS = [
    NATIVE_SEARCH_ENABLED_ENV,
    GAP_FILL_ENABLED_ENV,
    GAP_FILL_V2_ENABLED_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV,
    BINARY_STACKING_ENABLED_ENV,
]


@pytest.fixture(autouse=True)
def _fresh_free_state(monkeypatch: pytest.MonkeyPatch):
    free_mode.reset_free_mode_state()
    # No pacing sleeps in unit tests.
    monkeypatch.setattr(free_mode, "FREE_MODE_MIN_INTERVAL_S", dict.fromkeys(free_mode.FREE_MODE_MIN_INTERVAL_S, 0.0))
    yield
    free_mode.reset_free_mode_state()


@pytest.fixture
def free_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    monkeypatch.delenv("FREE_MODE_ALLOW_GEMINI_GROUNDING", raising=False)
    monkeypatch.delenv("FREE_MODE_ENSEMBLE_SIZE", raising=False)
    return monkeypatch


# --- config: paid flags are hard-off, free flags untouched, paid path unchanged ---


@pytest.mark.parametrize("flag", _PAID_FLAGS)
def test_paid_flags_forced_off_in_free_mode(free_env: pytest.MonkeyPatch, flag: str) -> None:
    free_env.setenv(flag, "true")
    assert env_flag_enabled(flag) is False


@pytest.mark.parametrize("flag", _PAID_FLAGS)
def test_paid_flags_untouched_without_free_mode(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.delenv("FREE_MODE", raising=False)
    monkeypatch.setenv(flag, "true")
    assert env_flag_enabled(flag) is True


@pytest.mark.parametrize("flag", [PREDICTION_MARKETS_ENABLED_ENV, RESOLUTION_SOURCE_ENABLED_ENV])
def test_free_research_flags_still_honoured(free_env: pytest.MonkeyPatch, flag: str) -> None:
    free_env.setenv(flag, "true")
    assert env_flag_enabled(flag) is True


def test_gemini_search_needs_explicit_opt_in(free_env: pytest.MonkeyPatch) -> None:
    free_env.setenv(GEMINI_SEARCH_ENABLED_ENV, "true")
    assert env_flag_enabled(GEMINI_SEARCH_ENABLED_ENV) is False
    free_env.setenv("FREE_MODE_ALLOW_GEMINI_GROUNDING", "true")
    assert env_flag_enabled(GEMINI_SEARCH_ENABLED_ENV) is True


def test_gemini_search_model_pinned_to_free_grounding_model(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research.gemini_search import _resolve_model

    free_env.setenv("GEMINI_SEARCH_MODEL", "gemini-3.8-flash")
    assert _resolve_model("gemini-3.8-flash") == FREE_MODE_GEMINI_GROUNDING_MODEL
    free_env.delenv("FREE_MODE")
    assert _resolve_model(None) == "gemini-3.8-flash"


def test_ensemble_size_default_and_clamp(free_env: pytest.MonkeyPatch) -> None:
    assert free_mode_ensemble_size() == 4
    free_env.setenv("FREE_MODE_ENSEMBLE_SIZE", "99")
    assert free_mode_ensemble_size() == len(FREE_MODE_FORECASTER_CHAINS)
    free_env.setenv("FREE_MODE_ENSEMBLE_SIZE", "0")
    assert free_mode_ensemble_size() == 1


# --- the free-model whitelist ---


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-6.1-sol",
        "openrouter/google/gemini-3.1-pro-preview",
        "gemini/gemini-3.1-pro-preview",
        "anthropic/claude-opus-5.5",
        "perplexity/sonar-reasoning-pro",
    ],
)
def test_paid_models_refused(model: str) -> None:
    with pytest.raises(NotFreeModelError):
        assert_free_model(model)


def test_every_configured_free_model_passes_the_whitelist() -> None:
    for chain in (*FREE_MODE_FORECASTER_CHAINS, FREE_MODE_UTILITY_CHAIN):
        for model in chain:
            assert_free_model(model)


def test_every_forecaster_chain_ends_on_a_flash_lite_floor_and_slots_lead_distinct_models() -> None:
    leads = [chain[0] for chain in FREE_MODE_FORECASTER_CHAINS]
    assert len(set(leads)) == len(leads)
    for chain in FREE_MODE_FORECASTER_CHAINS:
        assert chain[-1].startswith("gemini/")
        assert "flash-lite" in chain[-1]
        assert len(chain) >= 3


def test_openrouter_leads_at_most_one_default_slot() -> None:
    """OpenRouter's 50 RPD account-wide budget cannot carry more than one primary slot."""
    default_leads = [chain[0] for chain in FREE_MODE_FORECASTER_CHAINS[:4]]
    assert sum(lead.startswith("openrouter/") for lead in default_leads) <= 1


# --- building chains ---


def test_build_drops_links_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    chain = build_free_chain(FREE_MODE_FORECASTER_CHAINS[0], role="forecaster:free1", timeout=10, max_tokens=100)
    assert [link.model for link in chain.links] == ["openrouter/thinkingmachines/inkling:free"]
    assert chain.links[0].litellm_kwargs["api_key"] == "or-test"
    assert chain.links[0].allowed_tries == 1


def test_build_without_any_key_stays_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    chain = build_free_chain(FREE_MODE_UTILITY_CHAIN, role="parser", timeout=10, max_tokens=100)
    assert [link.model for link in chain.links] == list(FREE_MODE_UTILITY_CHAIN)


def test_build_refuses_a_paid_model_even_without_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(NotFreeModelError):
        build_free_chain(("openrouter/openai/gpt-6-luna",), role="parser", timeout=10, max_tokens=100)


def test_nim_links_cap_max_tokens(free_env: pytest.MonkeyPatch) -> None:
    chain = build_free_chain(("nvidia_nim/z-ai/glm-5.3",), role="x", timeout=10, max_tokens=999_999)
    assert chain.links[0].litellm_kwargs["max_tokens"] == free_mode.FREE_MODE_NIM_MAX_TOKENS


def test_forecaster_llms_follow_ensemble_size(free_env: pytest.MonkeyPatch) -> None:
    free_env.setenv("FREE_MODE_ENSEMBLE_SIZE", "3")
    llms = build_free_forecaster_llms()
    assert [llm.model for llm in llms] == [chain[0] for chain in FREE_MODE_FORECASTER_CHAINS[:3]]
    assert all(isinstance(llm, FreeModelChainLlm) and llm.annotate_fallback for llm in llms)


class _Schema(BaseModel):
    value: int


def test_builder_returns_free_chain_in_free_mode(free_env: pytest.MonkeyPatch) -> None:
    llm = build_llm_with_openrouter_fallback(
        "openrouter/openai/gpt-6-luna",
        role="parser",
        reasoning={"effort": "low"},
        timeout=30,
        response_format=_Schema,
        extra_body={"provider": {"require_parameters": True}},
    )
    assert isinstance(llm, FreeModelChainLlm)
    # free_env sets no GROQ_API_KEY, so the optional Groq utility link is dropped.
    assert [link.model for link in llm.links] == [m for m in FREE_MODE_UTILITY_CHAIN if not m.startswith("groq/")]
    for link in llm.links:
        assert link.litellm_kwargs["response_format"] is _Schema
        assert link.litellm_kwargs["timeout"] == 30
        assert "extra_body" not in link.litellm_kwargs
        assert "reasoning" not in link.litellm_kwargs


def test_builder_unchanged_without_free_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREE_MODE", raising=False)
    llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-6-luna", role="parser")
    assert not isinstance(llm, FreeModelChainLlm)
    assert llm.model == "openrouter/openai/gpt-6-luna"


# --- the fallback chain at call time ---


def _rate_limit(msg: str) -> litellm.exceptions.RateLimitError:
    return litellm.exceptions.RateLimitError(message=msg, llm_provider="openrouter", model="m")


def _stub_chain(monkeypatch: pytest.MonkeyPatch, behaviours: dict[str, list]) -> tuple[FreeModelChainLlm, list[str]]:
    """A chain whose links pop their next behaviour (an exception to raise or text to return)."""
    calls: list[str] = []
    models = list(behaviours)
    for key in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.setenv(key, "k")
    chain = build_free_chain(tuple(models), role="forecaster:free1", timeout=10, max_tokens=100, annotate_fallback=True)
    for link in chain.links:

        async def _invoke(prompt, system_prompt=None, _model=link.model):
            calls.append(_model)
            outcome = behaviours[_model].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(link, "invoke", _invoke)
    return chain, calls


async def test_falls_back_on_429_and_annotates(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "openrouter/thinkingmachines/inkling:free": [_rate_limit("Rate limit exceeded: free-models-per-min")],
            "nvidia_nim/moonshotai/kimi-k3": ["PROBABILITY: 42%"],
        },
    )
    text = await chain.invoke("q")
    assert calls == ["openrouter/thinkingmachines/inkling:free", "nvidia_nim/moonshotai/kimi-k3"]
    assert text.endswith("PROBABILITY: 42%")
    assert "answered by `nvidia_nim/moonshotai/kimi-k3`" in text.splitlines()[0]


async def test_primary_success_is_not_annotated(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, _ = _stub_chain(monkeypatch, {"gemini/gemini-3.8-flash": ["ok"], "gemini/gemini-3.5-flash-lite": []})
    assert await chain.invoke("q") == "ok"


async def test_daily_quota_marks_bucket_exhausted_for_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "openrouter/thinkingmachines/inkling:free": [
                _rate_limit("Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000")
            ],
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free": ["never called"],
            "gemini/gemini-3.1-flash-lite": ["first", "second"],
        },
    )
    assert (await chain.invoke("q")).endswith("first")
    assert (await chain.invoke("q")).endswith("second")
    # The second OpenRouter link and the second call's OpenRouter attempt were skipped.
    assert calls == [
        "openrouter/thinkingmachines/inkling:free",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.1-flash-lite",
    ]


async def test_empty_reply_and_server_error_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    server_error = litellm.exceptions.InternalServerError(message="boom", llm_provider="nvidia_nim", model="m")
    chain, _ = _stub_chain(
        monkeypatch,
        {
            "nvidia_nim/z-ai/glm-5.3": [server_error],
            "gemini/gemini-3.8-flash": ["   "],
            "gemini/gemini-3.5-flash-lite": ["floor"],
        },
    )
    assert (await chain.invoke("q")).endswith("floor")


async def test_all_links_failing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, _ = _stub_chain(
        monkeypatch,
        {
            "openrouter/apodex/apodex-1.1-mini:free": [_rate_limit("Rate limit exceeded")],
            "gemini/gemini-3.5-flash-lite": [RuntimeError("down")],
        },
    )
    with pytest.raises(FreeChainExhaustedError, match="exhausted"):
        await chain.invoke("q")


async def test_pacing_spaces_call_starts_per_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(free_mode.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(free_mode.time, "monotonic", lambda: 100.0)
    for _ in range(3):
        await free_mode._pace("openrouter", 4.0)
    await free_mode._pace("nvidia_nim", 2.0)
    assert sleeps == [4.0, 8.0]  # first openrouter call and the nim call start immediately


def test_quota_buckets() -> None:
    assert free_mode.quota_bucket("openrouter/a/b:free") == "openrouter"
    assert free_mode.quota_bucket("nvidia_nim/a/b") == "nvidia_nim"
    assert free_mode.quota_bucket("gemini/gemini-3.8-flash") == "gemini/gemini-3.8-flash"


# --- research: no paid primary and no paid AskNews fallback ---


def test_no_paid_primary_research_provider_in_free_mode(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research.providers import choose_provider_with_name

    free_env.delenv("ASKNEWS_CLIENT_ID", raising=False)
    free_env.delenv("ASKNEWS_SECRET", raising=False)
    free_env.setenv("EXA_API_KEY", "e")
    free_env.setenv("PERPLEXITY_API_KEY", "p")
    _, name = choose_provider_with_name(None)
    assert name == "none"
    free_env.setenv("RESEARCH_PROVIDER", "perplexity")
    _, name = choose_provider_with_name(None)
    assert name == "none"


def test_asknews_used_when_its_secrets_exist(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research.providers import choose_provider_with_name

    free_env.delenv("RESEARCH_PROVIDER", raising=False)
    free_env.setenv("ASKNEWS_CLIENT_ID", "id")
    free_env.setenv("ASKNEWS_SECRET", "s")
    _, name = choose_provider_with_name(None)
    assert name == "asknews"


async def test_asknews_paid_fallback_skipped_in_free_mode(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research.orchestrator import ResearchOrchestrator

    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("paid research fallback must not run in FREE_MODE")

    free_env.setattr(orchestrator, "_call_perplexity", _boom, raising=False)
    free_env.setattr(orchestrator, "_call_exa_smart_searcher", _boom, raising=False)
    assert await orchestrator._attempt_research_fallback("q") == (None, None)


# --- the real module-level roster under FREE_MODE (fresh interpreter: llm_configs builds at import) ---


def test_every_llm_the_cli_wires_is_a_free_chain_under_free_mode() -> None:
    import os
    import subprocess
    import sys

    script = (
        "from metaculus_bot import llm_configs as c\n"
        "from metaculus_bot.free_mode import FreeModelChainLlm, assert_free_model\n"
        "llms = list(c.FORECASTER_LLMS) + [c.PARSER_LLM, c.SUMMARIZER_LLM, c.RESEARCHER_LLM, c.STACKER_LLM,\n"
        "    c.STACKER_FALLBACK_LLM, c.DISAGREEMENT_ANALYZER_LLM]\n"
        "assert len(c.FORECASTER_LLMS) == 4, len(c.FORECASTER_LLMS)\n"
        "for llm in llms:\n"
        "    assert isinstance(llm, FreeModelChainLlm), type(llm)\n"
        "    for link in llm.links:\n"
        "        assert_free_model(link.model)\n"
        "print('OK')\n"
    )
    env = {
        **os.environ,
        "FREE_MODE": "true",
        "OPENROUTER_API_KEY": "or-test",
        "GOOGLE_API_KEY": "g-test",
        "NVIDIA_API_KEY": "nv-test",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    env.pop("FREE_MODE_ENSEMBLE_SIZE", None)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("OK")


# --- optional Groq / Mistral free tiers (added 2026-10-12) ---


@pytest.mark.parametrize(
    "model",
    [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
        "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest",
    ],
)
def test_groq_and_mistral_free_models_allowed(model: str) -> None:
    assert_free_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "groq/moonshotai/kimi-k3-0905",
        "groq/compound",
        "mistral/codestral-2508",
        "mistral/mistral-large-latest",
        "github/openai/gpt-5",
        "cohere/command-a",
    ],
)
def test_groq_and_mistral_off_list_and_unwired_providers_refused(model: str) -> None:
    with pytest.raises(NotFreeModelError):
        assert_free_model(model)


def test_groq_and_mistral_links_drop_without_their_keys(free_env: pytest.MonkeyPatch) -> None:
    free_env.delenv("GROQ_API_KEY", raising=False)
    free_env.delenv("MISTRAL_API_KEY", raising=False)
    for chain in FREE_MODE_FORECASTER_CHAINS:
        built = build_free_chain(chain, role="x", timeout=10, max_tokens=100)
        assert not any(link.model.startswith(("groq/", "mistral/")) for link in built.links)


def test_groq_link_uses_its_key_and_caps_max_tokens(free_env: pytest.MonkeyPatch) -> None:
    free_env.setenv("GROQ_API_KEY", "gq-test")
    chain = build_free_chain(("groq/openai/gpt-oss-120b",), role="x", timeout=10, max_tokens=999_999)
    assert chain.links[0].litellm_kwargs["api_key"] == "gq-test"
    assert chain.links[0].litellm_kwargs["max_tokens"] == free_mode.FREE_MODE_GROQ_MAX_TOKENS


def test_groq_daily_cap_marks_bucket_exhausted() -> None:
    exc = litellm.RateLimitError(
        message="Rate limit reached for model `openai/gpt-oss-120b` on requests per day (RPD): Limit 1000",
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    assert free_mode._classify_and_mark("groq/openai/gpt-oss-120b", exc) == "daily_quota"
    assert "groq" in free_mode._EXHAUSTED


def test_groq_and_mistral_sit_before_the_floor_never_as_slot_leads() -> None:
    for chain in FREE_MODE_FORECASTER_CHAINS:
        assert not chain[0].startswith(("groq/", "mistral/"))
        extras = [i for i, m in enumerate(chain) if m.startswith(("groq/", "mistral/"))]
        assert all(i == len(chain) - 2 for i in extras)
