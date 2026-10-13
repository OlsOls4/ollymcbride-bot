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
    FREE_MODE_OPTIONAL_PROVIDERS,
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
    chain = build_free_chain(FREE_MODE_FORECASTER_CHAINS[1], role="forecaster:free2", timeout=10, max_tokens=100)
    assert [link.model for link in chain.links] == ["openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"]
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
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free": [
                _rate_limit("Rate limit exceeded: free-models-per-min")
            ],
            "nvidia_nim/moonshotai/kimi-k3": ["PROBABILITY: 42%"],
        },
    )
    text = await chain.invoke("q")
    assert calls == ["openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia_nim/moonshotai/kimi-k3"]
    assert text.endswith("PROBABILITY: 42%")
    assert "answered by `nvidia_nim/moonshotai/kimi-k3`" in text.splitlines()[0]


async def test_primary_success_is_not_annotated(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, _ = _stub_chain(monkeypatch, {"gemini/gemini-3.8-flash": ["ok"], "gemini/gemini-3.5-flash-lite": []})
    assert await chain.invoke("q") == "ok"


async def test_daily_quota_marks_bucket_exhausted_for_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "openrouter/apodex/apodex-1.1-mini:free": [
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
        "openrouter/apodex/apodex-1.1-mini:free",
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


def test_optional_key_providers_sit_in_the_tail_never_lead_or_floor() -> None:
    optional = tuple(f"{p}/" for p in FREE_MODE_OPTIONAL_PROVIDERS)
    for chain in FREE_MODE_FORECASTER_CHAINS:
        assert not chain[0].startswith(optional)
        assert not chain[-1].startswith(optional)
        assert all(i >= 2 for i, m in enumerate(chain) if m.startswith(optional))


# --- 2026-10-13 serving fixes: denied models, budgets, streaming, retries, new providers ---


@pytest.mark.parametrize(
    "model", ["openrouter/thinkingmachines/inkling:free", "openrouter/thinkingmachines/inkling-small:free"]
)
def test_agentic_harness_only_models_are_denied(model: str) -> None:
    with pytest.raises(NotFreeModelError, match="agentic"):
        assert_free_model(model)


def test_no_live_chain_carries_a_dead_or_hopeless_link() -> None:
    every = [m for chain in FREE_MODE_FORECASTER_CHAINS for m in chain]
    assert not any("inkling" in m for m in every)
    assert not any(m.startswith("mistral/") for m in every)  # free key: 0 RPM on medium/small
    assert "groq/qwen/qwen3.8-27b" not in every  # 1K output-tokens/min on Groq Free


def test_floors_try_3_5_flash_lite_before_3_1() -> None:
    for chain in FREE_MODE_FORECASTER_CHAINS:
        assert chain[-2:] == ("gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-flash-lite")


def test_links_have_no_hidden_sdk_retries(free_env: pytest.MonkeyPatch) -> None:
    chain = build_free_chain(FREE_MODE_FORECASTER_CHAINS[0], role="x", timeout=10, max_tokens=100)
    assert chain.links
    assert all(link.litellm_kwargs["max_retries"] == 0 for link in chain.links)


def test_slow_nim_models_are_streamed_with_a_longer_cap(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.constants import FREE_MODE_NIM_SLOW_LINK_TIMEOUT_S, FREE_MODE_NIM_SLOW_MODELS

    chain = build_free_chain(
        (*FREE_MODE_NIM_SLOW_MODELS, "nvidia_nim/moonshotai/kimi-k3"), role="x", timeout=240, max_tokens=100
    )
    slow, plain = chain.links[:-1], chain.links[-1]
    assert all(isinstance(link, free_mode._StreamedLink) for link in slow)
    assert all(link.total_timeout_s == FREE_MODE_NIM_SLOW_LINK_TIMEOUT_S for link in slow)
    assert not isinstance(plain, free_mode._StreamedLink)


def test_structured_output_links_are_never_streamed(free_env: pytest.MonkeyPatch) -> None:
    chain = build_free_chain(
        ("nvidia_nim/z-ai/glm-5.3",), role="x", timeout=10, max_tokens=100, response_format=_Schema
    )
    assert not isinstance(chain.links[0], free_mode._StreamedLink)


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


async def test_streamed_link_reassembles_content_and_uses_an_idle_timeout(
    free_env: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    async def _fake_acompletion(**kwargs):
        seen.update(kwargs)

        async def _gen():
            for piece in ("Reasoning done. ", None, "Probability: ", "37%"):
                yield _Chunk(piece)

        return _gen()

    free_env.setattr(free_mode.litellm, "acompletion", _fake_acompletion)
    chain = build_free_chain(("nvidia_nim/z-ai/glm-5.3",), role="x", timeout=10, max_tokens=100)
    text = await chain.links[0].invoke("q")
    assert text == "Reasoning done. Probability: 37%"
    assert seen["stream"] is True
    assert seen["timeout"] == free_mode.FREE_MODE_STREAM_IDLE_TIMEOUT_S


async def test_streamed_link_empty_answer_raises(free_env: pytest.MonkeyPatch) -> None:
    async def _fake_acompletion(**kwargs):
        async def _gen():
            yield _Chunk(None)

        return _gen()

    free_env.setattr(free_mode.litellm, "acompletion", _fake_acompletion)
    chain = build_free_chain(
        ("nvidia_nim/z-ai/glm-5.3", "gemini/gemini-3.5-flash-lite"), role="x", timeout=10, max_tokens=10
    )

    async def _floor(prompt, system_prompt=None):
        return "floor"

    free_env.setattr(chain.links[1], "invoke", _floor)
    assert await chain.invoke("q") == "floor"


def _tight_budget(monkeypatch: pytest.MonkeyPatch, reserve: float = 0.3, min_link: float = 0.05) -> None:
    monkeypatch.setattr(free_mode, "FREE_MODE_FLOOR_RESERVE_S", reserve)
    monkeypatch.setattr(free_mode, "FREE_MODE_MIN_LINK_S", min_link)


async def test_budget_cuts_a_stalled_head_and_keeps_the_floors_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import time

    _tight_budget(monkeypatch)
    for key in ("NVIDIA_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "k")
    chain = build_free_chain(
        ("nvidia_nim/moonshotai/kimi-k3", "gemini/gemini-3.5-flash-lite"),
        role="x",
        timeout=60,
        max_tokens=10,
        budget_s=1.0,
    )

    async def _stall(prompt, system_prompt=None):
        await asyncio.sleep(30)
        return "never"

    async def _floor(prompt, system_prompt=None):
        return "floor"

    monkeypatch.setattr(chain.links[0], "invoke", _stall)
    monkeypatch.setattr(chain.links[1], "invoke", _floor)
    t0 = time.monotonic()
    text = await chain.invoke("q")
    assert text.endswith("floor")
    assert time.monotonic() - t0 < 1.0


async def test_budget_skips_links_that_cannot_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    _tight_budget(monkeypatch, reserve=5.0)
    chain, calls = _stub_chain(
        monkeypatch,
        {"nvidia_nim/moonshotai/kimi-k3": ["never"], "gemini/gemini-3.5-flash-lite": ["floor"]},
    )
    chain.budget_s = 1.0  # head allowance = 1 - 5 < min link: skipped, the floor still runs
    assert (await chain.invoke("q")).endswith("floor")
    assert calls == ["gemini/gemini-3.5-flash-lite"]


async def test_nim_plain_429_is_retried_on_the_same_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_mode, "FREE_MODE_RATE_LIMIT_BACKOFF_S", (0.0, 0.0))
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "nvidia_nim/moonshotai/kimi-k3": [_rate_limit("Too Many Requests"), "PROBABILITY: 40%"],
            "gemini/gemini-3.5-flash-lite": ["floor"],
        },
    )
    assert await chain.invoke("q") == "PROBABILITY: 40%"
    assert calls == ["nvidia_nim/moonshotai/kimi-k3", "nvidia_nim/moonshotai/kimi-k3"]


async def test_nim_429_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_mode, "FREE_MODE_RATE_LIMIT_BACKOFF_S", (0.0, 0.0))
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "nvidia_nim/moonshotai/kimi-k3": [_rate_limit("Too Many Requests")] * 3,
            "gemini/gemini-3.5-flash-lite": ["floor"],
        },
    )
    assert (await chain.invoke("q")).endswith("floor")
    assert calls.count("nvidia_nim/moonshotai/kimi-k3") == 3  # first try + 2 retries


async def test_non_nim_429_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_mode, "FREE_MODE_RATE_LIMIT_BACKOFF_S", (0.0, 0.0))
    chain, calls = _stub_chain(
        monkeypatch,
        {
            "openrouter/apodex/apodex-1.1-mini:free": [_rate_limit("Rate limit exceeded: free-models-per-min")],
            "gemini/gemini-3.5-flash-lite": ["floor"],
        },
    )
    assert (await chain.invoke("q")).endswith("floor")
    assert calls == ["openrouter/apodex/apodex-1.1-mini:free", "gemini/gemini-3.5-flash-lite"]


async def test_pacing_slot_beyond_the_allowance_is_skipped_not_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_mode.time, "monotonic", lambda: 100.0)
    assert await free_mode._pace("groq", 55.0, max_wait_s=10.0) is True  # first call starts now
    assert await free_mode._pace("groq", 55.0, max_wait_s=10.0) is False  # next slot is 55 s away
    assert free_mode._NEXT_START["groq"] == 155.0  # the refused call reserved nothing


def test_groq_request_fits_the_free_8k_tokens_per_minute(free_env: pytest.MonkeyPatch) -> None:
    free_env.setenv("GROQ_API_KEY", "gq")
    chain = build_free_chain(("groq/openai/gpt-oss-120b",), role="x", timeout=10, max_tokens=32_000)
    kwargs = chain.links[0].litellm_kwargs
    assert kwargs["max_tokens"] == 2_500
    assert kwargs["reasoning_effort"] == "low"
    assert 4_600 + kwargs["max_tokens"] <= 8_000  # ~4.5K-token forecasting prompt


def test_free_mode_deadlines_nest() -> None:
    from metaculus_bot import constants as c

    assert (
        c.FREE_MODE_FORECASTER_CHAIN_BUDGET_S + c.FREE_MODE_UTILITY_LINK_TIMEOUT_S
        <= c.FREE_MODE_FORECASTER_SOFT_DEADLINE_S
    )
    assert c.FREE_MODE_NIM_SLOW_LINK_TIMEOUT_S + c.FREE_MODE_FLOOR_RESERVE_S <= c.FREE_MODE_FORECASTER_CHAIN_BUDGET_S
    assert c.FREE_MODE_FORECASTER_SOFT_DEADLINE_S > c.FORECASTER_SOFT_DEADLINE


def test_forecaster_chains_carry_the_budget(free_env: pytest.MonkeyPatch) -> None:
    from metaculus_bot.constants import FREE_MODE_FORECASTER_CHAIN_BUDGET_S

    assert all(llm.budget_s == FREE_MODE_FORECASTER_CHAIN_BUDGET_S for llm in build_free_forecaster_llms())


@pytest.mark.parametrize(
    "model",
    [
        "zai/glm-4.7-flash",
        "zai/glm-4.5-flash",
        "cloudflare/@cf/openai/gpt-oss-120b",
        "cloudflare/@cf/zai-org/glm-4.7-flash",
        "gemini/gemma-4-31b-it",
    ],
)
def test_new_free_models_allowed(model: str) -> None:
    assert_free_model(model)


@pytest.mark.parametrize(
    "model",
    ["zai/glm-5.3-flash", "zai/glm-5.3", "cloudflare/@cf/moonshotai/kimi-k2.6", "cloudflare/@cf/zai-org/glm-5.2"],
)
def test_paid_or_paid_plan_models_on_new_providers_refused(model: str) -> None:
    with pytest.raises(NotFreeModelError):
        assert_free_model(model)


def test_zai_and_cloudflare_links_drop_without_their_secrets(free_env: pytest.MonkeyPatch) -> None:
    for env in ("ZAI_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
        free_env.delenv(env, raising=False)
    free_env.setenv("CLOUDFLARE_API_TOKEN", "cf")  # token without account id is not enough
    for chain in FREE_MODE_FORECASTER_CHAINS:
        built = build_free_chain(chain, role="x", timeout=10, max_tokens=100)
        assert not any(link.model.startswith(("zai/", "cloudflare/")) for link in built.links)


def test_zai_and_cloudflare_links_are_openai_compatible(free_env: pytest.MonkeyPatch) -> None:
    free_env.setenv("ZAI_API_KEY", "z")
    free_env.setenv("CLOUDFLARE_API_TOKEN", "cf")
    free_env.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    chain = build_free_chain(
        ("zai/glm-4.7-flash", "cloudflare/@cf/openai/gpt-oss-120b"), role="x", timeout=10, max_tokens=100
    )
    zai, cf = chain.links
    assert zai.model == "zai/glm-4.7-flash"
    assert zai.litellm_kwargs["model"] == "openai/glm-4.7-flash"
    assert zai.litellm_kwargs["api_base"] == "https://api.z.ai/api/paas/v4"
    assert zai.litellm_kwargs["api_key"] == "z"
    assert cf.litellm_kwargs["model"] == "openai/@cf/openai/gpt-oss-120b"
    assert cf.litellm_kwargs["api_base"] == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1"
    assert cf.litellm_kwargs["api_key"] == "cf"
    assert free_mode.quota_bucket(cf.model) == "cloudflare"


def test_gemma_has_its_own_pacing_and_quota_bucket() -> None:
    assert free_mode.quota_bucket("gemini/gemma-4-31b-it") == "gemini/gemma-4-31b-it"
    from metaculus_bot.constants import FREE_MODE_MIN_INTERVAL_S

    assert FREE_MODE_MIN_INTERVAL_S["gemma"] > 0
