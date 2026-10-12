"""FREE_MODE free research add-on (GDELT, Wikipedia, optional keyed web search). Offline only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from metaculus_bot.research import free_sources
from metaculus_bot.research.free_sources import (
    extract_query_terms,
    free_research_enabled,
    free_sources_provider,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    free_sources.reset_free_sources_state()
    monkeypatch.setattr(free_sources, "_GDELT_MIN_INTERVAL_S", 0.0)
    for env in ("TAVILY_API_KEY", "EXA_FREE_API_KEY", "SERPER_API_KEY", "FREE_RESEARCH_ENABLED"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("FREE_MODE", "true")
    yield
    free_sources.reset_free_sources_state()


def _question(title: str = "Will the Federal Reserve cut interest rates before December 2026?") -> Any:
    return SimpleNamespace(question_text=title, id_of_question=1)


def test_extract_terms_prefers_capitalised_phrases_and_drops_noise() -> None:
    terms = extract_query_terms("Will the Federal Reserve cut interest rates before December 31, 2026?")
    assert terms[0] == "Federal Reserve"
    assert "december" not in [t.lower() for t in terms]
    assert all(len(t) >= 3 for t in terms)
    assert len(terms) <= 4


def test_extract_terms_empty_title() -> None:
    assert extract_query_terms("") == []


def test_enabled_by_default_only_in_free_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert free_research_enabled() is True
    monkeypatch.setenv("FREE_RESEARCH_ENABLED", "false")
    assert free_research_enabled() is False
    monkeypatch.delenv("FREE_RESEARCH_ENABLED")
    monkeypatch.delenv("FREE_MODE")
    assert free_research_enabled() is False


async def test_benchmarking_hard_disables_without_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network must not be touched under is_benchmarking")

    monkeypatch.setattr(free_sources, "gather_free_sources", _boom)
    assert await free_sources_provider(is_benchmarking=True)(_question()) == ""


def _fake_get(responses: dict[str, Any]):
    async def _get_json(_session: Any, url: str, *, params: dict[str, str]) -> Any:
        if "gdelt" in url:
            return responses["gdelt"](params)
        if params.get("list") == "search":
            return responses["wiki_search"]
        return responses["wiki_pages"]

    return _get_json


async def test_renders_gdelt_and_wikipedia_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        free_sources,
        "_get_json",
        _fake_get(
            {
                "gdelt": lambda p: {
                    "articles": [
                        {
                            "title": "Fed holds rates",
                            "url": "https://x.test/a",
                            "seendate": "20261010T120000Z",
                            "domain": "x.test",
                        }
                    ]
                },
                "wiki_search": {"query": {"search": [{"title": "Federal funds rate"}]}},
                "wiki_pages": {"query": {"pages": {"1": {"title": "Federal funds rate", "extract": "The rate..."}}}},
            }
        ),
    )
    out = await free_sources_provider()(_question())
    assert "GDELT Project" in out
    assert "Fed holds rates (x.test, 2026-10-10) <https://x.test/a>" in out
    assert "**Federal funds rate**: The rate..." in out
    assert "Web search" not in out


async def test_gdelt_retries_with_fewer_terms_then_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def _gdelt(params: dict[str, str]) -> Any:
        queries.append(params["query"])
        return {"articles": []}

    async def _wiki_fail(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("wiki down")

    monkeypatch.setattr(free_sources, "_get_json", _fake_get({"gdelt": _gdelt, "wiki_search": {}, "wiki_pages": {}}))
    monkeypatch.setattr(free_sources, "_wikipedia", _wiki_fail)
    out = await free_sources_provider()(_question())
    assert out == ""
    assert len(queries) == 2
    assert len(queries[1]) < len(queries[0])


async def test_gdelt_throttle_notice_is_a_soft_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gdelt(_p: dict[str, str]) -> Any:
        raise RuntimeError("non-JSON reply: Please limit requests to one every 5 seconds")

    monkeypatch.setattr(
        free_sources,
        "_get_json",
        _fake_get({"gdelt": _gdelt, "wiki_search": {"query": {"search": []}}, "wiki_pages": {}}),
    )
    assert await free_sources_provider()(_question()) == ""


async def test_gdelt_skipped_when_queue_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_sources, "_GDELT_MIN_INTERVAL_S", 1000.0)
    assert await free_sources._gdelt_slot() is True
    assert await free_sources._gdelt_slot() is False


async def test_keyed_search_falls_through_dead_backend_and_respects_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tv")
    monkeypatch.setenv("SERPER_API_KEY", "sp")
    monkeypatch.setenv("FREE_WEB_SEARCH_MAX_PER_RUN", "3")
    calls: list[str] = []

    async def _tavily(_s: Any, _k: str, _q: str) -> Any:
        calls.append("tavily")
        raise free_sources._BackendHttpError(432, "plan limit")

    async def _serper(_s: Any, key: str, _q: str) -> Any:
        calls.append("serper")
        assert key == "sp"
        return [free_sources._Hit(title="T", url="https://s.test", snippet="snip")]

    monkeypatch.setattr(
        free_sources,
        "_BACKENDS",
        (("tavily", "TAVILY_API_KEY", _tavily), ("serper", "SERPER_API_KEY", _serper)),
    )
    name, hits = await free_sources._web_search(None, "q")  # type: ignore[arg-type]
    assert name == "serper"
    assert hits[0].url == "https://s.test"
    assert "tavily" in free_sources._dead_search_backends
    name2, _ = await free_sources._web_search(None, "q")  # type: ignore[arg-type]
    assert name2 == "serper"
    assert calls == ["tavily", "serper", "serper"]
    assert await free_sources._web_search(None, "q") == ("", [])  # type: ignore[arg-type]  # cap of 3 reached


def test_configured_backends_follow_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    assert free_sources.configured_search_backends() == []
    monkeypatch.setenv("EXA_FREE_API_KEY", "ex")
    monkeypatch.setenv("EXA_API_KEY", "paid-key-must-be-ignored")
    assert free_sources.configured_search_backends() == ["exa"]


def test_orchestrator_selects_free_sources_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research.orchestrator import ResearchOrchestrator

    for flag in (
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED",
    ):
        monkeypatch.setenv(flag, "false")
    from unittest.mock import MagicMock

    def _orch() -> ResearchOrchestrator:
        return ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

    names = [n for _, n in _orch()._select_research_providers()]
    assert "free_sources" in names
    monkeypatch.setenv("FREE_RESEARCH_ENABLED", "false")
    names = [n for _, n in _orch()._select_research_providers()]
    assert "free_sources" not in names
