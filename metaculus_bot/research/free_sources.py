"""FREE_MODE research add-on: zero-cost news, background and (optional) keyed web search.

The ollymcbride-bot fork runs with no paid research primary, so this provider adds three cheap
sources, each failing soft (any error or empty result just drops that sub-section):

* **GDELT DOC 2.0** (no key). Recent news headlines matching the question's key terms.
  Terms (gdeltproject.org/about.html#termsofuse, checked 2026-10-12): "unlimited and unrestricted
  use for any academic, commercial, or governmental use of any kind without fee", with a citation
  to the GDELT Project. The API asks for at most one request every 5 seconds, so calls are gated
  process-wide (``_GDELT_MIN_INTERVAL_S``) and a question that would wait too long skips GDELT.
* **Wikipedia** (no key). Lead-section extracts of the top search hits for background. Requests
  carry a descriptive User-Agent with contact info, per the Wikimedia User-Agent policy.
* **Keyed web search** (optional; first key present wins, the next is tried on failure):
  Tavily (1,000 free credits/month, no card), Exa (free tier $10 credits/month, no card), Serper
  (2,500 one-time free queries, no card). None can bill without a payment method on the account.
  A per-process cap (``FREE_WEB_SEARCH_MAX_PER_RUN``) keeps a MiniBench burst from draining a
  monthly allowance in one run.

Google News RSS is deliberately NOT used: the feed states it is "made available solely for ...
a personal feed reader for personal, non-commercial use. Any other use of the feed is expressly
prohibited", and news.google.com/robots.txt disallows /rss.

Leakage: ``is_benchmarking=True`` hard-disables the whole provider (as ``prediction_market``
does). All three sources return CURRENT content, which for a resolved question can reveal the
outcome, so backtests never see them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import env_flag_enabled, free_mode_enabled
from metaculus_bot.research.providers import ResearchCallable

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "FREE_RESEARCH_ENABLED_ENV",
    "extract_query_terms",
    "free_research_enabled",
    "free_sources_provider",
    "reset_free_sources_state",
]

FREE_RESEARCH_ENABLED_ENV = "FREE_RESEARCH_ENABLED"
FREE_RESEARCH_GDELT_ENV = "FREE_RESEARCH_GDELT"
FREE_RESEARCH_WIKIPEDIA_ENV = "FREE_RESEARCH_WIKIPEDIA"
FREE_WEB_SEARCH_MAX_PER_RUN_ENV = "FREE_WEB_SEARCH_MAX_PER_RUN"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
# Distinct from the paid path's EXA_API_KEY so a paid Exa key can never be picked up here.
EXA_FREE_API_KEY_ENV = "EXA_FREE_API_KEY"
SERPER_API_KEY_ENV = "SERPER_API_KEY"

USER_AGENT = "ollymcbride-bot/1.0 (https://github.com/OlsOls4/ollymcbride-bot; forecasting bot) aiohttp"
_HTTP_TIMEOUT_S = 25.0
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_MIN_INTERVAL_S = 5.5
_GDELT_MAX_WAIT_S = 90.0
_GDELT_TIMESPAN = "30d"
_GDELT_MAX_RECORDS = 10
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_PAGES = 2
_WIKI_EXTRACT_CHARS = 1200
_WEB_RESULTS = 5
_WEB_SNIPPET_CHARS = 500
_DEFAULT_WEB_SEARCH_CAP = 40

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "between",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "each",
        "either",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "least",
        "less",
        "more",
        "most",
        "no",
        "not",
        "of",
        "on",
        "or",
        "other",
        "over",
        "per",
        "same",
        "should",
        "since",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "those",
        "through",
        "to",
        "under",
        "until",
        "up",
        "upon",
        "was",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "would",
        "after",
        "again",
        "against",
        "all",
        "any",
        "above",
        "below",
        "both",
        "new",
        "next",
        "least",
        "many",
        "much",
        "only",
        "own",
        "very",
        "also",
        "amount",
        "number",
        "total",
        "value",
        "according",
        "reported",
        "report",
        "resolve",
        "resolves",
        "resolved",
        "question",
        "say",
        "says",
        "said",
        "end",
        "start",
        "date",
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "week",
        "weeks",
        "today",
        "yes",
    ]
)
_MONTHS = frozenset(
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&'.\-]*")


# --- process-wide state (GDELT pacing clock, keyed-search budget and dead keys) ---
_gdelt_next_start: list[float] = [0.0]
_web_search_used: list[int] = [0]
_dead_search_backends: set[str] = set()


def reset_free_sources_state() -> None:
    """Clear the GDELT clock, the web-search counter and the dead-backend set. Used by tests."""
    _gdelt_next_start[0] = 0.0
    _web_search_used[0] = 0
    _dead_search_backends.clear()


def free_research_enabled() -> bool:
    """On by default under FREE_MODE; ``FREE_RESEARCH_ENABLED=false`` turns it off."""
    if not free_mode_enabled():
        return env_flag_enabled(FREE_RESEARCH_ENABLED_ENV, default=False)
    return env_flag_enabled(FREE_RESEARCH_ENABLED_ENV, default=True)


def _web_search_cap() -> int:
    raw = os.getenv(FREE_WEB_SEARCH_MAX_PER_RUN_ENV, "").strip()
    try:
        return max(0, int(raw)) if raw else _DEFAULT_WEB_SEARCH_CAP
    except ValueError:
        return _DEFAULT_WEB_SEARCH_CAP


def extract_query_terms(title: str, *, max_terms: int = 4) -> list[str]:
    """Salient search terms from a question title: capitalised phrases first, then other words.

    Consecutive capitalised words become one quoted phrase ("Federal Reserve"); stopwords, months,
    numbers and words shorter than 3 characters (GDELT rejects them) are dropped.
    """
    tokens = _WORD_RE.findall(title)
    phrases: list[str] = []
    others: list[str] = []
    run: list[str] = []

    def _flush() -> None:
        if run:
            phrases.append(" ".join(run))
            run.clear()

    for i, raw in enumerate(tokens):
        word = raw.strip(".'-")
        low = word.lower()
        skip = len(word) < 3 or low in _STOPWORDS or low in _MONTHS
        if skip:
            _flush()
            continue
        if word[0].isupper() and i > 0:
            run.append(word)
            continue
        _flush()
        if word[0].isupper():  # first word of the title: only a phrase if it is not a stopword
            run.append(word)
        else:
            others.append(low)
    _flush()
    seen: set[str] = set()
    terms: list[str] = []
    for term in phrases + others:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms[:max_terms]


def _gdelt_query(terms: list[str]) -> str:
    parts = [f'"{t.lower()}"' if " " in t else t.lower() for t in terms]
    return " ".join(parts) + " sourcelang:english"


@dataclass(frozen=True)
class _Hit:
    title: str
    url: str
    when: str = ""
    source: str = ""
    snippet: str = ""


def _format_hits(hits: list[_Hit], *, with_snippets: bool) -> str:
    lines: list[str] = []
    for hit in hits:
        meta = ", ".join(x for x in (hit.source, hit.when) if x)
        line = f"- {hit.title.strip()}" + (f" ({meta})" if meta else "") + f" <{hit.url}>"
        lines.append(line)
        if with_snippets and hit.snippet:
            lines.append(f"  > {hit.snippet.strip()[:_WEB_SNIPPET_CHARS]}")
    return "\n".join(lines)


async def _get_json(session: aiohttp.ClientSession, url: str, *, params: dict[str, str]) -> Any:
    async with session.get(url, params=params) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        if not text.lstrip().startswith(("{", "[")):
            # GDELT answers throttled or malformed queries with a plain-text notice and HTTP 200.
            raise RuntimeError(f"non-JSON reply: {text[:200]}")
        return await resp.json(content_type=None)


async def _gdelt_slot() -> bool:
    """Reserve the next GDELT start slot; False if the wait would exceed ``_GDELT_MAX_WAIT_S``."""
    now = time.monotonic()
    start = max(now, _gdelt_next_start[0])
    if start - now > _GDELT_MAX_WAIT_S:
        return False
    _gdelt_next_start[0] = start + _GDELT_MIN_INTERVAL_S
    if start > now:
        await asyncio.sleep(start - now)
    return True


def _gdelt_date(seen: str) -> str:
    # "20261012T120000Z" -> "2026-10-12"
    return f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 and seen[:8].isdigit() else ""


async def _gdelt_news(session: aiohttp.ClientSession, terms: list[str]) -> list[_Hit]:
    """Recent headlines; retries once with the two strongest terms if the full query finds nothing."""
    attempts = [terms] if len(terms) <= 2 else [terms, terms[:2]]
    for attempt in attempts:
        if not attempt or not await _gdelt_slot():
            return []
        params = {
            "query": _gdelt_query(attempt),
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(_GDELT_MAX_RECORDS),
            "sort": "hybridrel",
            "timespan": _GDELT_TIMESPAN,
        }
        data = await _get_json(session, _GDELT_URL, params=params)
        articles = (data.get("articles") or []) if isinstance(data, dict) else []
        hits = [
            _Hit(
                title=str(a.get("title", "")),
                url=str(a.get("url", "")),
                when=_gdelt_date(str(a.get("seendate", ""))),
                source=str(a.get("domain", "")),
            )
            for a in articles
            if a.get("title") and a.get("url")
        ]
        if hits:
            return hits
    return []


async def _wikipedia(session: aiohttp.ClientSession, terms: list[str]) -> list[tuple[str, str]]:
    """(title, lead extract) for the top Wikipedia search hits on the question's key terms."""
    search = await _get_json(
        session,
        _WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": " ".join(terms),
            "srlimit": str(_WIKI_PAGES),
            "format": "json",
        },
    )
    titles = [str(r["title"]) for r in search.get("query", {}).get("search", []) if r.get("title")]
    if not titles:
        return []
    pages = await _get_json(
        session,
        _WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": "|".join(titles),
            "format": "json",
        },
    )
    out: list[tuple[str, str]] = []
    for page in (pages.get("query", {}).get("pages", {}) or {}).values():
        extract = str(page.get("extract", "")).strip()
        if extract:
            out.append((str(page.get("title", "")), extract[:_WIKI_EXTRACT_CHARS]))
    return out


# --- keyed web search backends -------------------------------------------------------------


async def _post_json(session: aiohttp.ClientSession, url: str, *, headers: dict[str, str], body: dict) -> Any:
    async with session.post(url, json=body, headers=headers) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise _BackendHttpError(resp.status, text[:200])
        return await resp.json(content_type=None)


class _BackendHttpError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


async def _tavily(session: aiohttp.ClientSession, key: str, query: str) -> list[_Hit]:
    data = await _post_json(
        session,
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        body={"query": query, "max_results": _WEB_RESULTS, "search_depth": "basic"},
    )
    return [
        _Hit(
            title=str(r.get("title", "")),
            url=str(r.get("url", "")),
            when=str(r.get("published_date", "") or "")[:10],
            snippet=str(r.get("content", "")),
        )
        for r in data.get("results", [])
        if r.get("url")
    ]


async def _exa(session: aiohttp.ClientSession, key: str, query: str) -> list[_Hit]:
    data = await _post_json(
        session,
        "https://api.exa.ai/search",
        headers={"x-api-key": key},
        body={
            "query": query,
            "numResults": _WEB_RESULTS,
            "type": "auto",
            "contents": {"text": {"maxCharacters": _WEB_SNIPPET_CHARS}},
        },
    )
    return [
        _Hit(
            title=str(r.get("title", "") or r.get("url", "")),
            url=str(r.get("url", "")),
            when=str(r.get("publishedDate", "") or "")[:10],
            snippet=str(r.get("text", "")),
        )
        for r in data.get("results", [])
        if r.get("url")
    ]


async def _serper(session: aiohttp.ClientSession, key: str, query: str) -> list[_Hit]:
    data = await _post_json(
        session,
        "https://google.serper.dev/news",
        headers={"X-API-KEY": key},
        body={"q": query, "num": _WEB_RESULTS},
    )
    return [
        _Hit(
            title=str(r.get("title", "")),
            url=str(r.get("link", "")),
            when=str(r.get("date", "")),
            source=str(r.get("source", "")),
            snippet=str(r.get("snippet", "")),
        )
        for r in data.get("news", [])
        if r.get("link")
    ]


_Backend = Callable[[aiohttp.ClientSession, str, str], Awaitable[list[_Hit]]]
_BACKENDS: tuple[tuple[str, str, _Backend], ...] = (
    ("tavily", TAVILY_API_KEY_ENV, _tavily),
    ("exa", EXA_FREE_API_KEY_ENV, _exa),
    ("serper", SERPER_API_KEY_ENV, _serper),
)
# Out of credits / bad key / over quota: stop calling that backend for the rest of the process.
_DEAD_STATUSES = frozenset({401, 402, 403, 429, 432, 433})


def configured_search_backends() -> list[str]:
    """Names of the keyed backends whose key is set (order = preference)."""
    return [name for name, env, _ in _BACKENDS if os.getenv(env)]


async def _web_search(session: aiohttp.ClientSession, query: str) -> tuple[str, list[_Hit]]:
    if _web_search_used[0] >= _web_search_cap():
        return "", []
    for name, env, backend in _BACKENDS:
        key = os.getenv(env)
        if not key or name in _dead_search_backends:
            continue
        _web_search_used[0] += 1
        try:
            hits = await backend(session, key, query)
        except _BackendHttpError as exc:
            if exc.status in _DEAD_STATUSES:
                _dead_search_backends.add(name)
            logger.warning("FREE_RESEARCH: %s search failed (HTTP %s); trying next backend", name, exc.status)
            continue
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            logger.warning("FREE_RESEARCH: %s search failed (%s); trying next backend", name, type(exc).__name__)
            continue
        if hits:
            return name, hits
    return "", []


async def _soft(label: str, coro: Awaitable[Any], default: Any) -> Any:
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # each free source fails soft by design
        logger.warning("FREE_RESEARCH: %s failed (%s: %s)", label, type(exc).__name__, str(exc)[:200])
        return default


async def gather_free_sources(question: MetaculusQuestion) -> str:
    """Run the enabled free sources concurrently and render one markdown body ("" if nothing)."""
    title = str(getattr(question, "question_text", "") or "")
    terms = extract_query_terms(title)
    timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
        news_task = (
            _soft("gdelt", _gdelt_news(session, terms), [])
            if terms and env_flag_enabled(FREE_RESEARCH_GDELT_ENV, default=True)
            else asyncio.sleep(0, result=[])
        )
        wiki_task = (
            _soft("wikipedia", _wikipedia(session, terms), [])
            if terms and env_flag_enabled(FREE_RESEARCH_WIKIPEDIA_ENV, default=True)
            else asyncio.sleep(0, result=[])
        )
        web_task = (
            _soft("web_search", _web_search(session, title[:300]), ("", []))
            if title and configured_search_backends()
            else asyncio.sleep(0, result=("", []))
        )
        news, wiki, (web_name, web_hits) = await asyncio.gather(news_task, wiki_task, web_task)

    sections: list[str] = []
    if web_hits:
        sections.append(f"### Web search ({web_name})\n" + _format_hits(web_hits, with_snippets=True))
    if news:
        sections.append(
            f"### Recent news headlines (GDELT Project, last {_GDELT_TIMESPAN}; https://www.gdeltproject.org/)\n"
            + _format_hits(news, with_snippets=False)
        )
    if wiki:
        sections.append(
            "### Background (Wikipedia lead sections; may lag recent events)\n"
            + "\n\n".join(f"**{t}**: {x}" for t, x in wiki)
        )
    return "\n\n".join(sections)


def free_sources_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory for the free research add-on. Hard-disabled under ``is_benchmarking`` (leakage)."""

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking or not free_research_enabled():
            return ""
        # An add-on must never redden a run: a raise here would count as an errored provider.
        return await _soft("free_sources", gather_free_sources(question), "")

    return _fetch
