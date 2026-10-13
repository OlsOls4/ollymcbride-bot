"""Infrastructure health check for the free LLM / search keys (run by free_keys_health.yaml).

Small, fixed probes through the providers' own endpoints. Nothing here reads a Metaculus
question, forecasts anything, or publishes: every chat probe uses ONE synthetic, long-resolved
historical question (``_SYNTHETIC_QUESTION``) purely to measure latency and token limits at a
realistic prompt size. Prints only provider names, model ids, HTTP statuses, timings, token
counts and rate-limit header VALUES. Never a key, never a URL that carries one.

Probe groups (each skipped when its key is unset):

* search (opt-in, ``--search``): Tavily / Exa / Serper through the exact live backends in
  ``metaculus_bot.research.free_sources``, one call each.
* groq: ``/models`` presence of the wired ids, then one realistic-size chat call with the live
  Groq output cap, printing the ``x-ratelimit-*`` headers (TPM is the binding Free-plan limit).
* mistral: ``/models`` presence of candidate ids, then one tiny chat call per candidate, 2 s
  apart, printing status, Mistral's error ``type``/``code`` and its per-model
  ``x-ratelimit-limit-req-minute`` / ``-tokens-minute`` headers (a 0 means "not on this tier").
* nim: one STREAMED realistic-size call per candidate model, concurrently (2 s apart), with
  time-to-first-token, total time, reasoning/content sizes and finish_reason. This is what
  shows whether a slow model can answer inside the bot's per-forecaster budget.
* gemma: AI Studio ``/models`` presence of the free Gemma ids (no quota), then ONE short call
  to the wired Gemma model with a system prompt (Gemma's own quota, not Gemini Flash's).
* chain (``--chain``): one call through the bot's OWN ``build_free_chain`` for each slow NIM
  model, proving the streamed link + chain budget path end to end.
* zai / cloudflare: one tiny chat call each, only once their optional secrets exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaculus_bot import constants as C
from metaculus_bot import free_mode
from metaculus_bot.research import free_sources as fs

QUERY = "Federal Reserve interest rate decision"

# One long-resolved question (resolved Sep 2024), padded with neutral background so the prompt is
# roughly the size of a live forecasting prompt (~4.5K tokens). Never an open question.
_SYNTHETIC_QUESTION = (
    "You are a professional forecaster. Question (historical, for a latency test): Will the US "
    "Federal Reserve lower the federal funds target range at its FOMC meeting ending 18 September "
    "2024? Resolution: YES if the FOMC statement of 18 Sep 2024 announces a lower target range.\n\n"
    "Background notes:\n"
)
_PAD = (
    "The Federal Open Market Committee meets eight times a year. Markets price policy moves through "
    "fed funds futures, and officials signal intentions in speeches, minutes and the dot plot. "
    "Inflation, labour-market data and financial conditions all feed the decision. "
)
_TAIL = "\n\nThink briefly, then finish with one line exactly of the form 'Probability: NN%'."


def synthetic_prompt(target_chars: int = 17_000) -> str:
    body = _SYNTHETIC_QUESTION
    while len(body) < target_chars:
        body += _PAD
    return body + _TAIL


NIM_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_CANDIDATES = (
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
)
NIM_PROBE_MAX_TOKENS = 8_192
NIM_PROBE_CAP_S = 540.0

MISTRAL_CANDIDATES = (
    "mistral-medium-latest",
    "mistral-small-latest",
    "magistral-small-latest",
    "ministral-14b-latest",
    "ministral-8b-latest",
    "open-mistral-nemo",
)
GEMMA_IDS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")

_HDR_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after")


def _rl_headers(headers: Any) -> str:
    pairs = [f"{k.lower()}={v}" for k, v in headers.items() if k.lower().startswith(_HDR_PREFIXES)]
    return ", ".join(sorted(pairs)) or "no rate-limit headers"


async def _error_fields(resp: aiohttp.ClientResponse) -> str:
    """Provider error type/code/short message: never echoes the request."""
    try:
        data = json.loads(await resp.text())
    except Exception:  # noqa: BLE001
        return ""
    err = data.get("error", data) if isinstance(data, dict) else {}
    if not isinstance(err, dict):
        return str(err)[:160]
    parts = [f"{k}={str(err[k])[:160]}" for k in ("type", "code", "message") if err.get(k)]
    return " ".join(parts)


# Search backends probes.
async def probe_search(session: aiohttp.ClientSession) -> list[str]:
    rows: list[str] = []
    for name, env, backend in fs._BACKENDS:  # the live backends, by design
        key = os.getenv(env, "").strip()
        if not key:
            rows.append(f"{name}: NOT CONFIGURED ({env} empty)")
            continue
        try:
            hits = await backend(session, key, QUERY)
            rows.append(f"{name}: OK ({len(hits)} results)")
        except fs._BackendHttpError as exc:
            rows.append(f"{name}: ERROR HTTP {exc.status}")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"{name}: ERROR {type(exc).__name__}")
    return rows


async def _model_ids(session: aiohttp.ClientSession, url: str, key: str) -> tuple[int, set[str]]:
    async with session.get(url, headers={"Authorization": f"Bearer {key}"}) as resp:
        if resp.status != 200:
            return resp.status, set()
        data: Any = await resp.json(content_type=None)
    ids: set[str] = set()
    for m in data.get("data", []) if isinstance(data, dict) else []:
        ids.add(str(m.get("id", "")))
        ids.update(str(a) for a in (m.get("aliases") or []))
    return 200, ids


# Groq probes.
async def probe_groq(session: aiohttp.ClientSession) -> list[str]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return ["groq: NOT CONFIGURED"]
    rows: list[str] = []
    status, ids = await _model_ids(session, "https://api.groq.com/openai/v1/models", key)
    missing = sorted(m for m in C.FREE_MODE_GROQ_FREE_TIER_MODELS if m not in ids)
    rows.append(
        f"groq models: HTTP {status}, wired ids {'all present' if not missing else 'MISSING ' + ', '.join(missing)}"
    )
    model = "openai/gpt-oss-120b"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt()}],
        "max_tokens": C.FREE_MODE_GROQ_MAX_TOKENS,
        **C.FREE_MODE_GROQ_EXTRA_PARAMS,
    }
    t0 = time.monotonic()
    async with session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=body,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        dt = time.monotonic() - t0
        hdr = _rl_headers(resp.headers)
        if resp.status != 200:
            rows.append(
                f"groq chat {model} (max_tokens={body['max_tokens']}): HTTP {resp.status} {await _error_fields(resp)} | {hdr}"
            )
            return rows
        data = await resp.json(content_type=None)
    usage = data.get("usage", {})
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    rows.append(
        f"groq chat {model}: OK {dt:.0f}s prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')} finish={choice.get('finish_reason')} "
        f"has_probability={'probability' in text.lower()} | {hdr}"
    )
    return rows


# Mistral probes.
async def probe_mistral(session: aiohttp.ClientSession) -> list[str]:
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        return ["mistral: NOT CONFIGURED"]
    rows: list[str] = []
    status, ids = await _model_ids(session, "https://api.mistral.ai/v1/models", key)
    present = [m for m in MISTRAL_CANDIDATES if m in ids]
    rows.append(f"mistral models: HTTP {status}, candidates listed: {', '.join(present) or 'none'}")
    for model in MISTRAL_CANDIDATES:
        await asyncio.sleep(2.0)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 16,
        }
        try:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                hdr = _rl_headers(resp.headers)
                extra = "" if resp.status == 200 else " " + await _error_fields(resp)
                rows.append(f"mistral chat {model}: HTTP {resp.status}{extra} | {hdr}")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"mistral chat {model}: ERROR {type(exc).__name__}")
    return rows


# Nim (streamed) probes.
class _StreamStats:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.ttft: float | None = None
        self.n_content = 0
        self.n_reason = 0
        self.finish: str | None = None
        self.tail = ""

    def feed(self, line: str) -> bool:
        """Consume one SSE line; False once the stream says [DONE]."""
        if not line.startswith("data:"):
            return True
        payload = line[5:].strip()
        if payload == "[DONE]":
            return False
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return True
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            c = delta.get("content") or ""
            r = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (c or r) and self.ttft is None:
                self.ttft = time.monotonic() - self.t0
            self.n_content += len(c)
            self.n_reason += len(r)
            self.tail = (self.tail + c)[-400:]
            self.finish = ch.get("finish_reason") or self.finish
        return True

    def describe(self) -> str:
        ttft = f"{self.ttft:.0f}s" if self.ttft is not None else "none"
        return (
            f"ttft={ttft} content_chars={self.n_content} reasoning_chars={self.n_reason} "
            f"finish={self.finish} has_probability={'probability' in self.tail.lower()}"
        )


async def _nim_stream(session: aiohttp.ClientSession, key: str, model: str, delay: float) -> str:
    await asyncio.sleep(delay)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt()}],
        "max_tokens": NIM_PROBE_MAX_TOKENS,
        "stream": True,
    }
    st = _StreamStats()
    try:
        async with session.post(
            NIM_BASE,
            headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"},
            json=body,
            timeout=aiohttp.ClientTimeout(total=NIM_PROBE_CAP_S, sock_read=180),
        ) as resp:
            if resp.status != 200:
                elapsed = time.monotonic() - st.t0
                return f"nim {model}: HTTP {resp.status} after {elapsed:.0f}s {await _error_fields(resp)} | {_rl_headers(resp.headers)}"
            async for raw in resp.content:
                if not st.feed(raw.decode("utf-8", "ignore").strip()):
                    break
    except TimeoutError:
        return f"nim {model}: TIMEOUT at {time.monotonic() - st.t0:.0f}s ({st.describe()})"
    except Exception as exc:  # noqa: BLE001
        return f"nim {model}: ERROR {type(exc).__name__} after {time.monotonic() - st.t0:.0f}s"
    return f"nim {model}: OK total={time.monotonic() - st.t0:.0f}s {st.describe()}"


async def probe_nim(session: aiohttp.ClientSession) -> list[str]:
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not key:
        return ["nim: NOT CONFIGURED"]
    tasks = [_nim_stream(session, key, m, 2.0 * i) for i, m in enumerate(NIM_CANDIDATES)]
    return list(await asyncio.gather(*tasks))


# Gemma (ai studio) probes.
async def probe_gemma(session: aiohttp.ClientSession) -> list[str]:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return ["gemma: NOT CONFIGURED"]
    rows: list[str] = []
    try:
        async with session.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key},
            params={"pageSize": "1000"},
        ) as resp:
            data: Any = await resp.json(content_type=None) if resp.status == 200 else {}
            names = {str(m.get("name", "")).removeprefix("models/") for m in data.get("models", [])}
            rows.append(
                f"gemma models: HTTP {resp.status}, listed: {', '.join(g for g in GEMMA_IDS if g in names) or 'none'}"
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(f"gemma models: ERROR {type(exc).__name__}")
    for gid in GEMMA_IDS:
        model = "gemini/" + gid
        t0 = time.monotonic()
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Reply with the single word OK."},
                ],
                max_tokens=32,
                timeout=90,
                api_key=key,
                max_retries=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            rows.append(
                f"gemma chat {model} (with system prompt): OK {time.monotonic() - t0:.0f}s "
                f"({'non-empty' if text else 'EMPTY'})"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                f"gemma chat {model}: ERROR {type(exc).__name__} {getattr(exc, 'status_code', '')} {str(exc)[:160]}"
            )
        await asyncio.sleep(C.FREE_MODE_MIN_INTERVAL_S.get("gemma", 20.0))
    return rows


# Optional new providers probes.
async def probe_optional_openai_compat(session: aiohttp.ClientSession) -> list[str]:
    rows: list[str] = []
    # Every allow-listed Z.ai and Cloudflare id (a handful of 32-token calls), not just the probe model.
    probes = [("zai", m) for m in sorted(C.FREE_MODE_ZAI_FREE_TIER_MODELS)]
    probes += [("cloudflare", m) for m in sorted(C.FREE_MODE_CLOUDFLARE_FREE_TIER_MODELS)]
    for name, model in probes:
        base, key = C.free_openai_compat_endpoint(name)
        if not (base and key):
            rows.append(f"{name}: NOT CONFIGURED")
            continue
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 32,
        }
        try:
            async with session.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                extra = "" if resp.status == 200 else " " + await _error_fields(resp)
                rows.append(f"{name} chat {model}: HTTP {resp.status}{extra} | {_rl_headers(resp.headers)}")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"{name} chat {model}: ERROR {type(exc).__name__}")
    return rows


# The bot's own chain path probes: the slow NIM links, every non-floor link of the live chains
# (end to end through build_free_chain with a realistic-size prompt), and the allow-listed
# candidates that were swapped out after a failed health run.
SWAPPED_OUT_CANDIDATES: tuple[str, ...] = (
    "zai/glm-4.7-flash",
    "zai/glm-4.5-flash",
    "gemini/gemma-4-31b-it",
    "nvidia_nim/moonshotai/kimi-k2.6",
)


def _chain_probe_models() -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for m in C.FREE_MODE_NIM_SLOW_MODELS:
        seen[m] = None
    for chain in C.FREE_MODE_FORECASTER_CHAINS[: C.FREE_MODE_ENSEMBLE_SIZE_DEFAULT]:
        for m in chain:
            # Skip OpenRouter (never used here) and Gemini Flash / Flash-Lite (keep their quota).
            if m.startswith(("openrouter/", "gemini/gemini-")):
                continue
            seen[m] = None
    for m in SWAPPED_OUT_CANDIDATES:
        seen[m] = None
    return tuple(seen)


CHAIN_PROBE_MODELS: tuple[str, ...] = _chain_probe_models()

_KEY_ENV: dict[str, str] = {
    "nvidia_nim": "NVIDIA_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _has_key(model: str) -> bool:
    provider = model.split("/", 1)[0]
    if provider in ("zai", "cloudflare"):
        return all(C.free_openai_compat_endpoint(provider))
    env = _KEY_ENV.get(provider)
    return bool(env and os.getenv(env, "").strip())


async def probe_chain() -> list[str]:
    rows: list[str] = []
    if not os.getenv("NVIDIA_API_KEY", "").strip():
        return ["chain: NOT CONFIGURED"]

    async def one(model: str) -> str:
        chain = free_mode.build_free_chain(
            (model,),
            role="health:chain",
            timeout=C.FREE_MODE_FORECASTER_LINK_TIMEOUT_S,
            max_tokens=C.FREE_MODE_FORECASTER_MAX_TOKENS,
            budget_s=C.FREE_MODE_FORECASTER_CHAIN_BUDGET_S,
        )
        t0 = time.monotonic()
        try:
            text = await chain.invoke(synthetic_prompt())
            return f"chain {model}: OK {time.monotonic() - t0:.0f}s has_probability={'probability' in text.lower()}"
        except Exception as exc:  # noqa: BLE001
            return f"chain {model}: FAILED {time.monotonic() - t0:.0f}s {type(exc).__name__} {str(exc)[:200]}"

    rows += await asyncio.gather(*(one(m) for m in CHAIN_PROBE_MODELS if _has_key(m)))
    return rows


def _append(path: str, text: str) -> None:
    with open(path, "a") as fh:
        fh.write(text)


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true", help="also probe Tavily/Exa/Serper (1 credit each)")
    p.add_argument("--chain", action="store_true", help="also run the bot's own chain for the slow NIM models")
    args = p.parse_args(argv)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": fs.USER_AGENT}) as session:
        groups = [probe_groq(session), probe_mistral(session), probe_nim(session), probe_gemma(session)]
        groups.append(probe_optional_openai_compat(session))
        if args.search:
            groups.append(probe_search(session))
        if args.chain:
            groups.append(probe_chain())
        results = await asyncio.gather(*groups, return_exceptions=True)
    rows: list[str] = []
    for res in results:
        if isinstance(res, BaseException):
            rows.append(f"probe group crashed: {type(res).__name__}")
        else:
            rows.extend(res)
    for r in rows:
        bad = any(s in r for s in ("ERROR", "MISSING", "TIMEOUT", "FAILED", "HTTP 4", "HTTP 5", "crashed"))
        print(f"::{'warning' if bad else 'notice'} title=Free key health::{r}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        text = "### Free key health\n\n" + "\n".join(f"- {r}" for r in rows) + "\n"
        await asyncio.to_thread(_append, summary, text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
