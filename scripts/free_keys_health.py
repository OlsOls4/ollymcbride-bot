"""Infrastructure health check for the optional free keys (run by free_keys_health.yaml).

One tiny request per configured provider, through the bot's own code paths where possible:

* Tavily / Exa / Serper: the exact backends ``metaculus_bot.research.free_sources`` uses in live
  runs (same endpoint, headers, body), with a fixed generic query and ONE call each.
* Groq / Mistral: the provider's free ``/models`` list (checks every allow-listed id exists) plus
  one "Reply OK" chat call through litellm with the same model ids the chains use.

Prints only provider names, HTTP statuses, result counts and model-id presence: never a key,
never question content. It touches no Metaculus question and publishes nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaculus_bot.constants import (
    FREE_MODE_GROQ_FREE_TIER_MODELS,
    FREE_MODE_MISTRAL_FREE_TIER_MODELS,
)
from metaculus_bot.research import free_sources as fs

QUERY = "Federal Reserve interest rate decision"
_MODEL_LISTS = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/models", FREE_MODE_GROQ_FREE_TIER_MODELS),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/models", FREE_MODE_MISTRAL_FREE_TIER_MODELS),
}
_CHAT_PROBE = {"groq": "groq/openai/gpt-oss-20b", "mistral": "mistral/mistral-small-latest"}


async def _search_backends(session: aiohttp.ClientSession) -> list[str]:
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
        except Exception as exc:  # noqa: BLE001  # report every failure kind, never raise
            rows.append(f"{name}: ERROR {type(exc).__name__}")
    return rows


async def _model_lists(session: aiohttp.ClientSession) -> list[str]:
    rows: list[str] = []
    for name, (env, url, wanted) in _MODEL_LISTS.items():
        key = os.getenv(env, "").strip()
        if not key:
            rows.append(f"{name} models: NOT CONFIGURED ({env} empty)")
            continue
        try:
            async with session.get(url, headers={"Authorization": f"Bearer {key}"}) as resp:
                if resp.status != 200:
                    rows.append(f"{name} models: ERROR HTTP {resp.status}")
                    continue
                data: Any = await resp.json(content_type=None)
        except Exception as exc:  # noqa: BLE001
            rows.append(f"{name} models: ERROR {type(exc).__name__}")
            continue
        ids: set[str] = set()
        for m in data.get("data", []) if isinstance(data, dict) else []:
            ids.add(str(m.get("id", "")))
            ids.update(str(a) for a in (m.get("aliases") or []))
        missing = sorted(w for w in wanted if w not in ids)
        rows.append(f"{name} models: OK, wired ids {'all present' if not missing else 'MISSING ' + ', '.join(missing)}")
    return rows


async def _chat_probes() -> list[str]:
    rows: list[str] = []
    for name, model in _CHAT_PROBE.items():
        if not os.getenv(_MODEL_LISTS[name][0], "").strip():
            rows.append(f"{name} chat: NOT CONFIGURED")
            continue
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=64,
                timeout=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            rows.append(f"{name} chat ({model}): OK ({'non-empty' if text else 'EMPTY'} reply)")
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", "")
            rows.append(f"{name} chat ({model}): ERROR {type(exc).__name__} {status}".strip())
    return rows


def _append(path: str, text: str) -> None:
    with open(path, "a") as fh:
        fh.write(text)


async def main() -> int:
    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": fs.USER_AGENT}) as session:
        rows = await _search_backends(session)
        rows += await _model_lists(session)
    rows += await _chat_probes()
    for r in rows:
        level = "warning" if "ERROR" in r or "MISSING" in r else "notice"
        print(f"::{level} title=Free key health::{r}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        text = "### Free key health\n\n" + "\n".join(f"- {r}" for r in rows) + "\n"
        await asyncio.to_thread(_append, summary, text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
