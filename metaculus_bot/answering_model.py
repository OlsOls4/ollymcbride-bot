"""Which model actually answered a forecaster slot (4 Nov 2026 review fix: truthful attribution).

A FREE_MODE slot is a fallback chain (``free_mode.FreeModelChainLlm``) whose ``model`` is its
first-choice link. When a later link answers, the chain puts :data:`FALLBACK_NOTE_TEMPLATE` at the
top of the reply. Everything that names a forecaster in a comment or a log line (the ``Model:``
line, the summary bullet, ``MEMBER_FORECAST``, ``FORECASTERS_SURVIVED``) reads the answering model
from that note through :func:`answering_model`, so a fallback is never credited to the slot's
first choice. Pure string helpers: no provider imports, safe on the paid path.
"""

from __future__ import annotations

import re

FALLBACK_NOTE_TEMPLATE: str = (
    "_(Answered by `{answered}`: this slot's first-choice model `{primary}` was unavailable.)_"
)

# Current note, plus the pre-4-Nov wording ("Free-mode fallback: answered by `x`; slot primary
# `y` was unavailable.") so re-rendering an older reasoning still attributes it correctly.
_NOTE_RE: re.Pattern[str] = re.compile(
    r"_\((?:Answered|Free-mode fallback: answered) by `([^`]+)`[:;] "
    r"(?:this slot's first-choice model|slot primary) `([^`]+)` was unavailable\.\)_"
)
# The note sits on the first line of the chain's reply, after at most the ``Model:`` line.
_SEARCH_PREFIX_CHARS: int = 1_000


def format_fallback_note(answered: str, primary: str) -> str:
    return FALLBACK_NOTE_TEMPLATE.format(answered=answered, primary=primary)


def fallback_models(reasoning: str) -> tuple[str, str] | None:
    """``(answered, first_choice)`` when the reasoning carries a fallback note, else None."""
    match = _NOTE_RE.search(reasoning[:_SEARCH_PREFIX_CHARS])
    return (match.group(1), match.group(2)) if match else None


def answering_model(reasoning: str, default: str) -> str:
    """The model that wrote ``reasoning``: the fallback note's model, else ``default`` (the slot's)."""
    found = fallback_models(reasoning)
    return found[0] if found else default


def short_model_name(model: str) -> str:
    """Display name: the last slash segment (``nvidia_nim/moonshotai/kimi-k3`` -> ``kimi-k3``)."""
    return model.rsplit("/", 1)[-1].strip() or model
