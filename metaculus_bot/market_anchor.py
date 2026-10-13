"""FREE_MODE market anchor: blend a binary forecast toward a liquid same-question market price.

Evidence (ForecastBench backtest, run 38324248999 plus the market-context rerun; numbers in
docs/free-mode.md "Market anchor"): on resolved market-source questions the market price at
freeze beat the free-model chain median clearly, and the post-hoc blend
``p = w * market + (1 - w) * model`` beat the chain beyond its paired bootstrap CI.

Design (all deterministic, no human in the loop):

* The prediction-market research provider calls :func:`select_market_anchor` on its live
  snapshot and :func:`record_anchor` stores the result per question id. Only a row the ranker
  graded ``same_quantity_same_date`` qualifies, and only when it is open, unresolved, a plain
  binary price (no multi-outcome children, no scalar value, no withheld price), not demoted by
  the staleness cap, liquid (real money: at least ``ANCHOR_MIN_REAL_MONEY_USD`` of volume or
  open interest; Manifold: at least ``MANIFOLD_THIN_BETTORS`` bettors and non-zero volume) and,
  when bid and ask are known, with a spread of at most ``ANCHOR_MAX_SPREAD``. PredictIt rows
  carry no liquidity data and never qualify. Several qualifying rows must agree within
  ``ANCHOR_MAX_DISAGREEMENT``; otherwise there is no anchor (ambiguous).
* At aggregation the forecaster pops the anchor for that question and, for a binary question
  only, publishes :func:`blend_toward_market` of the ensemble value, clamped to
  [BINARY_PROB_MIN, BINARY_PROB_MAX]. No anchor means the published value is unchanged.
* The comment carries :func:`format_anchor_note`, which states the venue, its price, the
  weight, the ensemble value before blending and the published value.

Off unless FREE_MODE is on; ``MARKET_ANCHOR_ENABLED=false`` turns it off inside FREE_MODE.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, free_mode_enabled

logger: logging.Logger = logging.getLogger(__name__)

MARKET_ANCHOR_ENABLED_ENV: str = "MARKET_ANCHOR_ENABLED"
# Weight on the market price. Chosen from {0.3, 0.5, 0.7} on resolved ForecastBench questions
# (docs/free-mode.md "Market anchor"); 0.7 was best overall and under leave-one-out selection.
MARKET_ANCHOR_WEIGHT: float = 0.7
MARKET_ANCHOR_WEIGHT_GRID: tuple[float, ...] = (0.3, 0.5, 0.7)
ANCHOR_TIER: str = "same_quantity_same_date"
ANCHOR_MIN_REAL_MONEY_USD: float = 10_000.0  # "strong" tier floor (market_retrieval.types)
ANCHOR_MIN_MANIFOLD_BETTORS: int = 20  # MANIFOLD_THIN_BETTORS
ANCHOR_MAX_SPREAD: float = 0.10
ANCHOR_MAX_DISAGREEMENT: float = 0.15
_VENUE_LABEL: dict[str, str] = {
    "polymarket": "Polymarket",
    "kalshi": "Kalshi",
    "manifold": "Manifold (play money)",
}


@dataclass(frozen=True)
class MarketAnchor:
    platform: str
    prob: float  # the anchor price (mean of the qualifying rows' YES prices)
    url: str
    title: str
    n_markets: int = 1


def market_anchor_enabled() -> bool:
    if not free_mode_enabled():
        return False
    return os.getenv(MARKET_ANCHOR_ENABLED_ENV, "true").strip().lower() not in ("false", "0", "no", "off")


def clamp_binary(p: float) -> float:
    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, float(p)))


def blend_toward_market(model_p: float, market_p: float, weight: float = MARKET_ANCHOR_WEIGHT) -> float:
    """``weight * market + (1 - weight) * model``, clamped to the published binary range."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"market anchor weight must be in [0, 1], got {weight}")
    return clamp_binary(weight * float(market_p) + (1.0 - weight) * float(model_p))


def _is_liquid(row: Any) -> bool:
    platform = getattr(row, "platform", "")
    volume = getattr(row, "total_volume", None)
    if platform == "manifold":
        bettors = getattr(row, "num_bettors", None)
        if volume is not None and volume == 0.0:
            return False
        return bettors is not None and bettors >= ANCHOR_MIN_MANIFOLD_BETTORS
    if platform in ("polymarket", "kalshi"):
        oi = getattr(row, "open_interest", None)
        if volume is None and oi is None:
            return False
        return max(volume or 0.0, oi or 0.0) >= ANCHOR_MIN_REAL_MONEY_USD
    return False  # PredictIt (no liquidity data) and anything unknown


def _qualifies(row: Any, now: datetime) -> bool:
    if getattr(row, "relation_tier", "") != ANCHOR_TIER or getattr(row, "tier_cap_note", ""):
        return False
    if getattr(row, "is_resolved", True) or getattr(row, "price_withheld", False):
        return False
    if getattr(row, "children", ()) or getattr(row, "scalar_estimate", None) is not None:
        return False
    p = getattr(row, "implied_prob_yes", None)
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= float(p) <= 1.0:
        return False
    close = getattr(row, "close_time", None)
    if isinstance(close, datetime):
        close_utc = close if close.tzinfo else close.replace(tzinfo=UTC)
        if close_utc < now:
            return False
    spread = getattr(row, "spread", None)
    if isinstance(spread, (int, float)) and spread > ANCHOR_MAX_SPREAD:
        return False
    return _is_liquid(row)


def select_market_anchor(snapshot: Any, *, now: datetime | None = None) -> MarketAnchor | None:
    """The anchor for a live market snapshot, or None (see the module docstring for the rules)."""
    rows = list(getattr(snapshot, "matches", None) or [])
    now = now or datetime.now(UTC)
    ok = [r for r in rows if _qualifies(r, now)]
    if not ok:
        return None
    probs = [float(r.implied_prob_yes) for r in ok]
    if max(probs) - min(probs) > ANCHOR_MAX_DISAGREEMENT:
        logger.info(
            "MARKET_ANCHOR: %d same-question markets disagree (%.2f-%.2f); no anchor",
            len(ok),
            *(min(probs), max(probs)),
        )
        return None
    ok.sort(key=lambda r: r.rank if isinstance(getattr(r, "rank", None), int) else 10**6)
    lead = ok[0]
    return MarketAnchor(
        platform=str(lead.platform),
        prob=sum(probs) / len(probs),
        url=str(getattr(lead, "market_url", "") or ""),
        title=str(getattr(lead, "market_title", "") or ""),
        n_markets=len(ok),
    )


_lock = threading.Lock()
_anchors: dict[int, MarketAnchor] = {}


def record_anchor(qid: int | None, anchor: MarketAnchor | None) -> None:
    """Store (or clear, with None) the anchor for a question id. No-op without a qid."""
    if qid is None:
        return
    with _lock:
        if anchor is None:
            _anchors.pop(qid, None)
        else:
            _anchors[qid] = anchor


def pop_anchor(qid: int | None) -> MarketAnchor | None:
    if qid is None:
        return None
    with _lock:
        return _anchors.pop(qid, None)


def reset_anchors() -> None:
    with _lock:
        _anchors.clear()


def format_anchor_note(anchor: MarketAnchor, *, model_p: float, published_p: float, weight: float) -> str:
    venue = _VENUE_LABEL.get(anchor.platform, anchor.platform)
    more = f" (mean of {anchor.n_markets} matching markets)" if anchor.n_markets > 1 else ""
    link = f" {anchor.url}" if anchor.url else ""
    return (
        f"**Market anchor:** a liquid {venue} market on this same question{more} traded at "
        f"{anchor.prob:.1%}{link}. The published probability is {weight:.0%} that market price and "
        f"{1 - weight:.0%} the model ensemble ({model_p:.1%}), clamped to "
        f"{BINARY_PROB_MIN:.0%}-{BINARY_PROB_MAX:.0%}: {published_p:.1%}."
    )


def insert_anchor_note(comment: str, note: str) -> str:
    """Put ``note`` right under the comment's leading ``#`` header line (kept first for the
    comment validator and inside the summary head the trimmer preserves); append it when the
    comment has no leading header."""
    if comment.startswith("#"):
        head, sep, rest = comment.partition("\n")
        return f"{head}\n{note}\n\n{rest}" if sep else f"{head}\n{note}\n"
    return f"{comment.rstrip()}\n\n{note}\n"


__all__ = [
    "MARKET_ANCHOR_WEIGHT",
    "MARKET_ANCHOR_WEIGHT_GRID",
    "MarketAnchor",
    "blend_toward_market",
    "clamp_binary",
    "format_anchor_note",
    "insert_anchor_note",
    "market_anchor_enabled",
    "pop_anchor",
    "record_anchor",
    "reset_anchors",
    "select_market_anchor",
]
