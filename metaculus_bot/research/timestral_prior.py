"""Timestral prior: blend a time-series model's forecast into series-tied numeric questions.

FREE_MODE add-on, zero quota: Mistral's Timestral-28M (Apache-2.0, released 2026-10-15) runs on
CPU inside the GitHub runner, no API key. Evidence (docs/free-mode.md "Timestral prior"): on 2,892
resolved ForecastBench "will the series rise" questions it scored Brier 0.207 vs 0.250 for 50%,
and in the head-to-head on 60 resolved questions (run 39127193293) the blend
0.3 x Timestral + 0.7 x chain median was the only variant that beat the chain beyond its paired
95% CI (dBrier -0.013 [-0.024, -0.002]). So the weight here is 0.3.

Scope (deliberately narrow, every guard fails OPEN to the unchanged ensemble):
- numeric questions only, routed by the time-series anchor to ONE series (no spreads), with
  derivation "level", scale 1, not a forward-max question, and only when the anchor section
  itself passed its own band and bounds guards;
- history is the anchor's point-in-time series (cut at as_of), horizon = native steps from the
  last observation to the scheduled resolution date;
- the published CDF is 0.7 x ensemble + 0.3 x Timestral (renormalised onto closed bounds), then
  the usual min-step sweep; any constraint the mixture cannot meet drops the blend.

Switch off with TIMESTRAL_PRIOR_ENABLED=false (default off when unset). The torch/timestral
packages are NOT in the lockfile: the workflows install them in a continue-on-error step, and a
missing import simply means no prior.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from metaculus_bot.numeric.pchip_cdf import enforce_min_steps

logger = logging.getLogger(__name__)

TIMESTRAL_PRIOR_ENABLED_ENV = "TIMESTRAL_PRIOR_ENABLED"
TIMESTRAL_MODEL_ID: str = os.environ.get("TIMESTRAL_MODEL", "mistralai/Timestral-28M")
TIMESTRAL_PRIOR_WEIGHT: float = 0.3
TIMESTRAL_TIMEOUT_S: float = float(os.environ.get("TIMESTRAL_TIMEOUT", "45"))
TIMESTRAL_MIN_HISTORY: int = 60
TIMESTRAL_MAX_CONTEXT: int = 4096
TIMESTRAL_MAX_HORIZON: int = 1024

_PENDING: dict[int, tuple[np.ndarray, int, str]] = {}
_PRIORS: dict[int, TimestralPrior] = {}
_MODEL: Any = None
_MODEL_FAILED = False


@dataclass(frozen=True)
class TimestralPrior:
    levels: np.ndarray  # quantile levels, increasing
    values: np.ndarray  # quantile values at the resolution step, increasing
    label: str
    horizon: int


def enabled() -> bool:
    return os.environ.get(TIMESTRAL_PRIOR_ENABLED_ENV, "false").strip().lower() in ("1", "true", "yes", "on")


def reset() -> None:
    _PENDING.clear()
    _PRIORS.clear()


def stage(qid: int | None, series: Any, *, route: Any, scheduled_date: Any) -> None:
    """Record the history + horizon for one routed question (called on the anchor's success path)."""
    if qid is None or not enabled():
        return
    try:
        if route.kind != "single" or route.derivation != "level" or route.is_max or float(route.scale) != 1.0:
            return
        if not route.model_target:
            return
        s = series.dropna()
        if len(s) < TIMESTRAL_MIN_HISTORY:
            return
        last = s.index[-1].date() if hasattr(s.index[-1], "date") else s.index[-1]
        days = (scheduled_date - last).days
        if days <= 0:
            return
        # Native steps from the series' own density over its last year (as in the study).
        yr = s[s.index >= s.index[-1] - np.timedelta64(365, "D")]
        per_day = max(len(yr), 2) / 365.0
        h = max(1, round(days * per_day))
        if h > TIMESTRAL_MAX_HORIZON:
            return
        vals = np.asarray(s.values[-TIMESTRAL_MAX_CONTEXT:], dtype=float)
        if not np.all(np.isfinite(vals)):
            return
        _PENDING[qid] = (vals, h, str(route.label or route.spec.series_id))
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.info("timestral: stage skipped for qid=%s (%s)", qid, exc)


def _load_model() -> Any:
    global _MODEL, _MODEL_FAILED  # noqa: PLW0603 - per-process model cache
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        from timestral import TimestralForecaster  # type: ignore[import-not-found]  # noqa: PLC0415 - optional dep

        _MODEL = TimestralForecaster.from_pretrained(TIMESTRAL_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - not installed / no weights: no prior
        _MODEL_FAILED = True
        logger.warning("timestral: model unavailable (%s: %s); prior off for this run", type(exc).__name__, exc)
    return _MODEL


def _forecast(vals: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    model = _load_model()
    if model is None:
        return None
    f = model.forecast(vals, prediction_length=h)
    q = np.asarray(f.quantiles, dtype=float)[..., -1, :].reshape(-1)
    lv = np.asarray(f.quantile_levels, dtype=float).reshape(-1)
    if q.size != lv.size or q.size < 3 or not np.all(np.isfinite(q)):
        return None
    q = np.maximum.accumulate(q)
    return lv, q


async def compute(qid: int | None) -> None:
    """Run Timestral for a staged question (own timeout, never raises)."""
    if qid is None or qid not in _PENDING:
        return
    vals, h, label = _PENDING.pop(qid)
    try:
        out = await asyncio.wait_for(asyncio.to_thread(_forecast, vals, h), timeout=TIMESTRAL_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - timeout or model error: no prior
        logger.warning("timestral: forecast failed for qid=%s (%s: %s)", qid, type(exc).__name__, exc)
        return
    if out is None:
        return
    lv, q = out
    _PRIORS[qid] = TimestralPrior(levels=lv, values=q, label=label, horizon=h)
    logger.info(
        "timestral: qid=%s series=%s h=%s P10~%.6g P50~%.6g P90~%.6g",
        qid,
        label,
        h,
        float(np.interp(0.1, lv, q)),
        float(np.interp(0.5, lv, q)),
        float(np.interp(0.9, lv, q)),
    )


def pop_prior(qid: int | None) -> TimestralPrior | None:
    return _PRIORS.pop(qid, None) if qid is not None else None


def prior_cdf(prior: TimestralPrior, x: np.ndarray) -> np.ndarray:
    """Piecewise-linear CDF through the quantiles, tails extended linearly to 0 and 1."""
    lv, q = prior.levels, prior.values
    lo_slope = (q[1] - q[0]) / max(lv[1] - lv[0], 1e-9)
    hi_slope = (q[-1] - q[-2]) / max(lv[-1] - lv[-2], 1e-9)
    xs = np.concatenate(([q[0] - lv[0] * max(lo_slope, 1e-12)], q, [q[-1] + (1 - lv[-1]) * max(hi_slope, 1e-12)]))
    ps = np.concatenate(([0.0], lv, [1.0]))
    # strictly increasing x for interp
    for i in range(1, xs.size):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-12 * max(1.0, abs(xs[i - 1]))
    return np.interp(x, xs, ps, left=0.0, right=1.0)


def blend_cdf(  # noqa: C901 - flat list of guards
    ens: np.ndarray,
    x: np.ndarray,
    prior: TimestralPrior,
    *,
    open_lower: bool,
    open_upper: bool,
    min_step: float,
    max_step: float,
    weight: float = TIMESTRAL_PRIOR_WEIGHT,
) -> np.ndarray | None:
    """0.7 x ensemble + 0.3 x Timestral on the question's grid; None when the result is not server-legal."""
    ts = prior_cdf(prior, x)
    a, b = float(ts[0]), float(ts[-1])
    if not open_lower or not open_upper:
        lo = 0.0 if open_lower else a
        hi = 1.0 if open_upper else b
        if hi - lo < 0.05:  # Timestral puts almost nothing inside the bounds: wrong units, skip
            return None
        ts = (ts - lo) / (hi - lo)
        if not open_lower:
            ts[0] = 0.0
        if not open_upper:
            ts[-1] = 1.0
    if open_lower and open_upper and (b - a) < 0.05:
        return None
    mix = (1 - weight) * ens + weight * np.clip(ts, 0.0, 1.0)
    # Keep at least the ensemble's own open-tail mass (it already met the server's tail rules).
    if open_lower:
        mix[0] = max(mix[0], ens[0])
    if open_upper:
        mix[-1] = min(mix[-1], ens[-1])
    mix = enforce_min_steps(
        mix, min_step, upper_cap=float(mix[-1]) if open_upper else 1.0, lower_cap=float(mix[0]) if open_lower else 0.0
    )
    steps = np.diff(mix)
    if not np.all(np.isfinite(mix)) or np.any(steps < min_step * 0.999) or np.any(steps > max_step):
        return None
    if not open_lower and abs(mix[0]) > 1e-9:
        return None
    if not open_upper and abs(mix[-1] - 1.0) > 1e-9:
        return None
    return mix


def format_note(prior: TimestralPrior, weight: float = TIMESTRAL_PRIOR_WEIGHT) -> str:
    p10, p50, p90 = (float(np.interp(p, prior.levels, prior.values)) for p in (0.1, 0.5, 0.9))
    return (
        f"**Time-series prior:** Mistral's Timestral-28M model, run on the {prior.label} history "
        f"({prior.horizon} steps ahead), forecast a median of {p50:.6g} (10-90%: {p10:.6g} to {p90:.6g}). "
        f"The published distribution is {1 - weight:.0%} the model ensemble and {weight:.0%} this forecast."
    )


__all__ = [
    "TIMESTRAL_PRIOR_ENABLED_ENV",
    "TIMESTRAL_PRIOR_WEIGHT",
    "TimestralPrior",
    "blend_cdf",
    "compute",
    "enabled",
    "format_note",
    "pop_prior",
    "prior_cdf",
    "reset",
    "stage",
]
