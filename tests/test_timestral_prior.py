"""Pure tests for the Timestral prior blend (no torch needed)."""

import numpy as np
from scipy.stats import norm

from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.research import timestral_prior as tp


def _prior(mu: float, sd: float) -> tp.TimestralPrior:
    lv = np.linspace(0.05, 0.95, 19)
    return tp.TimestralPrior(levels=lv, values=norm.ppf(lv, mu, sd), label="T", horizon=5)


def _ens(x: np.ndarray, mu: float, sd: float, *, open_: bool) -> np.ndarray:
    c = norm.cdf(x, mu, sd)
    if open_:
        return np.clip(c, 0.001, 0.999)
    return (c - c[0]) / (c[-1] - c[0])


def test_blend_open_bounds_is_legal_and_moves_median() -> None:
    x = np.linspace(80, 130, 201)
    lo, hi = grid_step_constraints(201)
    ens = _ens(x, 110, 6, open_=True)
    mix = tp.blend_cdf(ens, x, _prior(95, 6), open_lower=True, open_upper=True, min_step=lo, max_step=hi)
    assert mix is not None
    assert np.all(np.diff(mix) >= lo * 0.999)
    assert np.all(np.diff(mix) <= hi)
    assert mix[0] >= ens[0]
    assert mix[-1] <= ens[-1]
    assert np.interp(0.5, mix, x) < np.interp(0.5, ens, x)


def test_blend_closed_bounds_pins_endpoints() -> None:
    x = np.linspace(80, 130, 201)
    lo, hi = grid_step_constraints(201)
    mix = tp.blend_cdf(
        _ens(x, 105, 6, open_=False), x, _prior(100, 6), open_lower=False, open_upper=False, min_step=lo, max_step=hi
    )
    assert mix is not None
    assert mix[0] == 0.0
    assert abs(mix[-1] - 1.0) < 1e-9


def test_blend_rejects_prior_outside_bounds() -> None:
    x = np.linspace(80, 130, 201)
    lo, hi = grid_step_constraints(201)
    mix = tp.blend_cdf(
        _ens(x, 105, 6, open_=False), x, _prior(5000, 6), open_lower=False, open_upper=False, min_step=lo, max_step=hi
    )
    assert mix is None


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(tp.TIMESTRAL_PRIOR_ENABLED_ENV, raising=False)
    assert not tp.enabled()
