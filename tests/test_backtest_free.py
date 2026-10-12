"""Pure helpers of backtest_free.py (offline ensemble aggregation, scoring, report)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import backtest_free as bf


def _bin(p: float) -> bf.MemberForecast:
    return bf.MemberForecast("binary", binary=p)


def test_import_does_not_touch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert bf._REMOVED_AT_START == []


def test_sanitize_refuses_without_free_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREE_MODE", raising=False)
    with pytest.raises(SystemExit):
        bf._sanitize_environment()


def test_sanitize_strips_openrouter_and_paid_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY_DONATED", "y")
    monkeypatch.setenv("EXA_API_KEY", "z")
    monkeypatch.setenv("NVIDIA_API_KEY", "keep")
    # Registered with monkeypatch so the sanitizer's writes are rolled back after the test.
    for flag in (
        "FREE_RESEARCH_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "GEMINI_SEARCH_ENABLED",
    ):
        monkeypatch.setenv(flag, "unset-by-test")
    removed = bf._sanitize_environment()
    assert {"OPENROUTER_API_KEY", "OPENROUTER_API_KEY_DONATED", "EXA_API_KEY"} <= set(removed)
    assert "OPENROUTER_API_KEY" not in os.environ
    assert os.environ["NVIDIA_API_KEY"] == "keep"
    assert os.environ["FREE_RESEARCH_ENABLED"] == "false"


def _register_sanitizer_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in (
        "FREE_RESEARCH_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "BACKTEST_TOKEN_SOURCE",
        "METACULUS_TOKEN",
    ):
        monkeypatch.setenv(flag, "unset-by-test")


def test_sanitize_prefers_backtest_token_for_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_sanitizer_writes(monkeypatch)
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.setenv("METACULUS_TOKEN", "bot")
    monkeypatch.setenv("METACULUS_BACKTEST_TOKEN", " personal ")
    bf._sanitize_environment()
    assert os.environ["METACULUS_TOKEN"] == "personal"
    assert "METACULUS_BACKTEST_TOKEN" not in os.environ
    assert os.environ["BACKTEST_TOKEN_SOURCE"] == "METACULUS_BACKTEST_TOKEN"


def test_sanitize_falls_back_to_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_sanitizer_writes(monkeypatch)
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.setenv("METACULUS_TOKEN", "bot")
    monkeypatch.setenv("METACULUS_BACKTEST_TOKEN", "")
    bf._sanitize_environment()
    assert os.environ["METACULUS_TOKEN"] == "bot"
    assert os.environ["BACKTEST_TOKEN_SOURCE"] == "METACULUS_TOKEN"


def test_backtest_token_only_in_backtest_workflow() -> None:
    from pathlib import Path

    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    users = sorted(f.name for f in wf.glob("*.y*ml") if "METACULUS_BACKTEST_TOKEN" in f.read_text())
    assert users == ["backtest_free.yaml"]


def test_binary_median_and_scores() -> None:
    agg = bf.aggregate([_bin(0.2), _bin(0.9), _bin(0.6)])
    assert agg.binary == pytest.approx(0.6)
    s = bf.score(agg, bf.Truth("binary", True))
    assert s["brier"] == pytest.approx(0.16)
    assert s["baseline"] == pytest.approx(s["binary_log"])


def test_binary_zero_probability_is_not_treated_as_missing() -> None:
    s = bf.score(_bin(0.0), bf.Truth("binary", False))
    assert s["brier"] < 0.01


def test_mc_median_renormalises() -> None:
    q = SimpleNamespace(options=["A", "B"])
    f1 = bf.MemberForecast("multiple_choice", mc={"A": 0.9, "B": 0.1}, question=q)
    f2 = bf.MemberForecast("multiple_choice", mc={"A": 0.5, "B": 0.5}, question=q)
    agg = bf.aggregate([f1, f2])
    assert sum(agg.mc.values()) == pytest.approx(1.0)
    assert bf.score(agg, bf.Truth("multiple_choice", "A"))["mc_log"] > 0


def test_numeric_pointwise_median_cdf_scores() -> None:
    q = SimpleNamespace(
        lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False, zero_point=None
    )
    xs = [i * 10 / 200 for i in range(201)]
    lin = [i / 200 for i in range(201)]
    sharp = [min(1.0, max(0.0, (x - 4) / 2)) * 0.98 + 0.01 for x in xs]
    f1 = bf.MemberForecast("numeric", cdf_x=xs, cdf_p=lin, question=q)
    f2 = bf.MemberForecast("numeric", cdf_x=xs, cdf_p=sharp, question=q)
    agg = bf.aggregate([f1, f2, f2])
    assert agg.cdf_p == pytest.approx(sharp)
    s = bf.score(agg, bf.Truth("numeric", 5.0))
    assert "crps" in s
    assert "baseline" in s


def test_evaluate_variant_uses_common_questions_and_bootstrap() -> None:
    truths = {
        1: bf.Truth("binary", True),
        2: bf.Truth("binary", False),
        3: bf.Truth("binary", True),
        4: bf.Truth("binary", False),
    }
    forecasts = {
        "a": {1: _bin(0.8), 2: _bin(0.3), 3: _bin(0.7), 4: _bin(0.2)},
        "b": {1: _bin(0.6), 2: _bin(0.5), 3: _bin(0.5)},
    }
    va = bf.evaluate_variant("a", ["a"], forecasts, truths)
    vab = bf.evaluate_variant("ab", ["a", "b"], forecasts, truths)
    assert va.n == 4
    assert vab.n == 3
    mean, lo, hi, n = bf.paired_bootstrap(va.per_question, vab.per_question, iters=200)
    assert n == 3
    assert lo <= mean <= hi


def test_variant_definitions_and_report_render() -> None:
    members = list(bf.DEFAULT_MEMBERS)
    defs = dict(bf.variant_definitions(members, members))
    assert "live_proxy(kimi,kimi,glm,glm)" in defs
    assert any(name.startswith("all_minus_weakest") for name in defs)
    v = bf.VariantResult("x", ["a"], 3, {"baseline": 1.0, "brier": 0.2})
    text = bf.render_report(
        variants=[v],
        reference="x",
        deltas={"x": (0.0, 0.0, 0.0, 3)},
        meta={
            "n_questions": 3,
            "type_counts": {},
            "tournaments": [],
            "resolved_after": "d",
            "members": ["a"],
            "coverage": {},
        },
    )
    assert "| x | 3 |" in text
