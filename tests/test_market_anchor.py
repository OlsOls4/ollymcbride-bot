"""FREE_MODE market anchor: selection rules, blend + clamp, comment note, and the forecaster seam."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from metaculus_bot import market_anchor as ma
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch, MarketSnapshot, ScalarEstimate

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _row(**kw: object) -> MarketMatch:
    base = MarketMatch(
        platform="polymarket",
        market_title="Will X happen by 2026-12-31?",
        market_url="https://polymarket.com/event/x",
        implied_prob_yes=0.40,
        bid=0.39,
        ask=0.41,
        spread=0.02,
        volume_24h=1_000.0,
        close_time=NOW + timedelta(days=60),
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
        total_volume=50_000.0,
        relation_tier="same_quantity_same_date",
        rank=0,
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def _snap(*rows: MarketMatch) -> MarketSnapshot:
    return MarketSnapshot(matches=list(rows))


@pytest.fixture(autouse=True)
def _free_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.delenv(ma.MARKET_ANCHOR_ENABLED_ENV, raising=False)
    ma.reset_anchors()


def test_liquid_same_question_market_is_an_anchor() -> None:
    a = ma.select_market_anchor(_snap(_row()), now=NOW)
    assert a is not None
    assert a.platform == "polymarket"
    assert a.prob == pytest.approx(0.40)
    assert a.url == "https://polymarket.com/event/x"


@pytest.mark.parametrize(
    "kw",
    [
        {"relation_tier": "same_quantity_other_cut"},
        {"relation_tier": "driver_or_consequence"},
        {"relation_tier": ""},
        {"tier_cap_note": "capped: closed long before the question opened"},
        {"is_resolved": True},
        {"price_withheld": True},
        {"implied_prob_yes": None},
        {"implied_prob_yes": 1.5},
        {"close_time": NOW - timedelta(days=1)},
        {"spread": 0.2},
        {"total_volume": 5_000.0},  # thin/decent real money: below the $10k floor
        {"total_volume": None, "open_interest": None},
        {"platform": "predictit"},
        {"children": (MarketChild(title="a", implied_prob_yes=0.3),)},
        {"scalar_estimate": ScalarEstimate(value=3.0)},
    ],
)
def test_rows_that_never_anchor(kw: dict[str, object]) -> None:
    assert ma.select_market_anchor(_snap(_row(**kw)), now=NOW) is None


def test_open_interest_counts_as_liquidity() -> None:
    assert ma.select_market_anchor(_snap(_row(total_volume=None, open_interest=20_000.0)), now=NOW) is not None


def test_manifold_needs_bettors_and_volume() -> None:
    m = _row(platform="manifold", total_volume=500.0, num_bettors=25, bid=None, ask=None, spread=None)
    assert ma.select_market_anchor(_snap(m), now=NOW) is not None
    assert ma.select_market_anchor(_snap(replace(m, num_bettors=5)), now=NOW) is None
    assert ma.select_market_anchor(_snap(replace(m, num_bettors=None)), now=NOW) is None
    assert ma.select_market_anchor(_snap(replace(m, total_volume=0.0)), now=NOW) is None


def test_empty_or_missing_snapshot() -> None:
    assert ma.select_market_anchor(_snap(), now=NOW) is None
    assert ma.select_market_anchor(SimpleNamespace(), now=NOW) is None
    assert ma.select_market_anchor(None, now=NOW) is None


def test_agreeing_markets_average_and_disagreeing_ones_abstain() -> None:
    a = ma.select_market_anchor(_snap(_row(rank=1), _row(implied_prob_yes=0.50, rank=0, platform="kalshi")), now=NOW)
    assert a is not None
    assert a.prob == pytest.approx(0.45)
    assert a.platform == "kalshi"  # best-ranked row leads
    assert a.n_markets == 2
    assert ma.select_market_anchor(_snap(_row(), _row(implied_prob_yes=0.80)), now=NOW) is None


def test_non_qualifying_rows_do_not_block_a_qualifying_one() -> None:
    a = ma.select_market_anchor(_snap(_row(relation_tier="weak", implied_prob_yes=0.9), _row()), now=NOW)
    assert a is not None
    assert a.prob == pytest.approx(0.40)


def test_blend_and_clamp() -> None:
    assert ma.blend_toward_market(0.20, 0.60, 0.5) == pytest.approx(0.40)
    assert ma.blend_toward_market(0.20, 0.60, 0.7) == pytest.approx(0.48)
    assert ma.blend_toward_market(0.001, 0.0, 0.7) == pytest.approx(0.02)
    assert ma.blend_toward_market(0.999, 1.0, 0.7) == pytest.approx(0.98)
    assert ma.blend_toward_market(0.3, 0.9, 0.0) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="weight must be in"):
        ma.blend_toward_market(0.3, 0.5, 1.2)
    assert ma.MARKET_ANCHOR_WEIGHT in ma.MARKET_ANCHOR_WEIGHT_GRID


def test_enabled_only_in_free_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ma.market_anchor_enabled()
    monkeypatch.setenv(ma.MARKET_ANCHOR_ENABLED_ENV, "false")
    assert not ma.market_anchor_enabled()
    monkeypatch.delenv(ma.MARKET_ANCHOR_ENABLED_ENV)
    monkeypatch.setenv("FREE_MODE", "false")
    assert not ma.market_anchor_enabled()


def test_registry_pop_once_and_clear() -> None:
    a = ma.MarketAnchor(platform="polymarket", prob=0.4, url="", title="")
    ma.record_anchor(7, a)
    assert ma.pop_anchor(7) == a
    assert ma.pop_anchor(7) is None
    ma.record_anchor(7, a)
    ma.record_anchor(7, None)  # a later snapshot with no anchor clears a stale one
    assert ma.pop_anchor(7) is None
    ma.record_anchor(None, a)
    assert ma.pop_anchor(None) is None


def test_note_states_the_anchor_truthfully_and_sits_under_the_header() -> None:
    a = ma.MarketAnchor(platform="manifold", prob=0.6, url="https://manifold.markets/x", title="X")
    note = ma.format_anchor_note(a, model_p=0.2, published_p=0.48, weight=0.7)
    for s in ("Manifold (play money)", "60.0%", "70%", "30%", "20.0%", "48.0%", "2%-98%", "https://manifold.markets/x"):
        assert s in note
    out = ma.insert_anchor_note("# SUMMARY\n*Question*: q\n", note)
    assert out.startswith("# SUMMARY\n**Market anchor:**")
    assert out.endswith("*Question*: q\n")
    assert ma.insert_anchor_note("plain", note).endswith(note + "\n")


def _bot() -> object:
    from metaculus_bot.forecaster import TemplateForecaster

    bot = TemplateForecaster.__new__(TemplateForecaster)
    bot._market_anchor_notes = {}  # type: ignore[attr-defined]
    return bot


def _binary(qid: int) -> object:
    from forecasting_tools import BinaryQuestion

    return BinaryQuestion(question_text="Will X happen?", id_of_question=qid, id_of_post=qid)


def test_forecaster_blends_binary_only_when_an_anchor_exists() -> None:
    bot = _bot()
    q = _binary(11)
    assert bot._apply_market_anchor(0.2, q) == 0.2  # type: ignore[attr-defined]  # no anchor: unchanged
    assert bot._market_anchor_notes == {}  # type: ignore[attr-defined]
    ma.record_anchor(11, ma.MarketAnchor(platform="polymarket", prob=0.6, url="", title=""))
    out = bot._apply_market_anchor(0.2, q)  # type: ignore[attr-defined]
    assert out == pytest.approx(0.7 * 0.6 + 0.3 * 0.2)
    assert "Polymarket" in bot._market_anchor_notes[11]  # type: ignore[attr-defined]
    assert ma.pop_anchor(11) is None  # consumed


def test_forecaster_leaves_non_binary_and_disabled_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecasting_tools import MultipleChoiceQuestion

    bot = _bot()
    mc = MultipleChoiceQuestion(question_text="Which?", id_of_question=12, id_of_post=12, options=["a", "b"])
    ma.record_anchor(12, ma.MarketAnchor(platform="polymarket", prob=0.6, url="", title=""))
    sentinel = object()
    assert bot._apply_market_anchor(sentinel, mc) is sentinel  # type: ignore[attr-defined]
    monkeypatch.setenv(ma.MARKET_ANCHOR_ENABLED_ENV, "false")
    ma.record_anchor(13, ma.MarketAnchor(platform="polymarket", prob=0.6, url="", title=""))
    assert bot._apply_market_anchor(0.2, _binary(13)) == 0.2  # type: ignore[attr-defined]


def test_research_provider_records_the_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    from metaculus_bot.research import prediction_market as pm

    snap = _snap(_row(close_time=datetime.now(UTC) + timedelta(days=30)))

    async def fake_fetch(question: object, **_kw: object) -> MarketSnapshot:
        return snap

    monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
    monkeypatch.setattr(pm, "fetch_market_snapshot", fake_fetch)
    monkeypatch.setattr(pm, "format_snapshot_for_research", lambda s, qid=None: "rendered")
    q = SimpleNamespace(id_of_question=21, title="t", resolution_criteria="", fine_print="")
    assert asyncio.run(pm.prediction_market_provider()(q)) == "rendered"  # type: ignore[arg-type]
    got = ma.pop_anchor(21)
    assert got is not None
    assert got.prob == pytest.approx(0.40)
    # Benchmarking: provider is off, so no anchor is ever recorded.
    assert asyncio.run(pm.prediction_market_provider(is_benchmarking=True)(q)) == ""  # type: ignore[arg-type]
    assert ma.pop_anchor(21) is None
