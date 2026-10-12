"""ForecastBench source of backtest_free.py: data join, sampling, leakage filter, chain simulation."""

from __future__ import annotations

import asyncio
import os
from datetime import date
from pathlib import Path

import pytest
import yaml

import backtest_free as bf
from metaculus_bot.backtest import forecastbench as fb

_WF = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "backtest_free.yaml"


def _qset(due: str, questions: list[dict]) -> dict:
    return {"forecast_due_date": due, "question_set": f"{due}-llm.json", "questions": questions}


def _q(qid: str, source: str = "polymarket", **kw: object) -> dict:
    base: dict[str, object] = {
        "id": qid,
        "source": source,
        "question": f"Will {qid} happen?",
        "resolution_criteria": "Resolves to the outcome of the market.",
        "background": "<p>Some &nbsp;background</p>",
        "market_info_open_datetime": "2026-05-01",
        "market_info_close_datetime": "Thu 2026-10-01 6:00 AM BST (UTC+01:00)",
        "market_info_resolution_criteria": "N/A",
        "freeze_datetime_value": "0.3",
        "source_intro": "Predict the market.",
        "url": f"https://example.com/{qid}",
    }
    base.update(kw)
    return base


def _r(qid: str, source: str = "polymarket", rdate: str = "2026-09-01", to: float = 1.0, resolved: bool = True) -> dict:
    return {"id": qid, "source": source, "resolution_date": rdate, "resolved_to": to, "resolved": resolved}


def test_parse_fb_date_handles_both_formats() -> None:
    assert fb.parse_fb_date("2025-12-11") == date(2025, 12, 11)
    assert fb.parse_fb_date("Thu 2026-06-11 12:59 AM BST (UTC+01:00)") == date(2026, 6, 11)
    assert fb.parse_fb_date("N/A") is None
    assert fb.parse_fb_date(None) is None


def test_candidate_due_dates_follow_the_14_day_cadence() -> None:
    ds = fb.candidate_due_dates(date(2026, 9, 1), date(2026, 10, 12))
    assert ds == [date(2026, 9, 13), date(2026, 9, 27), date(2026, 10, 11)]


def test_join_keeps_only_resolved_market_binary_after_floor() -> None:
    qset = _qset(
        "2026-06-07",
        [_q("a"), _q("b"), _q("c"), _q("d", source="acled"), _q("e"), _q("f"), _q("g")],
    )
    rset = {
        "resolutions": [
            _r("a", rdate="2026-09-01", to=1.0),  # kept
            _r("b", rdate="2026-10-11", to=0.42, resolved=False),  # unresolved market value: dropped
            _r("c", rdate="2026-05-20", to=0.0),  # before the floor: dropped
            _r("d", source="acled", to=1.0),  # dataset source: dropped
            _r("e", rdate="2026-06-07", to=0.0),  # resolved on the due date itself: dropped
            _r("zz", rdate="2026-09-01", to=1.0),  # no matching question: dropped
            _r("f", rdate="2026-07-01", to=0.0),  # kept, NO
            _r("f", rdate="2026-07-01", to=0.0),  # duplicate (id, date): dropped
        ]
    }
    items = fb.join_market_binary(qset, rset, resolved_after=date(2026, 6, 1))
    assert [(i.fb_id, i.outcome) for i in items] == [("a", True), ("f", False)]
    a = items[0]
    assert a.due_date == date(2026, 6, 7)
    assert a.resolution_date == date(2026, 9, 1)
    assert a.market_value == pytest.approx(0.3)
    assert a.background == "Some  background"
    assert a.close_date == date(2026, 10, 1)


def test_load_forecastbench_skips_missing_releases() -> None:
    files = {
        f"{fb.FB_RAW_BASE}/question_sets/2026-09-13-llm.json": _qset("2026-09-13", [_q("a")]),
        f"{fb.FB_RAW_BASE}/resolution_sets/2026-09-13_resolution_set.json": {
            "resolutions": [_r("a", rdate="2026-10-01")]
        },
    }
    items, diag = fb.load_forecastbench(
        since=date(2026, 9, 1), until=date(2026, 10, 12), resolved_after=date(2026, 6, 1), fetch=files.get
    )
    assert [i.fb_id for i in items] == ["a"]
    assert diag["due_dates"]["2026-09-27"] == "missing"


def _item(fid: str, src: str, rdate: date = date(2026, 9, 1), due: date = date(2026, 8, 2)) -> fb.FBQuestion:
    return fb.FBQuestion(fid, src, due, rdate, True, "q", "b", "c")


def test_dedupe_one_instance_per_question() -> None:
    items = [
        _item("a", "manifold", due=date(2026, 7, 5)),
        _item("a", "manifold", due=date(2026, 7, 19)),
        _item("b", "metaculus"),
    ]
    out = fb.dedupe_by_question(items, seed=1)
    assert sorted(i.fb_id for i in out) == ["a", "b"]
    assert fb.dedupe_by_question(items, seed=1)[0].due_date == out[0].due_date  # deterministic


def test_stratified_sample_equal_shares_and_refill() -> None:
    items = [_item(f"p{i}", "polymarket") for i in range(50)] + [_item(f"m{i}", "metaculus") for i in range(30)]
    items += [_item(f"i{i}", "infer") for i in range(3)]
    s = fb.stratified_sample(items, 60, seed=7)
    counts: dict[str, int] = {}
    for it in s:
        counts[it.source] = counts.get(it.source, 0) + 1
    assert len(s) == 60
    assert counts["infer"] == 3  # all it has; the shortfall is refilled from the others
    assert counts["metaculus"] >= 20
    assert [i.fb_id for i in fb.stratified_sample(items, 60, seed=7)] == [i.fb_id for i in s]


def test_cutoff_filter_excludes_questions_resolved_before_cutoff_and_unknown_models() -> None:
    early = _item("x", "polymarket", rdate=date(2026, 8, 1))
    late = _item("y", "polymarket", rdate=date(2026, 9, 20))
    assert not fb.eligible_for("nvidia_nim/z-ai/glm-5.3", early)  # released 2026-08-14
    assert fb.eligible_for("nvidia_nim/z-ai/glm-5.3", late)
    assert not fb.eligible_for("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", _item("z", "x", rdate=date(2026, 9, 9)))
    assert not fb.eligible_for("some/unknown-model", late)
    for m in (*bf.FB_DEFAULT_MEMBERS, bf.FLASH_MODEL):
        assert fb.effective_cutoff(m) is not None, m


def test_live_chain_members_all_have_cutoffs() -> None:
    from metaculus_bot.constants import FREE_MODE_FORECASTER_CHAINS

    for chain in FREE_MODE_FORECASTER_CHAINS:
        for link in chain:
            if not link.startswith("openrouter/"):
                assert fb.effective_cutoff(link) is not None, link


def test_simulate_chains_falls_through_like_the_live_chain() -> None:
    chains = [("a", "b"), ("c", "b"), ("d",)]
    forecasts = {"a": {"q": 0.2}, "b": {"q": 0.6}, "d": {"q": 0.9}}
    assert fb.simulate_chains(chains, forecasts, "q") == [0.2, 0.6, 0.9]
    assert fb.simulate_chains(chains, forecasts, "q", drop=frozenset({"a"})) == [0.6, 0.6, 0.9]
    assert fb.simulate_chains(chains, forecasts, "missing") is None


def test_scores_and_summary() -> None:
    assert fb.brier(0.8, True) == pytest.approx(0.04)
    assert fb.log_score(0.5, False) == pytest.approx(-0.6931, abs=1e-4)
    assert fb.log_score(0.0, True) == pytest.approx(-6.9078, abs=1e-3)  # clipped, finite
    outcomes = {"a": True, "b": False, "c": True, "d": False}
    ref = fb.BinaryVariant("ref", dict.fromkeys(outcomes, 0.5))
    good = fb.BinaryVariant("good", {"a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2})
    rows = {r["name"]: r for r in fb.summarize_variants([ref, good], outcomes, reference="ref", seed=0, iters=300)}
    assert rows["ref"]["brier"] == pytest.approx(0.25)
    assert rows["good"]["d_brier"] < 0 < rows["good"]["d_log"]
    assert rows["good"]["brier_lo"] <= rows["good"]["brier"] <= rows["good"]["brier_hi"]
    assert fb._median([1.0, 3.0, 2.0, 10.0]) == 2.5


def test_sanitize_forecastbench_removes_metaculus_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in (
        "FREE_RESEARCH_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "BACKTEST_TOKEN_SOURCE",
    ):
        monkeypatch.setenv(flag, "unset-by-test")
    monkeypatch.setenv("FREE_MODE", "true")
    monkeypatch.setenv("METACULUS_TOKEN", "bot")
    monkeypatch.setenv("METACULUS_BACKTEST_TOKEN", "personal")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    removed = bf._sanitize_environment("forecastbench")
    assert {"METACULUS_TOKEN", "METACULUS_BACKTEST_TOKEN", "OPENROUTER_API_KEY"} <= set(removed)
    for k in ("METACULUS_TOKEN", "METACULUS_BACKTEST_TOKEN", "OPENROUTER_API_KEY"):
        assert k not in os.environ
    assert os.environ["FREE_RESEARCH_ENABLED"] == "false"


def test_source_from_argv() -> None:
    assert bf._source_from_argv(["x.py", "--source", "forecastbench"]) == "forecastbench"
    assert bf._source_from_argv(["x.py", "--source=forecastbench"]) == "forecastbench"
    assert bf._source_from_argv(["x.py"]) == "metaculus"


def test_flash_call_budget_is_hard() -> None:
    class _Chain:
        model = "gemini/gemini-3.8-flash"

        async def invoke(self, prompt: object, system_prompt: str | None = None) -> str:
            return "ok"

    chain = bf._budgeted(_Chain(), 2)

    async def go() -> None:
        assert await chain.invoke("p") == "ok"
        assert await chain.invoke("p") == "ok"
        with pytest.raises(bf._CallBudgetExceededError):
            await chain.invoke("p")

    asyncio.run(go())
    assert chain._budget_used[0] == 2
    assert bf.FLASH_CALL_CAP < 25


def test_workflow_forecastbench_path_needs_no_metaculus_token() -> None:
    text = _WF.read_text()
    wf = yaml.safe_load(text)
    triggers = wf.get("on", wf.get(True))
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["source"]["default"] == "forecastbench"
    steps = wf["jobs"]["backtest"]["steps"]
    gate = next(st for st in steps if str(st.get("name", "")).startswith("Require the read-only"))
    assert "inputs.source == 'metaculus'" in str(gate.get("if"))
    run = next(st for st in steps if "backtest_free.py" in str(st.get("run", "")))
    for key in ("METACULUS_TOKEN", "METACULUS_BACKTEST_TOKEN"):
        assert run["env"][key].startswith("${{ inputs.source == 'metaculus' &&"), key
    assert "secrets.OPENROUTER" not in text


def test_attribution_text() -> None:
    assert "CC BY-SA 4.0" in fb.FB_ATTRIBUTION
    assert "forecastbench-datasets" in fb.FB_ATTRIBUTION
