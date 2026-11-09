"""Tests for Agent A's statistics, severity bands, and the A -> B graph."""
import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from driftmon.data import monitoring
from driftmon.config import load_config
from driftmon.drift import severity as sev
from driftmon.drift.compare import eligible_segments
from driftmon.graph import FULL_EVALUATION, REGRESSION_CHECK, build_graph, route_after_drift
from driftmon.schemas import DriftReport

# ---------- Agent A statistics ----------


def test_category_mix_test_effect_size():
    base = pd.Series({"POLITICS": 50, "WELLNESS": 50})
    assert monitoring.category_mix_test(base, base)["cramers_v"] == pytest.approx(0.0)
    disjoint = monitoring.category_mix_test(base, pd.Series({"SPORTS": 100}))
    assert disjoint["cramers_v"] == pytest.approx(1.0) and disjoint["p_value"] < 1e-6


def test_eligible_segments_reasons():
    docs = pd.DataFrame({"period": ["b"] * 60 + ["b"] * 5 + ["p"] * 60 + ["p"] * 60 + ["p"] * 10,
                         "category": ["A"] * 60 + ["C"] * 5 + ["A"] * 60 + ["NEW"] * 60 + ["C"] * 10})
    seg = eligible_segments(docs, "b", ["p"], min_size=50).set_index("category")
    assert seg.loc["A", "eligible"]
    assert seg.loc["NEW", "reason"] == "not in baseline"
    assert seg.loc["C", "reason"] == "too few in baseline"


# ---------- severity ----------


def test_band_kind_and_overall():
    th = {"medium": 1.0, "high": 2.0}
    assert [sev.band(v, th) for v in (0.5, 1.5, 2.5)] == ["low", "medium", "high"]
    assert sev.band(1.0, th) == "low"  # must exceed the threshold
    assert sev.drift_kind("high", "low") == "meaning"
    assert sev.drift_kind("low", "medium") == "wording"
    assert sev.drift_kind("medium", "high") == "both"
    assert sev.drift_kind("low", "low") == "none"
    assert sev.overall("medium", "high") == "high"


# ---------- synthetic archive for the graph ----------

RNG = np.random.default_rng(0)
DIM = 16
WORDS_OLD = ["senate", "budget", "vote", "court", "policy", "tax", "governor", "bill"]
WORDS_NEW = ["virus", "mask", "lockdown", "vaccine", "outbreak", "quarantine"]


def _synthetic_corpus(per_month=30, n_new=40, shift=1.5):
    rows, vecs = [], []
    center = RNG.normal(size=DIM)

    def add(day, period, words, offset):
        v = center + RNG.normal(size=DIM)
        v[0] += offset
        vecs.append(v / np.linalg.norm(v))
        title = " ".join(RNG.choice(words, 4))
        rows.append({"doc_id": f"d{len(rows)}", "date": day, "period": period,
                     "category": "POLITICS", "category_raw": "POLITICS",
                     "title": title, "text": " ".join(RNG.choice(words, 6)), "words": 10})

    for year in (2016, 2017):
        for month in range(1, 13):
            for k in range(per_month):
                add(date(year, month, 1) + timedelta(days=k % 28), "2016-17", WORDS_OLD, 0.0)
    for k in range(n_new):
        add(date(2020, 1, 1) + timedelta(days=k), "2020", WORDS_OLD[:3] + WORDS_NEW, shift)
    docs = pd.DataFrame(rows)
    docs["in_drift_sample"] = False
    for p, n in (("2016-17", n_new), ("2020", n_new)):
        idx = docs.index[docs["period"] == p]
        docs.loc[RNG.choice(idx, n, replace=False), "in_drift_sample"] = True
    return docs, np.array(vecs, dtype=np.float32)


def _config():
    cfg = copy.deepcopy(load_config())
    cfg["periods"]["baseline"] = {"name": "2016-17", "start": "2016-01-01", "end": "2017-12-31"}
    cfg["periods"]["later"] = [{"name": "2020", "start": "2020-01-01", "end": "2020-12-31"}]
    cfg["periods"]["sample_per_period"] = 40
    cfg["periods"]["min_segment_size"] = 20
    cfg["drift"].update(mmd_permutations=99, null_draws=2, null_permutations=20)
    cfg["paths"]["runs"] = "runs/_test_does_not_exist"
    return cfg


def test_build_null_uses_month_pairs():
    from driftmon.drift.compare import prepare
    docs, vecs = _synthetic_corpus()
    inp = prepare(_config(), docs, vecs)
    null = sev.build_null(inp, n=25, draws=2, permutations=10)
    assert null.window == "month" and null.pairs == 12 and len(null.values) == 24
    big = sev.build_null(inp, n=60, draws=1, permutations=10)  # months hold 30: widen to quarters
    assert big.window == "quarter" and big.pairs == 4
    assert sev.build_null(inp, n=500, draws=1, permutations=10).n < 500  # shrinks to fit


def test_graph_flags_the_shifted_period_and_routes_to_full_evaluation():
    docs, vecs = _synthetic_corpus()
    state = build_graph().invoke({"config": _config(), "docs": docs, "vectors": vecs})
    mon, drift = state["monitoring"], state["drift"]
    assert mon.baseline == "2016-17" and mon.periods == ["2020"]
    assert [s["category"] for s in mon.eligible_segments()] == ["POLITICS"]
    whole = drift.whole_period[0]
    assert whole.segment_id == "2020|ALL|vs 2016-17"
    assert whole.flagged and whole.embedding_band == "high"
    assert set(whole.emerging_terms) <= set(WORDS_OLD + WORDS_NEW)
    assert any(t in WORDS_NEW for t in whole.emerging_terms)
    assert "2020|ALL|vs 2016-17" in drift.drifted_segments
    assert state["next_step"] == FULL_EVALUATION
    # Agent E runs last and reports on every action, even without answers to judge
    recs = state["recommendations"].recommendations
    assert len(recs) == 9 and {r.status for r in recs} <= {"not checked", "not needed", "recommended"}


def test_graph_reports_the_query_stream_and_query_drift():
    from tests.test_queries import _stream
    docs, vecs = _synthetic_corpus()
    stream, qvecs = _stream()
    state = build_graph().invoke({"config": _config(), "docs": docs, "vectors": vecs,
                                  "queries": stream, "query_vectors": qvecs})
    assert [r["period"] for r in state["monitoring"].query_stream] == ["2016-17", "2020"]
    qd = state["drift"].query_drift
    assert [r["period"] for r in qd] == ["2017 (inside baseline)", "2020"] and qd[1]["significant"]
    without = build_graph().invoke({"config": _config(), "docs": docs, "vectors": vecs})
    assert isinstance(without["drift"].query_drift, str)  # no question set: says so, no crash


def test_graph_routes_to_regression_check_without_drift():
    docs, vecs = _synthetic_corpus(shift=0.0)
    docs.loc[docs["period"] == "2020", "title"] = [" ".join(RNG.choice(WORDS_OLD, 4))
                                                   for _ in range((docs["period"] == "2020").sum())]
    docs.loc[docs["period"] == "2020", "text"] = [" ".join(RNG.choice(WORDS_OLD, 6))
                                                  for _ in range((docs["period"] == "2020").sum())]
    state = build_graph().invoke({"config": _config(), "docs": docs, "vectors": vecs})
    assert not state["drift"].any_flagged
    assert state["next_step"] == REGRESSION_CHECK


def test_route_after_drift_reads_the_report():
    report = DriftReport(whole_period=[], segments=[], drifted_segments=[], control_segments=[],
                         reference={}, any_flagged=False)
    assert route_after_drift({"drift": report}) == REGRESSION_CHECK
    assert route_after_drift({"drift": report.model_copy(update={"any_flagged": True})}) == FULL_EVALUATION
