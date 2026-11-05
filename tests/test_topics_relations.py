"""Tests for emerging topics, term-context shift and the P6 correlations."""
from datetime import date

import numpy as np
import pandas as pd

from driftmon.drift import relations as rel
from driftmon.drift import topics as top
from driftmon.drift.report import metric_correlations
from driftmon.periods import Period


def _unit(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_emerging_topic_is_found_by_growth_not_novelty():
    rng = np.random.default_rng(0)
    # two topics in the baseline; in "2020" topic B grows from 5% to 60%
    a = _unit(rng.normal([5, 0, 0], 0.3, size=(190, 3)))
    b = _unit(rng.normal([0, 5, 0], 0.3, size=(10, 3)))
    a2 = _unit(rng.normal([5, 0, 0], 0.3, size=(80, 3)))
    b2 = _unit(rng.normal([0, 5, 0], 0.3, size=(120, 3)))
    vectors = np.vstack([a, b, a2, b2])
    period = np.array(["base"] * 200 + ["2020"] * 200)
    cats = np.array(["POLITICS"] * 190 + ["WELLNESS"] * 10 + ["POLITICS"] * 80
                    + ["WELLNESS"] * 60 + ["FOOD"] * 60)
    tokens = ([["senate", "vote"]] * 200 + [["senate", "vote"]] * 80 + [["virus", "mask"]] * 120)
    topic = top.cluster_documents(vectors, k=2, seed=0)
    labels = top.topic_labels(topic, tokens, min_count=1)
    titles = np.array([f"t{i}" for i in range(400)])
    df = top.emerging_topics(topic, period, cats, "base", ["2020"], labels, titles, min_docs=15)
    em = df[df["emerging"]]
    assert len(em) == 1
    row = em.iloc[0]
    assert row["docs_period"] == 120 and row["docs_baseline"] == 10
    assert row["ratio"] > 10 and "virus" in row["label"]
    assert row["sections"] == 2  # spans WELLNESS and FOOD


def test_term_context_shift_detects_a_changed_meaning():
    rng = np.random.default_rng(1)
    n = 40
    old = _unit(rng.normal([5, 0, 0, 0], 0.5, size=(2 * n, 4)))   # "mask" in the baseline
    new = _unit(rng.normal([0, 5, 0, 0], 0.5, size=(n, 4)))       # "mask" in 2020, a new context
    stable_old = _unit(rng.normal([0, 0, 5, 0], 0.5, size=(2 * n, 4)))
    stable_new = _unit(rng.normal([0, 0, 5, 0], 0.5, size=(n, 4)))
    vectors = np.vstack([old, new, stable_old, stable_new])
    dates = ([date(2016, 3, 1)] * n + [date(2017, 3, 1)] * n + [date(2020, 3, 1)] * n) * 2
    period = (["base"] * 2 * n + ["2020"] * n) * 2
    docs = pd.DataFrame({"date": dates, "period": period})
    tokens = ([["mask", "halloween"]] * 2 * n + [["mask", "coronavirus"]] * n
              + [["court", "ruling"]] * 3 * n)
    halves = (Period("h1", date(2016, 1, 1), date(2016, 12, 31)),
              Period("h2", date(2017, 1, 1), date(2017, 12, 31)))
    out = rel.term_context_shifts(["mask", "court", "missing"], docs, vectors, tokens, "base",
                                  halves, ["2020"], min_docs=20, permutations=99).set_index("term")
    assert set(out.index) == {"mask", "court"}  # "missing" has no articles
    assert out.loc["mask", "p_value"] < 0.05 and out.loc["mask", "times_ordinary"] > 10
    assert out.loc["court", "p_value"] > 0.05
    assert "coronavirus" in out.loc["mask", "context_after"]
    assert "halloween" in out.loc["mask", "context_before"]


def test_metric_correlations_needs_variation():
    from driftmon.schemas import SegmentDrift
    base = dict(segment_id="x", period="2020", compared_with="b", n_reference=1, n_period=1,
                mmd_q=0.0, jsd_q=0.0, centroid_shift=0.0, novelty_mean=0.0, novel_share=0.0,
                times_ordinary_embedding=1, times_ordinary_lexical=1, embedding_band="low",
                lexical_band="low", severity="low", kind="none", significant=True,
                flagged=False, null_window="month", null_n=1)
    one = [SegmentDrift(**base, mmd2=0.1, jsd_excess=0.1)]
    assert metric_correlations(one).isna().all().all()
    many = [SegmentDrift(**{**base, "centroid_shift": v, "novelty_mean": v, "novel_share": v,
                            "new_term_rate": v}, mmd2=v, jsd_excess=v) for v in (0.1, 0.2, 0.3, 0.4)]
    corr = metric_correlations(many)
    assert corr.loc["mmd2", "jsd_excess"] == 1.0
