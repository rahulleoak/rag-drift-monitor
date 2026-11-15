"""Keyword search (BM25) and the keyword vs meaning comparison (D40)."""
import math

import numpy as np
import pandas as pd
import pytest

from driftmon.config import load_config
from driftmon.evaluation import keyword as kw
from driftmon.rag.bm25 import BM25Index

CFG = load_config()


def test_bm25_matches_a_hand_worked_score():
    idx = BM25Index(["d1", "d2"], ["apple banana", "apple apple cherry"], k1=1.5, b=0.75)
    s = idx.scores(["banana"])[0]
    # banana: N=2, df=1 -> idf = ln(1 + 1.5/1.5) = ln 2; d1 has 2 words, average 2.5
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / 2.5)
    assert s[0] == pytest.approx(math.log(2) * 1 * 2.5 / (1 + norm)) and s[1] == 0
    # repeated words count with diminishing returns; a common word is worth less than a rare one
    a = idx.scores(["apple"])[0]
    assert a[1] > a[0] > 0 and a[0] < s[0]


def test_bm25_search_uses_the_vocabulary_tokens_and_is_repeatable():
    idx = BM25Index(["d1", "d2", "d3"], ["Coronavirus cases rise", "The vaccine works", "Trump's tweets"])
    ids, sc = idx.search(["coronavirus vaccine", "trump", "COVID"], 2)
    assert ids[0, 0] in {"d1", "d2"} and set(ids[0]) == {"d1", "d2"}
    assert ids[1, 0] == "d3"                                # possessive stripped, as in Agent B's tokens
    assert sc[2].max() == 0                                 # no shared word: nothing found
    again, _ = idx.search(["coronavirus vaccine", "trump", "COVID"], 2, batch=1)
    assert (again == ids).all()


def test_ranks_ignore_keyword_results_without_a_shared_word():
    ids = np.array([["a", "b"], ["c", "d"]])
    assert kw.ranks(ids, ["b", "x"]) == [2, 0]
    assert kw.ranks(ids, ["b", "c"], np.array([[3.0, 0.0], [1.0, 0.5]])) == [0, 1]


def _pq(shortfall_wording, shortfall_none):
    """Questions in wording-drift and no-drift sections with given meaning/keyword outcomes."""
    rows = []
    for kind, outcomes in (("wording", shortfall_wording), ("none", shortfall_none)):
        for i, (m, k) in enumerate(outcomes):
            rows.append({"question_id": f"{kind}{i}", "period": "2020", "category": f"C-{kind}",
                         "m": 1 if m else 0, "k": 1 if k else 0})
    df = pd.DataFrame(rows)
    kinds = {("2020", "C-wording"): "wording", ("2020", "C-none"): "none"}
    return kw.per_question(df, df["m"].tolist(), df["k"].tolist(), kinds, "2016-17", hit_k=5)


def test_wording_drift_hurting_keyword_more_is_detected():
    # wording-drift sections: keyword misses 15 of 30 that meaning finds; no-drift: keyword finds all
    pq = _pq([(True, i >= 15) for i in range(30)], [(True, True)] * 30)
    s = kw.summarise(pq).set_index("kind")
    assert s.at["wording", "shortfall"] == 0.5 and s.at["none", "shortfall"] == 0
    assert s.at["all", "only_meaning"] == 15 and s.at["all", "only_keyword"] == 0
    v = kw.verdict(pq, None, CFG)
    assert v["by_drift_kind"]["gap"] == 0.5 and v["by_drift_kind"]["wording_drift_hurts_keyword_more"]
    assert v["overall"]["p_meaning_better"] < 0.05 and "newer_wording" not in v


def test_no_difference_is_not_called_a_difference():
    pq = _pq([(True, True)] * 20 + [(True, False)], [(True, True)] * 20 + [(True, False)])
    v = kw.verdict(pq, None, CFG)["by_drift_kind"]
    assert v["gap"] == pytest.approx(0) and v["wording_drift_hurts_keyword_more"] is False


def test_newer_wording_part():
    pq = _pq([(True, True)], [(True, True)])
    w = pd.DataFrame({"found_same_era_meaning": [True] * 12, "found_newer_meaning": [True] * 12,
                      "found_same_era_keyword": [True] * 12, "found_newer_keyword": [False] * 6 + [True] * 6})
    v = kw.verdict(pq, w, CFG)["newer_wording"]
    assert v["questions"] == 12 and v["keyword"]["lost"] == 6 and v["keyword"]["hurts"]
    assert not v["meaning"]["hurts"] and v["keyword_hurt_more"]
