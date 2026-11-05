"""Tests for the drift statistics, the indexes and the BH correction."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from driftmon.drift import embedding as emb
from driftmon.drift import lexical as lex
from driftmon.evaluation.stats import benjamini_hochberg
from driftmon.index import VectorIndex, build_study_indexes


def _normal_rows(rng, n, d, shift=0.0):
    x = rng.normal(size=(n, d))
    x[:, 0] += shift
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------- embedding drift ----------

def test_mmd_detects_a_shift_and_not_a_null():
    rng = np.random.default_rng(0)
    a, b = _normal_rows(rng, 120, 8), _normal_rows(rng, 120, 8)
    shifted = _normal_rows(rng, 120, 8, shift=1.5)
    gamma = emb.median_heuristic_gamma(a)
    same = emb.mmd_permutation_test(a, b, gamma, permutations=200, seed=1)
    diff = emb.mmd_permutation_test(a, shifted, gamma, permutations=200, seed=1)
    assert diff.p_value < 0.01 and diff.mmd2 > same.mmd2
    assert same.p_value > 0.05
    assert abs(same.mmd2) < 0.01  # unbiased estimate is close to 0 under the null


def test_mmd_matches_direct_formula():
    rng = np.random.default_rng(3)
    x, y = _normal_rows(rng, 15, 4), _normal_rows(rng, 12, 4, shift=0.5)
    g = 0.7
    kxx = np.exp(-g * ((x[:, None] - x[None]) ** 2).sum(-1))
    kyy = np.exp(-g * ((y[:, None] - y[None]) ** 2).sum(-1))
    kxy = np.exp(-g * ((x[:, None] - y[None]) ** 2).sum(-1))
    m, n = len(x), len(y)
    direct = ((kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1))
              - 2 * kxy.mean())
    assert emb.mmd_permutation_test(x, y, g, permutations=10).mmd2 == pytest.approx(direct)


def test_mmd_p_value_is_repeatable():
    rng = np.random.default_rng(5)
    x, y = _normal_rows(rng, 40, 4), _normal_rows(rng, 40, 4, shift=0.3)
    r1 = emb.mmd_permutation_test(x, y, 1.0, permutations=99, seed=7)
    r2 = emb.mmd_permutation_test(x, y, 1.0, permutations=99, seed=7)
    assert r1 == r2


def test_centroid_shift_and_dispersion():
    e1, e2 = np.eye(3)[0], np.eye(3)[1]
    assert emb.centroid_shift(np.stack([e1, e1]), np.stack([e1, e1])) == pytest.approx(0.0)
    assert emb.centroid_shift(np.stack([e1]), np.stack([e2])) == pytest.approx(1.0)
    assert emb.dispersion(np.stack([e1, e1, e1])) == pytest.approx(0.0)
    assert emb.dispersion(np.stack([e1, e2])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = _normal_rows(rng, 30, 5)
    pairwise = 1 - (x @ x.T)[np.triu_indices(30, 1)]
    assert emb.dispersion(x) == pytest.approx(pairwise.mean())


# ---------- lexical drift ----------

def test_tokenize():
    assert lex.tokenize("Trump's COVID-19 Plan: The Masks Are Back!") == [
        "trump", "covid-19", "plan", "masks"]
    assert lex.tokenize("It’s 2020 and we're 100% in") == []


def test_jensen_shannon_bounds():
    a = Counter({"x": 3, "y": 1})
    assert lex.jensen_shannon(a, Counter({"x": 6, "y": 2})) == pytest.approx(0.0)
    assert lex.jensen_shannon(a, Counter({"z": 4})) == pytest.approx(1.0)


def test_jsd_permutation_test_separates_vocabularies():
    rng = np.random.default_rng(0)
    words_a, words_b = ["apple", "pear", "plum"], ["covid", "mask", "vaccine"]
    common = ["news", "today", "report"]
    docs_a = [list(rng.choice(words_a + common, 6)) for _ in range(40)]
    docs_b = [list(rng.choice(words_b + common, 6)) for _ in range(40)]
    docs_a2 = [list(rng.choice(words_a + common, 6)) for _ in range(40)]
    diff = lex.jsd_permutation_test(docs_a, docs_b, permutations=200, seed=1)
    same = lex.jsd_permutation_test(docs_a, docs_a2, permutations=200, seed=1)
    assert diff.p_value < 0.01 and diff.jsd > same.jsd
    assert diff.excess > 0.05 and abs(same.excess) < 0.02  # noise removed
    assert same.p_value > 0.05


def test_log_odds_ranks_emerging_and_declining_terms():
    # equal totals (200 words each), so "news" has the same share in both
    old = Counter({"flu": 50, "election": 50, "news": 100})
    new = Counter({"covid": 60, "flu": 40, "news": 100})
    prior = old + new
    t = lex.log_odds_dirichlet(new, old, prior, prior_size=100, min_count=5)
    assert t.index[0] == "covid"            # most emerging
    assert t.index[-1] == "election"        # most declining
    assert abs(t.loc["news", "z"]) < 1.0    # unchanged term near zero


def test_new_term_rate():
    assert lex.new_term_rate([["a", "b"], ["c", "a"]], {"a"}) == 0.5


# ---------- indexes ----------

def test_vector_index_search_and_cumulative_refresh():
    vecs = np.eye(4, dtype=np.float32)
    docs = pd.DataFrame({"doc_id": ["b1", "b2", "n1", "n2"],
                         "period": ["base", "base", "2020", "2021"]})
    idx = build_study_indexes(docs, vecs, "base", ["2020", "2021"])
    assert len(idx.stale) == 2
    assert len(idx.refreshed["2020"]) == 3 and len(idx.refreshed["2021"]) == 4
    ids, scores = idx.refreshed["2021"].search(vecs[[2]], k=2)
    assert ids[0, 0] == "n1" and scores[0, 0] == pytest.approx(1.0)
    stale_ids, _ = idx.stale.search(vecs[[2]], k=5)
    assert set(stale_ids[0]) <= {"b1", "b2"}  # the stale index never has new documents
    with pytest.raises(ValueError):
        VectorIndex(["a"], vecs)


# ---------- multiple testing ----------

def test_benjamini_hochberg():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    # explicit values: sorted p = .01,.03,.04,.5 -> .04,.06,.0533,.5 -> monotone .04,.0533,.0533,.5
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], rel=1e-4)
    assert benjamini_hochberg([]).size == 0
