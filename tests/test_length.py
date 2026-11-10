"""Tests for the length check (D33)."""
import numpy as np
import pandas as pd

from driftmon.config import load_config
from driftmon.drift import length as ln


def test_matched_rows_copy_the_target_lengths_without_replacement():
    rng = np.random.default_rng(0)
    pool = np.arange(100)
    words = np.array([10] * 50 + [20] * 30 + [30] * 20)
    rows = ln.length_matched_rows(pool, words, [20] * 10 + [30] * 5, rng)
    assert len(set(rows)) == 15
    assert sorted(words[rows]) == [20] * 10 + [30] * 5


def test_matched_rows_fall_back_to_the_nearest_length():
    rng = np.random.default_rng(0)
    pool, words = np.arange(6), np.array([10, 10, 20, 20, 40, 40])
    rows = ln.length_matched_rows(pool, words, [21, 21, 21], rng)
    assert len(set(rows)) == 3 and sorted(words[rows]) == [10, 20, 20]


def _draws(kept):
    return pd.DataFrame({"period": ["2018", "2020"], "random_x_embedding": [2.0, 4.0],
                         "matched_x_embedding": [2.0 * kept, 4.0 * kept], "random_x_lexical": [1.0, 1.0],
                         "matched_x_lexical": [1.0, 1.0], "median_words_period": [25, 30],
                         "median_words_random": [24, 24], "median_words_matched": [25, 30]})


def test_verdict_by_the_fixed_rule():
    cfg = load_config()
    small = ln.verdict(ln.summarise(_draws(0.9)), cfg)
    assert not small["length_explains_a_large_part"] and abs(small["embedding_drop"] - 0.1) < 1e-9
    large = ln.verdict(ln.summarise(_draws(0.5)), cfg)
    assert large["length_explains_a_large_part"] and large["by_period"] == {"2018": 0.5, "2020": 0.5}
