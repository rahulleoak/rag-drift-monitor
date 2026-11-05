"""Tests for the cleaning rules (driftmon.cleaning) and periods (driftmon.periods)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from driftmon import cleaning
from driftmon.periods import Period, assign_period, study_periods


def _df(rows):
    return pd.DataFrame(rows, columns=["title", "text"])


def test_word_count_rules():
    df = _df([("One two", ""), ("One two three", "four five six seven eight"), ("a " * 50, "b " * 31)])
    assert list(cleaning.word_counts(df)) == [2, 8, 81]
    assert list(cleaning.empty_description(df)) == [True, False, False]
    assert list(cleaning.too_short(df, 8)) == [True, False, False]  # 8 words is kept
    assert list(cleaning.too_long(df, 80)) == [False, False, True]  # 81 is too long


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_near_duplicates_keeps_earliest_copy():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    a_copy = _unit(1, 0.05, 0)  # cosine with a is about 0.999
    vectors = np.stack([a, b, a_copy, a])
    assert list(cleaning.near_duplicates(vectors, 0.95)) == [False, False, True, True]


def test_near_duplicates_compares_with_kept_rows_only():
    # x and z are each 0.96 similar to y but only about 0.85 to each other.
    # y is dropped as a copy of x, so z (compared only with kept rows) is kept.
    x = _unit(1, 0, 0)
    y = _unit(np.cos(0.29), np.sin(0.29), 0)
    z = _unit(np.cos(0.58), np.sin(0.58), 0)
    assert float(x @ y) >= 0.95 and float(x @ z) < 0.95
    assert list(cleaning.near_duplicates(np.stack([x, y, z]), 0.95)) == [False, True, False]


def test_near_duplicates_result_does_not_depend_on_chunk_size():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 8)).astype(np.float32)
    noisy = base[rng.integers(0, 30, size=70)] + 0.05 * rng.normal(size=(70, 8)).astype(np.float32)
    vectors = np.vstack([base, noisy])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = cleaning.near_duplicates(vectors, 0.95, chunk_size=1000)
    assert expected.sum() > 0
    for size in (1, 7, 32):
        assert (cleaning.near_duplicates(vectors, 0.95, chunk_size=size) == expected).all()


def test_equal_samples_size_seed_and_shortfall():
    periods = pd.Series(["a"] * 10 + ["b"] * 5)
    s1 = cleaning.equal_samples(periods, 5, seed=1)
    assert s1.groupby(periods).sum().to_dict() == {"a": 5, "b": 5}
    assert (cleaning.equal_samples(periods, 5, seed=1) == s1).all()
    with pytest.raises(ValueError, match="fewer than"):
        cleaning.equal_samples(periods, 6, seed=1)


def test_periods_from_config_and_assignment():
    cfg = {"periods": {"baseline": {"name": "base", "start": "2016-01-01", "end": "2017-12-31"},
                       "later": [{"name": "2020", "start": "2020-01-01", "end": "2020-12-31"}]}}
    periods = study_periods(cfg)
    assert [p.name for p in periods] == ["base", "2020"]
    assert assign_period(date(2017, 12, 31), periods) == "base"
    assert assign_period(date(2020, 6, 1), periods) == "2020"
    assert assign_period(date(2018, 6, 1), periods) is None


def test_overlapping_periods_rejected():
    cfg = {"periods": {"baseline": {"name": "a", "start": "2016-01-01", "end": "2017-12-31"},
                       "later": [{"name": "b", "start": "2017-06-01", "end": "2018-12-31"}]}}
    with pytest.raises(ValueError, match="overlap"):
        study_periods(cfg)


def test_period_halves():
    first, second = Period("2016-17", date(2016, 1, 1), date(2017, 12, 31)).halves()
    assert (first.end, second.start) == (date(2016, 12, 31), date(2017, 1, 1))
