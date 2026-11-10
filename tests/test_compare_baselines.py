"""Tests for the baseline comparison (scripts/exploration/compare_baselines.py)."""
import math
from collections import Counter
from datetime import date
from pathlib import Path

from driftmon.data.records import iter_records
from driftmon.data.periods import Period
from scripts.exploration.compare_baselines import (
    count_categories,
    main,
    mix_difference,
    mix_difference_table,
    stability_table,
    unmatched_share,
)

FIXTURE = Path(__file__).parent / "fixtures" / "huffpost_tiny.jsonl"
# Fixture records: 2014-03-01 POLITICS, 2014-07-15 POLITICS, 2015-01-02 WELLNESS,
# 2020-04-10 WELLNESS, 2020-11-30 U.S. NEWS.


def test_mix_difference_bounds_and_symmetry():
    a = Counter({"POLITICS": 3, "WELLNESS": 1})
    assert mix_difference(a, Counter({"POLITICS": 30, "WELLNESS": 10})) == 0.0  # scale-free
    assert mix_difference(a, Counter({"SPORTS": 5})) == 1.0
    b = Counter({"POLITICS": 1, "WELLNESS": 1})
    assert mix_difference(a, b) == mix_difference(b, a) == 0.25  # 75/25 vs 50/50
    assert math.isnan(mix_difference(a, Counter()))


def test_unmatched_share_counts_categories_missing_from_baseline():
    base = Counter({"POLITICS": 10})
    later = Counter({"POLITICS": 6, "U.S. NEWS": 4})
    assert unmatched_share(base, later) == 0.4
    assert math.isnan(unmatched_share(base, Counter()))


def test_period_halves_split_two_years_into_years():
    p = Period("2016-17", date(2016, 1, 1), date(2017, 12, 31))
    first, second = p.halves()
    assert (first.start, first.end) == (date(2016, 1, 1), date(2016, 12, 31))
    assert (second.start, second.end) == (date(2017, 1, 1), date(2017, 12, 31))


def test_count_categories_applies_renaming_and_overlapping_periods():
    periods = [Period("2014-15", date(2014, 1, 1), date(2015, 12, 31)),
               Period("2015", date(2015, 1, 1), date(2015, 12, 31)),
               Period("2020", date(2020, 1, 1), date(2020, 12, 31))]
    counts = count_categories(iter_records("huffpost", FIXTURE), periods,
                              normalise={"WELLNESS": "HEALTH"})
    assert counts["2014-15"] == Counter({"POLITICS": 2, "HEALTH": 1})
    assert counts["2015"] == Counter({"HEALTH": 1})  # a record can be in two periods
    assert counts["2020"] == Counter({"HEALTH": 1, "U.S. NEWS": 1})


def test_tables_on_fixture():
    cand = [Period("2014-15", date(2014, 1, 1), date(2015, 12, 31))]
    later = [Period("2020", date(2020, 1, 1), date(2020, 12, 31))]
    periods = cand + later + list(cand[0].halves())
    counts = count_categories(iter_records("huffpost", FIXTURE), periods, normalise={})
    mix = mix_difference_table(counts, cand, later)
    # 2014-15: 2/3 POLITICS, 1/3 WELLNESS; 2020: 1/2 WELLNESS, 1/2 U.S. NEWS
    # TVD = 0.5 * (2/3 + 1/6 + 1/2) = 2/3
    assert mix.loc["2014-15", "2020"] == 66.7
    assert mix.loc["2014-15", "articles"] == 3
    # halves: 2014 (all POLITICS) vs 2015 (all WELLNESS) -> 100%
    assert stability_table(counts, cand).loc["2014-15", "mix_difference_pct"] == 100.0


def test_main_writes_csvs(tmp_path):
    code = main(["--dataset", "huffpost", "--path", str(FIXTURE), "--out-dir", str(tmp_path)])
    assert code == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "mix_difference.csv", "internal_stability.csv", "unmatched_share.csv",
        "category_shares.csv"}


def test_main_reports_missing_file(tmp_path):
    assert main(["--path", str(tmp_path / "missing.json")]) == 2
