"""Tests for the exploration aggregates (scripts/explore_data.py)."""
from pathlib import Path

from driftmon.data import iter_records
from scripts.explore_data import (
    category_year_table,
    main,
    summarize,
    top_categories_table,
    year_table,
)

FIXTURE = Path(__file__).parent / "fixtures" / "huffpost_tiny.jsonl"


def _summary(k=5, seed=42):
    return summarize(iter_records("huffpost", FIXTURE), samples_per_year=k, seed=seed)


def test_counts_per_year_and_category():
    s = _summary()
    assert dict(s.per_year) == {2014: 2, 2015: 1, 2020: 2}
    table = category_year_table(s)
    assert table.loc["POLITICS", 2014] == 2
    assert table.loc["WELLNESS", 2020] == 1
    assert table.loc["WELLNESS", "total"] == 2
    assert list(year_table(s)["articles"]) == [2, 1, 2]
    assert top_categories_table(s).iloc[0]["category"] in {"POLITICS", "WELLNESS"}


def test_summary_stats():
    s = _summary()
    assert s.empty_text == 1
    assert str(s.first_date) == "2014-03-01" and str(s.last_date) == "2020-11-30"
    assert s.median_words() == 6  # word counts: 6, 3, 6, 6, 6


def test_reservoir_sampling_is_bounded_and_repeatable():
    a, b = _summary(k=1, seed=7), _summary(k=1, seed=7)
    assert all(len(v) == 1 for v in a.samples.values())
    assert {y: [r.doc_id for r in v] for y, v in a.samples.items()} == \
           {y: [r.doc_id for r in v] for y, v in b.samples.items()}


def test_main_writes_csvs(tmp_path, capsys):
    code = main(["--dataset", "huffpost", "--path", str(FIXTURE), "--out-dir", str(tmp_path)])
    assert code == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "articles_per_year.csv", "articles_per_category_year.csv",
        "top_categories.csv", "sample_titles.csv"}
    assert "Articles per year" in capsys.readouterr().out


def test_main_reports_missing_file(tmp_path):
    assert main(["--dataset", "huffpost", "--path", str(tmp_path / "missing.json")]) == 2
