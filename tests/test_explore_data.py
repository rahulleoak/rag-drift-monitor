"""Tests for the exploration aggregates (scripts/exploration/explore_data.py)."""
from pathlib import Path

from driftmon.data.records import iter_records
from scripts.exploration.explore_data import (
    category_year_table,
    main,
    summarize,
    term_rate_table,
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


def test_watch_term_rates_per_year():
    # Fixture texts: 2014 has "one" and "two"; 2015 "three"; 2020 "five" and "six".
    s = summarize(iter_records("huffpost", FIXTURE), samples_per_year=1, seed=0,
                  watch_terms={"one_or_five": r"\b(one|five)\b", "upper": r"SYNTHETIC"})
    table = term_rate_table(s)
    assert table.loc["one_or_five"].to_dict() == {2014: 50.0, 2015: 0.0, 2020: 50.0}
    assert (table.loc["upper"] == 100.0).all()  # matching is case-insensitive
    # "one" appears twice in record a1 but the record is counted once.
    assert s.term_year[("one_or_five", 2014)] == 1


def test_no_watch_terms_gives_empty_table():
    assert term_rate_table(_summary()).empty


def test_main_writes_csvs(tmp_path, capsys):
    code = main(["--dataset", "huffpost", "--path", str(FIXTURE), "--out-dir", str(tmp_path)])
    assert code == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "articles_per_year.csv", "articles_per_category_year.csv",
        "top_categories.csv", "sample_titles.csv", "watch_term_rates.csv"}
    assert "Articles per year" in capsys.readouterr().out


def test_main_reports_missing_file(tmp_path):
    assert main(["--dataset", "huffpost", "--path", str(tmp_path / "missing.json")]) == 2
