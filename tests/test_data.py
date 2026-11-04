"""Tests for dataset loading (driftmon.data) using synthetic fixtures."""
from datetime import date
from pathlib import Path

import pytest

from driftmon.data import LoadStats, iter_records

FIXTURES = Path(__file__).parent / "fixtures"


def test_huffpost_parses_valid_lines_and_counts_skipped():
    stats = LoadStats()
    records = list(iter_records("huffpost", FIXTURES / "huffpost_tiny.jsonl", stats=stats))
    # 7 non-blank lines: 5 valid, 1 invalid JSON, 1 missing date
    assert stats.lines_read == 7
    assert stats.records == 5
    assert stats.skipped == 2
    assert [r.date.year for r in records] == [2014, 2014, 2015, 2020, 2020]


def test_huffpost_field_mapping_and_whitespace_cleanup():
    records = list(iter_records("huffpost", FIXTURES / "huffpost_tiny.jsonl"))
    third = records[2]
    assert third.title == "Synthetic headline three"
    assert third.category == "WELLNESS"
    assert third.text == "Synthetic description three."
    assert records[1].text == ""  # empty descriptions are kept, not dropped


def test_arxiv_uses_first_version_date_and_primary_category():
    stats = LoadStats()
    records = list(iter_records("arxiv", FIXTURES / "arxiv_tiny.jsonl", stats=stats))
    assert stats.records == 2 and stats.skipped == 1  # empty categories skipped
    first = records[0]
    assert first.date == date(2007, 4, 2)  # v1 created, not update_date (2008)
    assert first.category == "hep-ph"
    assert first.title == "Synthetic title one"
    assert first.text == "Synthetic abstract one."
    assert records[1].date == date(2020, 1, 5)  # falls back to update_date


def test_limit_stops_early():
    records = list(iter_records("huffpost", FIXTURES / "huffpost_tiny.jsonl", limit=2))
    assert len(records) == 2


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        list(iter_records("nope", FIXTURES / "huffpost_tiny.jsonl"))
