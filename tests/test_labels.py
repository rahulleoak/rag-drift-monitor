"""Tests for the hand-label sheet and agreement scoring (P8)."""
import pandas as pd
import pytest

from driftmon import labels


def _table():
    rows = []
    for i in range(6):
        rows.append({"question_id": f"k{i}", "set": "answerable", "rejected_by": "", "question": f"Kept {i}?",
                     "reference_answer": "A.", "title": f"T{i}", "text": "x"})
    for i in range(4):
        rows.append({"question_id": f"r{i}", "set": "", "rejected_by": "review 1: not natural",
                     "question": f"Rej {i}?", "reference_answer": "A.", "title": f"R{i}", "text": "x"})
    rows.append({"question_id": "c0", "set": "", "rejected_by": "copies the headline", "question": "Code?",
                 "reference_answer": "A.", "title": "C", "text": "x"})
    return pd.DataFrame(rows)


def test_sheet_hides_verdicts_and_samples_both_kinds():
    t = _table()
    docs = pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "date": ["2020-01-01"] * 3, "title": ["A", "B", "C"],
                         "text": ["a", "b", "c"]})
    grades = pd.DataFrame({"question_id": ["k0", "k0", "k1", "k1"], "doc_id": ["d1", "d2", "d3", "d1"],
                           "grade": [2, 0, 1, float("nan")]})
    q = labels.sample_questions(t, kept=3, rejected=10, seed=0)
    assert (q["verdict"] == 1).sum() == 3 and (q["verdict"] == 0).sum() == 4  # all reviewer rejections
    assert "Code?" not in set(q["question"])           # code-check rejections are not the reviewer's
    r = labels.sample_relevance(grades, t, docs, per_grade=5, seed=0)
    assert sorted(r["verdict"]) == [0, 1, 2]           # ungraded pairs are not sampled
    sheet, key = labels.build_sheet([q, r], seed=0)
    assert "verdict" not in sheet.columns and list(sheet.columns) == labels.SHEET_COLUMNS
    assert set(sheet["item_id"]) == set(key["item_id"]) and len(sheet) == 10


def test_kappa():
    assert labels.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert labels.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_score():
    key = pd.DataFrame({"item_id": ["L1", "L2", "L3", "L4", "L5"],
                        "part": ["relevance", "relevance", "relevance", "questions", "questions"],
                        "source_id": ["a"] * 5, "verdict": [2, 1, 0, 1, 1]})
    sheet = pd.DataFrame({"item_id": ["L1", "L2", "L3", "L4", "L5"], "your_label": ["2", "0", "0", "0", ""]})
    s = labels.score(sheet, key).set_index("part")
    assert s.loc["relevance", "agreement"] == pytest.approx(2 / 3)
    assert s.loc["relevance", "answers_agreement"] == 1.0 and s.loc["relevance", "llm_more_lenient"] == 1
    assert s.loc["questions", "labelled"] == 1 and s.loc["questions", "unlabelled"] == 1
    assert s.loc["questions", "llm_kept_you_rejected"] == 1


def test_fact_items_are_all_flagged_by_the_check():
    changes = pd.DataFrame({"page_id": [1, 2, 3], "title": ["A", "B", "C"], "period": ["2020"] * 3,
                            "before": ["x 1."] * 3, "after": ["x 2."] * 3})
    f = labels.sample_facts(changes, n=2, seed=0)
    assert len(f) == 2 and (f["verdict"] == 1).all() and f["text"].str.startswith("BEFORE:").all()
    assert "facts" in labels.INSTRUCTIONS
