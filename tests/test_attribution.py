"""Why answers went wrong: knowledge, retrieval or producing (D38)."""
import numpy as np
import pandas as pd

from driftmon.evaluation import attribution as at

NAN = np.nan
QUESTIONS = pd.DataFrame({
    "question_id": ["k", "r", "p", "ok", "base", "unj", "un"],
    "set": ["answerable"] * 6 + ["unanswerable"],
    "period": ["2020", "2020", "2020", "2020", "2016-17", "2020", "2020"],
    "category": ["WELLNESS"] * 7})


def _answers():
    rows = [  # question, role, index, correct, context_has_answer
        ("k", "stale", "stale", 0.0, 0.0),                  # 2020 article, stale index: never indexed
        ("k", "refreshed", "refreshed_2020", 1.0, 1.0),
        ("r", "refreshed", "refreshed_2020", 0.0, 0.0),     # indexed, but the top 5 held no answer
        ("p", "refreshed", "refreshed_2020", 0.0, 1.0),     # answer in the top 5, still wrong
        ("ok", "refreshed", "refreshed_2020", 1.0, 1.0),
        ("base", "stale", "stale", 0.0, 0.0),               # baseline article on the stale index: retrieval
        ("unj", "refreshed", "refreshed_2020", NAN, 1.0),   # judge failed: left out
        ("un", "stale", "stale", 1.0, 0.0),                 # unanswerable: left out
    ]
    return pd.DataFrame(rows, columns=["question_id", "role", "index", "correct", "context_has_answer"])


def test_index_periods_are_cumulative():
    held = at.index_periods("2016-17", ["2018", "2019", "2020"])
    assert held["stale"] == {"2016-17"}
    assert held["refreshed_2019"] == {"2016-17", "2018", "2019"}
    assert "2020" not in held["refreshed_2019"]


def test_each_wrong_answer_gets_one_bucket():
    out = at.attribute(_answers(), QUESTIONS, at.index_periods("2016-17", ["2018", "2019", "2020"]))
    got = {(r["question_id"], r["role"]): r["bucket"] for r in out.to_dict("records")}
    assert got[("k", "stale")] == at.KNOWLEDGE
    assert got[("r", "refreshed")] == at.RETRIEVAL
    assert got[("p", "refreshed")] == at.PRODUCING
    assert got[("base", "stale")] == at.RETRIEVAL
    assert pd.isna(got[("k", "refreshed")]) and pd.isna(got[("ok", "refreshed")])   # right answers: no bucket
    assert ("unj", "refreshed") not in got and ("un", "stale") not in got


def test_missing_relevance_grade_is_not_guessed():
    assert at.bucket(0.0, True, NAN) is None          # in the index, but no grade: cannot tell retrieval from producing
    assert at.bucket(0.0, False, NAN) == at.KNOWLEDGE  # not in the index needs no grade


def test_summary_counts_per_index():
    out = at.attribute(_answers(), QUESTIONS, at.index_periods("2016-17", ["2018", "2019", "2020"]))
    s = at.summary(out).set_index("role")
    assert s.loc["stale", "wrong"] == 2 and s.loc["stale", "knowledge"] == 1 and s.loc["stale", "retrieval"] == 1
    assert s.loc["refreshed", "answers"] == 4 and s.loc["refreshed", "wrong"] == 2
    assert s.loc["refreshed", "producing"] == 1 and s.loc["refreshed", "unplaced"] == 0
