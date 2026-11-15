"""Why answers went wrong: knowledge, retrieval or producing (D38)."""
import numpy as np
import pandas as pd

from driftmon.evaluation import attribution as at
from driftmon.evaluation.retrieval import retrieve_all
from driftmon.rag.fault import without_documents
from driftmon.rag.index import build_study_indexes

NAN = np.nan
QUESTIONS = pd.DataFrame({
    "question_id": ["k", "r", "p", "ok", "base", "unj", "un", "gone"],
    "set": ["answerable"] * 6 + ["unanswerable", "answerable"],
    "period": ["2020", "2020", "2020", "2020", "2016-17", "2020", "2020", "2020"],
    "category": ["WELLNESS"] * 8})

ROWS = [  # question, role, index, correct, context_has_answer, source in the index searched
    ("k", "stale", "stale", 0.0, 0.0, False),                  # 2020 article, stale index: never indexed
    ("k", "refreshed", "refreshed_2020", 1.0, 1.0, True),
    ("r", "refreshed", "refreshed_2020", 0.0, 0.0, True),      # indexed, but the top 5 held no answer
    ("p", "refreshed", "refreshed_2020", 0.0, 1.0, True),      # answer in the top 5, still wrong
    ("ok", "refreshed", "refreshed_2020", 1.0, 1.0, True),
    ("base", "stale", "stale", 0.0, 0.0, True),                # baseline article on the stale index: retrieval
    ("unj", "refreshed", "refreshed_2020", NAN, 1.0, True),    # judge failed: left out
    ("un", "stale", "stale", 1.0, 0.0, False),                 # unanswerable: left out
    ("gone", "refreshed", "refreshed_2020", 0.0, 0.0, False),  # a failed re-index left the article out: knowledge
]


def _tables():
    df = pd.DataFrame(ROWS, columns=["question_id", "role", "index", "correct", "context_has_answer",
                                     "source_in_index"])
    return df.drop(columns="source_in_index"), df[["question_id", "index", "source_in_index"]]


def test_each_wrong_answer_gets_one_bucket():
    answers, runs = _tables()
    out = at.attribute(answers, QUESTIONS, runs)
    got = {(r["question_id"], r["role"]): r["bucket"] for r in out.to_dict("records")}
    assert got[("k", "stale")] == at.KNOWLEDGE
    assert got[("r", "refreshed")] == at.RETRIEVAL
    assert got[("p", "refreshed")] == at.PRODUCING
    assert got[("base", "stale")] == at.RETRIEVAL
    assert got[("gone", "refreshed")] == at.KNOWLEDGE   # membership comes from the index searched, not the year
    assert pd.isna(got[("k", "refreshed")]) and pd.isna(got[("ok", "refreshed")])   # right answers: no bucket
    assert ("unj", "refreshed") not in got and ("un", "stale") not in got


def test_unknown_is_not_guessed():
    assert at.bucket(0.0, True, NAN) is None          # in the index, but no grade: cannot tell retrieval from producing
    assert at.bucket(0.0, False, NAN) == at.KNOWLEDGE  # not in the index needs no grade
    assert at.bucket(0.0, None, 0.0) is None           # membership unknown
    answers, runs = _tables()
    out = at.attribute(answers, QUESTIONS, runs.drop(columns="source_in_index"))   # a run without membership
    s = at.summary(out).set_index("role")
    assert s[["knowledge", "retrieval", "producing"]].to_numpy().sum() == 0
    assert s["unplaced"].sum() == s["wrong"].sum() > 0


def test_summary_counts_per_index():
    answers, runs = _tables()
    s = at.summary(at.attribute(answers, QUESTIONS, runs)).set_index("role")
    assert s.loc["stale", "wrong"] == 2 and s.loc["stale", "knowledge"] == 1 and s.loc["stale", "retrieval"] == 1
    assert s.loc["refreshed", "answers"] == 5 and s.loc["refreshed", "wrong"] == 3
    assert s.loc["refreshed", "producing"] == 1 and s.loc["refreshed", "knowledge"] == 1
    assert s.loc["refreshed", "unplaced"] == 0


def test_retrieval_records_whether_the_source_is_in_the_index_searched():
    docs = pd.DataFrame({"doc_id": ["b1", "b2", "n1", "n2"], "period": ["2016-17", "2016-17", "2020", "2020"],
                         "category": ["WELLNESS"] * 4})
    vecs = np.eye(4, dtype=np.float32)
    idx = build_study_indexes(docs, vecs, "2016-17", ["2020"])
    q = pd.DataFrame({"question_id": ["qb", "qn1", "qn2"], "doc_id": ["b1", "n1", "n2"],
                      "period": ["2016-17", "2020", "2020"]})
    held = retrieve_all(q, vecs[[0, 2, 3]], idx, "2016-17", 2).set_index(["question_id", "index"])["source_in_index"]
    assert held[("qb", "stale")] and held[("qb", "refreshed_2020")]
    assert not held[("qn1", "stale")] and held[("qn1", "refreshed_2020")]
    broken = without_documents(idx, docs, vecs, {"n2"})                       # a failed re-index drops n2
    held = retrieve_all(q, vecs[[0, 2, 3]], broken, "2016-17", 2).set_index(["question_id", "index"])["source_in_index"]
    assert held[("qn1", "refreshed_2020")] and not held[("qn2", "refreshed_2020")]
