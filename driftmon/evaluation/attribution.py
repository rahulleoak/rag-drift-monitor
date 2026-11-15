"""Why answers went wrong: knowledge, retrieval or producing (Agent D; D38).

One score can drop for different reasons, and each reason has a different
fix. For every wrong answer to an answerable question, code asks where the
chain broke, in order:

    knowledge   the question's source article is not in the index at all
                (the stale index never got that year): update or re-index;
    retrieval   it is in the index, but the top k the answer step read held
                no answer (Agent C's relevance grades): chunking, embeddings,
                retrieval settings;
    producing   the top k held an answer and the answer is still wrong:
                the answer step (model, prompt, conflicting context).

Query drift (people asking in new words) is measured separately, on the
queries themselves (Agent B, and the wording test F23). It is one cause of
retrieval misses, not a fourth outcome for an answer, so it is not a bucket
here.

Answers whose grades failed stay out (never counted as wrong), and
unanswerable questions are left out: declining is their right answer.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

KNOWLEDGE, RETRIEVAL, PRODUCING = "knowledge", "retrieval", "producing"
BUCKETS = (KNOWLEDGE, RETRIEVAL, PRODUCING)


def bucket(correct: float, in_index: bool | None, context_has_answer: float) -> str | None:
    """Where one answer's chain broke; None when it is right or cannot be placed
    (index membership or the relevance grade unknown)."""
    if correct is None or np.isnan(correct) or correct == 1 or in_index is None:
        return None
    if not in_index:
        return KNOWLEDGE
    if np.isnan(context_has_answer):
        return None
    return PRODUCING if context_has_answer == 1 else RETRIEVAL


def attribute(answers: pd.DataFrame, questions: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """One row per scored answer to an answerable question, with ``bucket``
    (knowledge / retrieval / producing, or None when the answer is right).

    ``answers``: Agent D's per-answer table (question_id, role, index,
    correct, context_has_answer). ``questions``: question_id, set, period,
    category. ``runs``: Agent C's retrieval runs, with ``source_in_index``
    from the index actually searched, so an article a failed re-index left
    out counts as knowledge, not as a search miss.
    """
    q = questions[questions["set"] == "answerable"][["question_id", "period", "category"]]
    held = (runs.set_index(["question_id", "index"])["source_in_index"].to_dict()
            if "source_in_index" in runs.columns else {})
    a = answers.merge(q, on="question_id")
    a = a[a["correct"].notna()].copy()
    a["in_index"] = [held.get((qid, i)) for qid, i in zip(a["question_id"], a["index"])]   # None: unknown
    a["bucket"] = [bucket(c, ii, h) for c, ii, h in zip(a["correct"], a["in_index"], a["context_has_answer"])]
    return a[["question_id", "role", "index", "period", "category", "correct", "in_index",
              "context_has_answer", "bucket"]].reset_index(drop=True)


def summary(attributed: pd.DataFrame, by: Sequence[str] = ("role",)) -> pd.DataFrame:
    """Counts per group: answers scored, wrong, and wrong per bucket (wrong
    answers that could not be placed are counted as ``unplaced``)."""
    rows = []
    for key, g in attributed.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        wrong = g[g["correct"] == 0]
        row = dict(zip(by, key))
        row.update(answers=len(g), wrong=len(wrong))
        for b in BUCKETS:
            row[b] = int((wrong["bucket"] == b).sum())
        row["unplaced"] = int(wrong["bucket"].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
