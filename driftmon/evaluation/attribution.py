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

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

KNOWLEDGE, RETRIEVAL, PRODUCING = "knowledge", "retrieval", "producing"
BUCKETS = (KNOWLEDGE, RETRIEVAL, PRODUCING)


def index_periods(baseline: str, later: Sequence[str]) -> dict[str, set[str]]:
    """The periods each study index holds: the stale index the baseline only,
    ``refreshed_<period>`` the baseline and every period up to that one."""
    out, held = {"stale": {baseline}}, {baseline}
    for p in later:
        held = held | {p}
        out[f"refreshed_{p}"] = held
    return out


def bucket(correct: float, in_index: bool, context_has_answer: float) -> str | None:
    """Where one answer's chain broke; None when it is right or cannot be scored."""
    if correct is None or np.isnan(correct) or correct == 1:
        return None
    if not in_index:
        return KNOWLEDGE
    if np.isnan(context_has_answer):
        return None
    return PRODUCING if context_has_answer == 1 else RETRIEVAL


def attribute(answers: pd.DataFrame, questions: pd.DataFrame,
              periods_in: Mapping[str, set[str]]) -> pd.DataFrame:
    """One row per scored answer to an answerable question, with ``bucket``
    (knowledge / retrieval / producing, or None when the answer is right).

    ``answers``: Agent D's per-answer table (question_id, role, index,
    correct, context_has_answer). ``questions``: question_id, set, period,
    category. ``periods_in``: index name -> the periods it holds.
    """
    q = questions[questions["set"] == "answerable"][["question_id", "period", "category"]]
    a = answers.merge(q, on="question_id")
    a = a[a["correct"].notna()].copy()
    a["in_index"] = [p in periods_in.get(i, set()) for p, i in zip(a["period"], a["index"])]
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
