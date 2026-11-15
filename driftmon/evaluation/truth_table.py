"""The truth table: change one thing at a time and see where wrong answers land (D39).

The four-kinds split (D38) found no "producing" errors in the demo: every
answer whose top 5 held an answer was right. That is only convincing if the
split *would* catch a worse answer step. So the same questions are answered
again from the **same retrieved articles** (a saved pipeline run's top k),
changing one thing at a time:

    baseline   the pipeline's answer model and prompt (already in the cache)
    prompt     the same model, with one realistic bad edit to the prompt
               ("under 10 words" instead of "one to three sentences")
    model      a much smaller answer model, the same prompt
    both       both changes

The same judge grades every answer, and the same code splits the wrong ones.
Because the articles do not change, knowledge and retrieval counts should
not move (the isolation check); any loss must land in producing. Code
decides, with thresholds set before any result (``truth_table`` in the
config): producing is "caught" when the share of wrong answers among those
with an answer in front of the model rises by at least ``min_rise`` and a
paired exact test (McNemar, one-sided) gives p < ``alpha``.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from driftmon.evaluation import answers as an
from driftmon.evaluation import attribution as at
from driftmon.evaluation.wording import mcnemar_p
from driftmon.llm.client import LLMRequest
from driftmon.llm.prompts import ANSWER_SHORT_SYSTEM, ANSWER_SHORT_VERSION, RagAnswerShort

BASELINE = "baseline"


@dataclass(frozen=True)
class Cell:
    name: str
    label: str
    small_model: bool    # the model change
    short_prompt: bool   # the prompt change


CELLS = (Cell(BASELINE, "pipeline model, pipeline prompt", False, False),
         Cell("prompt", "pipeline model, 'under 10 words' prompt", False, True),
         Cell("model", "smaller model, pipeline prompt", True, False),
         Cell("both", "smaller model, 'under 10 words' prompt", True, True))


def parse_ids(value: Any) -> list[str]:
    """Doc ids as saved in retrieval_runs.csv ("[np.str_('hp-1'), ...]" or a list)."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    return re.findall(r"'([^']+)'", str(value))


def load_plan(saved_answers: pd.DataFrame, runs: pd.DataFrame, k: int) -> pd.DataFrame:
    """The saved run's answer plan for its answerable questions: one row per (question, role),
    its index, the top k it read and Agent C's grade of that top k (``context_has_answer``).
    Unanswerable questions are left out: the run does not keep their top k, and declining is
    their right answer, so they have no producing bucket (D38)."""
    ids = {(q, i): parse_ids(d)[:k] for q, i, d in zip(runs["question_id"], runs["index"], runs["doc_ids"])}
    keep = [(q, i) in ids for q, i in zip(saved_answers["question_id"], saved_answers["index"])]
    plan = saved_answers.loc[keep, ["question_id", "role", "index", "context_has_answer"]].reset_index(drop=True)
    plan["doc_ids"] = [ids[(q, i)] for q, i in zip(plan["question_id"], plan["index"])]
    return plan


def answer_requests(plan: pd.DataFrame, questions: pd.DataFrame, docs: pd.DataFrame,
                    cfg: Mapping[str, Any], cell: Cell) -> list[LLMRequest]:
    """The pipeline's own answer requests, with the cell's one change applied."""
    reqs = an.answer_requests(plan, questions, docs, cfg)
    changes: dict[str, Any] = {}
    if cell.small_model:
        changes["model"] = cfg["truth_table"]["small_model"]
    if cell.short_prompt:
        changes.update(system=ANSWER_SHORT_SYSTEM, schema=RagAnswerShort, prompt_version=ANSWER_SHORT_VERSION)
    return [dataclasses.replace(r, **changes) for r in reqs] if changes else reqs


def score(plan: pd.DataFrame, answers: Sequence[Any], judgements: Sequence[Any], questions: pd.DataFrame,
          runs: pd.DataFrame, cell: str) -> pd.DataFrame:
    """One row per (question, role): the judge's grades (as in Agent D), the four-kinds bucket
    of a wrong answer to an answerable question, and the answer's length in words."""
    has = {(q, r): h for q, r, h in zip(plan["question_id"], plan["role"], plan["context_has_answer"])}
    pa = an.per_answer(plan, answers, judgements, questions, has)
    why = at.attribute(pa, questions, runs)[["question_id", "role", "in_index", "bucket"]]
    pa = pa.merge(why, on=["question_id", "role"], how="left")
    pa = pa.merge(questions[["question_id", "set"]], on="question_id")
    pa["context"] = ["|".join(d) for d in plan["doc_ids"]]
    pa["words"] = [len(a.split()) if isinstance(a, str) else np.nan for a in pa["answer"]]
    pa.insert(0, "cell", cell)
    return pa


def _had_answer(s: pd.DataFrame) -> pd.Series:
    """Answerable, source article in the index, and the top k held an answer: the answers
    only the answer step can get wrong."""
    return (s["set"] == "answerable") & s["in_index"].eq(True) & (s["context_has_answer"] == 1) & s["correct"].notna()


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Per cell and index role: the four-kinds counts, how often an answer in front of the model
    still went wrong, answer length and groundedness."""
    rows = []
    for (cell, role), g in scored.groupby(["cell", "role"], sort=False):
        a = g[(g["set"] == "answerable") & g["correct"].notna()]
        wrong = a[a["correct"] == 0]
        had = g[_had_answer(g)]
        row = {"cell": cell, "role": role, "answers": len(a), "wrong": len(wrong)}
        row.update({b: int((wrong["bucket"] == b).sum()) for b in at.BUCKETS})
        row.update(unplaced=int(wrong["bucket"].isna().sum()), had_answer=len(had),
                   producing_share=float((had["correct"] == 0).mean()) if len(had) else np.nan,
                   partly_right=int((had["correctness"] == 1).sum()),
                   mean_score=float(a["score"].mean()) if len(a) else np.nan,
                   median_words=float(a["words"].median()) if len(a) else np.nan,
                   hallucinated=float(a["hallucinated"].mean()) if len(a) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _unique(s: pd.DataFrame) -> pd.DataFrame:
    """One row per (question, context): the two indexes often hand over the same top k, which is
    one request, so it counts once in the test."""
    return s.drop_duplicates(["question_id", "context"]).set_index(["question_id", "context"])


def verdict(scored: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Per changed cell, by the thresholds fixed before results: did the split stay isolated
    (knowledge and retrieval unchanged), and was the producing loss caught?"""
    t = cfg["truth_table"]
    tol, min_rise, alpha = float(t["isolation_tolerance"]), float(t["min_rise"]), float(t["alpha"])
    summ = summarise(scored).set_index(["cell", "role"])
    base = _unique(scored[(scored["cell"] == BASELINE) & _had_answer(scored)])
    out: dict[str, Any] = {"thresholds": {"isolation_tolerance": tol, "min_rise": min_rise, "alpha": alpha}}
    for cell in [c.name for c in CELLS if c.name != BASELINE and c.name in set(scored["cell"])]:
        moves = {}
        for role in summ.loc[cell].index:
            b, c = summ.loc[(BASELINE, role)], summ.loc[(cell, role)]
            n = max(int(c["answers"]), 1)
            moves[role] = {k: int(c[k] - b[k]) for k in (at.KNOWLEDGE, at.RETRIEVAL, at.PRODUCING)}
            moves[role]["within_tolerance"] = bool(abs(c[at.KNOWLEDGE] - b[at.KNOWLEDGE]) / n <= tol
                                                   and abs(c[at.RETRIEVAL] - b[at.RETRIEVAL]) / n <= tol)
        isolated = all(m["within_tolerance"] for m in moves.values())
        mine = _unique(scored[(scored["cell"] == cell) & _had_answer(scored)])
        both = base[["correct"]].rename(columns={"correct": "correct_base"}).join(mine[["correct"]], how="inner")
        lost = int(((both["correct_base"] == 1) & (both["correct"] == 0)).sum())
        gained = int(((both["correct_base"] == 0) & (both["correct"] == 1)).sum())
        rise = float((both["correct"] == 0).mean() - (both["correct_base"] == 0).mean()) if len(both) else np.nan
        p = mcnemar_p(lost, gained)
        caught = bool(len(both) and rise >= min_rise and p < alpha)
        out[cell] = {"isolated": isolated, "moves": moves, "paired_answers": len(both), "lost": lost,
                     "gained": gained, "producing_rise": rise, "p": p, "producing_caught": caught,
                     "reading": _reading(isolated, caught, rise)}
    return out


def _reading(isolated: bool, caught: bool, rise: float) -> str:
    if not isolated:
        return "not isolated: knowledge or retrieval counts moved although the articles did not change"
    if caught:
        return f"caught: the loss lands in producing (+{rise:.1%} wrong with an answer in front of the model)"
    if rise > 0:
        return f"no reliable change: +{rise:.1%}, below the threshold or not significant"
    return "no loss: the answer step holds with this change"
