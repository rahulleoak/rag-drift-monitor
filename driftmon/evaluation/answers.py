"""Answer quality, stale vs refreshed (Agent D; D13, D28).

For each test question, the RAG pipeline answers twice: from the top k of
the stale index and of the refreshed index (Agent C's saved retrieval). An
LLM judge grades each answer on its own (correctness 0/1/2, grounded,
declined), without being told which index it came from.

Code then decides everything:
    - per answer: correct, hallucinated (not grounded), and whether a
      decline was right (from Agent C's relevance grades: did the top k
      hold an answer?);
    - per segment (period x section): the loss in correct answers on the
      stale index, with a bootstrap interval; "material" when the loss is at
      least ``min_answer_quality_drop`` and the interval excludes zero;
    - the four classes (drift that hurts / drift that does not / harm
      without drift / healthy) and a likely cause (D13);
    - the rank correlation between drift scores and quality loss.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from driftmon.rag.embeddings import document_text
from driftmon.evaluation.stats import bootstrap_mean_ci
from driftmon.llm.client import LLMRequest
from driftmon.llm.prompts import (ANSWER_JUDGE_SYSTEM, ANSWER_JUDGE_TEMPLATE, ANSWER_JUDGE_VERSION,
                              ANSWER_SYSTEM, ANSWER_TEMPLATE, ANSWER_VERSION, AnswerJudgement,
                              RagAnswer)

ROLES = ("stale", "refreshed")
DRIFT_HURTS, DRIFT_HARMLESS = "drift that hurts", "drift that does not hurt"
HARM_WITHOUT_DRIFT, HEALTHY = "harm without drift", "healthy"
BORDERLINE = "borderline (sent back for more evidence)"   # not harmless: the loop decides (D28, D34)
DRIFT_METRICS = ("mmd2", "jsd_excess", "centroid_shift", "novelty_mean", "novel_share", "new_term_rate",
                 "times_ordinary_embedding", "times_ordinary_lexical")


def answer_plan(questions: pd.DataFrame, runs: pd.DataFrame, pairs: Mapping[str, tuple[str, str]],
                k: int) -> pd.DataFrame:
    """One row per (question, role): which index, and its top-k doc ids.

    ``pairs`` maps question_id -> (stale index name, refreshed index name).
    """
    by_run = runs.set_index(["question_id", "index"])["doc_ids"]
    rows = []
    for qid in questions["question_id"]:
        for role, name in zip(ROLES, pairs[qid]):
            rows.append({"question_id": qid, "role": role, "index": name,
                         "doc_ids": list(by_run[(qid, name)])[:k]})
    return pd.DataFrame(rows)


def article_listing(doc_ids: Sequence[str], docs: pd.DataFrame) -> str:
    """The retrieved articles as the assistant sees them: lettered, in rank order, dated."""
    d = docs.set_index("doc_id")
    return "\n".join(f"[{ascii_uppercase[i]}] (published {d.at[x, 'date']}) "
                     f"{document_text(d.at[x, 'title'], d.at[x, 'text'])}" for i, x in enumerate(doc_ids))


def answer_requests(plan: pd.DataFrame, questions: pd.DataFrame, docs: pd.DataFrame,
                    cfg: Mapping[str, Any]) -> list[LLMRequest]:
    """One Sonnet request per (question, role). Identical contexts give
    identical requests, so they are answered (and paid for) once."""
    job = cfg["llm"]["jobs"]["answers"]
    q = questions.set_index("question_id")["question"]
    return [LLMRequest(job="answers", model=cfg["llm"]["generator_model"], system=ANSWER_SYSTEM,
                       prompt=ANSWER_TEMPLATE.format(question=q[r["question_id"]],
                                                     articles=article_listing(r["doc_ids"], docs)),
                       schema=RagAnswer, prompt_version=ANSWER_VERSION, effort=job["effort"],
                       max_tokens=int(job["max_tokens"]))
            for r in plan.to_dict("records")]


def judge_requests(plan: pd.DataFrame, answers: Sequence[Any], questions: pd.DataFrame, docs: pd.DataFrame,
                   cfg: Mapping[str, Any]) -> list[LLMRequest | None]:
    """One Opus request per answer (None where answering failed). The index
    is not named; identical (context, answer) pairs share one request."""
    job = cfg["llm"]["jobs"]["answer_judge"]
    q = questions.set_index("question_id")
    out: list[LLMRequest | None] = []
    for r, a in zip(plan.to_dict("records"), answers):
        if a is None or not a.ok:
            out.append(None)
            continue
        out.append(LLMRequest(
            job="answer_judge", model=cfg["llm"]["judge_model"], system=ANSWER_JUDGE_SYSTEM,
            prompt=ANSWER_JUDGE_TEMPLATE.format(question=q.at[r["question_id"], "question"],
                                                reference_answer=q.at[r["question_id"], "reference_answer"],
                                                articles=article_listing(r["doc_ids"], docs),
                                                answer=a.parsed.answer),
            schema=AnswerJudgement, prompt_version=ANSWER_JUDGE_VERSION, effort=job["effort"],
            max_tokens=int(job["max_tokens"])))
    return out


def per_answer(plan: pd.DataFrame, answers: Sequence[Any], judgements: Sequence[Any],
               questions: pd.DataFrame, context_has_answer: Mapping[tuple[str, str], float]) -> pd.DataFrame:
    """One row per (question, role) with the judge's grades and the derived
    metrics. Failed answers or judgements stay as NaN, never scored 0.

    ``context_has_answer`` maps (question_id, role) -> 1/0 from Agent C's
    relevance grades (did the top k hold an answer?). Unanswerable questions
    have no answer anywhere by design, so their only correct response is to
    decline.
    """
    qset = questions.set_index("question_id")["set"]
    rows = []
    for r, a, j in zip(plan.to_dict("records"), answers, judgements):
        unanswerable = qset[r["question_id"]] == "unanswerable"
        row = {"question_id": r["question_id"], "role": r["role"], "index": r["index"],
               "answer": a.parsed.answer if a is not None and a.ok else None,
               "found": a.parsed.found if a is not None and a.ok else None,
               "context_has_answer": 0.0 if unanswerable else context_has_answer.get((r["question_id"], r["role"]),
                                                                                     np.nan)}
        if j is not None and j.ok:
            g = j.parsed
            row.update(correctness=g.correctness, grounded=float(g.grounded), declined=float(g.declined),
                       judge_reason=g.reason)
            row["correct"] = float(g.declined) if unanswerable else float(g.correctness == 2)
            row["score"] = float(g.declined) if unanswerable else g.correctness / 2
            row["hallucinated"] = float(not g.grounded)
            has = row["context_has_answer"]
            # a decline is right when the context holds no answer; answering is right when it does
            row["declined_ok"] = np.nan if np.isnan(has) else float(bool(g.declined) == (has == 0))
        else:
            row.update(correctness=np.nan, grounded=np.nan, declined=np.nan, judge_reason=None,
                       correct=np.nan, score=np.nan, hallucinated=np.nan, declined_ok=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(answers: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """One row per question: each metric on both indexes, and the loss on the
    stale index (refreshed minus stale; positive = the stale index is worse)."""
    metrics = ["correct", "score", "hallucinated", "declined_ok", "context_has_answer"]
    wide = answers.pivot(index="question_id", columns="role", values=metrics)
    wide.columns = [f"{role}_{m}" for m, role in wide.columns]
    wide = wide.reset_index()
    for m in ("correct", "score"):
        wide[f"loss_{m}"] = wide[f"refreshed_{m}"] - wide[f"stale_{m}"]
    keep = ["question_id", "period", "category", "set", "type_group", "sample_pool"]
    return questions[[c for c in keep if c in questions.columns]].merge(wide, on="question_id")


def classify(material: bool, flagged: bool) -> str:
    """The four classes of D13 / the brief's "which drift matters"."""
    if flagged:
        return DRIFT_HURTS if material else DRIFT_HARMLESS
    return HARM_WITHOUT_DRIFT if material else HEALTHY


def diagnose(material: bool, refreshed_correct: float, refreshed_retrieval: float,
             cfg: Mapping[str, Any]) -> str:
    """Likely cause (D13), from where quality is lost.

    Harm that the refreshed index recovers means the content was missing:
    re-index. If quality is poor on the refreshed index too, the cause is
    after the index: the answer step when retrieval finds answers, else
    retrieval or the embedding model.
    """
    imp = cfg["impact"]
    recovered = refreshed_correct >= float(imp["recovered_correct"])
    if material and recovered:
        return "missing content (re-index)"
    if recovered:
        return ""
    if not np.isnan(refreshed_retrieval) and refreshed_retrieval >= float(imp["retrieval_ok"]):
        return "answer step (articles found, answers wrong)"
    return "retrieval or embedding model (poor on the refreshed index too)"


def segments(pq: pd.DataFrame, baseline: str, flagged: Sequence[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per (period, section) of answerable questions: correct rates, the loss
    with its interval, material / borderline, class and likely cause.
    Baseline segments are the historical control (should barely move)."""
    imp, seed = cfg["impact"], int(cfg["seed"])
    threshold, min_n = float(imp["min_answer_quality_drop"]), int(imp["min_segment_questions"])
    resamples = int(imp["bootstrap_resamples"])
    pq = pq[(pq["set"] == "answerable") & pq["loss_correct"].notna()]
    rows = []
    for (period, cat), g in pq.groupby(["period", "category"]):
        seg = f"{period}|{cat}|vs {baseline}"
        loss, lo, hi = bootstrap_mean_ci(g["loss_correct"].to_numpy(float), resamples, seed=seed)
        row = {"segment_id": seg, "period": period, "category": cat, "questions": len(g),
               "stale_correct": g["stale_correct"].mean(), "refreshed_correct": g["refreshed_correct"].mean(),
               "loss_correct": loss, "loss_low": lo, "loss_high": hi,
               "stale_hallucinated": g["stale_hallucinated"].mean(),
               "refreshed_retrieval": g["refreshed_context_has_answer"].mean(),
               "flagged": seg in flagged, "baseline": period == baseline}
        enough = len(g) >= min_n
        row["material"] = bool(enough and loss >= threshold and lo > 0)
        row["borderline"] = bool(enough and loss >= threshold and lo <= 0)
        row["class"] = ("historical control" if row["baseline"] else
                        "too few questions" if not enough else
                        BORDERLINE if row["borderline"] else classify(row["material"], row["flagged"]))
        row["cause"] = diagnose(row["material"], row["refreshed_correct"], row["refreshed_retrieval"], cfg) \
            if enough else ""
        rows.append(row)
    return pd.DataFrame(rows)


def by_group(pq: pd.DataFrame, by: str | Sequence[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Means and the stale-index loss with its interval, per group."""
    resamples, seed = int(cfg["impact"]["bootstrap_resamples"]), int(cfg["seed"])
    rows = []
    for key, g in pq.groupby(by, dropna=False):
        row = dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)))
        row["questions"] = len(g)
        for c in ("stale_correct", "refreshed_correct", "stale_hallucinated", "refreshed_hallucinated",
                  "stale_declined_ok", "refreshed_declined_ok"):
            row[c] = g[c].mean()
        row["loss_correct"], row["loss_low"], row["loss_high"] = bootstrap_mean_ci(
            g["loss_correct"].to_numpy(float), resamples, seed=seed)
        rows.append(row)
    return pd.DataFrame(rows)


def drift_quality_correlation(seg: pd.DataFrame, drift_segments: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each drift measure and the answer-quality
    loss, across the classified (non-baseline) segments."""
    m = seg[~seg["baseline"] & (seg["class"] != "too few questions")].merge(drift_segments, on="segment_id")
    rows = []
    for metric in DRIFT_METRICS:
        if metric not in m.columns:
            continue
        x, y = m[metric].to_numpy(float), m["loss_correct"].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            rows.append({"measure": metric, "segments": int(ok.sum()), "spearman": np.nan, "p_value": np.nan})
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"measure": metric, "segments": int(ok.sum()), "spearman": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)


CONFIRMED = "harm confirmed with more context"
JUST_OUTSIDE = "answers just outside the top k (update retrieval configuration)"
UNCLEAR = "still unclear (escalate to a human)"


def evidence_conclusion(stale_gain: float, material_deep: bool, threshold: float) -> str:
    """What the evidence loop shows for a borderline segment.

    ``stale_gain``: how much the stale index's correct rate rose when the
    answer step read the top 10 instead of the top 5. ``material_deep``:
    whether the loss is material with the top 10.
    """
    if material_deep:
        return CONFIRMED
    if stale_gain >= threshold:
        return JUST_OUTSIDE
    return UNCLEAR
