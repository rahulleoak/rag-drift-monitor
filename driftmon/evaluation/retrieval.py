"""Retrieval evaluation, stale vs refreshed (Agent C; D2, D7, D25).

Three comparisons, each answering a different question:
    (a) Did not re-indexing hurt? Each later period's questions on the stale
        index (baseline only) and on that period's refreshed index. The LLM
        judge grades the pooled top-k of both (0/1/2), so an older article
        that still answers the question gets credit. Metrics: precision@k,
        answer@k (the context contains an answer), nDCG@k against the pooled
        ideal; the paired difference with a bootstrap interval.
        Baseline questions are judged on the stale index vs the latest
        refreshed index: the historical control, which should barely change.
    (b) Does the retriever cope with drifted content? On the same refreshed
        index, each period's questions vs the baseline's: does each question
        find its source article (MRR, Recall@k)? Same index, so index size is
        not the explanation. Weighted down: questions are written from their
        source, which makes it easy to find (F11).
    (c) Do old questions get crowded out? Baseline questions on the stale vs
        each refreshed index: top-k overlap, share of the top-k now from
        newer periods, and whether the source article fell out of the top-k.
        No LLM needed.

Statistics and metrics are plain code; the LLM only grades relevance.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Any

import numpy as np
import pandas as pd

from driftmon.embeddings import document_text
from driftmon.evaluation import retrieval_metrics as rm
from driftmon.evaluation.stats import bootstrap_mean_ci
from driftmon.index import StudyIndexes
from driftmon.llm import LLMClient, LLMRequest
from driftmon.prompts import (RELEVANCE_SYSTEM, RELEVANCE_TEMPLATE, RELEVANCE_VERSION,
                              RelevanceJudgement)


def drift_status(period: str, category: str, baseline: str, flagged: Sequence[str],
                 controls: Sequence[str]) -> str:
    """'flagged', 'control' or 'other', from Agent B's segment lists."""
    seg = f"{period}|{category}|vs {baseline}"
    if seg in flagged:
        return "flagged"
    if seg in controls:
        return "control"
    if period == baseline:
        return "baseline"
    return "other"


def retrieve_all(questions: pd.DataFrame, qvecs: np.ndarray, indexes: StudyIndexes, baseline: str,
                 k: int) -> pd.DataFrame:
    """Top-k for every (question, index) the three comparisons need.

    Later-period questions: the stale index and their period's refreshed
    index. Baseline questions: the stale index and every refreshed index.
    Returns one row per run: question_id, index, doc_ids, scores.
    """
    period = questions["period"].to_numpy()
    plan: dict[str, list[int]] = {"stale": list(range(len(questions)))}
    for p in indexes.refreshed:
        plan[f"refreshed_{p}"] = [i for i in range(len(questions)) if period[i] in (p, baseline)]
    rows = []
    for name, idx in plan.items():
        if not idx:
            continue
        index = indexes.stale if name == "stale" else indexes.refreshed[name[len("refreshed_"):]]
        ids, scores = index.search(qvecs[idx], k)
        for i, d, s in zip(idx, ids, scores):
            rows.append({"question_id": questions["question_id"].iloc[i], "index": name,
                         "doc_ids": list(d), "scores": [float(x) for x in s]})
    return pd.DataFrame(rows)


def judged_pair(period: str, baseline: str, later: Sequence[str]) -> tuple[str, str]:
    """The two indexes compared with the judge for a question from ``period``."""
    return "stale", f"refreshed_{later[-1] if period == baseline else period}"


def _shuffle_seed(question_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()[:8], 16)


def relevance_requests(questions: pd.DataFrame, runs: pd.DataFrame, docs: pd.DataFrame,
                       cfg: Mapping[str, Any]) -> tuple[list[LLMRequest], list[dict[str, str]]]:
    """One judge request per question: the pooled top-``judge_k`` of its two
    indexes, de-duplicated, shuffled, without saying which index found what.

    Returns the requests and, per request, the letter -> doc_id mapping.
    """
    base, later = _periods(cfg)
    k = int(cfg["retrieval"]["judge_k"])
    job = cfg["llm"]["jobs"]["relevance"]
    by_run = runs.set_index(["question_id", "index"])["doc_ids"]
    text = docs.set_index("doc_id")
    reqs, letters = [], []
    for q in questions.to_dict("records"):
        a, b = judged_pair(q["period"], base, later)
        pool = list(dict.fromkeys(list(by_run[(q["question_id"], a)])[:k] + list(by_run[(q["question_id"], b)])[:k]))
        rng = np.random.default_rng(_shuffle_seed(q["question_id"], int(cfg["seed"])))
        pool = [pool[i] for i in rng.permutation(len(pool))]
        mapping = {ascii_uppercase[i]: d for i, d in enumerate(pool)}
        listing = "\n".join(
            f"[{L}] (published {text.at[d, 'date']}) {document_text(text.at[d, 'title'], text.at[d, 'text'])}"
            for L, d in mapping.items())
        reqs.append(LLMRequest(job="relevance", model=cfg["llm"]["judge_model"], system=RELEVANCE_SYSTEM,
                               prompt=RELEVANCE_TEMPLATE.format(question=q["question"],
                                                                reference_answer=q["reference_answer"],
                                                                articles=listing),
                               schema=RelevanceJudgement, prompt_version=RELEVANCE_VERSION,
                               effort=job["effort"], max_tokens=int(job["max_tokens"])))
        letters.append(mapping)
    return reqs, letters


def collect_grades(questions: pd.DataFrame, results: Sequence[Any], letters: Sequence[Mapping[str, str]]
                   ) -> tuple[pd.DataFrame, dict[str, int]]:
    """(question_id, doc_id, grade) rows from the judge, plus counts of problems.

    A failed request or an article the judge skipped gets no grade (NaN), so
    it is visible rather than silently scored 0.
    """
    rows, problems = [], {"failed_requests": 0, "missing_grades": 0, "unknown_letters": 0}
    for qid, res, mapping in zip(questions["question_id"], results, letters):
        graded: dict[str, int] = {}
        if res.ok:
            for g in res.parsed.grades:
                if g.article in mapping and g.article not in graded:
                    graded[g.article] = int(g.grade)
                elif g.article not in mapping:
                    problems["unknown_letters"] += 1
        else:
            problems["failed_requests"] += 1
        for letter, doc in mapping.items():
            if res.ok and letter not in graded:
                problems["missing_grades"] += 1
            rows.append({"question_id": qid, "doc_id": doc, "grade": graded.get(letter, np.nan)})
    return pd.DataFrame(rows, columns=["question_id", "doc_id", "grade"]), problems


def judged_metrics(questions: pd.DataFrame, runs: pd.DataFrame, grades: pd.DataFrame,
                   cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per question: precision@k, answer@k and nDCG@k on both judged indexes.

    Questions with any ungraded article in their top-k are left out of the
    judged metrics (marked ``judged=False``) rather than scored as if 0.
    """
    base, later = _periods(cfg)
    k = int(cfg["retrieval"]["judge_k"])
    g = grades.set_index(["question_id", "doc_id"])["grade"]
    by_run = runs.set_index(["question_id", "index"])["doc_ids"]
    out = []
    for q in questions.to_dict("records"):
        qid = q["question_id"]
        a, b = judged_pair(q["period"], base, later)
        tops = {name: list(by_run[(qid, name)])[:k] for name in (a, b)}
        pool = {d: g.get((qid, d), np.nan) for d in tops[a] + tops[b]}
        row = {"question_id": qid, "stale_index": a, "refreshed_index": b,
               "judged": bool(pool) and not any(np.isnan(v) for v in pool.values())}
        for side, name in (("stale", a), ("refreshed", b)):
            grades_k = [pool[d] for d in tops[name]]
            if row["judged"]:
                gi = [int(x) for x in grades_k]
                row[f"{side}_precision"] = rm.precision_at_k(gi, k)
                row[f"{side}_answer"] = rm.answer_at_k(gi, k)
                row[f"{side}_ndcg"] = rm.ndcg_at_k(gi, [int(v) for v in pool.values()], k)
            else:
                row[f"{side}_precision"] = row[f"{side}_answer"] = row[f"{side}_ndcg"] = np.nan
        for m in ("precision", "answer", "ndcg"):
            row[f"delta_{m}"] = row[f"refreshed_{m}"] - row[f"stale_{m}"]
        out.append(row)
    return pd.DataFrame(out)


def known_item(questions: pd.DataFrame, runs: pd.DataFrame, k: int) -> pd.DataFrame:
    """Rank of each question's source article in every run (0 = not in the top-k)."""
    src = questions.set_index("question_id")["doc_id"]
    out = runs[["question_id", "index"]].copy()
    rr = [rm.reciprocal_rank(d, src[q]) for q, d in zip(runs["question_id"], runs["doc_ids"])]
    out["reciprocal_rank"] = rr
    out["recall"] = [rm.recall_at_k(d, src[q], k) for q, d in zip(runs["question_id"], runs["doc_ids"])]
    out["rank"] = [int(round(1 / r)) if r else 0 for r in rr]
    return out


def ranking_change(questions: pd.DataFrame, runs: pd.DataFrame, docs: pd.DataFrame, baseline: str,
                   k: int) -> pd.DataFrame:
    """(c) Baseline questions, stale vs each refreshed index.

    Per (question, refreshed index): top-k overlap, the share of the refreshed
    top-k from periods after the baseline, and whether the source article was
    in the stale top-k but fell out of the refreshed top-k (displaced).
    """
    base_q = questions[questions["period"] == baseline].set_index("question_id")
    period_of = docs.set_index("doc_id")["period"]
    by_run = runs.set_index(["question_id", "index"])["doc_ids"]
    rows = []
    for qid, q in base_q.iterrows():
        stale = list(by_run[(qid, "stale")])[:k]
        for name in sorted({i for (x, i) in by_run.index if x == qid and i != "stale"}):
            ref = list(by_run[(qid, name)])[:k]
            newer = [d for d in ref if period_of[d] != baseline]
            rows.append({"question_id": qid, "category": q["category"], "index": name,
                         "overlap": rm.overlap_at_k(stale, ref, k), "newer_share": len(newer) / k,
                         "source_displaced": q["doc_id"] in stale and q["doc_id"] not in ref,
                         "newer_doc_ids": newer})
    return pd.DataFrame(rows)


def summarise(per_q: pd.DataFrame, value_cols: Sequence[str], by: str | Sequence[str],
              resamples: int, seed: int) -> pd.DataFrame:
    """Mean and bootstrap interval of each value column within each group."""
    rows = []
    for key, grp in per_q.groupby(by, dropna=False):
        row = dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)))
        row["questions"] = len(grp)
        for c in value_cols:
            mean, lo, hi = bootstrap_mean_ci(grp[c].to_numpy(dtype=float), resamples, seed=seed)
            row[c], row[f"{c}_low"], row[f"{c}_high"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _periods(cfg: Mapping[str, Any]) -> tuple[str, list[str]]:
    from driftmon.periods import study_periods
    ps = study_periods(cfg)
    return ps[0].name, [p.name for p in ps[1:]]


def evaluate(cfg: Mapping[str, Any], questions: pd.DataFrame, qvecs: np.ndarray, docs: pd.DataFrame,
             indexes: StudyIndexes, flagged: Sequence[str], controls: Sequence[str],
             llm: LLMClient | None, batch: bool | None = None) -> dict[str, Any]:
    """Run the three comparisons. ``llm=None`` skips the judged part (a).

    Returns a dict of tables (DataFrames) and judge counts, used to build the
    RetrievalReport.
    """
    base, later = _periods(cfg)
    k = int(cfg["retrieval"]["top_k"])
    resamples, seed = int(cfg["impact"]["bootstrap_resamples"]), int(cfg["seed"])
    # unanswerable questions have no relevant article by design; they are for Agent D (abstention)
    q = questions[questions["set"] == "answerable"].reset_index(drop=True).copy()
    qvecs = qvecs[(questions["set"] == "answerable").to_numpy()]
    q["drift_status"] = [drift_status(p, c, base, flagged, controls) for p, c in zip(q["period"], q["category"])]
    runs = retrieve_all(q, qvecs, indexes, base, k)

    # (b) known item, refreshed index: each period's questions vs the baseline's on the same index
    ki = known_item(q, runs, k).merge(q[["question_id", "period", "category", "drift_status"]],
                                      on="question_id")
    ki_ref = ki[ki["index"] != "stale"].copy()
    ki_ref["asked_by"] = np.where(ki_ref["period"] == base, "baseline questions", "period questions")
    known = summarise(ki_ref, ["reciprocal_rank", "recall"], ["index", "asked_by"], resamples, seed)
    known_by_status = summarise(ki_ref[ki_ref["asked_by"] == "period questions"],
                                ["reciprocal_rank", "recall"], ["index", "drift_status"], resamples, seed)

    # (c) ranking change for baseline questions
    change = ranking_change(q, runs, docs, base, k)
    change_summary = (summarise(change.assign(source_displaced=change["source_displaced"].astype(float)),
                                ["overlap", "newer_share", "source_displaced"], "index", resamples, seed)
                      if len(change) else pd.DataFrame())

    out: dict[str, Any] = {"questions": q, "runs": runs, "known_item": ki, "known_summary": known,
                           "known_by_status": known_by_status, "ranking_change": change,
                           "ranking_summary": change_summary, "judge": {"requests": 0}}
    if llm is None:
        return out

    # (a) judged stale vs refreshed
    reqs, letters = relevance_requests(q, runs, docs, cfg)
    before = llm.spent()
    results = llm.complete_many(reqs, batch=batch)
    grades, problems = collect_grades(q, results, letters)
    per_q = judged_metrics(q, runs, grades, cfg).merge(
        q[["question_id", "period", "category", "drift_status", "sample_pool", "type_group"]], on="question_id")
    judged = per_q[per_q["judged"]]
    cols = ["stale_ndcg", "refreshed_ndcg", "delta_ndcg", "stale_answer", "refreshed_answer", "delta_answer",
            "stale_precision", "refreshed_precision", "delta_precision"]
    out.update(grades=grades, judged=per_q,
               judged_by_period=summarise(judged, cols, "period", resamples, seed),
               judged_by_status=summarise(judged, cols, ["period", "drift_status"], resamples, seed),
               judged_by_type=summarise(judged[judged["period"] != base], cols, "type_group", resamples, seed),
               judge={"requests": len(reqs), "judged_questions": int(per_q["judged"].sum()),
                      "cost_usd": round(llm.spent() - before, 4), **problems})
    return out


def worst_queries(tables: Mapping[str, Any], docs: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """Questions that lost most on the stale index, with what each index returned."""
    if "judged" not in tables:
        return []
    per_q = tables["judged"]
    per_q = per_q[per_q["judged"]].sort_values(["delta_ndcg", "delta_answer"], ascending=False).head(n)
    q = tables["questions"].set_index("question_id")
    by_run = tables["runs"].set_index(["question_id", "index"])["doc_ids"]
    title = docs.set_index("doc_id")["title"]
    out = []
    for r in per_q.to_dict("records"):
        qid = r["question_id"]
        out.append({"question_id": qid, "question": q.at[qid, "question"], "period": r["period"],
                    "category": r["category"], "drift_status": r["drift_status"],
                    "stale_ndcg": r["stale_ndcg"], "refreshed_ndcg": r["refreshed_ndcg"],
                    "stale_top3": [title[d] for d in list(by_run[(qid, r["stale_index"])])[:3]],
                    "refreshed_top3": [title[d] for d in list(by_run[(qid, r["refreshed_index"])])[:3]]})
    return out
