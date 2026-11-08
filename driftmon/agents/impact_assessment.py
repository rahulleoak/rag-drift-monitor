"""Agent D: RAG Impact Assessment (D13, D28).

Reads: Agent C's retrieval (the top k each index returned per question, and
its relevance grades: did the top k hold an answer?) and Agent B's
DriftReport (which segments are flagged, and their drift scores).
Does: the RAG pipeline answers every test question from the stale and the
refreshed index; the judge grades each answer on its own; code computes the
loss in correct answers on the stale index per segment with an interval,
classes each segment (drift that hurts / drift that does not / harm without
drift / healthy) with a likely cause, and correlates drift scores with the
loss. Unanswerable questions test whether the system declines rather than
inventing an answer.
Writes: ``ImpactReport`` and the per-answer and per-segment tables.
Consumed by Agent E.

The loop (D28): a borderline segment (loss at least the threshold, but its
interval includes zero) is sent back to Agent C once, which grades ranks
6-10; D then answers those questions from the top 10. The result, in
``ImpactReport.evidence``, says whether the harm is confirmed, whether the
answers sat just outside the top 5, or whether a human should look.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from driftmon.evaluation import answers as an
from driftmon.evaluation import retrieval as rv
from driftmon.llm import LLMResult
from driftmon.periods import study_periods
from driftmon.schemas import ImpactReport, PipelineState

ESTIMATE_ANSWER = ("The archive says the measure passed with broad support after a long debate, "
                   "according to the article published that week.")   # typical length, for cost estimates


def _records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or not len(df):
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def pilot_sample(answerable: pd.DataFrame, unanswerable: pd.DataFrame, n: int, seed: int
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small sample for a pilot: about one in ten unanswerable, the rest
    answerable spread over the periods."""
    rng = np.random.default_rng(seed)
    n_un = min(len(unanswerable), max(1, n // 10)) if len(unanswerable) else 0
    n_ans = n - n_un
    per = max(1, n_ans // answerable["period"].nunique())
    picks = [g.iloc[rng.permutation(len(g))[:per]] for _, g in answerable.groupby("period")]
    ans = pd.concat(picks).head(n_ans)
    return ans.reset_index(drop=True), unanswerable.iloc[rng.permutation(len(unanswerable))[:n_un]]


def _run(llm: Any, requests: list[Any], batch: bool | None) -> list[LLMResult | None]:
    """Run the non-empty requests; keep None where there was no request."""
    todo = [r for r in requests if r is not None]
    results = iter(llm.complete_many(todo, batch=batch) if todo else [])
    return [next(results) if r is not None else None for r in requests]


def impact_assessment(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: build the ImpactReport."""
    cfg, drift, llm = state["config"], state["drift"], state.get("llm")
    tables = state.get("retrieval_tables") or {}
    threshold = float(cfg["impact"]["min_answer_quality_drop"])
    if llm is None or "judged" not in tables:
        return {"impact": ImpactReport(questions=0, threshold=threshold,
                                       notes=["No LLM client or no judged retrieval: answers were not assessed."]),
                "impact_tables": {}}

    if state.get("deepen") and "deep" in tables:
        return _evidence_round(state)
    periods = study_periods(cfg)
    base, later = periods[0].name, [p.name for p in periods[1:]]
    judged = tables["judged"]
    answerable = tables["questions"].merge(judged[["question_id", "stale_index", "refreshed_index"]],
                                           on="question_id")
    allq = state["questions"]
    unanswerable = allq[allq["set"] == "unanswerable"].reset_index(drop=True).copy()
    unanswerable["drift_status"] = [rv.drift_status(p, c, base, drift.drifted_segments, drift.control_segments)
                                    for p, c in zip(unanswerable["period"], unanswerable["category"])]
    sample, batch = "all", None          # None: the config decides (Batch API)
    n = state.get("impact_sample")
    if n:
        answerable, unanswerable = pilot_sample(answerable, unanswerable, int(n), int(cfg["seed"]))
        sample = f"pilot: {len(answerable)} answerable + {len(unanswerable)} unanswerable"
        batch = False                    # a pilot is small: live calls, results in minutes
    pairs = {r["question_id"]: (r["stale_index"], r["refreshed_index"]) for r in answerable.to_dict("records")}
    pairs.update({q: rv.judged_pair(p, base, later) for q, p in zip(unanswerable["question_id"],
                                                                      unanswerable["period"])})
    questions = pd.concat([answerable, unanswerable], ignore_index=True)
    runs = pd.concat([tables["runs"], tables.get("runs_unanswerable", pd.DataFrame())], ignore_index=True)
    plan = an.answer_plan(questions, runs, pairs, int(cfg["impact"]["answer_k"]))

    before = llm.spent()
    answers = llm.complete_many(an.answer_requests(plan, questions, state["docs"], cfg), batch=batch)
    if getattr(llm, "estimate_only", False):   # price the judge as if every answer existed
        answers = [a if a.ok else _placeholder(a) for a in answers]
    judgements = _run(llm, an.judge_requests(plan, answers, questions, state["docs"], cfg), batch)

    has = {}
    for r in judged.to_dict("records"):
        has[(r["question_id"], "stale")] = r["stale_answer"]
        has[(r["question_id"], "refreshed")] = r["refreshed_answer"]
    per_answer = an.per_answer(plan, answers, judgements, questions, has)
    pq = an.paired(per_answer, questions)
    pq = pq.merge(questions[["question_id", "drift_status"]], on="question_id")
    ans_pq = pq[pq["set"] == "answerable"]
    seg = an.segments(pq, base, drift.drifted_segments, cfg)
    drift_segs = pd.DataFrame([s.model_dump() for s in drift.segments])
    corr = an.drift_quality_correlation(seg, drift_segs) if len(seg) and len(drift_segs) else pd.DataFrame()
    un = pq[pq["set"] == "unanswerable"]
    unanswerable_summary = [{"index": role, "questions": len(un), "declined": un[f"{role}_correct"].mean(),
                             "hallucinated": un[f"{role}_hallucinated"].mean()} for role in an.ROLES] if len(un) else []

    failed_answers = sum(1 for a in answers if not a.ok)
    failed_judge = sum(1 for j in judgements if j is not None and not j.ok)
    notes = []
    if failed_answers or failed_judge:
        notes.append(f"{failed_answers} answers and {failed_judge} judgements failed; they are left out, "
                     "not scored as wrong.")
    report = ImpactReport(
        questions=len(questions), sample=sample, threshold=threshold,
        by_period=_records(an.by_group(ans_pq, "period", cfg)),
        by_status=_records(an.by_group(ans_pq, "drift_status", cfg)),
        by_type=_records(an.by_group(ans_pq[ans_pq["period"] != base], "type_group", cfg)),
        unanswerable=unanswerable_summary, segments=_records(seg),
        classes=seg["class"].value_counts().to_dict() if len(seg) else {},
        borderline=seg.loc[seg["borderline"], "segment_id"].tolist() if len(seg) else [],
        correlations=_records(corr), examples=_examples(per_answer, questions),
        judge={"answer_requests": len(plan), "judge_requests": sum(j is not None for j in judgements),
               "failed_answers": failed_answers, "failed_judgements": failed_judge,
               "cost_usd": round(llm.spent() - before, 4)},
        notes=notes)
    return {"impact": report, "impact_tables": {"answers": per_answer, "paired": pq, "segments": seg,
                                                "correlations": corr}}


def _evidence_round(state: PipelineState) -> dict[str, Any]:
    """Answer the borderline segments' questions from the top 10 (Agent C
    has graded ranks 6-10) and conclude per segment, in code."""
    cfg, tables, llm, drift = state["config"], state["retrieval_tables"], state["llm"], state["drift"]
    base = study_periods(cfg)[0].name
    k10, threshold = int(cfg["retrieval"]["top_k"]), float(cfg["impact"]["min_answer_quality_drop"])
    deep = tables["deep"]
    q = tables["deep_questions"].merge(tables["judged"][["question_id", "stale_index", "refreshed_index"]],
                                       on="question_id")
    pairs = {r["question_id"]: (r["stale_index"], r["refreshed_index"]) for r in q.to_dict("records")}
    plan = an.answer_plan(q, tables["runs"], pairs, k10)
    before = llm.spent()
    answers = llm.complete_many(an.answer_requests(plan, q, state["docs"], cfg))
    judgements = _run(llm, an.judge_requests(plan, answers, q, state["docs"], cfg), None)
    has = {}
    for r in deep["answer_deep"].to_dict("records"):
        has[(r["question_id"], "stale")] = r[f"stale_answer_at_{k10}"]
        has[(r["question_id"], "refreshed")] = r[f"refreshed_answer_at_{k10}"]
    per_answer = an.per_answer(plan, answers, judgements, q, has)
    seg10 = an.segments(an.paired(per_answer, q), base, drift.drifted_segments, cfg).set_index("segment_id")
    seg5 = state["impact_tables"]["segments"].set_index("segment_id")
    rows = []
    for sid in state["deepen"]:
        if sid not in seg10.index:
            continue
        s5, s10 = seg5.loc[sid], seg10.loc[sid]
        gain = float(s10["stale_correct"] - s5["stale_correct"])
        rows.append({"segment_id": sid, "questions": int(s10["questions"]),
                     "loss_top5": float(s5["loss_correct"]), "loss_top10": float(s10["loss_correct"]),
                     "loss_top10_low": float(s10["loss_low"]), "loss_top10_high": float(s10["loss_high"]),
                     "stale_correct_top5": float(s5["stale_correct"]), "stale_correct_top10": float(s10["stale_correct"]),
                     "conclusion": an.evidence_conclusion(gain, bool(s10["material"]), threshold)})
    report = state["impact"]
    judge = {**report.judge, "evidence_cost_usd": round(llm.spent() - before, 4),
             "evidence_relevance": deep["judge"]}
    report = report.model_copy(update={"evidence": rows, "judge": judge})
    return {"impact": report, "deepen": [],
            "impact_tables": {**state["impact_tables"], "evidence": pd.DataFrame(rows),
                              "evidence_answers": per_answer}}


def _placeholder(a: LLMResult) -> LLMResult:
    from driftmon.prompts import RagAnswer
    return LLMResult(key=a.key, parsed=RagAnswer(found=True, answer=ESTIMATE_ANSWER, sources=["A"]), text="",
                     stop_reason="", model=a.model, input_tokens=0, output_tokens=0, cost_usd=0.0)


def _examples(per_answer: pd.DataFrame, questions: pd.DataFrame, n: int = 8) -> list[dict[str, Any]]:
    """Questions answered correctly from the refreshed index but not the stale
    one, invented stale answers first: concrete evidence of the harm."""
    w = per_answer.pivot(index="question_id", columns="role", values=["answer", "correct", "hallucinated"])
    w.columns = [f"{r}_{m}" for m, r in w.columns]
    w = w.reset_index()
    lost = w[(w["refreshed_correct"] == 1) & (w["stale_correct"] == 0)].sort_values(
        "stale_hallucinated", ascending=False).head(n)
    q = questions.set_index("question_id")
    return [{"question_id": r["question_id"], "question": q.at[r["question_id"], "question"],
             "period": q.at[r["question_id"], "period"], "category": q.at[r["question_id"], "category"],
             "stale_answer": r["stale_answer"], "stale_invented": bool(r["stale_hallucinated"] == 1),
             "refreshed_answer": r["refreshed_answer"]} for r in lost.to_dict("records")]
