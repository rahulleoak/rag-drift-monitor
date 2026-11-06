"""Agent C: Retrieval Evaluation.

Reads: ``DriftReport`` (which segments are flagged and which are controls),
the golden question set with its query embeddings, the documents and their
embeddings, and the routing decision after Agent B.
Does: loads the saved stale and refreshed FAISS indexes (or builds and saves
them, D22); tags each question with its segment's drift status from Agent B;
runs the three comparisons in ``driftmon.evaluation.retrieval``: (a) judged
stale vs refreshed, (b) known-item retrieval on the same refreshed index for
period vs baseline questions, (c) ranking change for baseline questions;
lists the questions that lost most on the stale index.
On the regression-check route (no drift flagged) only the random-pool
questions are evaluated: a small check that still catches quality loss
without drift.
Writes: ``RetrievalReport`` and the per-question tables. Consumed by Agent D.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from driftmon.config import resolve_path
from driftmon.evaluation import retrieval as rv
from driftmon.index import load_or_build_study_indexes
from driftmon.periods import study_periods
from driftmon.schemas import PipelineState, RetrievalReport

REGRESSION_POOLS = ("random",)


def _records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or not len(df):
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def retrieval_evaluation(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: build the RetrievalReport."""
    cfg, drift = state["config"], state["drift"]
    mode = state.get("next_step", "full_evaluation")
    questions = state.get("questions")
    if questions is None or not len(questions):
        report = RetrievalReport(mode=mode, questions=0, questions_by_status={}, index_store={},
                                 known_item=[], known_item_by_status=[], ranking_change=[],
                                 notes=["No question set yet: retrieval was not evaluated."])
        return {"retrieval": report, "retrieval_tables": {}}

    qvecs = state["question_vectors"]
    if mode == "regression_check":
        keep = questions["sample_pool"].isin(REGRESSION_POOLS).to_numpy()
        questions, qvecs = questions[keep].reset_index(drop=True), qvecs[keep]

    periods = study_periods(cfg)
    directory = resolve_path(cfg["paths"]["indexes"])
    indexes, loaded = load_or_build_study_indexes(state["docs"], state["vectors"], periods[0].name,
                                                  [p.name for p in periods[1:]], directory,
                                                  cfg["embeddings"]["model"])
    tables = rv.evaluate(cfg, questions, qvecs, state["docs"], indexes, drift.drifted_segments,
                         drift.control_segments, state.get("llm"))
    q = tables["questions"]
    notes = ["Known-item numbers are weighted down: questions are written from their source "
             "article, which makes it easy to find (F11)."]
    if state.get("llm") is None:
        notes.append("No LLM client: the judged stale-vs-refreshed comparison was skipped.")
    report = RetrievalReport(
        mode=mode, questions=len(q), questions_by_status=q["drift_status"].value_counts().to_dict(),
        index_store={"directory": str(directory), "reloaded": loaded,
                     "documents": {"stale": len(indexes.stale),
                                   **{f"refreshed_{p}": len(i) for p, i in indexes.refreshed.items()}}},
        known_item=_records(tables["known_summary"]),
        known_item_by_status=_records(tables["known_by_status"]),
        ranking_change=_records(tables["ranking_summary"]),
        judged_by_period=_records(tables.get("judged_by_period")),
        judged_by_status=_records(tables.get("judged_by_status")),
        judged_by_type=_records(tables.get("judged_by_type")),
        judge=tables["judge"], worst_queries=rv.worst_queries(tables, state["docs"]), notes=notes)
    return {"retrieval": report, "retrieval_tables": tables}
