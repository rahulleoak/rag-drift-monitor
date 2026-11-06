"""Agent B: Semantic Drift Detection.

Reads: ``MonitoringReport`` (eligible segments), the documents, their
embeddings and the word tokens shared by Agent A.
Does: runs the embedding and lexical techniques per whole period (against
the baseline and the previous period) and per period x category (against
the same category in the baseline); corrects for multiple tests; assigns
severity bands calibrated on ordinary change inside the baseline; flags
segments that are significant *and* at least medium severity; picks
low-severity control segments; runs both techniques on the query stream
(each period's queries vs the baseline queries). (LLM-written explanations
come later and will only describe these numbers.)
Writes: ``DriftReport``. Consumed by Agents C, D and E.
"""
from __future__ import annotations

from typing import Any

from driftmon import queries
from driftmon.drift.report import build_drift_report
from driftmon.periods import study_periods
from driftmon.schemas import PipelineState


def drift_detection(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: build the DriftReport from Agent A's report."""
    report, inp, _ = build_drift_report(state["config"], state["docs"], state["vectors"],
                                      state["monitoring"], state.get("tokens"))
    if state.get("queries") is not None:
        cfg, periods = state["config"], study_periods(state["config"])
        table = queries.query_drift_table(
            state["queries"], state["query_vectors"], periods[0].name, [p.name for p in periods[1:]],
            permutations=int(cfg["drift"]["mmd_permutations"]), seed=int(cfg["seed"]),
            alpha=float(cfg["drift"]["alpha"]), archive_vocab=inp.baseline_vocab)
        if len(table):
            report.query_drift = table.astype(object).where(table.notna(), None).to_dict("records")
    return {"drift": report}
