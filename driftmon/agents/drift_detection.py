"""Agent B: Semantic Drift Detection.

Reads: ``MonitoringReport`` (eligible segments), the documents, their
embeddings and the word tokens shared by Agent A.
Does: runs the embedding and lexical techniques per whole period (against
the baseline and the previous period) and per period x category (against
the same category in the baseline); corrects for multiple tests; assigns
severity bands calibrated on ordinary change inside the baseline; flags
segments that are significant *and* at least medium severity; picks
low-severity control segments. (LLM-written explanations come later and
will only describe these numbers.)
Writes: ``DriftReport``. Consumed by Agents C, D and E.
"""
from __future__ import annotations

from typing import Any

from driftmon.drift.report import build_drift_report
from driftmon.schemas import PipelineState


def drift_detection(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: build the DriftReport from Agent A's report."""
    report, _, _ = build_drift_report(state["config"], state["docs"], state["vectors"],
                                      state["monitoring"], state.get("tokens"))
    return {"drift": report}
