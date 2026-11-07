"""LangGraph wiring for the five agents.

Flow: A Data Monitoring -> B Drift Detection -> C Retrieval Evaluation
-> D RAG Impact Assessment -> E Recommendation, over a typed shared state
(``schemas.PipelineState``).

Built so far: A -> B -> routing decision -> C. Both routes run C (the
regression check on a small question set); D and E are not built yet, so the
run ends after C. The decision is recorded in the state (``next_step``).

Conditional edges, decided in Python:
    - after B: no flagged drift -> C runs a small regression check only,
      then straight to E (still catches quality loss without drift)
    - after D: a borderline segment -> back to C once for more queries
      (bounded by ``impact.max_evidence_loops``)
    - after E: quality still poor on the refreshed index -> back to C once
      to evaluate the alternative embedding model (D7)
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from driftmon.agents.data_monitoring import data_monitoring
from driftmon.agents.drift_detection import drift_detection
from driftmon.agents.retrieval_eval import retrieval_evaluation
from driftmon.schemas import PipelineState

FULL_EVALUATION = "full_evaluation"
REGRESSION_CHECK = "regression_check"


def route_after_drift(state: PipelineState) -> str:
    """Full retrieval and impact evaluation if any segment was flagged, else a
    small regression check (which still catches quality loss without drift)."""
    return FULL_EVALUATION if state["drift"].any_flagged else REGRESSION_CHECK


def _record_route(state: PipelineState) -> dict[str, Any]:
    return {"next_step": route_after_drift(state)}


def build_graph():
    """Compile the agent graph built so far (A -> B -> routing decision -> C)."""
    g = StateGraph(PipelineState)
    g.add_node("data_monitoring", data_monitoring)
    g.add_node("drift_detection", drift_detection)
    g.add_node("route", _record_route)
    g.add_node("retrieval_evaluation", retrieval_evaluation)
    g.add_edge(START, "data_monitoring")
    g.add_edge("data_monitoring", "drift_detection")
    g.add_edge("drift_detection", "route")
    # both routes run C; C reads next_step to choose the full or the small question set
    g.add_conditional_edges("route", route_after_drift, {FULL_EVALUATION: "retrieval_evaluation",
                                                         REGRESSION_CHECK: "retrieval_evaluation"})
    g.add_edge("retrieval_evaluation", END)  # D and E are not built yet
    return g.compile()
