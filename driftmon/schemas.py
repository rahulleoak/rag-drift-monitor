"""Typed contracts passed between agents.

Each agent writes exactly one report into the LangGraph state, and the next
agents read it. Keeping them here makes the dependencies between agents
explicit and testable.

Planned models (pydantic):
    Record            one document: doc_id, date, category, title, text
    Segment           a period x category group with its document ids
    MonitoringReport  Agent A: snapshots, descriptive stats, eligible segments
    DriftReport       Agent B: per-segment drift scores, p-values, severity,
                      emerging terms, example documents, explanations
    RetrievalReport   Agent C: per-query and per-segment metrics for the stale
                      and refreshed indexes, deltas with CIs, failing queries
    ImpactReport      Agent D: answer-quality deltas, drift-vs-quality
                      relationships, one of four classes per segment
    Recommendation    Agent E: action, priority, evidence ids, explanation
    PipelineState     the LangGraph state holding all of the above
"""
