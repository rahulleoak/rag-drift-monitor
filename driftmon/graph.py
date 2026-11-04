"""LangGraph wiring for the five agents.

Flow: A Data Monitoring -> B Drift Detection -> C Retrieval Evaluation
-> D RAG Impact Assessment -> E Recommendation, over a typed shared state
(``schemas.PipelineState``).

Conditional edges, decided in Python:
    - after B: no significant drift -> C runs a small regression check only,
      then straight to E (still catches quality loss without drift)
    - after D: a borderline segment -> back to C once for more queries
      (bounded by ``impact.max_evidence_loops``)
"""
