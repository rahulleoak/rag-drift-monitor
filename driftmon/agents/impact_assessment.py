"""Agent D: RAG Impact Assessment.

Reads: ``RetrievalReport`` (retrieved contexts and selected queries) and
``DriftReport`` (drift scores).
Does: generates answers from both indexes and has them judged; computes
answer-quality deltas with confidence intervals; relates each drift metric to
quality loss across segments; classifies each segment as drift that hurts,
drift that does not, quality loss without drift, or healthy.
Writes: ``ImpactReport``. Consumed by Agent E; may route back to Agent C once
for more evidence.
"""
