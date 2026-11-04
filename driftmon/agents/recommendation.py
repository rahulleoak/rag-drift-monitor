"""Agent E: Recommendation.

Reads: ``ImpactReport`` plus supporting evidence from ``DriftReport`` and
``RetrievalReport``.
Does: ``policy`` maps evidence patterns to actions; the LLM writes a plain-
language explanation for each; a check rejects any recommendation that does
not cite evidence.
Writes: a list of ``Recommendation``.
"""
