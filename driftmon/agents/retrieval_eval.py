"""Agent C: Retrieval Evaluation.

Reads: ``DriftReport`` (which segments to test, plus controls), the golden
evaluation set, the stale and refreshed indexes.
Does: known-item MRR and Recall@K; LLM-judged Precision@K and nDCG@K on the
pooled results of both indexes; per-segment deltas with confidence intervals;
lists the worst queries with what was retrieved.
Writes: ``RetrievalReport``. Consumed by Agent D.
"""
