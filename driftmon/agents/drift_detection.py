"""Agent B: Semantic Drift Detection.

Reads: ``MonitoringReport`` (snapshots and eligible segments), embeddings.
Does: runs the embedding and lexical techniques per segment and on queries,
against the baseline (cumulative) and the previous period; corrects for
multiple tests; assigns severity from within-baseline calibration; asks the
LLM to explain each flagged segment from the numbers and examples.
Writes: ``DriftReport`` with drifted segments and low-drift control segments.
Consumed by Agents C, D and E.
"""
