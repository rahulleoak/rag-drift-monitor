"""Agent A: Data Monitoring.

Reads: per-period document batches and the query stream (from ``simulate``).
Does: fixes the baseline snapshot; tracks volumes, text lengths, category mix
(chi-square test), new-term rate and empty descriptions; marks which
period x category segments have enough data to analyse.
Writes: ``MonitoringReport``. Consumed by Agent B.
"""
