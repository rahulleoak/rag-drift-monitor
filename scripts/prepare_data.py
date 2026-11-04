"""Prepare the chosen dataset: clean, deduplicate, assign periods, sample.

Writes per-period parquet files to ``data/processed`` and logs a retention
table (records removed by each filter). To be written after exploration.
"""
