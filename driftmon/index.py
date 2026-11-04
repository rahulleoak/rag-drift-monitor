"""Vector indexes for the stale-vs-refreshed experiment.

Builds exact FAISS inner-product indexes (docs/questions.md Q13):
    - stale index: baseline-period documents only
    - refreshed index: baseline plus the new period's documents
Metadata (dates, categories, text) lives in a parquet side table keyed by
document id.
"""
