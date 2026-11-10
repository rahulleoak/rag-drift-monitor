"""Simulated failure: a re-index that silently dropped documents (D34).

A demonstration of the evidence loop on real data. In normal runs every
section's result is clear-cut, so the loop from Agent D back to Agent C never
fires. This simulates a realistic operations failure instead: an ingestion
batch that failed, so most of one section's new articles never reached the
refreshed index. Questions about that section then fail on both indexes,
the stale-vs-refreshed loss becomes small and uncertain (borderline), and the
graph sends the section back for more evidence.

Off unless ``fault_injection`` is in the config (``scripts/run_pipeline.py
--fault``). Applied in memory to the refreshed indexes only; the saved
indexes on disk are never changed.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from driftmon.rag.index import StudyIndexes, VectorIndex


def dropped_doc_ids(docs: pd.DataFrame, fault: Mapping[str, Any], seed: int) -> list[str]:
    """The articles the failed batch lost: a random ``drop_share`` of one
    period's articles in one section."""
    pool = docs.loc[(docs["period"] == fault["period"]) & (docs["category"] == fault["category"]), "doc_id"]
    n = int(round(float(fault["drop_share"]) * len(pool)))
    rng = np.random.default_rng([seed, 34])
    return sorted(rng.choice(pool.to_numpy(), size=n, replace=False).tolist()) if n else []


def without_documents(indexes: StudyIndexes, docs: pd.DataFrame, vectors: np.ndarray,
                      drop: Iterable[str]) -> StudyIndexes:
    """The same indexes with ``drop`` removed from every refreshed index (the
    stale index holds only baseline articles, so it is unaffected)."""
    drop = set(drop)
    row = {d: i for i, d in enumerate(docs["doc_id"].to_numpy())}
    refreshed = {}
    for p, idx in indexes.refreshed.items():
        keep = [d for d in idx.doc_ids if d not in drop]
        refreshed[p] = idx if len(keep) == len(idx) else VectorIndex(keep, vectors[[row[d] for d in keep]])
    return StudyIndexes(stale=indexes.stale, refreshed=refreshed)


def describe(fault: Mapping[str, Any], dropped: int, total: int) -> str:
    """One line for reports and the dashboard."""
    return (f"Simulated failure: {dropped} of {total} {fault['period']} {fault['category']} articles "
            f"({float(fault['drop_share']):.0%}) were left out of the refreshed index, as if an ingestion "
            f"batch had failed.")
