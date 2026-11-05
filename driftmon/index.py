"""Vector indexes for the stale-vs-refreshed experiment.

Builds exact FAISS inner-product indexes (docs/questions.md Q13; vectors are
L2-normalised, so inner product = cosine similarity):
    - stale index: baseline-period documents only (the archive as it was
      when the assistant was built, never updated)
    - refreshed index for period P: the baseline plus every later period up
      to and including P (the archive kept up to date; D10)
Metadata (dates, categories, text) stays in the documents table, keyed by
row position / document id.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd


class VectorIndex:
    """Exact cosine-similarity search over a fixed set of documents."""

    def __init__(self, doc_ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(doc_ids) != len(vectors):
            raise ValueError("doc_ids and vectors must have the same length")
        self.doc_ids = np.asarray(doc_ids)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return self._index.ntotal

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` document ids and cosine similarities for each query row."""
        k = min(k, len(self))
        scores, rows = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return self.doc_ids[rows], scores


@dataclass
class StudyIndexes:
    """The stale index and one refreshed index per later period."""

    stale: VectorIndex
    refreshed: dict[str, VectorIndex]


def build_study_indexes(docs: pd.DataFrame, vectors: np.ndarray, baseline: str,
                        later: Sequence[str]) -> StudyIndexes:
    """Build the stale index and cumulative refreshed indexes.

    Args:
        docs: one row per document with ``doc_id`` and ``period`` columns,
            aligned with ``vectors``.
        vectors: normalised embeddings, one row per document.
        baseline: name of the baseline period.
        later: later period names in time order.
    """
    def subset(periods: set[str]) -> VectorIndex:
        mask = docs["period"].isin(periods).to_numpy()
        return VectorIndex(docs.loc[mask, "doc_id"].tolist(), vectors[mask])

    included = {baseline}
    refreshed = {}
    for name in later:
        included.add(name)
        refreshed[name] = subset(set(included))
    return StudyIndexes(stale=subset({baseline}), refreshed=refreshed)
