"""Vector indexes for the stale-vs-refreshed experiment.

Builds exact FAISS inner-product indexes (vectors are
L2-normalised, so inner product = cosine similarity):
    - stale index: baseline-period documents only (the archive as it was
      when the assistant was built, never updated)
    - refreshed index for period P: the baseline plus every later period up
      to and including P (the archive kept up to date; D10)
Metadata (dates, categories, text) stays in the documents table, keyed by
row position / document id. Agent C saves the indexes to ``indexes/`` with a
manifest and reloads them while the documents and embeddings are unchanged
(D22).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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

    @classmethod
    def from_faiss(cls, doc_ids: Sequence[str], index: faiss.Index) -> "VectorIndex":
        """Wrap an index loaded from disk."""
        if len(doc_ids) != index.ntotal:
            raise ValueError("doc_ids and the saved index have different lengths")
        obj = cls.__new__(cls)
        obj.doc_ids, obj._index = np.asarray(doc_ids), index
        return obj

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


# ---------- saved indexes (D22) ----------

def fingerprint(doc_ids: Sequence[str], vectors: np.ndarray, model_name: str) -> str:
    """Identifies the exact documents and vectors an index was built from."""
    h = hashlib.sha256(model_name.encode())
    h.update("\n".join(map(str, doc_ids)).encode())
    h.update(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())
    return h.hexdigest()[:20]


def load_or_build_study_indexes(docs: pd.DataFrame, vectors: np.ndarray, baseline: str,
                                later: Sequence[str], directory: str | Path, model_name: str
                                ) -> tuple[StudyIndexes, bool]:
    """Load the saved indexes if they were built from these exact documents and
    vectors; otherwise build them and save them with a manifest.

    Saved as ``<directory>/stale.faiss``, ``refreshed_<period>.faiss`` and
    ``manifest.json`` (model, fingerprint, build time, and per index the
    periods and document ids it holds). Returns (indexes, loaded_from_disk).
    """
    directory = Path(directory)
    fp = fingerprint(docs["doc_id"].tolist(), vectors, model_name)
    manifest_path = directory / "manifest.json"
    names = {"stale": [baseline]}
    included = [baseline]
    for p in later:
        included = [*included, p]
        names[f"refreshed_{p}"] = included
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("fingerprint") == fp and set(m["indexes"]) == set(names) and all(
                m["indexes"][n]["periods"] == names[n] for n in names):
            loaded = {n: VectorIndex.from_faiss(m["indexes"][n]["doc_ids"],
                                                faiss.read_index(str(directory / f"{n}.faiss")))
                      for n in names}
            return StudyIndexes(stale=loaded["stale"],
                                refreshed={p: loaded[f"refreshed_{p}"] for p in later}), True
    built = build_study_indexes(docs, vectors, baseline, later)
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for n, idx in [("stale", built.stale), *[(f"refreshed_{p}", built.refreshed[p]) for p in later]]:
        faiss.write_index(idx._index, str(directory / f"{n}.faiss"))
        entries[n] = {"periods": names[n], "documents": len(idx), "doc_ids": idx.doc_ids.tolist()}
    manifest_path.write_text(json.dumps({
        "fingerprint": fp, "embedding_model": model_name, "index_type": "faiss IndexFlatIP (exact cosine)",
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "indexes": entries}),
        encoding="utf-8")
    return built, False
