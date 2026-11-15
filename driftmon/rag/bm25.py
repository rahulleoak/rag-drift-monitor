"""Keyword search: BM25 over the same articles the meaning-based index holds (D40).

BM25 scores a document by the query words it contains: each word counts more
when it is rare in the archive (idf) and when it appears often in the
document, with diminishing returns (``k1``) and a correction for long
documents (``b``). Words are the vocabulary measure's tokens (Agent B's
``tokenize``: lower case, no stop words, no stemming), so "vaccine" and
"vaccines" are different words, as they are for the wording-drift measure.

    score(q, d) = sum over query words t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))     (never negative)

Exact, in-memory, like the FAISS index: one sparse matrix product per batch.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from driftmon.drift.lexical import tokenize


class BM25Index:
    """BM25 over a fixed list of documents; ``search`` mirrors ``VectorIndex.search``."""

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = np.asarray(doc_ids)
        toks = [tokenize(t) for t in texts]
        self.vocab: dict[str, int] = {}
        rows, cols = [], []
        for i, ts in enumerate(toks):
            for t in ts:
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
        n = len(toks)
        tf = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(self.vocab)))
        tf.sum_duplicates()
        lengths = np.array([len(ts) for ts in toks], dtype=float)
        avg = lengths.mean() if n and lengths.mean() > 0 else 1.0
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lengths / avg)                       # per document
        tf = tf.tocoo()
        w = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """One row of document scores per query (a query word counts once per time it appears)."""
        rows, cols = [], []
        for i, q in enumerate(queries):
            for t in tokenize(q):
                if t in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[t])
        qm = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries), len(self.vocab)))
        return np.asarray((qm @ self.weights.T).todense())

    def search(self, queries: Sequence[str], k: int, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` document ids and scores per query, in batches (a score row per query and
        document). Ties keep document order, so results are repeatable; a document with
        score 0 shares no word with the query."""
        k = min(k, len(self))
        ids, scores = [], []
        for start in range(0, len(queries), batch):
            s = self.scores(queries[start:start + batch])
            top = np.argsort(-s, axis=1, kind="stable")[:, :k]
            ids.append(self.doc_ids[top])
            scores.append(np.take_along_axis(s, top, axis=1))
        if not ids:
            return np.empty((0, k), dtype=self.doc_ids.dtype), np.empty((0, k))
        return np.vstack(ids), np.vstack(scores)
