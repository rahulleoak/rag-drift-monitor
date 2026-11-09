"""Cleaning rules for the document collection (decision D8, finding F3).

Pure functions over a pandas DataFrame with one row per document and the
columns ``doc_id, date, period, category, title, text``. Each rule returns a
boolean mask of rows to *drop*, so the retention table can count what every
rule removed. Rules run in a fixed order and each sees only the rows that
survived the earlier ones.

The thresholds are first versions chosen from the data (F3) and are to be
revisited as later findings come in.
"""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd


def word_counts(df: pd.DataFrame) -> pd.Series:
    """Words in title + description, split on whitespace."""
    return (df["title"].str.split().str.len().fillna(0)
            + df["text"].str.split().str.len().fillna(0)).astype(int)


def empty_description(df: pd.DataFrame) -> pd.Series:
    """Rows with no description (headline only)."""
    return df["text"].str.strip() == ""


def too_short(df: pd.DataFrame, min_words: int) -> pd.Series:
    """Rows with fewer than ``min_words`` words in title + description."""
    return word_counts(df) < min_words


def too_long(df: pd.DataFrame, max_words: int) -> pd.Series:
    """Rows with more than ``max_words`` words in title + description."""
    return word_counts(df) > max_words


def near_duplicates(vectors: np.ndarray, threshold: float, chunk_size: int = 2048) -> np.ndarray:
    """Mark rows that are near-duplicates of an earlier kept row.

    Rows are visited in the given order (callers sort by date, so the
    earliest copy is kept). A row is dropped if its cosine similarity to any
    row already kept is at least ``threshold``. This is exact, not
    approximate: each chunk is compared with every kept row so far, then rows
    inside the chunk are compared with each other in order.

    Args:
        vectors: L2-normalised embeddings, one row per document, in visit order.
        threshold: cosine similarity at or above which a row counts as a copy.
        chunk_size: rows compared at once (memory vs speed; does not change
            the result).

    Returns:
        Boolean array, True for rows to drop.
    """
    n = len(vectors)
    drop = np.zeros(n, dtype=bool)
    if n == 0:
        return drop
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    kept = faiss.IndexFlatIP(vectors.shape[1])
    for start in range(0, n, chunk_size):
        chunk = vectors[start:start + chunk_size]
        if kept.ntotal:
            best, _ = kept.search(chunk, 1)
            drop[start:start + len(chunk)] = best[:, 0] >= threshold
        sims = chunk @ chunk.T
        kept_in_chunk: list[int] = []
        for i in range(len(chunk)):
            if drop[start + i]:
                continue
            if kept_in_chunk and sims[i, kept_in_chunk].max() >= threshold:
                drop[start + i] = True
                continue
            kept_in_chunk.append(i)
        kept.add(chunk[kept_in_chunk])
    return drop


def equal_samples(periods: pd.Series, n: int, seed: int) -> pd.Series:
    """Choose ``n`` rows at random from every period (True = in the sample).

    Equal sample sizes keep drift statistics comparable across periods
    (D6). Raises ``ValueError`` if a period has fewer than ``n`` rows,
    rather than silently using a smaller sample.
    """
    chosen = pd.Series(False, index=periods.index)
    rng = np.random.default_rng(seed)
    for name in sorted(periods.dropna().unique()):
        idx = periods.index[periods == name]
        if len(idx) < n:
            raise ValueError(f"period {name!r} has {len(idx)} documents, fewer than "
                             f"sample_per_period={n}")
        chosen.loc[rng.choice(idx, size=n, replace=False)] = True
    return chosen
