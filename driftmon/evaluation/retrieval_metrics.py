"""Retrieval metrics. Pure functions, unit-tested against hand-computed cases.

Two kinds of ground truth (D2):
    - known item: the question's source article (``reciprocal_rank``,
      ``recall_at_k``)
    - graded relevance from the LLM judge over the pooled results of both
      indexes: 0 = not relevant, 1 = on topic but does not answer, 2 = answers
      the question (``precision_at_k``, ``answer_at_k``, ``ndcg_at_k``)
And ranking change between two indexes (``overlap_at_k``).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reciprocal_rank(retrieved: Sequence[str], target: str) -> float:
    """1 / rank of ``target`` in ``retrieved`` (rank 1 = first), or 0 if absent."""
    for i, doc in enumerate(retrieved, start=1):
        if doc == target:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: Sequence[str], target: str, k: int) -> float:
    """1.0 if ``target`` is in the top ``k``, else 0.0 (one relevant item)."""
    return float(target in list(retrieved)[:k])


def precision_at_k(grades: Sequence[int], k: int, min_grade: int = 1) -> float:
    """Share of the top ``k`` graded at least ``min_grade`` (missing ranks count as 0)."""
    top = list(grades)[:k]
    return sum(g >= min_grade for g in top) / k


def answer_at_k(grades: Sequence[int], k: int) -> float:
    """1.0 if any of the top ``k`` answers the question (grade 2): the context
    the answer step reads contains the answer."""
    return float(any(g >= 2 for g in list(grades)[:k]))


def dcg(grades: Sequence[float]) -> float:
    """Discounted cumulative gain with gain 2^g - 1 and log2(rank + 1) discount."""
    g = np.asarray(grades, dtype=np.float64)
    return float(((2.0 ** g - 1.0) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(grades: Sequence[int], pool_grades: Sequence[int], k: int) -> float:
    """nDCG@k of a ranking, with the ideal ranking taken from every judged
    document in the pool for this question (both indexes together).

    So the stale and refreshed rankings are scored against the same ideal.
    Returns 0 when nothing in the pool is relevant.
    """
    ideal = dcg(sorted(pool_grades, reverse=True)[:k])
    return dcg(list(grades)[:k]) / ideal if ideal > 0 else 0.0


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """Share of the top ``k`` of ``a`` that is also in the top ``k`` of ``b``."""
    return len(set(list(a)[:k]) & set(list(b)[:k])) / k
