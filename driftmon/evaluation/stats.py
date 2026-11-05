"""Statistical helpers shared by the agents. Pure functions.

Benjamini-Hochberg correction now; bootstrap confidence intervals (including
paired, for stale vs refreshed), rank correlation with confidence intervals
and within-baseline null distributions are added with the agents that use
them.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order.

    Controls the false discovery rate when many segments are tested at once:
    a segment is flagged at level alpha if its q-value is <= alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
