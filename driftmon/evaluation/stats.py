"""Statistical helpers shared by the agents. Pure functions.

Benjamini-Hochberg correction; bootstrap confidence intervals for a mean
(pass per-question differences for a paired stale-vs-refreshed comparison).
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


def bootstrap_mean_ci(values: Sequence[float], resamples: int = 2000, level: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap confidence interval (questions resampled).

    For a paired comparison (stale vs refreshed on the same questions), pass
    the per-question differences. Returns (mean, low, high); NaNs if empty.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(v.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
