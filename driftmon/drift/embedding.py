"""Drift technique 1: embedding space (docs/questions.md Q5).

    - MMD with an RBF kernel and a permutation test (distribution-level score
      and p-value)
    - per-document novelty: 1 - highest cosine similarity to the reference
      corpus (gives example documents, and links directly to retrieval)
    - centroid cosine shift (the simple, intuitive number)
    - dispersion: mean pairwise cosine distance, to catch narrowing or
      widening distributions (proposal P3)

All inputs are L2-normalised embedding matrices (one row per document).
Pure functions; no LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MMDResult:
    """Unbiased squared MMD between two samples, with its permutation p-value."""

    mmd2: float
    p_value: float
    n_x: int
    n_y: int
    gamma: float
    permutations: int


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between normalised rows: 2 - 2 cos."""
    return np.clip(2.0 - 2.0 * (a @ b.T), 0.0, None)


def median_heuristic_gamma(x: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """RBF kernel width from the median squared distance: gamma = 1 / median.

    Computed once on a reference sample (the baseline) and reused for every
    comparison, so MMD values are on the same scale across periods.
    """
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), size=max_n, replace=False)]
    d = _sq_dists(x, x)[np.triu_indices(len(x), k=1)]
    med = float(np.median(d))
    return 1.0 / med if med > 0 else 1.0


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, gamma: float,
                         permutations: int = 500, seed: int = 0) -> MMDResult:
    """Unbiased MMD^2 (Gretton et al. 2012) with a label-permutation p-value.

    The kernel matrix of the pooled sample is computed once; every
    permutation only re-assigns which rows belong to which group, so all
    permutations are evaluated with one matrix product.

    p-value = (1 + #permutations with MMD^2 >= observed) / (1 + permutations).
    """
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError("each sample needs at least 2 rows")
    z = np.vstack([x, y]).astype(np.float64)
    k = np.exp(-gamma * _sq_dists(z, z))
    total = k.sum()
    trace = float(np.trace(k))  # 1.0 per row for an RBF kernel

    def stats(s: np.ndarray) -> np.ndarray:
        # s: (m + n, P) indicator of membership in group x, one column per labelling
        ks = k @ s
        sum_xx = (s * ks).sum(axis=0)
        sum_xy = ((1.0 - s) * ks).sum(axis=0)
        sum_yy = total - sum_xx - 2.0 * sum_xy
        diag_x = (s * np.diag(k)[:, None]).sum(axis=0)
        diag_y = trace - diag_x
        return ((sum_xx - diag_x) / (m * (m - 1)) + (sum_yy - diag_y) / (n * (n - 1))
                - 2.0 * sum_xy / (m * n))

    observed_labels = np.zeros((m + n, 1))
    observed_labels[:m] = 1.0
    observed = float(stats(observed_labels)[0])

    rng = np.random.default_rng(seed)
    perm = np.zeros((m + n, permutations))
    for j in range(permutations):
        perm[rng.permutation(m + n)[:m], j] = 1.0
    null = stats(perm)
    p = (1 + int((null >= observed).sum())) / (1 + permutations)
    return MMDResult(observed, p, m, n, gamma, permutations)


def centroid_shift(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance between the two mean vectors (0 = same direction)."""
    cx, cy = x.mean(axis=0), y.mean(axis=0)
    return float(1.0 - cx @ cy / (np.linalg.norm(cx) * np.linalg.norm(cy)))


def dispersion(x: np.ndarray) -> float:
    """Mean pairwise cosine distance within a sample (higher = more spread out).

    Exact and O(n·d) for normalised rows: the sum of all pairwise cosines is
    |sum of rows|^2 minus the n self-pairs.
    """
    n = len(x)
    if n < 2:
        return 0.0
    s = x.sum(axis=0)
    mean_cos = (float(s @ s) - n) / (n * (n - 1))
    return 1.0 - mean_cos


def novelty(best_similarity: np.ndarray) -> np.ndarray:
    """Novelty from each document's highest cosine similarity to the reference set."""
    return 1.0 - np.asarray(best_similarity, dtype=np.float64)
