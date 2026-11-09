"""Severity bands calibrated on ordinary change inside the baseline (P7, D11).

"Ordinary change" = the same calendar month one year apart inside the
two-year baseline (January 2016 vs January 2017, ...). Each window pair is
compared several times with fresh random samples of the size being judged,
so the null distribution holds both news-cycle variation and sampling
noise. A drift value above the null's 95th percentile is "medium", above
its 99th "high", otherwise "low".

When a category has too few articles per month, the windows widen to
quarters, and the sample size shrinks to what the windows hold; the
returned ``window`` and ``n`` say which was used.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftmon.drift.compare import DriftInputs, compare

BANDS = ("low", "medium", "high")
MIN_PAIRS = 4  # need at least this many window pairs for a usable null


@dataclass
class NullDistribution:
    """Ordinary-change values of each drift measure for one sample size."""

    values: pd.DataFrame  # columns: mmd2, jsd_excess; one row per (window pair, draw)
    window: str           # "month" or "quarter"
    n: int                # sample size per side used for the null
    pairs: int            # number of window pairs

    def thresholds(self, percentiles: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Band thresholds per measure, e.g. {"mmd2": {"medium": ..., "high": ...}}."""
        return {m: {b: float(np.percentile(self.values[m], q)) for b, q in percentiles.items()}
                for m in self.values.columns}

    def typical(self, measure: str) -> float:
        """Median ordinary change: the denominator for "× ordinary change"."""
        return float(np.median(self.values[measure]))


def _window_pairs(inp: DriftInputs, category: str | None, n: int, window: str
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rows of matching windows in the first and second half of the baseline."""
    first, second = inp.baseline.halves()
    rows = inp.rows(inp.baseline.name, category)
    dates = pd.to_datetime(inp.docs["date"].iloc[rows])
    size = 1 if window == "month" else 3
    pairs = []
    for half_idx, half in enumerate((first, second)):
        months = (dates.dt.year - half.start.year) * 12 + dates.dt.month - half.start.month
        in_half = dates.dt.date.map(half.contains).to_numpy()
        keys = (months // size).to_numpy()
        groups = {k: rows[in_half & (keys == k)] for k in np.unique(keys[in_half])}
        pairs.append(groups)
    return [(pairs[0][k], pairs[1][k]) for k in sorted(pairs[0].keys() & pairs[1].keys())
            if len(pairs[0][k]) >= n and len(pairs[1][k]) >= n]


def build_null(inp: DriftInputs, n: int, category: str | None = None, draws: int = 5,
               permutations: int = 50, min_n: int = 20) -> NullDistribution | None:
    """Null distribution of drift values for samples of size ``n``.

    Tries month pairs first, then quarter pairs; if neither has
    ``MIN_PAIRS`` pairs with ``n`` articles per side, ``n`` is reduced to the
    largest size quarters support (not below ``min_n``). Returns None if no
    usable null exists.
    """
    for window in ("month", "quarter"):
        pairs = _window_pairs(inp, category, n, window)
        if len(pairs) >= MIN_PAIRS:
            break
    else:
        # shrink n to what quarter windows can support
        all_q = _window_pairs(inp, category, 1, "quarter")
        sizes = sorted((min(len(a), len(b)) for a, b in all_q), reverse=True)
        if len(sizes) < MIN_PAIRS or sizes[MIN_PAIRS - 1] < min_n:
            return None
        n, window = sizes[MIN_PAIRS - 1], "quarter"
        pairs = _window_pairs(inp, category, n, window)

    values = []
    for p_idx, (a, b) in enumerate(pairs):
        for d in range(draws):
            rng = np.random.default_rng([inp.seed, p_idx, d, n])
            r = compare(inp, rng.choice(a, n, replace=False), rng.choice(b, n, replace=False),
                        permutations=permutations, seed=inp.seed + d)
            values.append({"mmd2": r["mmd2"], "jsd_excess": r["jsd_excess"]})
    return NullDistribution(pd.DataFrame(values), window, n, len(pairs))


def band(value: float, thresholds: Mapping[str, float]) -> str:
    """'high' above the high threshold, 'medium' above medium, else 'low'."""
    if value > thresholds["high"]:
        return "high"
    if value > thresholds["medium"]:
        return "medium"
    return "low"


def drift_kind(embedding_band: str, lexical_band: str) -> str:
    """Four kinds of drift (P2): meaning, wording, both, or none.

    "meaning" = the embedding distribution moved but the vocabulary did not;
    "wording" = new words, same meaning space; "both"; "none".
    """
    e, w = embedding_band != "low", lexical_band != "low"
    return {(True, True): "both", (True, False): "meaning",
            (False, True): "wording", (False, False): "none"}[(e, w)]


def overall(embedding_band: str, lexical_band: str) -> str:
    """The more severe of the two bands."""
    return max(embedding_band, lexical_band, key=BANDS.index)
