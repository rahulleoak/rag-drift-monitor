"""Shifts in semantic relationships: term-context shift (D7 item 1, D12).

For a term (e.g. "mask"), take the articles that use it in the baseline and
in a later period, and measure how far the centre of those articles moved in
embedding space (centroid cosine distance), with a permutation p-value.
"Ordinary" shift for the same term is the first half of the baseline vs the
second half (2016 vs 2017). The words that most distinguish the term's
later contexts from its baseline contexts show *how* its meaning changed
("mask": halloween, beauty -> coronavirus, face, wear).

Pure computation; no LLM calls.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from driftmon.drift import embedding as emb
from driftmon.drift import lexical as lex


def centroid_shift_test(x: np.ndarray, y: np.ndarray, permutations: int = 500,
                        seed: int = 0) -> tuple[float, float]:
    """Centroid cosine distance between two groups and its permutation p-value."""
    observed = emb.centroid_shift(x, y)
    z = np.vstack([x, y])
    m = len(x)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        idx = rng.permutation(len(z))
        if emb.centroid_shift(z[idx[:m]], z[idx[m:]]) >= observed:
            hits += 1
    return observed, (1 + hits) / (1 + permutations)


def context_words(rows_target: np.ndarray, rows_reference: np.ndarray,
                  tokens: Sequence[Sequence[str]], term: str, prior, k: int = 6
                  ) -> tuple[list[str], list[str]]:
    """Words that most distinguish the term's target contexts from its reference
    contexts (and vice versa), excluding the term itself."""
    def counts(rows):
        c = lex.term_counts(tokens[i] for i in rows)
        c.pop(term, None)
        return c
    lo = lex.log_odds_dirichlet(counts(rows_target), counts(rows_reference), prior, min_count=3)
    return lo.head(k).index.tolist(), lo.tail(k).index[::-1].tolist()


def term_context_shifts(terms: Sequence[str], docs: pd.DataFrame, vectors: np.ndarray,
                        tokens: Sequence[Sequence[str]], baseline: str,
                        baseline_halves: tuple, later: Sequence[str], min_docs: int = 20,
                        permutations: int = 500, seed: int = 0) -> pd.DataFrame:
    """One row per (term, later period) where the term has ``min_docs`` articles
    on both sides.

    Columns include the centroid shift and p-value, the ordinary shift for the
    same term inside the baseline (2016 vs 2017), their ratio, and the
    distinguishing context words before and after.
    """
    period = docs["period"].to_numpy()
    dates = pd.to_datetime(docs["date"]).dt.date
    first, second = baseline_halves
    in_first = dates.map(first.contains).to_numpy()
    in_second = dates.map(second.contains).to_numpy()
    prior = lex.term_counts(tokens)
    has = {t: np.array([t in set(toks) for toks in tokens]) for t in terms}
    rows = []
    for t in terms:
        base = np.where(has[t] & (period == baseline))[0]
        h1, h2 = np.where(has[t] & in_first)[0], np.where(has[t] & in_second)[0]
        for k, p in enumerate(later):
            tgt = np.where(has[t] & (period == p))[0]
            if len(base) < min_docs or len(tgt) < min_docs:
                continue
            shift, pval = centroid_shift_test(vectors[base], vectors[tgt], permutations, seed + k)
            ordinary, ordinary_n = ordinary_shift(vectors, h1, h2, len(tgt), seed=seed + k)
            after, before = context_words(tgt, base, tokens, t, prior)
            rows.append({"term": t, "period": p, "docs_baseline": len(base), "docs_period": len(tgt),
                         "centroid_shift": shift, "p_value": pval, "ordinary_shift": ordinary,
                         "ordinary_n": ordinary_n,
                         "times_ordinary": shift / ordinary if ordinary > 0 else float("nan"),
                         "context_before": ", ".join(before), "context_after": ", ".join(after)})
    return pd.DataFrame(rows)


ORDINARY_MIN_DOCS = 10  # articles needed in each baseline half to estimate ordinary shift


def ordinary_shift(vectors: np.ndarray, first: np.ndarray, second: np.ndarray, n_target: int,
                   draws: int = 20, seed: int = 0) -> tuple[float, int]:
    """Average centroid shift of a term between the two baseline halves, with
    the second group sampled to the later period's size.

    A small group's centroid is noisy, so a small later group looks shifted
    partly by chance. Matching the size makes "× ordinary" a fair ratio.
    Returns (shift, group size used), or (nan, 0) if a half has too few articles.
    """
    if len(first) < ORDINARY_MIN_DOCS or len(second) < ORDINARY_MIN_DOCS:
        return float("nan"), 0
    n = min(n_target, len(second))
    rng = np.random.default_rng(seed)
    shifts = [emb.centroid_shift(vectors[first], vectors[rng.choice(second, n, replace=False)])
              for _ in range(draws)]
    return float(np.mean(shifts)), n
