"""Emerging topics that cut across sections (D7 item 4, D12).

F4 showed COVID articles are not individually novel: each has a close
baseline neighbour (flu, Zika). So topics are found by *growth*, not
novelty: all drift samples (baseline and every later period, equal sizes)
are clustered together in embedding space, and a topic is emerging in a
period when its share of that period is much larger than its share of the
baseline. Each topic is labelled with its most characteristic words
(log-odds against all other articles) and the sections it spans.

Clusters depend on the random seed and on k; they are an exploratory view
used for explanation, not for the drift decisions themselves.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans

from driftmon.drift import lexical as lex
from driftmon.evaluation.stats import benjamini_hochberg


def cluster_documents(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """K-means topic id for each row (cosine geometry: rows are normalised)."""
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(vectors)


def topic_labels(topic: np.ndarray, tokens: Sequence[Sequence[str]], n_terms: int = 6,
                 min_count: int = 5) -> dict[int, list[str]]:
    """Most characteristic words of each topic vs all other documents."""
    all_counts = lex.term_counts(tokens)
    labels = {}
    for t in np.unique(topic):
        inside = lex.term_counts(tokens[i] for i in np.where(topic == t)[0])
        outside = all_counts - inside
        lo = lex.log_odds_dirichlet(inside, outside, all_counts, min_count=min_count)
        labels[int(t)] = lo.head(n_terms).index.tolist()
    return labels


def emerging_topics(topic: np.ndarray, period: np.ndarray, categories: np.ndarray,
                    baseline: str, later: Sequence[str], labels: dict[int, list[str]],
                    titles: np.ndarray, min_ratio: float = 3.0, min_docs: int = 15,
                    alpha: float = 0.05) -> pd.DataFrame:
    """Topics whose share of a later period is at least ``min_ratio`` times
    their baseline share, with at least ``min_docs`` articles in the period
    and a significant Fisher exact test after Benjamini-Hochberg.

    The share ratio uses add-one smoothing so a topic absent from the
    baseline gets a finite ratio.
    """
    rows = []
    n_base = int((period == baseline).sum())
    for p in later:
        n_p = int((period == p).sum())
        for t in np.unique(topic):
            in_t = topic == t
            c_p = int((in_t & (period == p)).sum())
            c_b = int((in_t & (period == baseline)).sum())
            ratio = ((c_p + 1) / (n_p + 1)) / ((c_b + 1) / (n_base + 1))
            _, pval = fisher_exact([[c_p, n_p - c_p], [c_b, n_base - c_b]], alternative="greater")
            sel = in_t & (period == p)
            cats = pd.Series(categories[sel]).value_counts(normalize=True)
            rows.append({"period": p, "topic": int(t), "label": ", ".join(labels[int(t)]),
                         "docs_period": c_p, "share_period": c_p / n_p if n_p else 0.0,
                         "docs_baseline": c_b, "share_baseline": c_b / n_base if n_base else 0.0,
                         "ratio": ratio, "p_value": pval,
                         "sections": int((cats >= 0.10).sum()),
                         "top_sections": ", ".join(f"{c} {s:.0%}" for c, s in cats.head(3).items()),
                         "examples": " | ".join(titles[sel][:3])})
    df = pd.DataFrame(rows)
    df["q_value"] = benjamini_hochberg(df["p_value"])
    df["emerging"] = (df["ratio"] >= min_ratio) & (df["docs_period"] >= min_docs) & (df["q_value"] <= alpha)
    return df.sort_values(["period", "ratio"], ascending=[True, False]).reset_index(drop=True)
