"""Is part of the drift just article length? (D33, D6 check 5)

Baseline articles are shorter (median 24 words) than later ones (29-31).
Longer texts can sit in a different part of the embedding space whatever
their topic, so some of the measured drift could be length. The check:
compare each later period with a baseline sample whose lengths match that
period, and see how much the drift falls.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def length_matched_rows(pool_rows: np.ndarray, pool_words: np.ndarray, target_words: Sequence[int],
                        rng: np.random.Generator) -> np.ndarray:
    """One pool row per target length, without replacement: the same word
    count where one is left, otherwise the nearest length still available."""
    by_len: dict[int, list[int]] = defaultdict(list)
    for row, w in zip(pool_rows, pool_words):
        by_len[int(w)].append(int(row))
    for rows in by_len.values():
        rng.shuffle(rows)
    lengths = np.array(sorted(by_len))
    out = []
    for w in rng.permutation(np.asarray(target_words, dtype=int)):
        if not by_len.get(int(w)):
            alive = lengths[[bool(by_len[int(x)]) for x in lengths]]
            if not len(alive):
                raise ValueError("the pool is smaller than the target sample")
            w = alive[np.argmin(np.abs(alive - w))]
        out.append(by_len[int(w)].pop())
    return np.array(out)


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Per period: mean drift with a random and with a length-matched
    baseline sample, as multiples of ordinary change, and the ratio."""
    g = rows.groupby("period", sort=False)
    out = pd.DataFrame({
        "random_x_ordinary": g["random_x_embedding"].mean(),
        "matched_x_ordinary": g["matched_x_embedding"].mean(),
        "random_x_lexical": g["random_x_lexical"].mean(),
        "matched_x_lexical": g["matched_x_lexical"].mean(),
        "median_words_period": g["median_words_period"].first(),
        "median_words_random": g["median_words_random"].mean(),
        "median_words_matched": g["median_words_matched"].mean(),
    })
    out["embedding_kept"] = out["matched_x_ordinary"] / out["random_x_ordinary"]
    out["lexical_kept"] = out["matched_x_lexical"] / out["random_x_lexical"]
    return out.reset_index()


def verdict(summary: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Does length explain a large part of the embedding drift? By the rule
    fixed before results: matching lowers it by at least ``min_drop`` on
    average over the later periods."""
    min_drop = float(cfg["drift"]["length_check"]["min_drop"])
    kept = float(summary["embedding_kept"].mean())
    return {"length_explains_a_large_part": bool(1 - kept >= min_drop), "embedding_drop": 1 - kept,
            "lexical_drop": 1 - float(summary["lexical_kept"].mean()), "min_drop": min_drop,
            "by_period": summary.set_index("period")["embedding_kept"].round(3).to_dict()}
