"""How chunking changes what drift detection sees, on long documents (D26).

Each Wikipedia article is compared with its own baseline version (end of
2017). The ground truth is textual: the share of the later version's words
that sit in sentences absent from the baseline version (new or rewritten
sentences). Each way of reading the article gives a change score, and we
check how well each score follows the ground truth:

    truncated   one vector of the first ~512 tokens (what a whole-document
                embedding actually reads); score = 1 - cosine
    averaged    the mean of the article's chunk vectors; score = 1 - cosine
    chunks_N    chunks of about N tokens; score = mean over the later
                version's chunks of (1 - best cosine to any baseline chunk)

Pure functions; embedding happens in scripts/longform_drift.py.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [" ".join(s.split()) for para in text.split("\n") for s in _SENT.split(para) if s.strip()]


def changed_share(old: str, new: str, cutoff_words: int | None = None) -> tuple[float, float]:
    """(share of ``new``'s words in sentences not found in ``old``,
    share of those changed words that come after word ``cutoff_words``).

    The second number says how much of the change a reader of only the
    first part of the article would miss (NaN if nothing changed).
    """
    seen = set(_sentences(old))
    pos, total, changed, late = 0, 0, 0, 0
    for s in _sentences(new):
        n = len(s.split())
        total += n
        if s not in seen:
            changed += n
            if cutoff_words is not None:
                late += max(0, pos + n - max(cutoff_words, pos))
        pos += n
    share = changed / total if total else 0.0
    beyond = late / changed if changed and cutoff_words is not None else float("nan")
    return share, beyond


_NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")


def sentences(text: str) -> list[str]:
    """Sentences of a text, whitespace-normalised (the unit of the ground truth)."""
    return _sentences(text)


def number_only_edits(old: str, new: str) -> dict[str, str]:
    """Sentences of ``new`` that differ from a sentence of ``old`` only in
    their numbers (counts, dates, amounts): {new sentence: old sentence}.

    These edits can change what an answer should say while barely changing
    the wording, which is the blind spot of embedding-based drift.
    """
    old_set = set(_sentences(old))
    by_mask: dict[str, str] = {}
    for s in old_set:
        by_mask.setdefault(_NUMBER.sub("#", s), s)
    out = {}
    for s in _sentences(new):
        if s in old_set:
            continue
        mask = _NUMBER.sub("#", s)
        if "#" in mask and mask in by_mask:
            out[s] = by_mask[mask]
    return out


def cosine_change(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine between two vectors."""
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_change(old: np.ndarray, new: np.ndarray) -> tuple[float, float]:
    """(mean, max) over the new version's chunks of 1 - best cosine to any old chunk.
    Rows must be normalised."""
    best = (new @ old.T).max(axis=1)
    nov = 1.0 - best
    return float(nov.mean()), float(nov.max())


def evaluate(table: pd.DataFrame, methods: Sequence[str], big_change: float) -> pd.DataFrame:
    """How well each method's score follows the true changed share.

    spearman: rank correlation with the true share, over all article x period pairs.
    auc: how often a "big change" pair (true share >= big_change) scores higher
         than a pair below it (0.5 = chance, 1 = perfect).
    Both are repeated on the pairs where most of the change lies beyond the
    first 512 tokens ("late change"), where truncation should fail.
    """
    rows = []
    late = table["beyond_512"] >= 0.5
    for m in methods:
        row = {"method": m}
        for name, part in [("all", table), ("late_change", table[late])]:
            row[f"spearman_{name}"] = float(spearmanr(part[m], part["true_changed"]).statistic) if len(part) > 2 else np.nan
            row[f"auc_{name}"] = auc(part[m].to_numpy(), (part["true_changed"] >= big_change).to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via ranks (ties count half)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))
