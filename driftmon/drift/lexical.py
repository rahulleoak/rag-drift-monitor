"""Drift technique 2: vocabulary (docs/questions.md Q6).

    - Jensen-Shannon divergence between term distributions (segment score),
      with a document-level permutation test
    - log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008):
      z-scores per term, giving ranked emerging and declining terms
    - new-term rate: share of word occurrences not in the reference vocabulary

Pure functions; no LLM calls.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*[a-z0-9]|[a-z0-9]")
_CURLY = str.maketrans({"’": "'", "‘": "'", "—": " ", "–": " "})


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stop words, contractions or numbers-only tokens.

    Hyphenated words stay whole ("covid-19"), possessives are stripped
    ("trump's" -> "trump").
    """
    out = []
    for tok in _TOKEN.findall(text.lower().translate(_CURLY)):
        if tok.endswith("'s"):
            tok = tok[:-2]
        tok = tok.strip("'-")
        if "'" in tok:  # contractions (we're, don't) are function words
            continue
        if len(tok) < 2 or tok in ENGLISH_STOP_WORDS or tok.replace("-", "").isdigit():
            continue
        out.append(tok)
    return out


def term_counts(token_lists: Iterable[Sequence[str]]) -> Counter:
    """Total occurrences of each term across documents."""
    c: Counter = Counter()
    for toks in token_lists:
        c.update(toks)
    return c


def _jsd_columns(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Base-2 Jensen-Shannon divergence between count columns (vocab x k)."""
    p = p / p.sum(axis=0, keepdims=True)
    q = q / q.sum(axis=0, keepdims=True)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=0)
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=0)
    return 0.5 * (kl_p + kl_q)


def jensen_shannon(a: Counter, b: Counter) -> float:
    """Base-2 Jensen-Shannon divergence between two term-count tables, in [0, 1]."""
    vocab = sorted(a.keys() | b.keys())
    p = np.array([[a.get(t, 0)] for t in vocab], dtype=np.float64)
    q = np.array([[b.get(t, 0)] for t in vocab], dtype=np.float64)
    return float(_jsd_columns(p, q)[0])


@dataclass(frozen=True)
class JSDResult:
    """JSD between two document sets, its permutation p-value, and the
    average JSD under random shuffling (what sampling noise alone gives)."""

    jsd: float
    p_value: float
    permutations: int
    null_mean: float

    @property
    def excess(self) -> float:
        """JSD above the shuffled baseline: comparable across sample sizes."""
        return self.jsd - self.null_mean


def jsd_permutation_test(docs_x: Sequence[Sequence[str]], docs_y: Sequence[Sequence[str]],
                         permutations: int = 500, seed: int = 0) -> JSDResult:
    """JSD between two document sets, with a p-value from shuffling documents.

    Documents (not single words) are shuffled between the groups, because
    words inside one article are not independent. Raw JSD on sparse word
    counts is dominated by sampling noise, so ``excess`` (JSD minus its
    shuffled average) is the number to compare across segments.
    """
    docs = list(docs_x) + list(docs_y)
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    rows, cols, vals = [], [], []
    for r, d in enumerate(docs):
        for t, c in Counter(d).items():
            rows.append(r)
            cols.append(vocab[t])
            vals.append(c)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float64)
    m = len(docs_x)
    total = np.asarray(mat.sum(axis=0)).ravel()[:, None]

    def stat(labels: np.ndarray) -> np.ndarray:
        cx = np.asarray(mat.T @ labels)          # vocab x k counts for group x
        return _jsd_columns(cx, total - cx)

    observed_labels = np.zeros((len(docs), 1))
    observed_labels[:m] = 1.0
    observed = float(stat(observed_labels)[0])
    rng = np.random.default_rng(seed)
    perm = np.zeros((len(docs), permutations))
    for j in range(permutations):
        perm[rng.permutation(len(docs))[:m], j] = 1.0
    null = stat(perm)
    p = (1 + int((null >= observed).sum())) / (1 + permutations)
    return JSDResult(observed, p, permutations, float(null.mean()))


def log_odds_dirichlet(target: Counter, reference: Counter, prior: Counter,
                       prior_size: float = 1000.0, min_count: int = 5) -> pd.DataFrame:
    """Weighted log-odds of each term, target vs reference (Monroe et al. 2008).

    The informative Dirichlet prior shrinks rare terms towards the background
    rate, so a word seen twice cannot top the list. Positive z = more typical
    of ``target`` (emerging); negative = more typical of ``reference``
    (declining).

    Args:
        target, reference: term counts of the two corpora being compared.
        prior: background term counts (e.g. the whole cleaned collection).
        prior_size: total pseudo-count of the prior (strength of shrinkage).
        min_count: drop terms seen fewer times than this in both corpora
            together.

    Returns:
        DataFrame indexed by term with columns ``z``, ``log_odds``,
        ``count_target``, ``count_reference``, sorted by ``z`` descending,
        ties broken alphabetically so repeated runs give the same order.
    """
    # The prior is defined over the whole vocabulary (alpha_0 = prior_size), as in
    # Monroe et al.; min_count only decides which terms are *reported*.
    full_vocab = prior.keys() | target.keys() | reference.keys()
    floor = 0.01  # tiny pseudo-count so a term absent from the prior still has one
    prior_total = sum(prior.values()) + floor * len(full_vocab)
    vocab = [t for t in (target.keys() | reference.keys())
             if target.get(t, 0) + reference.get(t, 0) >= min_count]
    a = np.array([prior.get(t, 0) for t in vocab], dtype=np.float64)
    a = prior_size * (a + floor) / prior_total
    a0 = float(prior_size)
    yt = np.array([target.get(t, 0) for t in vocab], dtype=np.float64)
    yr = np.array([reference.get(t, 0) for t in vocab], dtype=np.float64)
    nt, nr = sum(target.values()), sum(reference.values())
    delta = (np.log((yt + a) / (nt + a0 - yt - a)) - np.log((yr + a) / (nr + a0 - yr - a)))
    var = 1.0 / (yt + a) + 1.0 / (yr + a)
    out = pd.DataFrame({"z": delta / np.sqrt(var), "log_odds": delta,
                        "count_target": yt.astype(int), "count_reference": yr.astype(int)},
                       index=pd.Index(vocab, name="term"))
    return out.sort_values(["z", "term"], ascending=[False, True])


def new_term_rate(docs: Iterable[Sequence[str]], reference_vocab: set[str]) -> float:
    """Share of word occurrences in ``docs`` that never occur in the reference."""
    total = new = 0
    for d in docs:
        total += len(d)
        new += sum(1 for t in d if t not in reference_vocab)
    return new / total if total else 0.0
