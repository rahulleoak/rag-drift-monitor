"""The query stream: the simulated questions users ask in each period (D7).

There is no real query log, so each period's generated questions stand in for
the queries sent that period (D16). Only the sample pools drawn the same way
in every period count as the stream (``evaluation.query_stream_pools``,
random and oversample by default). Flagged and control articles are chosen
*because* of document drift, so their questions would inflate query drift by
design; they are used by Agent C, not here.

Used by:
    - Agent A: ``stream_summary`` (volume, length, new-term rate)
    - Agent B: ``query_drift_table`` (the same two techniques as for documents)

Limitation (stated in the README): the questions are written from each
period's articles, so query drift partly mirrors document drift by
construction. A real query log can drift on its own (new user groups, new
tasks); this one cannot.

Pure computation; no LLM calls.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from driftmon.config import resolve_path
from driftmon.drift import embedding as emb
from driftmon.drift import lexical as lex
from driftmon.embeddings import Embedder
from driftmon.evaluation.stats import benjamini_hochberg

KEPT = ("answerable", "unanswerable")  # a real stream also holds questions the archive cannot answer
# question wording that says nothing about the topic (not in the standard stop-word list)
QUESTION_WORDS = frozenset({"did", "does", "doing", "according", "happened", "kind", "type", "based",
                            "specific", "particular", "reportedly", "reported", "say", "said"})


def tokenize_question(question: str) -> list[str]:
    """Word tokens of a question, without stop words or question wording."""
    return [t for t in lex.tokenize(question) if t not in QUESTION_WORDS]


def load_questions(config: Mapping[str, Any], name: str | None = None) -> pd.DataFrame | None:
    """The kept questions of the configured question set, or None if not generated yet."""
    name = name or config["evaluation"].get("question_set", "full")
    path = resolve_path(config["paths"]["data_processed"]) / f"questions_{name}.parquet"
    if not path.exists():
        return None
    q = pd.read_parquet(path)
    return q[q["set"].isin(KEPT)].reset_index(drop=True)


def query_stream(config: Mapping[str, Any], questions: pd.DataFrame | None,
                 embedder: Embedder | None = None) -> tuple[pd.DataFrame, np.ndarray] | None:
    """The stream questions (stream pools only) and their query embeddings."""
    if questions is None or not len(questions):
        return None
    pools = config["evaluation"].get("query_stream_pools", ["random", "oversample"])
    stream = questions[questions["sample_pool"].isin(pools)].reset_index(drop=True)
    if not len(stream):
        return None
    embedder = embedder or Embedder.from_config(config)
    return stream, embedder.embed_queries(stream["question"].tolist())


def stream_summary(stream: pd.DataFrame, baseline: str, later: Sequence[str],
                   archive_vocab: set[str] | None = None) -> pd.DataFrame:
    """One row per period: query volume, length and new-term rate.

    ``archive_vocab``: the words of the baseline archive. A query word the
    archive never used is new; against the few dozen baseline *questions*
    almost every word would look new. Falls back to the baseline questions.
    """
    tokens = [tokenize_question(q) for q in stream["question"]]
    period = stream["period"].to_numpy()
    base_vocab = archive_vocab or {t for i in np.where(period == baseline)[0] for t in tokens[i]}
    rows = []
    for p in [baseline, *later]:
        idx = np.where(period == p)[0]
        sub = stream.iloc[idx]
        row = {"period": p, "queries": len(idx),
               "unanswerable_share": float((sub["set"] == "unanswerable").mean()) if len(idx) else 0.0,
               "median_words": float(sub["question"].str.split().str.len().median()) if len(idx) else 0.0}
        if p != baseline or archive_vocab:
            row["new_term_rate"] = lex.new_term_rate([tokens[i] for i in idx], base_vocab)
        rows.append(row)
    return pd.DataFrame(rows)


def _compare(x: np.ndarray, y: np.ndarray, tx: list[list[str]], ty: list[list[str]], gamma: float,
             permutations: int, seed: int, base_vocab: set[str]) -> dict[str, Any]:
    mmd = emb.mmd_permutation_test(x, y, gamma, permutations, seed)
    jsd = lex.jsd_permutation_test(tx, ty, permutations, seed)
    return {"n_reference": len(x), "n_period": len(y), "mmd2": mmd.mmd2, "mmd_p": mmd.p_value,
            "centroid_shift": emb.centroid_shift(x, y), "jsd": jsd.jsd, "jsd_excess": jsd.excess,
            "jsd_p": jsd.p_value, "new_term_rate": lex.new_term_rate(ty, base_vocab)}


def query_drift_table(stream: pd.DataFrame, vectors: np.ndarray, baseline: str, later: Sequence[str],
                      permutations: int = 500, seed: int = 0, alpha: float = 0.05,
                      prior: Mapping[str, int] | None = None, n_terms: int = 5,
                      archive_vocab: set[str] | None = None) -> pd.DataFrame:
    """Each period's queries vs the baseline queries, both techniques.

    The first row is the ordinary-change reference: the baseline's second
    year of queries vs its first (2017 vs 2016), so later rows can be read
    against what a year of ordinary change looks like for queries. A single
    pair is a rough scale, not a calibrated band (the document bands, D11,
    need far more data than the query stream has).

    Args:
        stream: stream questions with ``period``, ``date`` and ``question``.
        vectors: query embeddings aligned with ``stream``.
        baseline: baseline period name; later: later period names in order.
        prior: background term counts for the emerging-term ranking
            (defaults to all stream queries).
        archive_vocab: words of the baseline archive, for the new-term rate
            (share of query words the archive never used); defaults to the
            baseline questions' words, which is far too small a reference.
    """
    tokens = [tokenize_question(q) for q in stream["question"]]
    period = stream["period"].to_numpy()
    base = np.where(period == baseline)[0]
    if len(base) < 4:
        return pd.DataFrame()
    gamma = emb.median_heuristic_gamma(vectors[base], seed=seed)
    base_vocab = archive_vocab or {t for i in base for t in tokens[i]}
    prior = prior or lex.term_counts(tokens)

    years = pd.to_datetime(stream["date"]).dt.year.to_numpy()
    first_year = years[base].min()
    h1, h2 = base[years[base] == first_year], base[years[base] != first_year]
    rows = []
    if len(h1) >= 2 and len(h2) >= 2:
        ref = _compare(vectors[h1], vectors[h2], [tokens[i] for i in h1], [tokens[i] for i in h2],
                       gamma, permutations, seed, archive_vocab or {t for i in h1 for t in tokens[i]})
        ref.update(period=f"{first_year + 1} (inside baseline)", compared_with=str(first_year))
        rows.append(ref)
    for p in later:
        idx = np.where(period == p)[0]
        if len(idx) < 2:
            continue
        r = _compare(vectors[base], vectors[idx], [tokens[i] for i in base], [tokens[i] for i in idx],
                     gamma, permutations, seed, base_vocab)
        terms = lex.log_odds_dirichlet(lex.term_counts(tokens[i] for i in idx),
                                       lex.term_counts(tokens[i] for i in base), prior, min_count=3)
        top = terms.head(n_terms).index.tolist()
        r.update(period=p, compared_with=baseline, emerging_terms=", ".join(top),
                 examples=_examples(stream["question"].to_numpy()[idx], top[0] if top else None))
        rows.append(r)
    df = pd.DataFrame(rows)
    tested = ~df["period"].str.contains("inside baseline")
    for m in ("mmd", "jsd"):
        df[f"{m}_q"] = np.nan
        df.loc[tested, f"{m}_q"] = benjamini_hochberg(df.loc[tested, f"{m}_p"].to_numpy())
    if (~tested).any():  # rough scale: how many ordinary years of change
        ref = df.loc[~tested].iloc[0]
        for m, col in (("embedding", "mmd2"), ("lexical", "jsd_excess")):
            df[f"times_ordinary_{m}"] = df[col] / ref[col] if ref[col] > 0 else np.nan
    df["significant"] = (df["mmd_q"] <= alpha) | (df["jsd_q"] <= alpha)
    df.loc[~tested, "significant"] = False
    cols = ["period", "compared_with"]
    return df[cols + [c for c in df.columns if c not in cols]]


def _examples(questions: np.ndarray, term: str | None, k: int = 3) -> list[str]:
    """Up to ``k`` questions that use the period's top emerging term."""
    if not term:
        return []
    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return [q for q in questions if pat.search(q)][:k]
