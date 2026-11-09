"""Run both drift techniques on pairs of document samples (D10, D11).

Shared by Agent B and ``scripts/run_drift.py``:
    - ``prepare``: tokens, kernel width, baseline vocabulary, stale index and
      the novelty threshold, computed once per run
    - ``compare``: one reference sample vs one target sample, both techniques
    - ``whole_period_table``: each later period vs the baseline and vs the
      previous period, plus the ordinary-year reference row
    - ``segment_table``: each later period x category vs the same category in
      the baseline, using every article in the segment (capped)

Pure computation; no LLM calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from driftmon.drift import embedding as emb
from driftmon.drift import lexical as lex
from driftmon.rag.embeddings import document_text
from driftmon.rag.index import StudyIndexes, build_study_indexes
from driftmon.data.periods import Period, study_periods

NOVELTY_PERCENTILE = 95  # "novel" = less similar to the archive than 95% of baseline articles are


@dataclass
class DriftInputs:
    """Everything the comparisons need, computed once per run."""

    docs: pd.DataFrame
    vectors: np.ndarray
    tokens: list[list[str]]
    baseline: Period
    later: list[Period]
    gamma: float
    baseline_vocab: set[str]
    indexes: StudyIndexes
    novelty_cut: float
    baseline_novelty: np.ndarray  # novelty of the baseline drift sample (self excluded)
    permutations: int
    seed: int

    def rows(self, period: str, category: str | None = None, sample_only: bool = False) -> np.ndarray:
        mask = (self.docs["period"] == period).to_numpy()
        if category is not None:
            mask = mask & (self.docs["category"] == category).to_numpy()
        if sample_only:
            mask = mask & self.docs["in_drift_sample"].to_numpy()
        return np.where(mask)[0]


def tokenize_documents(docs: pd.DataFrame) -> list[list[str]]:
    """Word tokens of each document's headline and description."""
    return [lex.tokenize(document_text(t, x)) for t, x in zip(docs["title"], docs["text"])]


def prepare(config: Mapping[str, Any], docs: pd.DataFrame, vectors: np.ndarray,
            tokens: list[list[str]] | None = None) -> DriftInputs:
    """Tokenise documents (unless given), build indexes and fix the shared scales."""
    periods = study_periods(config)
    baseline, later = periods[0], periods[1:]
    seed = int(config["seed"])
    if tokens is None:
        tokens = tokenize_documents(docs)
    base_all = np.where((docs["period"] == baseline.name).to_numpy())[0]
    base_sample = np.where(((docs["period"] == baseline.name) & docs["in_drift_sample"]).to_numpy())[0]
    indexes = build_study_indexes(docs, vectors, baseline.name, [p.name for p in later])
    ids, sims = indexes.stale.search(vectors[base_sample], k=2)  # nearest *other* baseline article
    self_hit = ids[:, 0] == docs["doc_id"].to_numpy()[base_sample]
    base_nov = emb.novelty(np.where(self_hit, sims[:, 1], sims[:, 0]))
    return DriftInputs(
        docs=docs, vectors=vectors, tokens=tokens, baseline=baseline, later=list(later),
        gamma=emb.median_heuristic_gamma(vectors[base_sample], seed=seed),
        baseline_vocab={t for i in base_all for t in tokens[i]}, indexes=indexes,
        novelty_cut=float(np.percentile(base_nov, NOVELTY_PERCENTILE)), baseline_novelty=base_nov,
        permutations=int(config["drift"]["mmd_permutations"]), seed=seed)


def compare(inp: DriftInputs, ref_rows: np.ndarray, tgt_rows: np.ndarray,
            permutations: int | None = None, seed: int | None = None) -> dict[str, Any]:
    """Both techniques on one reference sample vs one target sample."""
    permutations = inp.permutations if permutations is None else permutations
    seed = inp.seed if seed is None else seed
    x, y = inp.vectors[ref_rows], inp.vectors[tgt_rows]
    mmd = emb.mmd_permutation_test(x, y, inp.gamma, permutations, seed)
    jsd = lex.jsd_permutation_test([inp.tokens[i] for i in ref_rows],
                                   [inp.tokens[i] for i in tgt_rows], permutations, seed)
    return {"n_reference": len(ref_rows), "n_period": len(tgt_rows),
            "mmd2": mmd.mmd2, "mmd_p": mmd.p_value, "centroid_shift": emb.centroid_shift(x, y),
            "dispersion_reference": emb.dispersion(x), "dispersion_period": emb.dispersion(y),
            "jsd": jsd.jsd, "jsd_excess": jsd.excess, "jsd_p": jsd.p_value,
            "new_term_rate": lex.new_term_rate([inp.tokens[i] for i in tgt_rows], inp.baseline_vocab)}


def novelty_of(inp: DriftInputs, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Novelty against the stale index, and each row's nearest baseline doc id."""
    ids, sims = inp.indexes.stale.search(inp.vectors[rows], k=1)
    return emb.novelty(sims[:, 0]), ids[:, 0]


def _with_novelty(inp: DriftInputs, row: dict[str, Any], nov: np.ndarray) -> dict[str, Any]:
    row.update(novelty_mean=float(nov.mean()), novel_share=float((nov > inp.novelty_cut).mean()))
    return row


def whole_period_table(inp: DriftInputs) -> pd.DataFrame:
    """Each later period vs the baseline and vs the previous period (drift samples).

    The first row is the ordinary-year reference: the second half of the
    baseline vs the first half (2017 vs 2016), at the same sample size.
    """
    rng = np.random.default_rng(inp.seed)
    base = inp.baseline.name
    n = len(inp.rows(base, sample_only=True))
    first, second = inp.baseline.halves()
    dates = pd.to_datetime(inp.docs["date"]).dt.date
    in_base = (inp.docs["period"] == base).to_numpy()
    h1 = np.where(in_base & dates.map(first.contains).to_numpy())[0]
    h2 = np.where(in_base & dates.map(second.contains).to_numpy())[0]
    ref = compare(inp, rng.choice(h1, n, replace=False), rng.choice(h2, n, replace=False))
    ref.update(period=f"{second.start.year} (inside baseline)", compared_with=str(first.start.year),
               new_term_rate=float("nan"))
    rows = [_with_novelty(inp, ref, inp.baseline_novelty)]

    previous = base
    for p in inp.later:
        tgt = inp.rows(p.name, sample_only=True)
        nov, _ = novelty_of(inp, tgt)
        for other in dict.fromkeys([base, previous]):  # the first later period's previous is the baseline
            r = compare(inp, inp.rows(other, sample_only=True), tgt)
            r.update(period=p.name, compared_with=other)
            rows.append(_with_novelty(inp, r, nov))
        previous = p.name
    cols = ["period", "compared_with"]
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]]


def eligible_segments(docs: pd.DataFrame, baseline: str, later: Sequence[str],
                      min_size: int) -> pd.DataFrame:
    """Later period x category groups with at least ``min_size`` articles whose
    category also has at least ``min_size`` articles in the baseline."""
    counts = docs.groupby(["period", "category"]).size()
    base_counts = counts.get(baseline, pd.Series(dtype=int))
    rows = []
    for p in later:
        if p not in counts.index.get_level_values(0):
            continue
        for cat, n in counts[p].items():
            nb = int(base_counts.get(cat, 0))
            rows.append({"period": p, "category": cat, "n_period": int(n), "n_baseline": nb,
                         "eligible": n >= min_size and nb >= min_size,
                         "reason": "" if n >= min_size and nb >= min_size else
                         ("not in baseline" if nb == 0 else
                          "too few in baseline" if nb < min_size else "too few in period")})
    return pd.DataFrame(rows)


def segment_table(inp: DriftInputs, segments: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Each eligible segment vs the same category in the baseline.

    Uses every article in the segment, capped at ``cap`` by random sampling,
    and an equal-size random sample from the baseline category.
    """
    out = []
    for k, s in enumerate(segments[segments["eligible"]].itertuples()):
        rng = np.random.default_rng([inp.seed, k])
        tgt = inp.rows(s.period, s.category)
        ref = inp.rows(inp.baseline.name, s.category)
        n = min(len(tgt), len(ref), cap)
        tgt = np.sort(rng.choice(tgt, n, replace=False))
        ref = np.sort(rng.choice(ref, n, replace=False))
        r = compare(inp, ref, tgt, seed=inp.seed + k)
        nov, _ = novelty_of(inp, tgt)
        r.update(period=s.period, category=s.category, compared_with=inp.baseline.name)
        out.append(_with_novelty(inp, r, nov))
    df = pd.DataFrame(out)
    cols = ["period", "category", "compared_with"]
    return df[cols + [c for c in df.columns if c not in cols]] if len(df) else df
