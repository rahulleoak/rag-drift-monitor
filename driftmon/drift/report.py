"""Assemble Agent B's DriftReport: measure, calibrate, band, flag (D11).

Steps:
    1. whole-period comparisons (vs baseline and previous period) and
       within-category comparisons (vs the same category in the baseline)
    2. Benjamini-Hochberg across every comparison, per technique
    3. severity bands from ordinary change inside the baseline: one null for
       whole periods, one per category (at the smallest segment size of that
       category, which gives a wider, more cautious null); a category whose
       own null cannot be built falls back to the pooled baseline
    4. a comparison is *flagged* when both techniques are significant and
       the severity is at least medium; drifted segments go to Agents C and
       D, low-severity ones are kept as controls

No LLM calls: explanations are added later from these numbers.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from driftmon.drift import lexical as lex
from driftmon.drift import severity as sev
from driftmon.drift.compare import DriftInputs, prepare, segment_table, whole_period_table
from driftmon.evaluation.stats import benjamini_hochberg
from driftmon.schemas import DriftReport, MonitoringReport, SegmentDrift


def _terms(inp: DriftInputs, rows: np.ndarray, ref_rows: np.ndarray, prior, k: int = 5
           ) -> tuple[list[str], list[str]]:
    t = lex.log_odds_dirichlet(lex.term_counts(inp.tokens[i] for i in rows),
                               lex.term_counts(inp.tokens[i] for i in ref_rows), prior)
    t = t[t["count_target"] + t["count_reference"] >= 5]
    return t.head(k).index.tolist(), t.tail(k).index[::-1].tolist()


def _examples(inp: DriftInputs, rows: np.ndarray, term: str | None, k: int = 3) -> list[str]:
    """Up to ``k`` headlines from the segment that use its top emerging term."""
    if not term:
        return []
    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    titles = inp.docs["title"].to_numpy()[rows]
    texts = inp.docs["text"].to_numpy()[rows]
    return [t for t, x in zip(titles, texts) if pat.search(f"{t} {x}")][:k]


def _to_segment(row: Mapping[str, Any], null: sev.NullDistribution, thresholds, alpha: float,
                terms: tuple[list[str], list[str]], examples: list[str]) -> SegmentDrift:
    e_band = sev.band(row["mmd2"], thresholds["mmd2"])
    l_band = sev.band(row["jsd_excess"], thresholds["jsd_excess"])
    severity = sev.overall(e_band, l_band)
    significant = row["mmd_q"] <= alpha and row["jsd_q"] <= alpha
    category = row.get("category")
    category = None if category is None or (isinstance(category, float) and np.isnan(category)) else category
    seg_id = f"{row['period']}|{category or 'ALL'}|vs {row['compared_with']}"
    ntr = row.get("new_term_rate")
    return SegmentDrift(
        segment_id=seg_id, period=str(row["period"]), category=category,
        compared_with=str(row["compared_with"]), n_reference=int(row["n_reference"]),
        n_period=int(row["n_period"]), mmd2=float(row["mmd2"]), mmd_q=float(row["mmd_q"]),
        jsd_excess=float(row["jsd_excess"]), jsd_q=float(row["jsd_q"]),
        centroid_shift=float(row["centroid_shift"]), novelty_mean=float(row["novelty_mean"]),
        novel_share=float(row["novel_share"]),
        new_term_rate=None if ntr is None or np.isnan(ntr) else float(ntr),
        times_ordinary_embedding=float(row["mmd2"] / max(null.typical("mmd2"), 1e-12)),
        times_ordinary_lexical=float(row["jsd_excess"] / max(null.typical("jsd_excess"), 1e-12)),
        embedding_band=e_band, lexical_band=l_band, severity=severity,
        kind=sev.drift_kind(e_band, l_band), significant=bool(significant),
        flagged=bool(significant and severity != "low"), null_window=null.window, null_n=null.n,
        emerging_terms=terms[0], declining_terms=terms[1], examples=examples)


def build_drift_report(config: Mapping[str, Any], docs: pd.DataFrame, vectors: np.ndarray,
                       monitoring: MonitoringReport, tokens: list[list[str]] | None = None
                       ) -> tuple[DriftReport, DriftInputs, dict[str, pd.DataFrame]]:
    """Run every comparison and return the report, the shared inputs and raw tables."""
    d = config["drift"]
    alpha = float(d["alpha"])
    pct = d["severity_null_percentiles"]
    draws, null_perm = int(d.get("null_draws", 5)), int(d.get("null_permutations", 50))
    cap = int(config["periods"]["sample_per_period"])
    inp = prepare(config, docs, vectors, tokens)
    base = inp.baseline.name

    whole = whole_period_table(inp)
    seg_info = pd.DataFrame(monitoring.segments)
    segs = segment_table(inp, seg_info, cap) if len(seg_info) else pd.DataFrame()

    # multiple-testing correction across every comparison except the reference row
    tested = pd.concat([whole.iloc[1:], segs], ignore_index=True)
    for m in ("mmd", "jsd"):
        tested[f"{m}_q"] = benjamini_hochberg(tested[f"{m}_p"])
    whole_t, segs_t = tested.iloc[:len(whole) - 1], tested.iloc[len(whole) - 1:]

    # nulls: whole periods at the drift-sample size; one per category otherwise
    n_whole = int(whole["n_period"].iloc[1])
    null_whole = sev.build_null(inp, n_whole, None, draws, null_perm)
    if null_whole is None:
        raise ValueError("the baseline is too small to calibrate severity bands")
    nulls: dict[str | None, sev.NullDistribution] = {None: null_whole}
    for cat, grp in segs_t.groupby("category") if len(segs_t) else []:
        n = int(grp["n_period"].min())
        nulls[cat] = (sev.build_null(inp, n, cat, draws, null_perm)
                      or sev.build_null(inp, n, None, draws, null_perm) or null_whole)

    prior = lex.term_counts(inp.tokens)
    whole_out, seg_out = [], []
    for row in whole_t.to_dict("records"):
        rows = inp.rows(row["period"], sample_only=True)
        ref_rows = inp.rows(row["compared_with"], sample_only=True)
        terms = _terms(inp, rows, ref_rows, prior)
        whole_out.append(_to_segment(row, null_whole, null_whole.thresholds(pct), alpha, terms,
                                     _examples(inp, rows, terms[0][0] if terms[0] else None)))
    for row in segs_t.to_dict("records"):
        null = nulls[row["category"]]
        rows = inp.rows(row["period"], row["category"])
        terms = _terms(inp, rows, inp.rows(base, row["category"]), prior)
        seg_out.append(_to_segment(row, null, null.thresholds(pct), alpha, terms,
                                   _examples(inp, rows, terms[0][0] if terms[0] else None)))

    def strength(s: SegmentDrift) -> float:
        return max(s.times_ordinary_embedding, s.times_ordinary_lexical)

    vs_baseline = [s for s in whole_out + seg_out if s.compared_with == base]
    drifted = sorted((s for s in vs_baseline if s.flagged), key=strength, reverse=True)
    controls = sorted((s for s in seg_out if s.severity == "low"), key=strength)
    ref = whole.iloc[0].to_dict()
    reference = {"ordinary_year": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                   for k, v in ref.items()},
                 "whole_period_null": {"window": null_whole.window, "n": null_whole.n,
                                       "pairs": null_whole.pairs,
                                       "thresholds": null_whole.thresholds(pct),
                                       "typical": {m: null_whole.typical(m) for m in ("mmd2", "jsd_excess")}},
                 "alpha": alpha, "percentiles": dict(pct)}
    report = DriftReport(whole_period=whole_out, segments=seg_out,
                         drifted_segments=[s.segment_id for s in drifted],
                         control_segments=[s.segment_id for s in controls],
                         reference=reference, any_flagged=bool(drifted))
    return report, inp, {"whole_period": whole, "segments": segs}
