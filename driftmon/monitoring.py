"""Descriptive monitoring statistics for Agent A (Data Monitoring).

Cheap, per-period descriptions of the incoming documents compared with the
baseline: volume, length, category mix (chi-square test with Cramér's V as
the effect size, because at these sample sizes every mix difference is
"significant"), sections that are new since the baseline, and the
new-term rate. Pure functions; no LLM calls.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from driftmon.drift import lexical as lex


def category_mix_test(baseline_counts: pd.Series, period_counts: pd.Series) -> dict[str, float]:
    """Chi-square test of baseline vs period category counts, with Cramér's V.

    Cramér's V is 0 for identical mixes and 1 for mixes with no category in
    common; for a 2 x K table it is sqrt(chi2 / N).
    """
    table = pd.concat([baseline_counts, period_counts], axis=1).fillna(0)
    table = table[(table.sum(axis=1) > 0)]
    chi2, p, dof, _ = chi2_contingency(table.T.to_numpy())
    n = table.to_numpy().sum()
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "cramers_v": float(np.sqrt(chi2 / n))}


def period_summary(docs: pd.DataFrame, baseline: str, periods: Sequence[str],
                   tokens: Sequence[Sequence[str]]) -> pd.DataFrame:
    """One row per period: volume, length, mix change vs baseline, new sections, new terms."""
    base = docs[docs["period"] == baseline]
    base_cats = base["category"].value_counts()
    base_vocab = {t for i in np.where((docs["period"] == baseline).to_numpy())[0] for t in tokens[i]}
    rows = []
    for p in [baseline, *periods]:
        d = docs[docs["period"] == p]
        cats = d["category"].value_counts()
        new_sections = [c for c in cats.index if c not in base_cats.index]
        row = {"period": p, "documents": len(d), "median_words": float(d["words"].median()),
               "mean_words": float(d["words"].mean()), "categories": int(cats.size),
               "new_sections": ", ".join(new_sections),
               "new_section_share": float(cats[new_sections].sum() / len(d)) if new_sections else 0.0}
        if p != baseline:
            row.update({f"mix_{k}": v for k, v in category_mix_test(base_cats, cats).items()})
            rows_p = np.where((docs["period"] == p).to_numpy())[0]
            row["new_term_rate"] = lex.new_term_rate([tokens[i] for i in rows_p], base_vocab)
        rows.append(row)
    return pd.DataFrame(rows)


def renamed_sections(docs: pd.DataFrame) -> pd.DataFrame:
    """Raw section labels that were merged into another name, with counts per period."""
    merged = docs[docs["category"] != docs["category_raw"]]
    return (merged.groupby(["category_raw", "category", "period"]).size()
            .rename("documents").reset_index())
