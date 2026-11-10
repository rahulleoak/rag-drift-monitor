"""Whole-period drift: every later period vs the baseline and vs the previous period.

Thin first version of Agent B's statistics (D10). For each comparison, on
the equal-size drift samples (1,398 documents per period):
    embedding: MMD^2 with permutation p-value, centroid shift, dispersion,
               novelty against the full stale (baseline) index
    lexical:   Jensen-Shannon divergence with permutation p-value, new-term
               rate, top emerging and declining terms (log-odds, vs baseline)
p-values are corrected across all comparisons with Benjamini-Hochberg.

A reference row compares 2017 with 2016 (both inside the baseline) at the
same sample size: what one ordinary year of news change looks like.

Writes to runs/drift/:
    period_drift.csv    one row per comparison
    terms_<period>.csv  emerging and declining terms vs the baseline
    novel_examples.csv  the most novel documents per period, with their
                        nearest baseline document

Usage (from the repository root):
    python scripts/run_drift.py
"""
from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_drift.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.drift import lexical as lex  # noqa: E402
from driftmon.drift.compare import NOVELTY_PERCENTILE, novelty_of, prepare, whole_period_table  # noqa: E402
from driftmon.evaluation.stats import benjamini_hochberg  # noqa: E402


def run(config: Mapping[str, Any], docs: pd.DataFrame, vectors: np.ndarray,
        out_dir: Path, top_terms: int = 30, top_examples: int = 10) -> pd.DataFrame:
    """Compute whole-period drift tables, write them to ``out_dir``, return the main table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = prepare(config, docs, vectors)
    table = whole_period_table(inp)
    table["mmd_q"] = benjamini_hochberg(table["mmd_p"])
    table["jsd_q"] = benjamini_hochberg(table["jsd_p"])
    table.attrs["novelty_cut"] = inp.novelty_cut

    prior = lex.term_counts(inp.tokens)
    base_rows = inp.rows(inp.baseline.name, sample_only=True)
    base_counts = lex.term_counts(inp.tokens[i] for i in base_rows)
    title_of = docs.set_index("doc_id")["title"]
    examples = []
    for p in inp.later:
        tgt = inp.rows(p.name, sample_only=True)
        terms = lex.log_odds_dirichlet(lex.term_counts(inp.tokens[i] for i in tgt), base_counts, prior)
        pd.concat([terms.head(top_terms).assign(direction="emerging"),
                   terms.tail(top_terms).iloc[::-1].assign(direction="declining")]
                  ).to_csv(out_dir / f"terms_{p.name}.csv")
        nov, nearest = novelty_of(inp, tgt)
        for j in np.argsort(-nov)[:top_examples]:
            i = tgt[j]
            examples.append({"period": p.name, "novelty": round(float(nov[j]), 3),
                             "title": docs.at[i, "title"], "category": docs.at[i, "category"],
                             "nearest_baseline_title": title_of[nearest[j]]})
    table.to_csv(out_dir / "period_drift.csv", index=False)
    pd.DataFrame(examples).to_csv(out_dir / "novel_examples.csv", index=False)
    return table


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    out_dir = resolve_path(config["paths"]["runs"]) / "drift"
    t0 = time.time()
    docs, vectors = load_corpus(config)
    table = run(config, docs, vectors, out_dir)
    pd.set_option("display.width", 250)
    cols = ["period", "compared_with", "mmd2", "mmd_q", "centroid_shift", "dispersion_period",
            "novelty_mean", "novel_share", "jsd_excess", "jsd_q", "new_term_rate"]
    print(f"\n=== Whole-period drift (n = {table['n_period'].iloc[0]:,} per side; "
          f"{config['drift']['mmd_permutations']} permutations; "
          f"novel = novelty above {table.attrs['novelty_cut']:.3f}, the baseline's "
          f"{NOVELTY_PERCENTILE}th percentile) ===")
    print(table[cols].round(4).to_string(index=False))
    print(f"\nTables in {out_dir} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
