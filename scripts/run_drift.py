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
from driftmon.corpus import load_corpus  # noqa: E402
from driftmon.drift import embedding as emb  # noqa: E402
from driftmon.drift import lexical as lex  # noqa: E402
from driftmon.embeddings import document_text  # noqa: E402
from driftmon.evaluation.stats import benjamini_hochberg  # noqa: E402
from driftmon.index import build_study_indexes  # noqa: E402
from driftmon.periods import study_periods  # noqa: E402

NOVELTY_PERCENTILE = 95  # "novel" = less similar to the archive than 95% of baseline articles are


def _compare(name: str, reference: str, ref_rows: np.ndarray, tgt_rows: np.ndarray,
             vectors: np.ndarray, tokens: list[list[str]], gamma: float,
             permutations: int, seed: int, baseline_vocab: set[str]) -> dict[str, Any]:
    x, y = vectors[ref_rows], vectors[tgt_rows]
    mmd = emb.mmd_permutation_test(x, y, gamma, permutations, seed)
    jsd = lex.jsd_permutation_test([tokens[i] for i in ref_rows], [tokens[i] for i in tgt_rows],
                                   permutations, seed)
    return {"period": name, "compared_with": reference, "n_reference": len(ref_rows),
            "n_period": len(tgt_rows), "mmd2": mmd.mmd2, "mmd_p": mmd.p_value,
            "centroid_shift": emb.centroid_shift(x, y),
            "dispersion_reference": emb.dispersion(x), "dispersion_period": emb.dispersion(y),
            "jsd": jsd.jsd, "jsd_excess": jsd.excess, "jsd_p": jsd.p_value,
            "new_term_rate": lex.new_term_rate([tokens[i] for i in tgt_rows], baseline_vocab)}


def run(config: Mapping[str, Any], docs: pd.DataFrame, vectors: np.ndarray,
        out_dir: Path, top_terms: int = 30, top_examples: int = 10) -> pd.DataFrame:
    """Compute whole-period drift tables, write them to ``out_dir``, return the main table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    permutations = int(config["drift"]["mmd_permutations"])
    periods = study_periods(config)
    baseline, later = periods[0].name, [p.name for p in periods[1:]]
    tokens = [lex.tokenize(document_text(t, x)) for t, x in zip(docs["title"], docs["text"])]

    sample = docs["in_drift_sample"].to_numpy()
    rows_of = {p: np.where(sample & (docs["period"] == p).to_numpy())[0] for p in [baseline] + later}
    base_all = np.where((docs["period"] == baseline).to_numpy())[0]
    baseline_vocab = {t for i in base_all for t in tokens[i]}
    gamma = emb.median_heuristic_gamma(vectors[rows_of[baseline]], seed=seed)

    # novelty: similarity of each document to its nearest baseline article
    indexes = build_study_indexes(docs, vectors, baseline, later)
    base_ids = docs["doc_id"].to_numpy()
    q = rows_of[baseline]
    ids, sims = indexes.stale.search(vectors[q], k=2)  # nearest other baseline article
    base_best = np.where(ids[:, 0] == base_ids[q], sims[:, 1], sims[:, 0])
    novelty_cut = np.percentile(emb.novelty(base_best), NOVELTY_PERCENTILE)

    results, examples = [], []
    # reference row: one ordinary year of change inside the baseline
    year = pd.to_datetime(docs["date"]).dt.year.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(rows_of[baseline])
    y16 = rng.choice(np.where((year == 2016) & (docs["period"] == baseline).to_numpy())[0], n, replace=False)
    y17 = rng.choice(np.where((year == 2017) & (docs["period"] == baseline).to_numpy())[0], n, replace=False)
    ref_row = _compare("2017 (inside baseline)", "2016", y16, y17, vectors, tokens, gamma,
                       permutations, seed, baseline_vocab)
    ref_row["new_term_rate"] = float("nan")  # 2017 is part of the baseline vocabulary
    ref_row.update(novelty_mean=float(emb.novelty(base_best).mean()),
                   novel_share=float((emb.novelty(base_best) > novelty_cut).mean()))
    results.append(ref_row)

    prior = lex.term_counts(tokens)
    title_of = docs.set_index("doc_id")["title"]
    base_counts = lex.term_counts(tokens[i] for i in rows_of[baseline])
    previous = baseline
    for p in later:
        tgt = rows_of[p]
        ids, sims = indexes.stale.search(vectors[tgt], k=1)
        nov = emb.novelty(sims[:, 0])
        for compared_with in dict.fromkeys([baseline, previous]):  # 2018: previous is the baseline
            r = _compare(p, compared_with, rows_of[compared_with], tgt, vectors, tokens, gamma,
                         permutations, seed, baseline_vocab)
            r.update(novelty_mean=float(nov.mean()), novel_share=float((nov > novelty_cut).mean()))
            results.append(r)
        terms = lex.log_odds_dirichlet(lex.term_counts(tokens[i] for i in tgt), base_counts, prior)
        terms = pd.concat([terms.head(top_terms).assign(direction="emerging"),
                           terms.tail(top_terms).iloc[::-1].assign(direction="declining")])
        terms.to_csv(out_dir / f"terms_{p}.csv")
        for j in np.argsort(-nov)[:top_examples]:
            i = tgt[j]
            examples.append({"period": p, "novelty": round(float(nov[j]), 3),
                             "title": docs.at[i, "title"], "category": docs.at[i, "category"],
                             "nearest_baseline_title": title_of[ids[j, 0]]})
        previous = p

    table = pd.DataFrame(results)
    table["mmd_q"] = benjamini_hochberg(table["mmd_p"])
    table["jsd_q"] = benjamini_hochberg(table["jsd_p"])
    table.attrs["novelty_cut"] = float(novelty_cut)
    table.attrs["gamma"] = gamma
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
