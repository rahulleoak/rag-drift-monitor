"""Is part of the drift just article length? (D33, D6 check 5)

    python scripts/length_check.py

No LLM calls. For each later period, compares its drift sample with
``seeds`` random baseline samples and ``seeds`` length-matched ones (same
word counts as the period's sample), using Agent B's comparison and its
ordinary-change yardstick. Writes to runs/length_check/:
    draws.csv      every comparison
    summary.csv    per period: drift as × ordinary change, random vs matched
    verdict.json   does length explain a large part, by the rule fixed before results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.drift import length as ln  # noqa: E402
from driftmon.drift import severity as sev  # noqa: E402
from driftmon.drift.compare import compare, prepare  # noqa: E402


def main() -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    d = cfg["drift"]
    seeds, perms = int(d["length_check"]["seeds"]), int(d["null_permutations"])
    docs, vectors = load_corpus(cfg)
    inp = prepare(cfg, docs, vectors)
    base = inp.baseline.name
    words = docs["words"].to_numpy()
    pool = inp.rows(base)
    n = len(inp.rows(base, sample_only=True))
    null = sev.build_null(inp, n, None, int(d.get("null_draws", 5)), perms)
    typical_e, typical_l = null.typical("mmd2"), null.typical("jsd_excess")
    print(f"Ordinary change (median of {len(null.values)} within-baseline comparisons, n = {null.n}): "
          f"MMD² {typical_e:.5f}, JSD excess {typical_l:.5f}")

    rows = []
    for p in inp.later:
        tgt = inp.rows(p.name, sample_only=True)
        for s in range(seeds):
            rng = np.random.default_rng([inp.seed, s, 33])
            samples = {"random": rng.choice(pool, len(tgt), replace=False),
                       "matched": ln.length_matched_rows(pool, words[pool], words[tgt], rng)}
            row = {"period": p.name, "seed": s, "median_words_period": float(np.median(words[tgt]))}
            for name, ref in samples.items():
                r = compare(inp, ref, tgt, permutations=perms, seed=inp.seed + s)
                row.update({f"{name}_mmd2": r["mmd2"], f"{name}_x_embedding": r["mmd2"] / typical_e,
                            f"{name}_x_lexical": r["jsd_excess"] / typical_l,
                            f"median_words_{name}": float(np.median(words[ref]))})
            rows.append(row)
            print(f"{p.name} seed {s}: random {row['random_x_embedding']:.2f}x, "
                  f"matched {row['matched_x_embedding']:.2f}x", flush=True)

    draws = pd.DataFrame(rows)
    summary = ln.summarise(draws)
    result = {**ln.verdict(summary, cfg), "seeds": seeds, "n": n, "ordinary_mmd2": typical_e,
              "ordinary_jsd_excess": typical_l}
    out = resolve_path(cfg["paths"]["runs"]) / "length_check"
    out.mkdir(parents=True, exist_ok=True)
    draws.to_csv(out / "draws.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
