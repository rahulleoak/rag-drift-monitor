"""Chunking experiment on the Wikipedia set: how the way a long article is
read changes what drift detection sees (D26).

    python scripts/longform_drift.py      (after scripts/fetch_wikipedia.py)

Writes to runs/longform/:
    article_change.csv    one row per article x later period: true changed share,
                          where the change is, and each method's change score
    chunking_eval.csv     how well each method follows the true change
    by_period.csv         average true change and scores per period; event set apart
    archive_mmd.csv       archive-level drift (MMD test) on article vectors vs chunk vectors

No LLM calls. Embeddings are cached (unchanged chunks are embedded once).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon import chunking, longform  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.drift.embedding import median_heuristic_gamma, mmd_permutation_test  # noqa: E402
from driftmon.embeddings import Embedder, document_text  # noqa: E402

TRUNCATE_TOKENS = 500   # leaves room for the title within the model's 512


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    lf, ch = cfg["longform"], cfg["longform"]["chunking"]
    snaps = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / "wiki_snapshots.parquet")
    count = chunking.token_counter(cfg["embeddings"]["model"])
    emb = Embedder.from_config(cfg)
    sizes = list(ch["compare_sizes"])

    # 1. Texts to embed for every snapshot, each way of reading it.
    trunc_text, chunk_text = {}, {s: {} for s in sizes}
    for r in snaps.itertuples(index=False):
        key = (r.page_id, r.period)
        trunc_text[key] = document_text(r.title, chunking.truncate(r.text, count, TRUNCATE_TOKENS))
        secs = json.loads(r.sections)
        for s in sizes:
            chunk_text[s][key] = [document_text(f"{r.title} - {c.section}", c.text)
                                  for c in chunking.chunk_document(secs, count, s, ch["overlap_tokens"],
                                                                   ch["min_tokens"])]
    print(f"{len(snaps)} snapshots; chunks per size: "
          f"{ {s: sum(map(len, chunk_text[s].values())) for s in sizes} }", flush=True)

    # 2. Embed (one call per kind; cached).
    keys = list(trunc_text)
    tv = dict(zip(keys, emb.embed_documents([trunc_text[k] for k in keys])))
    cv = {}
    for s in sizes:
        flat = [t for k in keys for t in chunk_text[s][k]]
        vecs = emb.embed_documents(flat)
        cv[s], i = {}, 0
        for k in keys:
            n = len(chunk_text[s][k])
            cv[s][k], i = vecs[i:i + n], i + n
    print(f"Embedded; {emb.model_calls} new texts sent to the model", flush=True)

    # 3. Each article vs its own baseline version.
    base = cfg["periods"]["baseline"]["name"]
    later = [p["name"] for p in cfg["periods"]["later"]]
    default = ch["size_tokens"]
    by_key = {(r.page_id, r.period): r for r in snaps.itertuples(index=False)}
    rows = []
    for (pid, period), r in by_key.items():
        if period == base:
            continue
        b = by_key.get((pid, base))
        if b is None:
            continue                                    # a new document since the baseline
        cutoff = len(trunc_text[(pid, period)].split()) - len(r.title.split())
        true, beyond = longform.changed_share(b.text, r.text, cutoff)
        row = {"page_id": pid, "title": r.title, "group": r.group, "event": r.event, "period": period,
               "true_changed": true, "beyond_512": beyond, "tokens": r.tokens,
               "truncated": longform.cosine_change(tv[(pid, base)], tv[(pid, period)]),
               "averaged": longform.cosine_change(cv[default][(pid, base)].mean(axis=0),
                                                  cv[default][(pid, period)].mean(axis=0))}
        for s in sizes:
            mean, mx = longform.chunk_change(cv[s][(pid, base)], cv[s][(pid, period)])
            row[f"chunks_{s}"], row[f"chunks_{s}_max"] = mean, mx
        rows.append(row)
    table = pd.DataFrame(rows)
    methods = ["truncated", "averaged"] + [f"chunks_{s}" for s in sizes] + [f"chunks_{s}_max" for s in sizes]
    ev = longform.evaluate(table, methods, lf["big_change"])
    new_docs = sorted(set(snaps.loc[snaps["period"] != base, "page_id"]) - set(snaps.loc[snaps["period"] == base, "page_id"]))

    by_period = (table.assign(set=np.where(table["event"], "event set", "vital sample"))
                 .groupby(["set", "period"])[["true_changed", "beyond_512"] + methods].mean().round(4).reset_index())

    # 4. Archive-level drift: the same MMD test on article vectors vs chunk vectors.
    rng = np.random.default_rng(int(cfg["seed"]))
    def stack(vecs: dict, period: str) -> np.ndarray:
        return np.vstack([v if v.ndim == 2 else v[None, :] for (p, per), v in vecs.items() if per == period])
    mmd_rows = []
    for name, vecs in [("article (truncated)", tv), (f"chunks_{default}", cv[default])]:
        x_all = stack(vecs, base)
        gamma = median_heuristic_gamma(x_all, seed=int(cfg["seed"]))
        for period in later:
            x, y = x_all, stack(vecs, period)
            n = lf["mmd_sample"]
            if len(x) > n:
                x = x[rng.choice(len(x), n, replace=False)]
            if len(y) > n:
                y = y[rng.choice(len(y), n, replace=False)]
            res = mmd_permutation_test(x, y, gamma, lf["mmd_permutations"], seed=int(cfg["seed"]))
            mmd_rows.append({"representation": name, "period": period, "n_baseline": len(x), "n_period": len(y),
                             "mmd2": round(res.mmd2, 5), "p_value": round(res.p_value, 4)})
    mmd = pd.DataFrame(mmd_rows)

    out = resolve_path(cfg["paths"]["runs"]) / "longform"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "article_change.csv", index=False)
    ev.to_csv(out / "chunking_eval.csv", index=False)
    by_period.to_csv(out / "by_period.csv", index=False)
    mmd.to_csv(out / "archive_mmd.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n{len(table)} article x period pairs; {len(new_docs)} articles are new since the baseline.")
    print(f"Pairs with most of the change beyond the first 512 tokens: {(table['beyond_512'] >= 0.5).mean():.0%}")
    print(f"Pairs with a big change (>= {lf['big_change']:.0%} of the text new or rewritten): "
          f"{(table['true_changed'] >= lf['big_change']).mean():.0%}\n")
    print("How well each method follows the true change:")
    print(ev.round(3).to_string(index=False))
    print("\nBy period (means):")
    print(by_period.to_string(index=False))
    print("\nArchive-level MMD vs the baseline:")
    print(mmd.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
