"""Apply Agent E's targeted re-index and check it worked (D32).

    python scripts/reindex_check.py --estimate   # no LLM spend: strata, work saved, cost
    python scripts/reindex_check.py              # run it (Batch API; three stages)
    python scripts/reindex_check.py --live       # same, live calls (faster, twice the price)

Needs scripts/fetch_wikipedia.py and scripts/longform_drift.py first (snapshots,
chunks, fact_changes.csv). Writes to runs/longform/:
    reindex_items.csv     every sampled edit: sentences, question, three answers, verdicts
    reindex_summary.csv   per stratum and weighted: outdated answers, stale vs targeted
    reindex_verdict.json  is the targeted re-index enough, by the fixed thresholds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.evaluation import outdated as od  # noqa: E402
from driftmon.evaluation import reindex as rx  # noqa: E402
from driftmon.llm.client import LLMClient  # noqa: E402
from driftmon.rag.embeddings import Embedder, document_text  # noqa: E402
from driftmon.rag.index import VectorIndex  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: strata, work saved and the cost")
    mode.add_argument("--live", action="store_true", help="live calls instead of the Batch API")
    args = parser.parse_args(argv)
    cfg = load_config()
    lf, r = cfg["longform"], cfg["longform"]["reindex"]
    baseline, latest = cfg["periods"]["baseline"]["name"], r["latest"]
    processed = resolve_path(cfg["paths"]["data_processed"])
    out = resolve_path(cfg["paths"]["runs"]) / "longform"
    snaps = pd.read_parquet(processed / "wiki_snapshots.parquet")
    chunks = pd.read_parquet(processed / "wiki_chunks.parquet")
    listed = rx.listed_pages(pd.read_csv(out / "fact_changes.csv"), baseline)

    all_items = rx.edit_strata(snaps, baseline, latest, listed, lf["fact_min_similarity"])
    population = all_items["stratum"].value_counts().to_dict()
    items = rx.sample_strata(all_items, r["per_stratum"], int(cfg["seed"]))
    tables = rx.index_chunks(chunks, baseline, latest, listed)
    effort = rx.work(tables)
    print(f"Listed articles: {len(listed)}. Edits {baseline} -> {latest} per stratum: {population}; "
          f"sampled {items['stratum'].value_counts().to_dict()}")
    print(f"Work: the targeted re-index re-embeds {effort['passages_targeted']:,} of "
          f"{effort['passages_full']:,} passages ({effort['targeted_share']:.0%})")

    batch = not args.live
    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv", log=lambda m: print(m, flush=True))

    # 1. Questions from the latest sentence (F21's prompt; the writer never sees the old one).
    q_reqs = od.question_requests(items, cfg)
    if args.estimate:
        q_cost = llm.estimate([q for q in q_reqs if llm.cached(q) is None], batch)
        a_cost = 3 * llm.estimate(od.answer_requests(items["after"].tolist(), ["x " * 600] * len(items), cfg), batch)
        ph = ["It was about 8,000 in 2019, according to passage A."] * len(items)
        j_cost = 2 * llm.estimate(od.change_requests(items["item_id"], items["after"], ph, ph, cfg), batch)
        print(f"Estimate: questions ${q_cost:.2f}, answers at most ${a_cost:.2f}, judge at most ${j_cost:.2f}; "
              f"total at most ${q_cost + a_cost + j_cost:.2f} ({'batch' if batch else 'live'} price). "
              f"Spent so far ${llm.spent():.2f}.")
        return 0
    q_res = llm.complete_many(q_reqs, batch=batch)
    items["question"] = [x.parsed.question if x.ok else None for x in q_res]
    items["question_answer"] = [x.parsed.answer if x.ok else None for x in q_res]
    items = items.dropna(subset=["question"]).reset_index(drop=True)
    questions = items["question"].tolist()

    # 2. Retrieve from each index (local, no LLM).
    emb = Embedder.from_config(cfg)
    qv = emb.embed_queries(questions)
    k = int(lf["outdated"]["top_k"])
    passages = {}
    for name in rx.INDEXES:
        c = tables[name].copy()
        c["embed_text"] = [document_text(f"{t} - {s}", x) for t, s, x in zip(c["title"], c["section"], c["text"])]
        index = VectorIndex(c["chunk_id"].tolist(), emb.embed_documents(c["embed_text"].tolist()))
        by_id = c.set_index("chunk_id")
        ids, _ = index.search(qv, k)
        passages[name] = []
        for i, row in enumerate(ids):
            top = by_id.loc[list(row)].reset_index()
            passages[name].append(od.passage_listing(top))
            if name != "targeted":
                sentence = items.loc[i, "before" if name == "stale" else "after"]
                items.loc[i, f"edit_retrieved_{name}"] = od.contains_sentence(sentence, top["text"].tolist())

    # 3. Answers from each index, then the judge: stale and targeted against full.
    a_reqs = {name: od.answer_requests(questions, passages[name], cfg) for name in rx.INDEXES}
    flat = [q for name in rx.INDEXES for q in a_reqs[name]]
    res = llm.complete_many(flat, batch=batch)
    n = len(items)
    for j, name in enumerate(rx.INDEXES):
        part = res[j * n:(j + 1) * n]
        items[f"answer_{name}"] = [x.parsed.answer if x.ok else None for x in part]
        items[f"found_{name}"] = [x.parsed.found if x.ok else None for x in part]
        items[f"same_passages_{name}"] = [p == f for p, f in zip(passages[name], passages["full"])]
    for name in ("stale", "targeted"):
        ok = items[f"answer_{name}"].notna() & items["answer_full"].notna()
        same = ok & items[f"same_passages_{name}"]  # same passages -> the same cached answer: no judge needed
        items.loc[same, f"verdict_{name}"] = "same"
        ask = ok & ~same
        j_reqs = od.change_requests(items.loc[ask, "item_id"], items.loc[ask, "question"],
                                    items.loc[ask, f"answer_{name}"], items.loc[ask, "answer_full"], cfg)
        j_res = llm.complete_many(j_reqs, batch=batch) if j_reqs else []
        items.loc[ask, f"verdict_{name}"] = [x.parsed.verdict if x.ok else None for x in j_res]
        items.loc[ask, f"reason_{name}"] = [x.parsed.reason if x.ok else None for x in j_res]
        items[f"outdated_{name}"] = rx.outdated(items[f"verdict_{name}"])

    summary = rx.summarise(items, population)
    by_stratum = {r_["stratum"]: {"items": int(r_["items"]), "outdated_stale": float(r_["outdated_stale"]),
                                  "outdated_targeted": float(r_["outdated_targeted"])}
                  for r_ in summary.to_dict("records") if r_["stratum"] in rx.STRATA}
    result = {**rx.verdict(items, population, cfg), **effort, "listed_articles": len(listed),
              "population": population, "by_stratum": by_stratum}
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "reindex_items.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "reindex_summary.csv", index=False)
    (out / "reindex_verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(json.dumps(result, indent=2))
    print(f"Spent so far ${llm.spent():.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
