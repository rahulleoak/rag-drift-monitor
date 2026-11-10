"""The outdated-answer test: do flagged changed facts change the answer? (D27, D28)

    python scripts/outdated_answers.py --estimate   # no LLM spend: print the cost
    python scripts/outdated_answers.py              # run it (Batch API; three stages)
    python scripts/outdated_answers.py --live       # same, live calls (faster, twice the price)

Needs ``scripts/fetch_wikipedia.py`` first (snapshots and chunks). Writes to
runs/longform/:
    outdated_items.csv     every sampled item: sentences, question, both answers, verdict
    outdated_summary.csv   per group: how often the answer changed, with an interval
    outdated_verdict.json  whether the changed-facts check is useful, by the fixed thresholds
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
from driftmon.rag.embeddings import Embedder, document_text  # noqa: E402
from driftmon.evaluation import outdated as od  # noqa: E402
from driftmon.rag.index import VectorIndex  # noqa: E402
from driftmon.llm.client import LLMClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: print the cost")
    mode.add_argument("--live", action="store_true", help="live calls instead of the Batch API")
    args = parser.parse_args(argv)
    cfg = load_config()
    lf, o = cfg["longform"], cfg["longform"]["outdated"]
    processed = resolve_path(cfg["paths"]["data_processed"])
    out = resolve_path(cfg["paths"]["runs"]) / "longform"
    snaps = pd.read_parquet(processed / "wiki_snapshots.parquet")
    chunks = pd.read_parquet(processed / "wiki_chunks.parquet")
    order = [lf["reference_period"]["name"], cfg["periods"]["baseline"]["name"],
             *[p["name"] for p in cfg["periods"]["later"]]]

    items = od.edit_items(snaps, order, lf["fact_min_similarity"])
    print(f"Edited sentences with a number: {items['group'].value_counts().to_dict()}")
    items = od.sample_items(items, int(o["items_per_group"]), int(cfg["seed"]))
    batch = not args.live
    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv", log=lambda m: print(m, flush=True))

    # 1. Questions from the new sentence (the writer never sees the group or the old sentence).
    q_reqs = od.question_requests(items, cfg)
    if args.estimate:
        cached = [r for r in q_reqs if llm.cached(r) is not None]
        q_cost = llm.estimate([r for r in q_reqs if llm.cached(r) is None], batch)
        # answers and the judge, priced with the sentence standing in for the question
        questions = items["after"].tolist()
    else:
        q_res = llm.complete_many(q_reqs, batch=batch)
        items["question"] = [r.parsed.question if r.ok else None for r in q_res]
        items["question_answer"] = [r.parsed.answer if r.ok else None for r in q_res]
        items = items.dropna(subset=["question"]).reset_index(drop=True)
        questions = items["question"].tolist()

    # 2. Retrieve from each version's chunk index (every article of that period; local, no LLM).
    emb = Embedder.from_config(cfg)
    chunks["embed_text"] = [document_text(f"{t} - {s}", x) for t, s, x in
                            zip(chunks["title"], chunks["section"], chunks["text"])]
    needed = sorted(set(items["previous"]) | set(items["period"]))
    indexes, by_period = {}, {}
    for p in needed:
        c = chunks[chunks["period"] == p].reset_index(drop=True)
        by_period[p] = c.set_index("chunk_id")
        indexes[p] = VectorIndex(c["chunk_id"].tolist(), emb.embed_documents(c["embed_text"].tolist()))
    qv = emb.embed_queries(questions)
    k = int(o["top_k"])
    passages = {side: [] for side in od.SIDES}
    for i, r in enumerate(items.to_dict("records")):
        for side, period, sentence in (("old", r["previous"], r["before"]), ("new", r["period"], r["after"])):
            ids, _ = indexes[period].search(qv[i:i + 1], k)
            top = by_period[period].loc[list(ids[0])].reset_index()
            passages[side].append(od.passage_listing(top))
            items.loc[i, f"edit_retrieved_{side}"] = od.contains_sentence(sentence, top["text"].tolist())

    # 3. Answers from each version, then the judge: same fact or not?
    a_reqs = {side: od.answer_requests(questions, passages[side], cfg) for side in od.SIDES}
    if args.estimate:
        a_cost = llm.estimate([r for s in od.SIDES for r in a_reqs[s] if llm.cached(r) is None], batch)
        placeholder = "It was about 8,000 in 2019, according to passage A."
        j_reqs = od.change_requests(items["item_id"], questions, [placeholder] * len(items),
                                    [placeholder] * len(items), cfg)
        j_cost = llm.estimate(j_reqs, batch)
        print(f"Estimate: questions ${q_cost:.2f} ({len(q_reqs) - len(cached)} uncached), answers ${a_cost:.2f}, "
              f"judge ${j_cost:.2f}; total about ${q_cost + a_cost + j_cost:.2f} "
              f"({'batch' if batch else 'live'} price). Spent so far ${llm.spent():.2f}.")
        print(f"Edited sentence among the retrieved passages (sentence as query): "
              f"old {items['edit_retrieved_old'].mean():.0%}, new {items['edit_retrieved_new'].mean():.0%}")
        return 0
    flat = a_reqs["old"] + a_reqs["new"]
    res = llm.complete_many(flat, batch=batch)
    n = len(items)
    for side, part in (("old", res[:n]), ("new", res[n:])):
        items[f"answer_{side}"] = [r.parsed.answer if r.ok else None for r in part]
        items[f"found_{side}"] = [r.parsed.found if r.ok else None for r in part]
    ok = items["answer_old"].notna() & items["answer_new"].notna()
    j_reqs = od.change_requests(items.loc[ok, "item_id"], items.loc[ok, "question"], items.loc[ok, "answer_old"],
                                items.loc[ok, "answer_new"], cfg)
    j_res = llm.complete_many(j_reqs, batch=batch)
    items.loc[ok, "verdict"] = [r.parsed.verdict if r.ok else None for r in j_res]
    items.loc[ok, "judge_reason"] = [r.parsed.reason if r.ok else None for r in j_res]

    summary = od.summarise(items, cfg)
    result = od.verdict(items, cfg)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "outdated_items.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "outdated_summary.csv", index=False)
    (out / "outdated_verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"Verdict: {result}")
    print(f"Failed: {int((~ok).sum())} items without both answers; "
          f"{int(items.loc[ok, 'verdict'].isna().sum())} without a judgement. Spent so far ${llm.spent():.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
