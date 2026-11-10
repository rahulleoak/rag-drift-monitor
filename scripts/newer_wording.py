"""The newer-wording test: does a question in later words still find an older article? (D31)

    python scripts/newer_wording.py --estimate   # no LLM spend: sample sizes and cost
    python scripts/newer_wording.py              # run it (Batch API; two stages)
    python scripts/newer_wording.py --live       # same, live calls (faster, twice the price)

Needs the question set (scripts/generate_questions.py) and a pipeline run (for
each section's drift status). Writes to runs/wording/:
    items.csv      every sampled question: both rewordings, checks, ranks
    summary.csv    per drift status and overall: how often each version finds the article
    verdict.json   the two answers, by the thresholds fixed before results
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
from driftmon.dashboard.runs import list_runs, load_run  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.data.queries import load_questions  # noqa: E402
from driftmon.evaluation import wording as wd  # noqa: E402
from driftmon.evaluation.retrieval import drift_status  # noqa: E402
from driftmon.llm.client import LLMClient  # noqa: E402
from driftmon.rag.embeddings import Embedder  # noqa: E402
from driftmon.rag.index import VectorIndex  # noqa: E402

SEARCH_DEPTH = 100  # ranks beyond this count as not found (reciprocal rank 0)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: print sample sizes and the cost")
    mode.add_argument("--live", action="store_true", help="live calls instead of the Batch API")
    args = parser.parse_args(argv)
    cfg = load_config()
    s = wd.settings(cfg)
    baseline = cfg["periods"]["baseline"]["name"]
    out = resolve_path(cfg["paths"]["runs"]) / "wording"

    runs_dir = resolve_path(cfg["paths"]["runs"])
    run_ids = list_runs(runs_dir)
    if not run_ids:
        raise SystemExit("No pipeline run found; run scripts/run_pipeline.py first.")
    drift = load_run(runs_dir, run_ids[0]).drift
    questions = load_questions(cfg)
    if questions is None:
        raise SystemExit("No question set found; run scripts/generate_questions.py first.")
    questions["drift_status"] = [drift_status(p, c, baseline, drift.drifted_segments, drift.control_segments)
                                 for p, c in zip(questions["period"], questions["category"])]
    items = wd.sample_questions(questions, cfg)
    print(f"Drift status from run {run_ids[0]}. Sampled questions: {items['group'].value_counts().to_dict()}")

    batch = not args.live
    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv", log=lambda m: print(m, flush=True))

    # 1. Sonnet: a same-era rewording and a version in later wording.
    r_reqs = wd.rewording_requests(items, cfg)
    if args.estimate:
        r_cost = llm.estimate([r for r in r_reqs if llm.cached(r) is None], batch)
        probe = items.assign(same_era=items["question"], newer=items["question"])
        m_cost = llm.estimate(wd.meaning_requests(probe, cfg)[0], batch)
        print(f"Estimate: rewordings ${r_cost:.2f}, meaning check ${m_cost:.2f} (if every question has a newer "
              f"version); total at most ${r_cost + m_cost:.2f} ({'batch' if batch else 'live'} price). "
              f"Spent so far ${llm.spent():.2f}.")
        return 0
    res = llm.complete_many(r_reqs, batch=batch)
    for col in ("same_era", "has_newer", "newer", "reason"):
        items[col] = [getattr(r.parsed, col) if r.ok else None for r in res]
    items["changed_words"] = [", ".join(r.parsed.changed_words) if r.ok else None for r in res]

    # 2. Code: the newer version must add a word that became more common by the query period.
    docs, vectors = load_corpus(cfg)
    counts = wd.period_counts(docs)
    terms = {p: wd.newer_terms(counts, p, s["query_period"], float(s["min_z"]), float(s["min_rate_change"]))
             for p in s["question_periods"]}
    items = wd.code_checks(items, terms)
    passed = items["rejected_by"] == ""
    print(f"Code checks: {int(passed.sum())} of {len(items)} pass; reasons: "
          f"{items.loc[~passed, 'rejected_by'].str.split('; ').explode().value_counts().to_dict()}")

    # 3. Opus: do both versions still ask the same thing as the original?
    m_reqs, flips = wd.meaning_requests(items[passed], cfg)
    m_res = llm.complete_many(m_reqs, batch=batch)
    checks = wd.unshuffle([r.parsed if r.ok else None for r in m_res], flips)
    checks.index = items.index[passed]
    items = items.join(checks)
    meaning_ok = items["same_era_same"].eq(True) & items["newer_same"].eq(True)
    items.loc[passed & ~meaning_ok, "rejected_by"] = "meaning changed (judge)"
    kept_mask = items["rejected_by"] == ""

    # 4. Search the query period's refreshed index (every article up to it); known item, no LLM.
    order = [baseline, *[p["name"] for p in cfg["periods"]["later"]]]
    upto = set(order[:order.index(s["query_period"]) + 1])
    mask = docs["period"].isin(upto).to_numpy()
    index = VectorIndex(docs.loc[mask, "doc_id"].tolist(), vectors[mask])
    emb = Embedder.from_config(cfg)
    kept = items[kept_mask].reset_index(drop=True)
    ranked = {}
    for v in wd.VERSIONS:
        col = "question" if v == "original" else v
        ids, _ = index.search(emb.embed_queries(kept[col].tolist()), SEARCH_DEPTH)
        ranked[v] = [list(x) for x in ids]
    kept = wd.add_ranks(kept, ranked, int(s["hit_k"]))
    items = items.merge(kept[["question_id", *[c for c in kept.columns if c.startswith(("rank_", "found_", "rr_"))]]],
                        on="question_id", how="left")

    summary = wd.summarise(kept, cfg)
    result = {**wd.verdict(kept, cfg), "drift_run": run_ids[0], "sampled": int(len(items))}
    out.mkdir(parents=True, exist_ok=True)
    keep_cols = ["question_id", "group", "period", "category", "doc_id", "title", "question", "same_era", "newer",
                 "changed_words", "newer_words", "reason", "rejected_by", "same_era_same", "newer_same",
                 "meaning_reason", *[c for c in items.columns if c.startswith(("rank_", "found_", "rr_"))]]
    items[keep_cols].to_csv(out / "items.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(json.dumps(result, indent=2))
    failed = sum(not r.ok for r in res) + sum(not r.ok for r in m_res)
    print(f"Kept {len(kept)} of {len(items)}; {failed} failed LLM calls. Spent so far ${llm.spent():.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
