"""Keyword search (BM25) next to meaning-based search: does wording drift hurt it more? (D40)

    python scripts/keyword_search.py             # no LLM calls; a few minutes
    python scripts/keyword_search.py --run ID    # drift kinds from another pipeline run

Same articles, same questions, same index per question as the pipeline's
judged comparison; only the search changes. Agent B's drift kind per section
comes from a saved pipeline run (default: the newest normal run), and the
newer-wording questions from runs/wording/items.csv (scripts/newer_wording.py),
if present. Check built in: the meaning-based ranks computed here must
reproduce the run's saved top 10, or the script stops.
Writes to runs/keyword/:
    per_question.csv   both ranks per question, with its drift kind
    summary.csv        per drift kind: how often each search finds the article
    wording.csv        the newer-wording questions, both versions, both searches
    verdict.json       the two answers, by the thresholds fixed before results
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
from driftmon.dashboard.runs import default_run, list_runs  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.data.periods import study_periods  # noqa: E402
from driftmon.data.queries import load_questions  # noqa: E402
from driftmon.evaluation import keyword as kw  # noqa: E402
from driftmon.evaluation.retrieval import judged_pair  # noqa: E402
from driftmon.evaluation.truth_table import parse_ids  # noqa: E402
from driftmon.rag.bm25 import BM25Index  # noqa: E402
from driftmon.rag.embeddings import Embedder, document_text  # noqa: E402
from driftmon.rag.index import build_study_indexes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="pipeline run id for the drift kinds (default: the newest normal run)")
    args = parser.parse_args(argv)
    cfg = load_config()
    s = cfg["keyword_search"]
    depth, hit_k = int(s["depth"]), int(s["hit_k"])
    runs_dir = resolve_path(cfg["paths"]["runs"])
    run_id = args.run or default_run(runs_dir, list_runs(runs_dir))
    if not run_id:
        raise SystemExit("No pipeline run found; run scripts/run_pipeline.py first.")
    periods = study_periods(cfg)
    base, later = periods[0].name, [p.name for p in periods[1:]]

    docs, vectors = load_corpus(cfg)
    texts = [document_text(t, x) for t, x in zip(docs["title"], docs["text"])]
    indexes = build_study_indexes(docs, vectors, base, later)
    emb = Embedder.from_config(cfg)
    questions = load_questions(cfg)
    q = questions[questions["set"] == "answerable"].reset_index(drop=True)
    q["index"] = [judged_pair(p, base, later)[1] for p in q["period"]]
    print(f"Run {run_id}: {len(q)} answerable questions; {len(docs)} articles.", flush=True)

    keyword: dict[str, BM25Index] = {}

    def keyword_index(name: str) -> BM25Index:
        if name not in keyword:
            mask = docs["period"].isin(set([base, *later[:later.index(name) + 1]])).to_numpy()
            keyword[name] = BM25Index(docs.loc[mask, "doc_id"].tolist(), [t for t, m in zip(texts, mask) if m],
                                      float(s["k1"]), float(s["b"]))
            print(f"  keyword index {name}: {len(keyword[name])} articles", flush=True)
        return keyword[name]

    # 1. Every question, both searches, on its judged refreshed index.
    qvecs = emb.embed_queries(q["question"].tolist())
    r_mean, r_key, saved_top = [None] * len(q), [None] * len(q), {}
    for name, g in q.groupby("index"):
        period = name[len("refreshed_"):]
        rows = g.index.to_numpy()
        ids, _ = indexes.refreshed[period].search(qvecs[rows], depth)
        for i, rk in zip(rows, kw.ranks(ids, q.loc[rows, "doc_id"])):
            r_mean[i] = rk
        saved_top.update({q.at[i, "question_id"]: list(d[:10]) for i, d in zip(rows, ids)})
        ids, sc = keyword_index(period).search(q.loc[rows, "question"].tolist(), depth)
        for i, rk in zip(rows, kw.ranks(ids, q.loc[rows, "doc_id"], sc)):
            r_key[i] = rk

    runs = pd.read_csv(runs_dir / run_id / "retrieval_runs.csv")
    runs = runs[runs["question_id"].isin(q["question_id"])]
    want = {(qq, i) for qq, i in zip(q["question_id"], q["index"])}
    saved = {qq: parse_ids(d)[:10] for qq, i, d in zip(runs["question_id"], runs["index"], runs["doc_ids"])
             if (qq, i) in want}
    mismatch = sum(saved.get(qq) != saved_top[qq] for qq in q["question_id"])
    if mismatch:
        raise SystemExit(f"{mismatch} questions' meaning-based top 10 differ from run {run_id}: not the same "
                         f"setup. Stopping.")
    print(f"Check: the meaning-based top 10 matches run {run_id} for all {len(q)} questions.", flush=True)

    segments = pd.read_csv(runs_dir / run_id / "drift_segments.csv")
    pq = kw.per_question(q, r_mean, r_key, kw.drift_kinds(segments, base), base, hit_k)
    pq["index"] = q["index"]

    # 2. The newer-wording questions (D31): both versions, both searches, on the query period's index.
    wording = None
    items_path = runs_dir / "wording" / "items.csv"
    if items_path.exists():
        items = pd.read_csv(items_path)
        wording = items[items["rejected_by"].isna() | (items["rejected_by"] == "")].reset_index(drop=True)
        qp = cfg["evaluation"]["newer_wording"]["query_period"]
        for version in ("same_era", "newer"):
            texts_v = wording[version].tolist()
            ids, _ = indexes.refreshed[qp].search(emb.embed_queries(texts_v), depth)
            rm_ = kw.ranks(ids, wording["doc_id"])
            ids, sc = keyword_index(qp).search(texts_v, depth)
            rk_ = kw.ranks(ids, wording["doc_id"], sc)
            for name, r in (("meaning", rm_), ("keyword", rk_)):
                wording[f"rank_{version}_{name}"] = r
                wording[f"found_{version}_{name}"] = [0 < x <= hit_k for x in r]
        old = wording[["rank_same_era", "rank_newer"]].fillna(0).astype(int)
        if not ((old["rank_same_era"] == wording["rank_same_era_meaning"]).all()
                and (old["rank_newer"] == wording["rank_newer_meaning"]).all()):
            print("Note: meaning-based ranks differ from runs/wording/items.csv.", flush=True)

    summary = kw.summarise(pq)
    result = {"run": run_id, **kw.verdict(pq, wording, cfg)}
    out = runs_dir / "keyword"
    out.mkdir(parents=True, exist_ok=True)
    pq.to_csv(out / "per_question.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False)
    if wording is not None:
        keep = ["question_id", "group", "period", "category", "doc_id", "question", "same_era", "newer",
                "changed_words", *[c for c in wording.columns if c.endswith(("_meaning", "_keyword"))]]
        wording[keep].to_csv(out / "wording.csv", index=False, encoding="utf-8-sig")
    (out / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
