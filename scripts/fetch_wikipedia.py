"""Download the long-form set (Wikipedia at period-end snapshots), clean it and chunk it (D26).

    python scripts/fetch_wikipedia.py

Writes:
    data/processed/wiki_snapshots.parquet   one row per article x period (clean sections)
    data/processed/wiki_chunks.parquet      one row per chunk (default chunk size)
    runs/longform/summary.csv               per period: articles, length, chunks, unchanged chunks
    runs/longform/articles.csv              the sampled articles and why each was picked

Every API response is cached under data/raw/wikipedia/api, so a rerun makes
no requests. No LLM calls.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon import chunking, wikipedia  # noqa: E402
from driftmon.chunking import token_counter  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402


def snapshot_times(cfg) -> list[tuple[str, str]]:
    """(period name, ISO end time) for the baseline and each later period."""
    periods = [cfg["periods"]["baseline"], *cfg["periods"]["later"]]
    return [(p["name"], f"{p['end']}T23:59:59Z") for p in periods]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    lf = cfg["longform"]
    client = wikipedia.WikiClient(lf["api"], lf["user_agent"], resolve_path(cfg["paths"]["data_raw"]) / "wikipedia" / "api",
                                  lf["request_pause_s"])

    # 1. Which articles: a stratified sample of the vital list, plus the event set.
    by_section = wikipedia.vital_titles(client.page_text(lf["vital_list"]))
    print(f"Vital list: {sum(map(len, by_section.values()))} titles in {len(by_section)} sections")
    picks = wikipedia.sample_titles(by_section, lf["per_section"], int(cfg["seed"]))
    events = set(lf["event_titles"])
    rows = [{"requested": t, "group": s, "event": t in events} for s, t in picks]
    rows += [{"requested": t, "group": "event", "event": True} for t in lf["event_titles"]
             if t not in {r["requested"] for r in rows}]
    ids = client.page_ids([r["requested"] for r in rows])
    articles = []
    for r in rows:
        hit = ids[r["requested"]]
        if hit is None:
            print(f"  not found, skipped: {r['requested']}")
            continue
        articles.append({**r, "page_id": hit[0], "title": hit[1]})
    articles = pd.DataFrame(articles).drop_duplicates("page_id")
    print(f"Articles: {len(articles)} ({int(articles['event'].sum())} in the event set)")

    # 2. Each article at the end of each period.
    count = token_counter(cfg["embeddings"]["model"])
    ch = lf["chunking"]
    snaps, chunks = [], []
    times = snapshot_times(cfg)
    for n, a in enumerate(articles.itertuples(index=False), 1):
        for period, ts in times:
            rev = client.revision_at(a.page_id, ts)
            if rev is None or wikipedia.is_redirect(rev["content"]):
                continue
            secs = wikipedia.sections(rev["content"], lf["skip_sections"])
            text = "\n\n".join(t for _, t in secs)
            words = len(text.split())
            if words < lf["min_words"]:
                continue
            snaps.append({"page_id": a.page_id, "title": a.title, "group": a.group, "event": a.event,
                          "period": period, "revid": rev["revid"], "rev_timestamp": rev["timestamp"],
                          "words": words, "tokens": count(text), "sections": json.dumps(secs, ensure_ascii=False),
                          "text": text})
            for i, c in enumerate(chunking.chunk_document(secs, count, ch["size_tokens"], ch["overlap_tokens"],
                                                           ch["min_tokens"])):
                chunks.append({"chunk_id": f"w{a.page_id}:{period}:{i}", "page_id": a.page_id, "title": a.title,
                               "group": a.group, "event": a.event, "period": period, "chunk_index": i,
                               "section": c.section, "text": c.text, "tokens": c.tokens,
                               "chunk_hash": hashlib.sha256(c.text.encode()).hexdigest()[:16]})
        if n % 20 == 0:
            print(f"  {n} of {len(articles)} articles ({client.requests} requests so far)", flush=True)

    snaps, chunks = pd.DataFrame(snaps), pd.DataFrame(chunks)
    out = resolve_path(cfg["paths"]["data_processed"])
    snaps.to_parquet(out / "wiki_snapshots.parquet", index=False)
    chunks.to_parquet(out / "wiki_chunks.parquet", index=False)

    # 3. Summary: how long, how much is cut off, how much changes.
    order = [p for p, _ in times]
    seen: dict[int, set[str]] = {}
    summary = []
    for period in order:
        s, c = snaps[snaps["period"] == period], chunks[chunks["period"] == period]
        unchanged = [h in seen.get(pid, set()) for pid, h in zip(c["page_id"], c["chunk_hash"])]
        summary.append({"period": period, "articles": len(s), "event_articles": int(s["event"].sum()),
                        "median_words": int(s["words"].median()), "median_tokens": int(s["tokens"].median()),
                        "share_over_512_tokens": round(float((s["tokens"] > 512).mean()), 3),
                        "share_of_text_beyond_512": round(float((1 - (512 / s["tokens"]).clip(upper=1)).mean()), 3),
                        "chunks": len(c), "median_chunks_per_article": int(c.groupby("page_id").size().median()),
                        "chunks_unchanged_since_previous": round(float(pd.Series(unchanged).mean()), 3)
                        if period != order[0] else None})
        seen = {pid: set(g["chunk_hash"]) for pid, g in c.groupby("page_id")}
    summary = pd.DataFrame(summary)
    runs = resolve_path(cfg["paths"]["runs"]) / "longform"
    runs.mkdir(parents=True, exist_ok=True)
    summary.to_csv(runs / "summary.csv", index=False)
    articles.to_csv(runs / "articles.csv", index=False)
    print(f"\n{client.requests} API requests this run\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
