"""Run the LangGraph pipeline and save every agent's report to ``runs/<run_id>/``.

Built so far: Agent A (Data Monitoring) -> Agent B (Drift Detection) -> the
routing decision -> Agent C (Retrieval Evaluation).

Writes:
    monitoring.json    Agent A's MonitoringReport
    drift.json         Agent B's DriftReport
    drift_whole.csv    whole-period drift with bands (one row per comparison)
    drift_segments.csv period x category drift with bands
    topics.csv         emerging topics across sections
    term_shifts.csv    term-context shifts
    correlations.csv   how the drift measures correlate across segments (P6)
    query_drift.csv    each period's queries vs the baseline queries (if a question set exists)
    route.txt          the routing decision after Agent B
    retrieval.json     Agent C's RetrievalReport
    retrieval_*.csv    Agent C's per-question tables (judged, known item, ranking change, runs)

Usage (from the repository root):
    python scripts/run_pipeline.py --estimate   # everything except the judge; prints its cost
    python scripts/run_pipeline.py              # including the relevance judge (LLM spend)
    python scripts/run_pipeline.py --no-judge   # Agent C without the judged comparison

The judge runs as a batch (half price). If the run is interrupted while it
waits, run the same command again: it collects the batch already submitted.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_pipeline.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.corpus import load_corpus  # noqa: E402
from driftmon.embeddings import Embedder  # noqa: E402
from driftmon.graph import build_graph  # noqa: E402
from driftmon.llm import LLMClient, LLMResult  # noqa: E402
from driftmon.queries import load_questions, query_stream  # noqa: E402

SHOW = ["segment_id", "times_ordinary_embedding", "times_ordinary_lexical", "embedding_band",
        "lexical_band", "kind", "flagged", "emerging_terms"]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: print the judge's cost")
    mode.add_argument("--no-judge", action="store_true", help="skip the LLM relevance judge")
    args = parser.parse_args(argv)
    config = load_config()
    run_dir = resolve_path(config["paths"]["runs"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs, vectors = load_corpus(config)
    embedder = Embedder.from_config(config)
    questions = load_questions(config)
    stream = query_stream(config, questions, embedder)
    queries, query_vectors = stream if stream else (None, None)
    print(f"Query stream: {len(queries)} questions" if stream is not None else
          "Query stream: no question set yet (query drift skipped)")
    question_vectors = embedder.embed_queries(questions["question"].tolist()) if questions is not None else None
    llm = None
    if questions is not None and not args.no_judge:
        llm = LLMClient(config=config["llm"], cache_dir=resolve_path(config["paths"]["cache"]) / "llm",
                        ledger=resolve_path(config["paths"]["runs"]) / "llm_costs.csv",
                        log=lambda m: print(m, flush=True),
                        client=_NoCalls() if args.estimate else None)
        if args.estimate:
            llm = _EstimateOnly(llm)
    state = build_graph().invoke({"config": config, "docs": docs, "vectors": vectors,
                                  "queries": queries, "query_vectors": query_vectors,
                                  "questions": questions, "question_vectors": question_vectors, "llm": llm})

    mon, drift = state["monitoring"], state["drift"]
    (run_dir / "monitoring.json").write_text(mon.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "drift.json").write_text(drift.model_dump_json(indent=2), encoding="utf-8")
    whole = pd.DataFrame([s.model_dump() for s in drift.whole_period])
    segs = pd.DataFrame([s.model_dump() for s in drift.segments])
    whole.to_csv(run_dir / "drift_whole.csv", index=False)
    segs.to_csv(run_dir / "drift_segments.csv", index=False)
    (run_dir / "route.txt").write_text(state["next_step"] + "\n", encoding="utf-8")
    topics = pd.DataFrame(drift.topics)
    shifts = pd.DataFrame(drift.term_shifts)
    corr = pd.DataFrame(drift.metric_correlations)
    topics.to_csv(run_dir / "topics.csv", index=False)
    shifts.to_csv(run_dir / "term_shifts.csv", index=False)
    corr.to_csv(run_dir / "correlations.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Agent A: periods ===")
    print(pd.DataFrame(mon.period_summary)[["period", "documents", "median_words", "categories",
                                           "new_sections", "new_section_share", "mix_cramers_v",
                                           "new_term_rate"]].round(3).to_string(index=False))
    elig = [s for s in mon.segments if s["eligible"]]
    print(f"\nEligible period x category segments: {len(elig)} of {len(mon.segments)}")
    print("\n=== Agent B: whole periods (bands vs ordinary change inside the baseline) ===")
    print(whole[SHOW].round(1).to_string(index=False))
    print(f"\n=== Agent B: segments ({len(segs)}) ===")
    print(segs[SHOW].round(1).to_string(index=False))
    if len(topics):
        print(f"\n=== Agent B: emerging topics ({len(topics)}) ===")
        print(topics[["period", "label", "docs_period", "docs_baseline", "ratio", "sections",
                      "top_sections"]].round(1).to_string(index=False))
    if len(shifts):
        print("\n=== Agent B: term-context shifts (q <= alpha) ===")
        sig = shifts[shifts["q_value"] <= config["drift"]["alpha"]]
        print(sig[["term", "period", "docs_baseline", "docs_period", "times_ordinary",
                   "context_before", "context_after"]].round(1).to_string(index=False))
    if not isinstance(drift.query_drift, str):
        qd = pd.DataFrame(drift.query_drift)
        qd.to_csv(run_dir / "query_drift.csv", index=False)
        print("\n=== Agent B: query drift (each period's questions vs the baseline questions) ===")
        print(qd[["period", "compared_with", "n_reference", "n_period", "mmd2", "mmd_q", "jsd_excess",
                  "jsd_q", "new_term_rate", "significant", "emerging_terms"]].round(4).to_string(index=False))
    print("\n=== Agent B: correlation of drift measures across segments (Spearman) ===")
    print(corr.set_index("measure").round(2).to_string())
    _report_retrieval(state, run_dir)
    if args.estimate and llm is not None:
        print(f"\nEstimate only: the judge would cost about ${llm.estimate:.2f} "
              f"({llm.uncached} uncached requests, batch price); spent so far ${llm.spent():.2f}")
    print(f"\nFlagged (vs baseline): {len(drift.drifted_segments)}; controls: "
          f"{len(drift.control_segments)}; next step: {state['next_step']}")
    print(f"Reports in {run_dir} ({time.time() - t0:.0f} s)")
    return 0


def _report_retrieval(state, run_dir: Path) -> None:
    """Save Agent C's report and tables, and print the headline numbers."""
    rep, tables = state["retrieval"], state.get("retrieval_tables", {})
    (run_dir / "retrieval.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    for name in ("judged", "known_item", "ranking_change", "runs", "grades"):
        if name in tables and len(tables[name]):
            tables[name].to_csv(run_dir / f"retrieval_{name}.csv", index=False)
    print(f"\n=== Agent C: retrieval ({rep.mode}; {rep.questions} questions: {rep.questions_by_status}) ===")
    if rep.index_store:
        print(f"Indexes: {'reloaded from' if rep.index_store.get('reloaded') else 'built and saved to'} "
              f"{rep.index_store.get('directory')}")

    def show(rows, cols):
        print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
    if rep.judged_by_period:
        print("(a) stale vs refreshed, judged (answer = the top k contains an answer):")
        show(rep.judged_by_period, ["period", "questions", "stale_answer", "refreshed_answer", "delta_answer",
                                    "delta_answer_low", "delta_answer_high", "stale_ndcg", "refreshed_ndcg",
                                    "delta_ndcg", "delta_ndcg_low", "delta_ndcg_high"])
        print("    by drift status:")
        show(rep.judged_by_status, ["period", "drift_status", "questions", "delta_answer", "delta_ndcg",
                                    "delta_ndcg_low", "delta_ndcg_high"])
    if rep.judge:
        print(f"    judge: {rep.judge}")
    if rep.known_item:
        print("(b) known item on the same refreshed index (weighted down, F11):")
        show(rep.known_item, ["index", "asked_by", "questions", "reciprocal_rank", "reciprocal_rank_low",
                              "reciprocal_rank_high", "recall"])
    if rep.ranking_change:
        print("(c) baseline questions, stale vs refreshed top-k:")
        show(rep.ranking_change, ["index", "questions", "overlap", "newer_share", "source_displaced"])
    for note in rep.notes:
        print(f"Note: {note}")


class _NoCalls:
    """Stand-in API client for --estimate: any call is an error."""

    def __getattr__(self, name):
        raise RuntimeError("estimate only: no API calls")


class _EstimateOnly:
    """Wraps the LLM client: serves cached replies, prices the rest, calls nothing."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm, self.estimate, self.uncached = llm, 0.0, 0

    def spent(self) -> float:
        return self.llm.spent()

    def complete_many(self, requests, batch=None):
        batch = bool(self.llm.config.get("use_batch_api", True)) if batch is None else batch
        out, todo = [], {}
        for r in requests:
            hit = self.llm.cached(r)
            if hit is None:
                todo[r.key] = r
                hit = LLMResult(key=r.key, parsed=None, text="", stop_reason="", model=r.model,
                                input_tokens=0, output_tokens=0, cost_usd=0.0, error="estimate only")
            out.append(hit)
        self.uncached += len(todo)
        self.estimate += self.llm.estimate(list(todo.values()), batch)
        return out


if __name__ == "__main__":
    raise SystemExit(main())
