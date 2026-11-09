"""Run the LangGraph pipeline and save every agent's report to ``runs/<run_id>/``.

Agent A (Data Monitoring) -> Agent B (Drift Detection) -> the routing
decision -> Agent C (Retrieval Evaluation) -> Agent D (RAG Impact) -> Agent E
(Recommendation). Agent E also reads the long-form evidence and the hand-label
checks saved outside the graph (``driftmon.longform.evidence``).

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
    impact.json        Agent D's ImpactReport
    impact_*.csv       Agent D's tables (answers, paired, segments, correlations)
    recommendations.json  Agent E's RecommendationReport
    recommendations.md    the same, readable
    longform/          a copy of the long-form results Agent E read (for the dashboard)
    run_info.json      what produced the run: models, question set, mode, cost, code version

Usage (from the repository root):
    python scripts/run_pipeline.py --estimate   # everything except the judge; prints its cost
    python scripts/run_pipeline.py              # including the relevance judge (LLM spend)
    python scripts/run_pipeline.py --no-judge   # Agents C and D without LLM calls
    python scripts/run_pipeline.py --pilot 20   # Agent D on a 20-question sample only

The judge runs as a batch (half price). If the run is interrupted while it
waits, run the same command again: it collects the batch already submitted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_pipeline.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.rag.embeddings import Embedder  # noqa: E402
from driftmon.longform.evidence import load_label_checks, load_longform, save_longform_snapshot  # noqa: E402
from driftmon.graph import build_graph  # noqa: E402
from driftmon.llm.client import LLMClient, LLMResult  # noqa: E402
from driftmon.data.queries import load_questions, query_stream  # noqa: E402

SHOW = ["segment_id", "times_ordinary_embedding", "times_ordinary_lexical", "embedding_band",
        "lexical_band", "kind", "flagged", "emerging_terms"]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: print the judge's cost")
    mode.add_argument("--no-judge", action="store_true", help="skip the LLM relevance judge")
    parser.add_argument("--pilot", type=int, help="Agent D on this many questions only")
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
    spent_before = llm.spent() if llm is not None else 0.0
    state = build_graph().invoke({"config": config, "docs": docs, "vectors": vectors,
                                  "queries": queries, "query_vectors": query_vectors,
                                  "questions": questions, "question_vectors": question_vectors, "llm": llm,
                                  "impact_sample": args.pilot, "longform": load_longform(config),
                                  "label_checks": load_label_checks(config)})

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
    _report_impact(state, run_dir)
    _report_recommendations(state, run_dir)
    _save_run_info(run_dir, config, args, llm, spent_before, questions is not None, time.time() - t0)
    if args.estimate and llm is not None:
        print(f"\nEstimate only: the uncached LLM calls would cost about ${llm.estimate:.2f} "
              f"({llm.uncached} uncached requests, batch price); spent so far ${llm.spent():.2f}")
    print(f"\nFlagged (vs baseline): {len(drift.drifted_segments)}; controls: "
          f"{len(drift.control_segments)}; next step: {state['next_step']}")
    print(f"Reports in {run_dir} ({time.time() - t0:.0f} s)")
    return 0


def _save_run_info(run_dir: Path, config: dict, args, llm, spent_before: float, has_questions: bool,
                   seconds: float) -> None:
    """Record what produced this run, so every run carries its provenance."""
    mode = "estimate" if args.estimate else "no_judge" if args.no_judge else "full"
    info = {
        "run_id": run_dir.name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(seconds),
        "mode": mode,
        "pilot_questions": args.pilot,
        "question_set": config["evaluation"].get("question_set", "full") if has_questions else None,
        "models": ({"writer": config["llm"]["generator_model"], "judge": config["llm"]["judge_model"]}
                   if llm is not None else None),
        "embedding_model": config["embeddings"]["model"],
        "llm_cost_usd": round(llm.spent() - spent_before, 4) if llm is not None and mode == "full" else None,
        "code_version": _git_commit(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "
", encoding="utf-8")


def _git_commit() -> str | None:
    """Short commit id of the code that produced the run, with '+changes' if uncommitted edits exist."""
    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=repo).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True, cwd=repo).stdout.strip()
        return sha + ("+changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


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


def _report_impact(state, run_dir: Path) -> None:
    """Save Agent D's report and tables, and print the headline numbers."""
    rep, tables = state.get("impact"), state.get("impact_tables", {})
    if rep is None:
        return
    (run_dir / "impact.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    for name, df in tables.items():
        if len(df):
            df.to_csv(run_dir / f"impact_{name}.csv", index=False)
    print(f"\n=== Agent D: answers, stale vs refreshed ({rep.questions} questions; {rep.sample}) ===")

    def show(rows, cols):
        if rows:
            df = pd.DataFrame(rows)
            print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))
    cols = ["questions", "stale_correct", "refreshed_correct", "loss_correct", "loss_low", "loss_high",
            "stale_hallucinated", "refreshed_hallucinated", "stale_declined_ok"]
    show(rep.by_period, ["period"] + cols)
    print("    by drift status:")
    show(rep.by_status, ["drift_status"] + cols)
    if rep.unanswerable:
        print("    unanswerable questions (should decline):")
        show(rep.unanswerable, ["index", "questions", "declined", "hallucinated"])
    if rep.classes:
        print(f"    segment classes (material loss >= {rep.threshold:.0%}, interval excludes 0): {rep.classes}")
        show(rep.segments, ["segment_id", "questions", "stale_correct", "refreshed_correct", "loss_correct",
                            "loss_low", "loss_high", "class", "cause"])
    if rep.correlations:
        print("    drift measure vs loss in correct answers (Spearman, across segments):")
        show(rep.correlations, ["measure", "segments", "spearman", "p_value"])
    print(f"    judge: {rep.judge}")
    for note in rep.notes:
        print(f"Note: {note}")


def _report_recommendations(state, run_dir: Path) -> None:
    """Save Agent E's report (JSON and readable) and print the actions."""
    rep = state.get("recommendations")
    if rep is None:
        return
    from driftmon.agents.recommendation import to_markdown
    (run_dir / "recommendations.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "recommendations.md").write_text(to_markdown(rep), encoding="utf-8")
    save_longform_snapshot(state["config"], run_dir / "longform")
    print(f"\n=== Agent E: recommendations ({rep.summary}) ===")
    for r in rep.recommendations:
        mark = f"{r.priority}." if r.priority else "-"
        print(f"  {mark} {r.action} [{r.status}]: {r.explanation if r.priority else r.finding}")
    print(f"    explanations: {rep.explanations}")


class _NoCalls:
    """Stand-in API client for --estimate: any call is an error."""

    def __getattr__(self, name):
        raise RuntimeError("estimate only: no API calls")


class _EstimateOnly:
    """Wraps the LLM client: serves cached replies, prices the rest, calls nothing."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm, self.estimate, self.uncached = llm, 0.0, 0
        self.estimate_only = True

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
