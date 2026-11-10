"""Generate golden evaluation questions (the simulator, D16).

Samples eligible articles per period, has Sonnet write one question each,
checks them in code, has Opus review the survivors, and keeps only questions
that pass every check. Every draft, check and reason is saved for review.

Writes:
    data/processed/questions_<name>.parquet  every draft with its decisions
    runs/questions/<name>/questions.csv      the same, for reading in a spreadsheet

Usage (from the repository root):
    python scripts/generate_questions.py --dry-run     # selection and cost estimate only
    python scripts/generate_questions.py               # the pilot (live calls)
    python scripts/generate_questions.py --name small --per-period 30 --batch
    python scripts/generate_questions.py --name full --full --batch   # the full set (D23)

With --batch there are two batches (writing, then reviewing). If the run is
interrupted while waiting, run the same command again: it collects the
batch already submitted instead of paying for a new one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/generate_questions.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.data import simulate  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.rag.embeddings import Embedder  # noqa: E402
from driftmon.llm.client import LLMClient  # noqa: E402
from driftmon.data.periods import study_periods  # noqa: E402
from driftmon.llm.prompts import DraftQuestion  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="pilot", help="name of this question set")
    parser.add_argument("--per-period", type=int, help="articles per period (default: the pilot counts)")
    parser.add_argument("--batch", action="store_true", help="use the Batch API (half price, slower)")
    parser.add_argument("--dry-run", action="store_true", help="show selection and estimate; no API calls")
    parser.add_argument("--oversample", action="store_true",
                        help="sample only articles likely to state a revisable fact (D20)")
    parser.add_argument("--full", action="store_true",
                        help="the full set: flagged, control, random and oversample articles (D23)")
    args = parser.parse_args(argv)

    sim = cfg["simulation"]
    counts = ({p.name: args.per_period for p in study_periods(cfg)} if args.per_period
              else dict(sim["pilot_counts"]))
    docs, vectors = load_corpus(cfg)
    ok = simulate.eligible_articles(docs, sim["exclude_title_patterns"], int(sim["min_words"]))
    print(f"Eligible source articles: {int(ok.sum()):,} of {len(docs):,} "
          f"({int((~ok).sum()):,} templated or too short)")
    revisable = simulate.revisable_fact_pool(docs, sim.get("oversample_patterns", []))
    if args.full:
        run_id, drift = _latest_drift(cfg)
        flagged = [s for s in drift.drifted_segments if "|ALL|" not in s]
        articles = simulate.full_sample(docs[ok], revisable, flagged, drift.control_segments,
                                        sim["full_set"], seed=int(cfg["seed"]))
        print(f"Full set from drift run {run_id}: {len(flagged)} flagged and "
              f"{len(drift.control_segments)} control segments")
        print(pd.crosstab(articles["period"], articles["sample_pool"], margins=True).to_string())
    else:
        pool_name = "random"
        if args.oversample:
            print(f"Oversample pool (likely revisable facts): {int((ok & revisable).sum()):,} eligible articles")
            ok, pool_name = ok & revisable, "oversample"
        articles = simulate.sample_articles(docs[ok], counts, seed=int(cfg["seed"]))
        articles = articles.assign(sample_pool=pool_name, sample_segment="")
        print(f"Sampled {len(articles)} articles: {counts}")

    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv",
                    client=_DryClient() if args.dry_run else None, log=lambda m: print(m, flush=True))
    rows = articles.to_dict("records")
    writes = [simulate.question_request(a, cfg) for a in rows]
    placeholder = DraftQuestion(question="x " * 15, reference_answer="y " * 25, question_type="evergreen")
    reviews = [simulate.review_request(a, placeholder, cfg) for a in rows] * 2  # up to 2 per article
    est = llm.estimate(writes, args.batch) + llm.estimate(reviews, args.batch)
    print(f"Estimated cost: ${est:.2f} (if every draft is reviewed); spent so far ${llm.spent():.2f} "
          f"of the ${llm.budget:.0f} budget")
    if args.dry_run:
        if not args.full:
            print(articles[["period", "category", "title"]].to_string(index=False))
        return 0

    before = llm.spent()
    table = simulate.generate_questions(cfg, articles, llm, Embedder.from_config(cfg), batch=args.batch)
    # why each article was picked: always reported apart, never mixed into archive-wide rates (D20, D23)
    tags = articles.set_index("doc_id")[["sample_pool", "sample_segment"]]
    table = table.join(tags, on="doc_id")
    out = resolve_path(cfg["paths"]["data_processed"]) / f"questions_{args.name}.parquet"
    table.to_parquet(out, index=False)
    report = resolve_path(cfg["paths"]["runs"]) / "questions" / args.name
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "questions.csv", index=False)

    kept = table[table["set"] == "answerable"]
    unans = table[table["set"] == "unanswerable"]
    pos = {d: i for i, d in enumerate(docs["doc_id"])}
    div = simulate.diversity(kept["question"].tolist(),
                             vectors[[pos[d] for d in kept["doc_id"]]], Embedder.from_config(cfg))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n{len(table)} questions drafted from {table['doc_id'].nunique()} articles: kept "
          f"{len(kept)} answerable + {len(unans)} unanswerable. Cost of this run: "
          f"${llm.spent() - before:.4f} (total ${llm.spent():.4f} of ${llm.budget:.0f})")
    print("\nKeep rates by period:\n" + simulate.keep_rates(table, "period").to_string())
    print("\nKeep rates by section:\n" + simulate.keep_rates(table, "category").to_string())
    reasons = table.loc[table["set"] == "", "rejected_by"].str.split("; ").explode().value_counts()
    if len(reasons):
        print("Rejection reasons:\n" + reasons.to_string())
    print(f"Label groups kept (writer and every review agree, else disputed): "
          f"{kept['type_group'].value_counts().to_dict()}")
    reviewed = table[table["type_agreed"].notna()]
    if len(reviewed):
        print(f"Writer and first reviewer agree on the type for {int(reviewed['type_agreed'].sum())} of "
              f"{len(reviewed)} reviewed questions ({reviewed['type_agreed'].mean():.0%})")
        both = reviewed[reviewed["reviews_agree"].notna()]
        if cfg["simulation"].get("reviews_per_question", 1) > 1 and len(both):
            labels_same = both["review_types"].map(lambda s: len(set(s.split(","))) == 1)
            print(f"Reviewer consistency (independent reviews of the same question): same keep/reject "
                  f"decision for {int(both['reviews_agree'].sum())} of {len(both)} "
                  f"({both['reviews_agree'].mean():.0%}); same type label for {int(labels_same.sum())} "
                  f"({labels_same.mean():.0%})")
    print(f"Diversity (P3): questions {div['questions']:.3f} vs source articles "
          f"{div['articles']:.3f} (ratio {div['ratio']:.2f})")
    print(f"\nSaved {out} and {report / 'questions.csv'}")
    return 0


def _latest_drift(cfg):
    """(run id, DriftReport) of the newest pipeline run: flagged and control segments."""
    from driftmon.dashboard.runs import list_runs, load_run
    runs_dir = resolve_path(cfg["paths"]["runs"])
    runs = list_runs(runs_dir)
    if not runs:
        raise SystemExit("No pipeline run found; run scripts/run_pipeline.py first.")
    run = load_run(runs_dir, runs[0])
    return run.run_id, run.drift


class _DryClient:
    """Stand-in client for --dry-run: any API call is an error."""

    def __getattr__(self, name):
        raise RuntimeError("dry run: no API calls")


if __name__ == "__main__":
    raise SystemExit(main())
