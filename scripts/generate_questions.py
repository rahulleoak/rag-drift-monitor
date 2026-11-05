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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/generate_questions.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon import simulate  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.corpus import load_corpus  # noqa: E402
from driftmon.embeddings import Embedder  # noqa: E402
from driftmon.llm import LLMClient  # noqa: E402
from driftmon.periods import study_periods  # noqa: E402
from driftmon.prompts import QuestionDraft  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="pilot", help="name of this question set")
    parser.add_argument("--per-period", type=int, help="articles per period (default: the pilot counts)")
    parser.add_argument("--batch", action="store_true", help="use the Batch API (half price, slower)")
    parser.add_argument("--dry-run", action="store_true", help="show selection and estimate; no API calls")
    args = parser.parse_args(argv)

    sim = cfg["simulation"]
    counts = ({p.name: args.per_period for p in study_periods(cfg)} if args.per_period
              else dict(sim["pilot_counts"]))
    docs, vectors = load_corpus(cfg)
    ok = simulate.eligible_articles(docs, sim["exclude_title_patterns"], int(sim["min_words"]))
    print(f"Eligible source articles: {int(ok.sum()):,} of {len(docs):,} "
          f"({int((~ok).sum()):,} templated or too short)")
    articles = simulate.sample_articles(docs[ok], counts, seed=int(cfg["seed"]))
    print(f"Sampled {len(articles)} articles: {counts}")

    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv",
                    client=_DryClient() if args.dry_run else None)
    rows = articles.to_dict("records")
    writes = [simulate.question_request(a, cfg) for a in rows]
    placeholder = QuestionDraft(question="x " * 15, reference_answer="y " * 25,
                                question_type="evergreen", answerable_from_article=True)
    reviews = [simulate.review_request(a, placeholder, cfg) for a in rows]
    est = llm.estimate(writes, args.batch) + llm.estimate(reviews, args.batch)
    print(f"Estimated cost: ${est:.2f} (if every draft is reviewed); spent so far ${llm.spent():.2f} "
          f"of the ${llm.budget:.0f} budget")
    if args.dry_run:
        print(articles[["period", "category", "title"]].to_string(index=False))
        return 0

    before = llm.spent()
    table = simulate.generate_questions(cfg, articles, llm, Embedder.from_config(cfg), batch=args.batch)
    out = resolve_path(cfg["paths"]["data_processed"]) / f"questions_{args.name}.parquet"
    table.to_parquet(out, index=False)
    report = resolve_path(cfg["paths"]["runs"]) / "questions" / args.name
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "questions.csv", index=False)

    kept = table[table["kept"]]
    pos = {d: i for i, d in enumerate(docs["doc_id"])}
    div = simulate.diversity(kept["question"].tolist(),
                             vectors[[pos[d] for d in kept["doc_id"]]], Embedder.from_config(cfg))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nKept {len(kept)} of {len(table)} questions. Cost of this run: ${llm.spent() - before:.4f} "
          f"(total ${llm.spent():.4f} of ${llm.budget:.0f})")
    reasons = table.loc[~table["kept"], "rejected_by"].str.split("; ").explode().value_counts()
    if len(reasons):
        print("Rejection reasons:\n" + reasons.to_string())
    print(f"Question types kept: {kept['question_type'].value_counts().to_dict()}")
    print(f"Diversity (P3): questions {div['questions']:.3f} vs source articles "
          f"{div['articles']:.3f} (ratio {div['ratio']:.2f})")
    print(f"\nSaved {out} and {report / 'questions.csv'}")
    return 0


class _DryClient:
    """Stand-in client for --dry-run: any API call is an error."""

    def __getattr__(self, name):
        raise RuntimeError("dry run: no API calls")


if __name__ == "__main__":
    raise SystemExit(main())
