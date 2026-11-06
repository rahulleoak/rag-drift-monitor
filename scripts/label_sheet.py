"""Make the hand-label sheet that checks the LLM judges, or score it (P8).

    python scripts/label_sheet.py make --run runs/<run_id>
        writes runs/labels/sheet.csv (fill in your_label; open in Excel)
        and runs/labels/key.csv (the LLM verdicts: do not open before labelling)
    python scripts/label_sheet.py score
        agreement between your labels and the LLM, per part

Parts: kept and rejected questions (the question reviewer),
(question, article) pairs (Agent C's relevance judge, from the run's
retrieval_grades.csv), and changed facts flagged by the D27 check (its
precision). Agent D's answers are added when D exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon import labels  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make")
    make.add_argument("--run", required=True, help="pipeline run folder with retrieval_grades.csv")
    make.add_argument("--kept", type=int, default=10)
    make.add_argument("--rejected", type=int, default=5)
    make.add_argument("--per-grade", type=int, default=5)
    make.add_argument("--facts", type=int, default=20, help="changed facts from runs/longform (D27); 0 to skip")
    sub.add_parser("score")
    args = parser.parse_args(argv)
    out = resolve_path(cfg["paths"]["runs"]) / "labels"
    seed = int(cfg["seed"])

    if args.cmd == "make":
        if (out / "sheet.csv").exists():
            raise SystemExit(f"{out / 'sheet.csv'} exists; move it away first (it may hold your labels).")
        name = cfg["evaluation"].get("question_set", "full")
        table = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / f"questions_{name}.parquet")
        docs = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / "documents.parquet")
        grades = pd.read_csv(Path(args.run) / "retrieval_grades.csv")
        parts = [labels.sample_questions(table, args.kept, args.rejected, seed),
                 labels.sample_relevance(grades, table, docs, args.per_grade, seed)]
        facts_file = resolve_path(cfg["paths"]["runs"]) / "longform" / "fact_changes.csv"
        if args.facts and facts_file.exists():
            parts.append(labels.sample_facts(pd.read_csv(facts_file), args.facts, seed))
        sheet, key = labels.build_sheet(parts, seed)
        out.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(out / "sheet.csv", index=False, encoding="utf-8-sig")  # BOM: Excel reads UTF-8
        key.to_csv(out / "key.csv", index=False)
        print(f"{len(sheet)} items: {sheet['part'].value_counts().to_dict()}\nFill in {out / 'sheet.csv'}; "
              f"do not open {out / 'key.csv'} until you are done.")
        return 0

    sheet = pd.read_csv(out / "sheet.csv", encoding="utf-8-sig", dtype=str)
    key = pd.read_csv(out / "key.csv", dtype={"item_id": str})
    result = labels.score(sheet, key)
    result.to_csv(out / "agreement.csv", index=False)
    print(result.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
