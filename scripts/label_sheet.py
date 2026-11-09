"""Make the hand-label sheet that checks the LLM judges, or score it (P8).

    python scripts/label_sheet.py make --run runs/<run_id>
        writes runs/labels/sheet.csv (fill in your_label; open in Excel)
        and runs/labels/key.csv (the LLM verdicts: do not open before labelling)
    python scripts/label_sheet.py score
        agreement between your labels and the LLM, per part
        (--sheet runs/labels/sheet_<name>.csv for another person's copy)
    python scripts/label_sheet.py compare runs/labels/sheet.csv runs/labels/sheet_<name>.csv
        agreement between people: the human baseline for the LLM's numbers

    python scripts/label_sheet.py make-answers --run runs/<run_id>
        a separate sheet for Agent D's answer judge, in runs/labels/answers/;
        score and compare it with --dir runs/labels/answers

Other labellers get a blank copy of the sheet plus docs/labelling-guide.md,
never the key or anyone else's labels.

Parts: kept and rejected questions (the question reviewer),
(question, article) pairs (Agent C's relevance judge, from the run's
retrieval_grades.csv), and changed facts flagged by the D27 check (its
precision). Agent D's answers are added when D exists.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.evaluation import labels  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.evaluation.answers import article_listing  # noqa: E402


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
    ma = sub.add_parser("make-answers")
    ma.add_argument("--run", required=True, help="pipeline run folder with impact_answers.csv")
    ma.add_argument("--correct", type=int, default=6)
    ma.add_argument("--declined", type=int, default=5)
    sc = sub.add_parser("score")
    sc.add_argument("--sheet", help="a labelled copy of the sheet (default runs/labels/sheet.csv)")
    cmp = sub.add_parser("compare")
    cmp.add_argument("sheets", nargs="+", help="two or more labelled copies of the sheet, one per person")
    for s in (sc, cmp):
        s.add_argument("--dir", help="folder with sheet.csv and key.csv (default runs/labels)")
    args = parser.parse_args(argv)
    out = Path(args.dir) if getattr(args, "dir", None) else resolve_path(cfg["paths"]["runs"]) / "labels"
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

    if args.cmd == "make-answers":
        out = out / "answers"
        if (out / "sheet.csv").exists():
            raise SystemExit(f"{out / 'sheet.csv'} exists; move it away first (it may hold labels).")
        run = Path(args.run)
        name = cfg["evaluation"].get("question_set", "full")
        table = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / f"questions_{name}.parquet")
        docs = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / "documents.parquet")
        runs = pd.read_csv(run / "retrieval_runs.csv")
        k = int(cfg["impact"]["answer_k"])
        articles = {(q, i): article_listing(re.findall(r"'([^']+)'", ids)[:k], docs)
                    for q, i, ids in zip(runs["question_id"], runs["index"], runs["doc_ids"])}
        part = labels.sample_answers(pd.read_csv(run / "impact_answers.csv"), table, articles,
                                     args.correct, args.declined, seed)
        sheet, key = labels.build_sheet([part], seed)
        out.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(out / "sheet.csv", index=False, encoding="utf-8-sig")
        key.to_csv(out / "key.csv", index=False)
        print(f"{len(sheet)} answer items in {out / 'sheet.csv'}; do not open key.csv until done.")
        return 0

    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    if args.cmd == "compare":
        paths = [Path(p) for p in args.sheets]
        if len(paths) < 2:
            raise SystemExit("give at least two sheets")
        tables = []
        for i, p in enumerate(paths):
            for q in paths[i + 1:]:
                tables.append(labels.compare_labellers(read(p), read(q)).assign(pair=f"{p.stem} vs {q.stem}"))
        result = pd.concat(tables, ignore_index=True)
        result.to_csv(out / "agreement_people.csv", index=False)
        print(result.round(3).to_string(index=False))
        return 0

    sheet_path = Path(args.sheet) if args.sheet else out / "sheet.csv"
    result = labels.score(read(sheet_path), pd.read_csv(out / "key.csv", dtype={"item_id": str}))
    name = "agreement.csv" if not args.sheet else f"agreement_{sheet_path.stem}.csv"
    result.to_csv(out / name, index=False)
    print(result.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
