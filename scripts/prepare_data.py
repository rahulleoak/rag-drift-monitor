"""Prepare the chosen dataset: assign periods, clean, deduplicate, sample.

Steps (rules and thresholds in ``data_cleaning`` in the config; D8, F3):
    1. check the raw file is the expected one (size and SHA-256 from config)
    2. keep documents inside the study periods; merge renamed categories
    3. drop empty descriptions, too-short and too-long documents
    4. drop near-duplicates within each period (embedding cosine, earliest kept)
    5. mark an equal-size random sample per period for the drift statistics

Writes:
    data/processed/documents.parquet  every kept document, with its period,
        normalised and raw category, word count and ``in_drift_sample`` flag
    runs/prepare/retention.csv        what each rule removed, per period
    runs/prepare/removed.csv          every removed document and the rule that removed it

Embeddings computed here go into the disk cache, so later steps reuse them.

Usage (from the repository root):
    python scripts/prepare_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/prepare_data.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.data import cleaning  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.records import iter_records  # noqa: E402
from driftmon.rag.embeddings import Embedder, document_text  # noqa: E402
from driftmon.data.periods import assign_period, study_periods  # noqa: E402

RULES = ["empty_description", "too_short", "too_long", "near_duplicate"]


def check_raw_file(path: Path, expected: Mapping[str, Any]) -> list[str]:
    """Compare the raw file with the size and SHA-256 recorded in the config.

    Returns a list of warnings (empty if it matches or nothing is recorded).
    """
    warnings = []
    if expected.get("bytes") and path.stat().st_size != int(expected["bytes"]):
        warnings.append(f"size {path.stat().st_size:,} bytes, expected {int(expected['bytes']):,}")
    if expected.get("sha256"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            warnings.append(f"SHA-256 {digest}, expected {expected['sha256']}")
    return warnings


def load_documents(dataset: str, path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Documents inside the study periods, with period and normalised category."""
    periods = study_periods(config)
    normalise = (config.get("categories") or {}).get("normalise") or {}
    rows = []
    for r in iter_records(dataset, path):
        period = assign_period(r.date, periods)
        if period is not None:
            rows.append({"doc_id": r.doc_id, "date": r.date, "period": period,
                         "category": normalise.get(r.category, r.category),
                         "category_raw": r.category, "title": r.title, "text": r.text})
    df = pd.DataFrame(rows, columns=["doc_id", "date", "period", "category", "category_raw",
                                     "title", "text"])
    order = {p.name: i for i, p in enumerate(periods)}
    df = df.sort_values(["period", "date", "doc_id"], key=lambda c: c.map(order) if c.name == "period"
                        else c).reset_index(drop=True)
    df["words"] = cleaning.word_counts(df)
    return df


def apply_rules(df: pd.DataFrame, rules: Mapping[str, Any], embedder: Embedder) -> pd.Series:
    """Name of the rule that removes each row, or "" if the row is kept.

    Rules run in ``RULES`` order; each sees only rows kept so far.
    """
    removed_by = pd.Series("", index=df.index)

    def mark(mask: pd.Series, rule: str) -> None:
        removed_by[(removed_by == "") & mask] = rule

    if rules.get("drop_empty_text", True):
        mark(cleaning.empty_description(df), "empty_description")
    if rules.get("min_words") is not None:
        mark(cleaning.too_short(df, int(rules["min_words"])), "too_short")
    if rules.get("max_words") is not None:
        mark(cleaning.too_long(df, int(rules["max_words"])), "too_long")
    if rules.get("near_duplicate_cosine") is not None:
        for _, group in df[removed_by == ""].groupby("period", sort=False):
            vectors = embedder.embed_documents(
                [document_text(t, x) for t, x in zip(group["title"], group["text"])])
            drop = cleaning.near_duplicates(vectors, float(rules["near_duplicate_cosine"]))
            removed_by[group.index[drop]] = "near_duplicate"
    return removed_by


def retention_table(df: pd.DataFrame, removed_by: pd.Series, in_sample: pd.Series,
                    period_order: list[str]) -> pd.DataFrame:
    """Per period: raw count, removals by rule, kept count and share, sample size."""
    t = pd.crosstab(df["period"], removed_by.replace("", "kept"))
    t = t.reindex(index=period_order, columns=RULES + ["kept"], fill_value=0)
    t.columns.name = None
    t.insert(0, "raw", t.sum(axis=1))
    t["kept_pct"] = (100 * t["kept"] / t["raw"]).round(1)
    t["in_drift_sample"] = in_sample.groupby(df["period"]).sum().reindex(period_order).fillna(0).astype(int)
    t.loc["total"] = t.sum(numeric_only=True)
    t.loc["total", "kept_pct"] = round(100 * t.loc["total", "kept"] / t.loc["total", "raw"], 1)
    return t


def run(config: Mapping[str, Any], path: Path, processed_dir: Path, report_dir: Path,
        embedder: Embedder, dataset: str = "huffpost") -> pd.DataFrame:
    """Prepare the dataset and write the outputs; returns the retention table."""
    df = load_documents(dataset, path, config)
    removed_by = apply_rules(df, config["data_cleaning"], embedder)
    kept = df[removed_by == ""].copy()
    in_sample = pd.Series(False, index=df.index)
    in_sample[kept.index] = cleaning.equal_samples(
        kept["period"], int(config["periods"]["sample_per_period"]), seed=int(config["seed"]))
    kept["in_drift_sample"] = in_sample[kept.index]

    period_order = [p.name for p in study_periods(config)]
    table = retention_table(df, removed_by, in_sample, period_order)

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    kept.reset_index(drop=True).to_parquet(processed_dir / "documents.parquet", index=False)
    table.to_csv(report_dir / "retention.csv", index_label="period")
    removed = df[removed_by != ""].assign(rule=removed_by[removed_by != ""])
    removed[["doc_id", "date", "period", "category", "rule", "words", "title", "text"]].to_csv(
        report_dir / "removed.csv", index=False)
    return table


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")  # titles include non-Latin characters
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=config.get("active_dataset") or "huffpost",
                        choices=sorted(config["datasets"]))
    parser.add_argument("--path", type=Path, help="override the raw dataset file")
    args = parser.parse_args(argv)

    spec = config["datasets"][args.dataset]
    path = resolve_path(args.path or spec["file"])
    if not path.exists():
        print(f"Dataset file not found: {path}\nDownload it from {spec['source']}.", file=sys.stderr)
        return 2
    for w in check_raw_file(path, spec):
        print(f"WARNING: raw file differs from the recorded one: {w}", file=sys.stderr)

    processed = resolve_path(config["paths"]["data_processed"])
    report = resolve_path(config["paths"]["runs"]) / "prepare"
    table = run(config, path, processed, report, Embedder.from_config(config), args.dataset)
    print("\n=== Retention table (documents removed by each rule, in order) ===")
    print(table.to_string())
    print(f"\nWrote {processed / 'documents.parquet'}\nReports in {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
