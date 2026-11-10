"""Explore a candidate dataset to choose the dataset, baseline and later periods.

Prints:
    1. a summary (records, date range, skipped lines, empty texts, text length)
    2. articles per year
    3. articles per category per year
    4. the 20 most common categories
    5. sample titles per year (5 by default, seeded so reruns match)
    6. watch-term rates: % of each year's records mentioning each term in
       ``exploration.watch_terms`` (a check that known drift is visible at all)

The same tables are written as CSV to ``runs/explore/<dataset>/`` so numbers
quoted in docs/findings.md can be traced back to a file.

Usage (from the repository root):
    python scripts/exploration/explore_data.py --dataset huffpost
    python scripts/exploration/explore_data.py --dataset arxiv
    python scripts/exploration/explore_data.py --dataset arxiv --limit 200000   # quick look

Expected files (paths set in config/config.yaml):
    huffpost: data/raw/News_Category_Dataset_v3.json
    arxiv:    data/raw/arxiv-metadata-oai-snapshot.json
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/exploration/explore_data.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.records import LoadStats, Record, iter_records  # noqa: E402

TOP_N_CATEGORIES = 20


@dataclass
class Summary:
    """Aggregates computed in a single streaming pass over the records."""

    per_year: Counter = field(default_factory=Counter)
    per_category: Counter = field(default_factory=Counter)
    per_category_year: Counter = field(default_factory=Counter)  # key: (category, year)
    word_counts: Counter = field(default_factory=Counter)  # words in title + text -> n records
    empty_text: int = 0
    first_date: date | None = None
    last_date: date | None = None
    samples: dict[int, list[Record]] = field(default_factory=dict)
    watch_terms: list[str] = field(default_factory=list)
    term_year: Counter = field(default_factory=Counter)  # key: (term, year) -> n records

    @property
    def total(self) -> int:
        return sum(self.per_year.values())

    def median_words(self) -> float:
        """Median of title + text word counts, computed from the histogram."""
        if not self.word_counts:
            return 0.0
        values = sorted(self.word_counts.items())
        n = self.total
        def nth(k: int) -> int:
            seen = 0
            for words, count in values:
                seen += count
                if seen > k:
                    return words
            return values[-1][0]
        return (nth((n - 1) // 2) + nth(n // 2)) / 2


def summarize(records: Iterable[Record], samples_per_year: int, seed: int,
              watch_terms: dict[str, str] | None = None) -> Summary:
    """Stream records once, counting and reservoir-sampling titles per year.

    Reservoir sampling keeps a uniform random sample per year without holding
    all records in memory; with a fixed seed and file order it is repeatable.
    ``watch_terms`` maps a name to a regex, matched case-insensitively against
    title + text; each record counts at most once per term.
    """
    rng = random.Random(seed)
    s = Summary()
    patterns = {name: re.compile(rx, re.IGNORECASE) for name, rx in (watch_terms or {}).items()}
    s.watch_terms = list(patterns)
    seen_per_year: Counter = Counter()
    reservoirs: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        year = r.date.year
        s.per_year[year] += 1
        s.per_category[r.category] += 1
        s.per_category_year[(r.category, year)] += 1
        full_text = f"{r.title} {r.text}"
        s.word_counts[len(full_text.split())] += 1
        for name, pattern in patterns.items():
            if pattern.search(full_text):
                s.term_year[(name, year)] += 1
        if not r.text:
            s.empty_text += 1
        s.first_date = r.date if s.first_date is None else min(s.first_date, r.date)
        s.last_date = r.date if s.last_date is None else max(s.last_date, r.date)

        seen_per_year[year] += 1
        reservoir = reservoirs[year]
        if len(reservoir) < samples_per_year:
            reservoir.append(r)
        else:
            j = rng.randrange(seen_per_year[year])
            if j < samples_per_year:
                reservoir[j] = r
    s.samples = {y: sorted(v, key=lambda r: r.date) for y, v in sorted(reservoirs.items())}
    return s


def year_table(s: Summary) -> pd.DataFrame:
    """Articles per year with each year's share of the total."""
    df = pd.DataFrame(sorted(s.per_year.items()), columns=["year", "articles"])
    df["share_pct"] = (100 * df["articles"] / max(s.total, 1)).round(2)
    return df


def category_year_table(s: Summary) -> pd.DataFrame:
    """Categories x years count matrix, rows sorted by total, plus a total column."""
    rows = [(c, y, n) for (c, y), n in s.per_category_year.items()]
    df = pd.DataFrame(rows, columns=["category", "year", "articles"])
    pivot = df.pivot_table(index="category", columns="year", values="articles",
                           aggfunc="sum", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False).astype(int)


def top_categories_table(s: Summary, n: int = TOP_N_CATEGORIES) -> pd.DataFrame:
    """The ``n`` most common categories with counts and share of all records."""
    df = pd.DataFrame(s.per_category.most_common(n), columns=["category", "articles"])
    df["share_pct"] = (100 * df["articles"] / max(s.total, 1)).round(2)
    df.index = range(1, len(df) + 1)
    return df


def samples_table(s: Summary) -> pd.DataFrame:
    """Sampled records per year as a flat table."""
    rows = [(y, r.date.isoformat(), r.category, r.title, r.doc_id)
            for y, recs in s.samples.items() for r in recs]
    return pd.DataFrame(rows, columns=["year", "date", "category", "title", "doc_id"])


def term_rate_table(s: Summary) -> pd.DataFrame:
    """Percent of each year's records mentioning each watch term (terms x years)."""
    years = sorted(s.per_year)
    data = {y: [round(100 * s.term_year[(t, y)] / s.per_year[y], 1) for t in s.watch_terms]
            for y in years}
    return pd.DataFrame(data, index=pd.Index(s.watch_terms, name="term"), columns=years)


def print_report(dataset: str, path: Path, s: Summary, stats: LoadStats) -> None:
    """Print all exploration tables to stdout."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)

    print(f"\n=== Dataset: {dataset} ({path}) ===")
    print(f"Lines read: {stats.lines_read:,}   records: {stats.records:,}   "
          f"skipped (malformed): {stats.skipped:,}")
    print(f"Date range: {s.first_date} to {s.last_date}")
    print(f"Categories: {len(s.per_category):,}")
    print(f"Empty text (no description/abstract): {s.empty_text:,} "
          f"({100 * s.empty_text / max(s.total, 1):.2f}%)")
    print(f"Median words per record (title + text): {s.median_words():.0f}")

    print("\n=== 1. Articles per year ===")
    print(year_table(s).to_string(index=False))

    print("\n=== 2. Articles per category per year ===")
    print(category_year_table(s).to_string())

    print(f"\n=== 3. Top {TOP_N_CATEGORIES} categories ===")
    print(top_categories_table(s).to_string())

    print("\n=== 4. Sample titles per year ===")
    for year, recs in s.samples.items():
        print(f"\n-- {year} --")
        for r in recs:
            print(f"  {r.date}  [{r.category}]  {r.title}")

    if s.watch_terms:
        print("\n=== 5. Watch terms: % of each year's records mentioning the term ===")
        print(term_rate_table(s).to_string())


def save_tables(s: Summary, out_dir: Path) -> None:
    """Write the exploration tables as CSV for traceability."""
    out_dir.mkdir(parents=True, exist_ok=True)
    year_table(s).to_csv(out_dir / "articles_per_year.csv", index=False)
    category_year_table(s).to_csv(out_dir / "articles_per_category_year.csv")
    top_categories_table(s).to_csv(out_dir / "top_categories.csv", index_label="rank")
    samples_table(s).to_csv(out_dir / "sample_titles.csv", index=False)
    if s.watch_terms:
        term_rate_table(s).to_csv(out_dir / "watch_term_rates.csv")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(config["datasets"]), default="huffpost")
    parser.add_argument("--path", type=Path, help="override the dataset file from config")
    parser.add_argument("--limit", type=int, help="only read this many records")
    parser.add_argument("--samples-per-year", type=int,
                        default=config["exploration"]["samples_per_year"])
    parser.add_argument("--out-dir", type=Path, help="where to write CSVs "
                        "(default: <runs>/explore/<dataset>)")
    args = parser.parse_args(argv)

    path = resolve_path(args.path or config["datasets"][args.dataset]["file"])
    if not path.exists():
        print(f"Dataset file not found: {path}\n"
              f"Download it from {config['datasets'][args.dataset]['source']} "
              f"and place it at that path.", file=sys.stderr)
        return 2

    stats = LoadStats()
    summary = summarize(iter_records(args.dataset, path, stats=stats, limit=args.limit),
                        samples_per_year=args.samples_per_year, seed=config["seed"],
                        watch_terms=config["exploration"].get("watch_terms"))
    print_report(args.dataset, path, summary, stats)

    out_dir = args.out_dir or resolve_path(config["paths"]["runs"]) / "explore" / args.dataset
    save_tables(summary, out_dir)
    print(f"\nTables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
