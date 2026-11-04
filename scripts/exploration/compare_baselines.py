"""Compare candidate baseline periods by how closely their topic mix matches the later periods.

Why: a drift detector compares later periods with the baseline. If the
baseline holds a very different mix of topics (say mostly lifestyle, while the
later years are mostly politics), the detector mostly measures that change in
mix, not a change in meaning. A good baseline has a mix close to the later
periods and is stable within itself, because the severity bands (P7) are set
from its own internal variation.

All numbers use category labels after the renaming map in
``categories.normalise`` (config). Mix difference is the total variation
distance: the share of articles that would have to change category to turn
one mix into the other (0% = identical mix, 100% = no category in common).

Prints and writes to ``runs/explore/<dataset>/baselines/``:
    1. mix_difference.csv   candidate x later period, plus mean and article count
    2. internal_stability.csv  each candidate's first half vs second half
    3. unmatched_share.csv  % of each later period in categories the candidate
       does not contain at all (no baseline can explain these articles)
    4. category_shares.csv  % of each period in each category (for charts)

Usage (from the repository root):
    python scripts/exploration/compare_baselines.py
    python scripts/exploration/compare_baselines.py --path tests/fixtures/huffpost_tiny.jsonl
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow running the file directly without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data import Record, iter_records  # noqa: E402


@dataclass(frozen=True)
class Period:
    """A named, inclusive date range."""

    name: str
    start: date
    end: date

    @classmethod
    def from_config(cls, d: Mapping[str, str]) -> Period:
        """Build from a config entry ``{name, start, end}`` with ISO dates."""
        return cls(str(d["name"]), date.fromisoformat(str(d["start"])),
                   date.fromisoformat(str(d["end"])))

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def halves(self) -> tuple[Period, Period]:
        """Split at the midpoint, e.g. 2016-17 into 2016 and 2017."""
        mid = self.start + (self.end - self.start) // 2
        return (Period(f"{self.name} (1st half)", self.start, mid),
                Period(f"{self.name} (2nd half)", mid + timedelta(days=1), self.end))


def count_categories(records: Iterable[Record], periods: Iterable[Period],
                     normalise: Mapping[str, str]) -> dict[str, Counter]:
    """Count normalised categories per period in one pass. Periods may overlap."""
    periods = list(periods)
    counts: dict[str, Counter] = {p.name: Counter() for p in periods}
    for r in records:
        category = normalise.get(r.category, r.category)
        for p in periods:
            if p.contains(r.date):
                counts[p.name][category] += 1
    return counts


def shares(counts: Counter) -> dict[str, float]:
    """Category counts as fractions summing to 1 (empty if there are no records)."""
    total = sum(counts.values())
    return {c: n / total for c, n in counts.items()} if total else {}


def mix_difference(a: Counter, b: Counter) -> float:
    """Total variation distance between two category mixes, in [0, 1].

    NaN if either side has no records, so empty periods are visible rather
    than silently scored as identical or as maximally different.
    """
    pa, pb = shares(a), shares(b)
    if not pa or not pb:
        return math.nan
    return 0.5 * sum(abs(pa.get(c, 0.0) - pb.get(c, 0.0)) for c in pa.keys() | pb.keys())


def unmatched_share(baseline: Counter, later: Counter) -> float:
    """Fraction of ``later`` records whose category never appears in ``baseline``."""
    total = sum(later.values())
    if not total:
        return math.nan
    return sum(n for c, n in later.items() if baseline.get(c, 0) == 0) / total


def mix_difference_table(counts: Mapping[str, Counter], candidates: list[Period],
                         later: list[Period]) -> pd.DataFrame:
    """Candidates x later periods, in percent, with the mean and the candidate's size."""
    rows = {c.name: {p.name: 100 * mix_difference(counts[c.name], counts[p.name]) for p in later}
            for c in candidates}
    df = pd.DataFrame(rows).T
    df["mean"] = df.mean(axis=1)
    df["articles"] = [sum(counts[c.name].values()) for c in candidates]
    return df.round(1).astype({"articles": int})


def stability_table(counts: Mapping[str, Counter], candidates: list[Period]) -> pd.DataFrame:
    """Mix difference between each candidate's first and second half, in percent."""
    rows = []
    for c in candidates:
        first, second = c.halves()
        rows.append({"candidate": c.name,
                     "halves": f"{first.start}..{first.end} vs {second.start}..{second.end}",
                     "mix_difference_pct": round(100 * mix_difference(
                         counts[first.name], counts[second.name]), 1)})
    return pd.DataFrame(rows).set_index("candidate")


def unmatched_table(counts: Mapping[str, Counter], candidates: list[Period],
                    later: list[Period]) -> pd.DataFrame:
    """Percent of each later period in categories missing from each candidate."""
    rows = {c.name: {p.name: round(100 * unmatched_share(counts[c.name], counts[p.name]), 1)
                     for p in later} for c in candidates}
    return pd.DataFrame(rows).T


def category_share_table(counts: Mapping[str, Counter], periods: list[Period]) -> pd.DataFrame:
    """Categories x periods, percent of each period's records, sorted by overall size."""
    df = pd.DataFrame({p.name: {c: 100 * s for c, s in shares(counts[p.name]).items()}
                       for p in periods}).fillna(0.0)
    order = pd.DataFrame({p.name: counts[p.name] for p in periods}).fillna(0).sum(axis=1)
    return df.loc[order.sort_values(ascending=False).index].round(1)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(config["datasets"]),
                        default=config.get("active_dataset") or "huffpost")
    parser.add_argument("--path", type=Path, help="override the dataset file from config")
    parser.add_argument("--out-dir", type=Path,
                        help="where to write CSVs (default: <runs>/explore/<dataset>/baselines)")
    args = parser.parse_args(argv)

    path = resolve_path(args.path or config["datasets"][args.dataset]["file"])
    if not path.exists():
        print(f"Dataset file not found: {path}", file=sys.stderr)
        return 2

    candidates = [Period.from_config(d) for d in config["exploration"]["baseline_candidates"]]
    later = [Period.from_config(d) for d in config["periods"]["later"]]
    halves = [h for c in candidates for h in c.halves()]
    normalise = (config.get("categories") or {}).get("normalise") or {}
    counts = count_categories(iter_records(args.dataset, path), candidates + later + halves,
                              normalise)

    mix = mix_difference_table(counts, candidates, later)
    stability = stability_table(counts, candidates)
    unmatched = unmatched_table(counts, candidates, later)
    category_shares = category_share_table(counts, candidates + later)

    pd.set_option("display.width", 200)
    print(f"\n=== Baseline candidates: {args.dataset} ({path}) ===")
    print("\n1. Category-mix difference vs each later period "
          "(% of articles; 0 = same mix, 100 = nothing in common)")
    print(mix.to_string())
    print("\n2. Internal stability: first half vs second half of each candidate (%)")
    print(stability.to_string())
    print("\n3. % of each later period in categories the candidate does not contain")
    print(unmatched.to_string())
    print("\n4. POLITICS share of each period (%)")
    if "POLITICS" in category_shares.index:
        print(category_shares.loc[["POLITICS"]].to_string())

    out_dir = args.out_dir or (resolve_path(config["paths"]["runs"]) / "explore"
                               / args.dataset / "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    mix.to_csv(out_dir / "mix_difference.csv", index_label="candidate")
    stability.to_csv(out_dir / "internal_stability.csv")
    unmatched.to_csv(out_dir / "unmatched_share.csv", index_label="candidate")
    category_shares.to_csv(out_dir / "category_shares.csv", index_label="category")
    print(f"\nTables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
