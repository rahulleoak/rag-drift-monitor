"""Make the drift charts from the latest pipeline run.

Reads ``runs/<run_id>/`` (from ``scripts/run_pipeline.py``) and the
exploration tables, and writes PNGs in two themes based on the Rivvun AI
brand: light (cream) to ``runs/<run_id>/charts/`` and ``docs/figures/``, and
dark (navy) to the ``dark/`` subfolders, for slides. The ``docs/figures``
copies are small and committed so the README and findings can show them.

Usage (from the repository root):
    python scripts/make_charts.py               # latest run
    python scripts/make_charts.py --run 20261105-112621
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/make_charts.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.dashboard import charts  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.data.periods import study_periods  # noqa: E402

MEASURE_NAMES = {"mmd2": "MMD²", "centroid_shift": "Centroid shift", "jsd_excess": "Word drift (JSD)",
                 "novelty_mean": "Novelty (mean)", "novel_share": "Novel share",
                 "new_term_rate": "New-term rate"}


def _table(path: Path, **kw) -> pd.DataFrame:
    """A saved CSV, or an empty table if the file is missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def latest_run(runs_dir: Path) -> Path:
    """The newest ``runs/<YYYYMMDD-HHMMSS>`` folder that has a drift report."""
    candidates = sorted(p for p in runs_dir.glob("[0-9]*-[0-9]*") if (p / "drift.json").exists())
    if not candidates:
        raise FileNotFoundError(f"no pipeline run in {runs_dir}; run scripts/run_pipeline.py first")
    return candidates[-1]


def make_all(config, run_dir: Path, out_dirs: list[Path], theme: str = "light") -> list[str]:
    """Render every chart for ``run_dir`` in one theme into each folder in ``out_dirs``;
    return the file names."""
    t = charts.THEMES[theme]
    drift = json.loads((run_dir / "drift.json").read_text(encoding="utf-8"))
    null = drift["reference"]["whole_period_null"]
    whole = pd.read_csv(run_dir / "drift_whole.csv", dtype={"period": str, "compared_with": str})
    segs = pd.read_csv(run_dir / "drift_segments.csv", dtype={"period": str, "compared_with": str})
    periods = study_periods(config)
    figures = {
        "drift_per_year.png": charts.drift_per_year(whole, null["thresholds"], null["typical"], t=t),
        "severity_by_section.png": charts.band_heatmap(segs, whole, [p.name for p in periods[1:]],
                                                       periods[0].name, t=t),
    }
    rates = resolve_path(config["paths"]["runs"]) / "explore" / config["active_dataset"] / "watch_term_rates.csv"
    if rates.exists():
        figures["pandemic_words.png"] = charts.term_rates(
            pd.read_csv(rates, index_col="term"), ["coronavirus", "covid", "vaccine"], t=t)
    shifts = _table(run_dir / "term_shifts.csv", dtype={"period": str})
    if len(shifts) and shifts["times_ordinary"].notna().any():
        figures["words_that_changed_meaning.png"] = charts.term_shift_bars(shifts, t=t)
    topics = _table(run_dir / "topics.csv", dtype={"period": str})
    if len(topics):
        figures["emerging_topics.png"] = charts.emerging_topics_bars(topics, t=t)
    corr = _table(run_dir / "correlations.csv", index_col="measure")
    if len(corr) and corr.notna().to_numpy().any():
        n = sum(1 for s in drift["whole_period"] + drift["segments"]
                if s["compared_with"] == periods[0].name)
        figures["measure_correlations.png"] = charts.correlation_heatmap(corr, MEASURE_NAMES, n, t=t)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(out_dirs[0] / name, dpi=160)
        for d in out_dirs[1:]:
            shutil.copyfile(out_dirs[0] / name, d / name)
        charts.plt.close(fig)
    return list(figures)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run id (folder name under runs/); default: latest")
    args = parser.parse_args(argv)
    runs = resolve_path(config["paths"]["runs"])
    run_dir = runs / args.run if args.run else latest_run(runs)
    names = make_all(config, run_dir, [run_dir / "charts", resolve_path("docs/figures")], "light")
    make_all(config, run_dir, [run_dir / "charts" / "dark", resolve_path("docs/figures/dark")], "dark")
    print(f"Charts from {run_dir.name}: {', '.join(names)}")
    print(f"Light theme: {run_dir / 'charts'} and docs/figures/; dark theme: .../dark/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
