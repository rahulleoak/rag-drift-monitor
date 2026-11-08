"""Copy one pipeline run into ``demo/runs/`` so the dashboard works after a fresh clone.

The dashboard reads ``runs/`` first and falls back to ``demo/runs/`` when
there are no local runs, so a reviewer can open it without an API key, a
GPU or a rerun. Only what the dashboard reads is copied (the reports, the
dark-theme charts and the long-form snapshot), about 1 MB; the large
per-question tables stay in ``runs/``.

Usage (from the repository root):
    python scripts/export_demo.py                     # the latest complete run
    python scripts/export_demo.py --run 20261107-202242
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/export_demo.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402

FILES = ["monitoring.json", "drift.json", "route.txt", "retrieval.json", "impact.json", "recommendations.json",
         "recommendations.md"]
FOLDERS = ["charts/dark", "longform"]


def latest_complete(runs: Path) -> Path:
    """The newest run with Agent E's report."""
    done = sorted(p.parent for p in runs.glob("*/recommendations.json"))
    if not done:
        raise SystemExit(f"No run with recommendations.json in {runs}: run scripts/run_pipeline.py first")
    return done[-1]


def export(run: Path, demo: Path) -> list[str]:
    """Replace ``demo`` with the dashboard's files from ``run``; return what was copied."""
    missing = [f for f in FILES if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run.name} is missing {', '.join(missing)}")
    if demo.exists():
        shutil.rmtree(demo)
    out = demo / run.name
    out.mkdir(parents=True)
    copied = []
    for f in FILES:
        shutil.copy2(run / f, out / f)
        copied.append(f)
    for d in FOLDERS:
        if (run / d).exists():
            shutil.copytree(run / d, out / d)
            copied += [f"{d}/{p.name}" for p in sorted((out / d).iterdir())]
    return copied


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run id (folder name under runs/); default: the latest complete run")
    args = parser.parse_args(argv)
    config = load_config()
    runs = resolve_path(config["paths"]["runs"])
    run = runs / args.run if args.run else latest_complete(runs)
    demo = resolve_path(config["paths"]["demo_runs"])
    copied = export(run, demo)
    size = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file())
    print(f"Copied {len(copied)} files from {run.name} to {demo} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
