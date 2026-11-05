"""Load saved pipeline runs for the dashboard (no computation, no LLM).

The dashboard only reads what ``scripts/run_pipeline.py`` and
``scripts/make_charts.py`` saved under ``runs/<run_id>/``, so a demo cannot
fail on a live model call (Q16).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from driftmon.schemas import DriftReport, MonitoringReport

RUN_PATTERN = "[0-9]*-[0-9]*"


def list_runs(runs_dir: Path) -> list[str]:
    """Run ids (folder names) that contain a drift report, newest first."""
    return sorted((p.name for p in runs_dir.glob(RUN_PATTERN) if (p / "drift.json").exists()),
                  reverse=True)


@dataclass
class Run:
    """Everything the dashboard shows for one run."""

    run_id: str
    path: Path
    monitoring: MonitoringReport
    drift: DriftReport
    next_step: str
    whole: pd.DataFrame
    segments: pd.DataFrame
    topics: pd.DataFrame
    term_shifts: pd.DataFrame
    correlations: pd.DataFrame
    charts: dict[str, Path] = field(default_factory=dict)

    @property
    def baseline(self) -> str:
        return self.monitoring.baseline

    def flagged(self) -> pd.DataFrame:
        """Flagged comparisons vs the baseline, most severe first."""
        both = pd.concat([self.whole.assign(category="ALL SECTIONS"), self.segments],
                         ignore_index=True)
        f = both[(both["flagged"]) & (both["compared_with"] == self.baseline)].copy()
        f["strength"] = f[["times_ordinary_embedding", "times_ordinary_lexical"]].max(axis=1)
        return f.sort_values("strength", ascending=False)


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"period": str, "compared_with": str}) if path.exists() else pd.DataFrame()


def load_run(runs_dir: Path, run_id: str) -> Run:
    """Read one saved run."""
    path = runs_dir / run_id
    mon = MonitoringReport.model_validate_json((path / "monitoring.json").read_text(encoding="utf-8"))
    drift = DriftReport.model_validate_json((path / "drift.json").read_text(encoding="utf-8"))
    route = path / "route.txt"
    whole = pd.DataFrame([s.model_dump() for s in drift.whole_period])
    segs = pd.DataFrame([s.model_dump() for s in drift.segments])
    corr = pd.DataFrame(drift.metric_correlations)
    charts = {}  # "<name>" for light PNGs, "dark/<name>" for the dark theme
    if (path / "charts").exists():
        charts = {p.relative_to(path / "charts").with_suffix("").as_posix(): p
                  for p in sorted((path / "charts").rglob("*.png"))}
    return Run(run_id=run_id, path=path, monitoring=mon, drift=drift,
               next_step=route.read_text(encoding="utf-8").strip() if route.exists() else "unknown",
               whole=whole, segments=segs, topics=pd.DataFrame(drift.topics),
               term_shifts=pd.DataFrame(drift.term_shifts),
               correlations=corr.set_index("measure") if "measure" in corr else corr,
               charts=charts)


def severity_grid(run: Run) -> pd.DataFrame:
    """Long table for the section x year severity chart (whole periods as 'ALL SECTIONS')."""
    seg = run.segments[run.segments["compared_with"] == run.baseline] if len(run.segments) else run.segments
    whole = run.whole[run.whole["compared_with"] == run.baseline].assign(category="ALL SECTIONS")
    cols = ["category", "period", "severity", "kind", "flagged", "times_ordinary_embedding",
            "times_ordinary_lexical", "n_period"]
    return pd.concat([whole[cols], seg[cols]], ignore_index=True) if len(seg) else whole[cols]
