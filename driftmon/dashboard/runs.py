"""Load saved pipeline runs for the dashboard (no computation, no LLM).

The dashboard only reads what ``scripts/run_pipeline.py`` and
``scripts/make_charts.py`` saved under ``runs/<run_id>/``, so a demo cannot
fail on a live model call (Q16). A copy of one run is committed under
``demo/runs/`` (``scripts/export_demo.py``), so the dashboard works straight
after cloning, without an API key or a rerun.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from driftmon.schemas import DriftReport, ImpactReport, MonitoringReport, RecommendationReport, RetrievalReport

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
    retrieval: RetrievalReport | None = None   # Agent C, if the run included it
    impact: ImpactReport | None = None         # Agent D
    recommendations: RecommendationReport | None = None  # Agent E
    longform: dict[str, Any] = field(default_factory=dict)  # the long-form evidence E used (run/longform/)
    info: dict[str, Any] = field(default_factory=dict)  # run_info.json: models, mode, cost (newer runs)

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
    rj = path / "retrieval.json"
    retrieval = RetrievalReport.model_validate_json(rj.read_text(encoding="utf-8")) if rj.exists() else None
    whole = pd.DataFrame([s.model_dump() for s in drift.whole_period])
    segs = pd.DataFrame([s.model_dump() for s in drift.segments])
    corr = pd.DataFrame(drift.metric_correlations)
    charts = {}  # "<name>" for light PNGs, "dark/<name>" for the dark theme
    if (path / "charts").exists():
        charts = {p.relative_to(path / "charts").with_suffix("").as_posix(): p
                  for p in sorted((path / "charts").rglob("*.png"))}
    impact = _model(path / "impact.json", ImpactReport)
    recs = _model(path / "recommendations.json", RecommendationReport)
    return Run(run_id=run_id, path=path, monitoring=mon, drift=drift, impact=impact, recommendations=recs,
               longform=load_longform_snapshot(path / "longform"),
               next_step=route.read_text(encoding="utf-8").strip() if route.exists() else "unknown",
               whole=whole, segments=segs, topics=pd.DataFrame(drift.topics),
               term_shifts=pd.DataFrame(drift.term_shifts),
               correlations=corr.set_index("measure") if "measure" in corr else corr,
               charts=charts, retrieval=retrieval,
               info=json.loads((path / "run_info.json").read_text(encoding="utf-8"))
               if (path / "run_info.json").exists() else {})


def model_name(model_id: str | None) -> str:
    """Readable model name: 'claude-sonnet-5-5' -> 'Claude Sonnet 5.5'; other ids unchanged."""
    if not model_id:
        return "unknown"
    m = re.fullmatch(r"claude-([a-z]+)-(\d+)-(\d+)", model_id)
    return f"Claude {m.group(1).title()} {m.group(2)}.{m.group(3)}" if m else model_id


def run_models(run: Run, config: dict[str, Any]) -> dict[str, str]:
    """The models behind a run: from its run_info.json, else the current config (older runs).

    Returns readable ``writer``, ``judge`` and ``embedding`` names and ``source``
    ("run" or "config"), so the dashboard can say where the names came from.
    """
    models = run.info.get("models")
    if run.info:
        return {"writer": model_name(models["writer"]) if models else "none (no LLM in this run)",
                "judge": model_name(models["judge"]) if models else "none (no LLM in this run)",
                "embedding": run.info.get("embedding_model", "unknown"), "source": "run"}
    return {"writer": model_name(config["llm"]["generator_model"]),
            "judge": model_name(config["llm"]["judge_model"]),
            "embedding": config["embeddings"]["model"], "source": "config"}


def _model(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


LONGFORM_FILES = {"outdated_summary": "outdated_summary.csv", "outdated_items": "outdated_items.csv",
                  "chunking_eval": "chunking_eval.csv", "longform_summary": "summary.csv",
                  "fact_summary": "fact_changes_summary.csv"}


def load_longform_snapshot(folder: Path) -> dict[str, Any]:
    """The long-form evidence saved with a run (empty if the run has none)."""
    out: dict[str, Any] = {}
    for key, name in LONGFORM_FILES.items():
        if (folder / name).exists():
            out[key] = pd.read_csv(folder / name)
    if (folder / "outdated_verdict.json").exists():
        out["outdated_verdict"] = json.loads((folder / "outdated_verdict.json").read_text(encoding="utf-8"))
    return out


REFRESHED = "Refreshed index"


def stale_label(baseline: str) -> str:
    """Legend name of the stale index, e.g. 'Stale index (2016-17 only)'."""
    return f"Stale index ({baseline} only)"


def _long(rows: list[dict[str, Any]], stale: str, refreshed: str, delta: str | None,
          low: str | None = None, high: str | None = None, baseline: str = "baseline") -> pd.DataFrame:
    out = []
    for r in rows:
        for side, label in ((stale, stale_label(baseline)), (refreshed, REFRESHED)):
            row = {"period": str(r["period"]), "index": label, "value": r[side], "questions": r["questions"]}
            if delta:
                row.update(delta=r[delta], delta_low=r[low or f"{delta}_low"], delta_high=r[high or f"{delta}_high"])
            out.append(row)
    return pd.DataFrame(out)


def stale_vs_refreshed(run: Run, measure: str = "answer") -> pd.DataFrame:
    """Long table for the retrieval stale-vs-refreshed chart: one row per period and index."""
    if run.retrieval is None or not run.retrieval.judged_by_period:
        return pd.DataFrame(columns=["period", "index", "value", "questions"])
    return _long(run.retrieval.judged_by_period, f"stale_{measure}", f"refreshed_{measure}", f"delta_{measure}",
                 baseline=run.baseline)


def answers_stale_vs_refreshed(run: Run) -> pd.DataFrame:
    """Long table for the answer chart: share answered correctly, per period and index.
    The delta is refreshed minus stale (Agent D reports the loss, the same number)."""
    if run.impact is None or not run.impact.by_period:
        return pd.DataFrame(columns=["period", "index", "value", "questions"])
    return _long(run.impact.by_period, "stale_correct", "refreshed_correct", "loss_correct", "loss_low",
                 "loss_high", baseline=run.baseline)


def harm_vs_drift(run: Run) -> pd.DataFrame:
    """Classified sections with their drift score: does more drift mean more harm?
    Sections drift detection could not score (e.g. new since the baseline) have no score."""
    if run.impact is None or not run.impact.segments:
        return pd.DataFrame()
    seg = pd.DataFrame(run.impact.segments)
    seg = seg[~seg["class"].isin(["too few questions", "historical control"])].copy()
    scores = run.segments.set_index("segment_id") if len(run.segments) else pd.DataFrame()
    for col in ("times_ordinary_embedding", "times_ordinary_lexical", "severity"):
        seg[col] = seg["segment_id"].map(scores[col]) if col in scores else None
    seg["section"] = seg["period"].astype(str) + " " + seg["category"]
    return seg.sort_values(["class", "period", "category"])


def severity_grid(run: Run) -> pd.DataFrame:
    """Long table for the section x year severity chart (whole periods as 'ALL SECTIONS')."""
    seg = run.segments[run.segments["compared_with"] == run.baseline] if len(run.segments) else run.segments
    whole = run.whole[run.whole["compared_with"] == run.baseline].assign(category="ALL SECTIONS")
    cols = ["category", "period", "severity", "kind", "flagged", "times_ordinary_embedding",
            "times_ordinary_lexical", "n_period"]
    return pd.concat([whole[cols], seg[cols]], ignore_index=True) if len(seg) else whole[cols]
