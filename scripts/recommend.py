"""Run Agent E (Recommendation) on a saved pipeline run.

Reads the reports a run saved (monitoring.json, drift.json, retrieval.json,
impact.json), the long-form evidence and the hand-label checks, runs Agent E
and writes, into the same run folder:
    recommendations.json   the RecommendationReport
    recommendations.md     a readable version

The full pipeline (``scripts/run_pipeline.py``) runs Agent E too; this script
re-runs only E, for example after a rule or the long-form evidence changes.

Usage (from the repository root):
    python scripts/recommend.py                       # the latest run, with LLM explanations
    python scripts/recommend.py --run runs/20261107-143926
    python scripts/recommend.py --no-llm              # rule text only, no LLM calls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/recommend.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.agents.recommendation import recommendation, to_markdown  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.evidence import load_label_checks, load_longform  # noqa: E402
from driftmon.llm import LLMClient  # noqa: E402
from driftmon.schemas import DriftReport, ImpactReport, MonitoringReport, RetrievalReport  # noqa: E402

REPORTS = {"monitoring": ("monitoring.json", MonitoringReport), "drift": ("drift.json", DriftReport),
           "retrieval": ("retrieval.json", RetrievalReport), "impact": ("impact.json", ImpactReport)}


def latest_run(runs: Path) -> Path:
    """The newest run folder that has Agent D's report."""
    done = sorted(p.parent for p in runs.glob("*/impact.json"))
    if not done:
        raise SystemExit(f"No run with impact.json in {runs}")
    return done[-1]


def load_state(run: Path, config: dict) -> dict:
    """The graph state Agent E needs, rebuilt from a run folder."""
    state = {"config": config, "longform": load_longform(config), "label_checks": load_label_checks(config)}
    for key, (name, model) in REPORTS.items():
        if (run / name).exists():
            state[key] = model.model_validate_json((run / name).read_text(encoding="utf-8"))
    return state


def save(report, run: Path) -> None:
    (run / "recommendations.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (run / "recommendations.md").write_text(to_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, help="run folder (default: the latest with impact.json)")
    parser.add_argument("--no-llm", action="store_true", help="no explanations from the LLM")
    args = parser.parse_args(argv)
    config = load_config()
    run = resolve_path(args.run) if args.run else latest_run(resolve_path(config["paths"]["runs"]))
    state = load_state(run, config)
    if not args.no_llm:
        state["llm"] = LLMClient(config=config["llm"], cache_dir=resolve_path(config["paths"]["cache"]) / "llm",
                                 ledger=resolve_path(config["paths"]["runs"]) / "llm_costs.csv",
                                 log=lambda m: print(m, flush=True))
    report = recommendation(state)["recommendations"]
    save(report, run)
    print(to_markdown(report))
    print(f"Explanations: {report.explanations}")
    print(f"Saved recommendations.json and recommendations.md in {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
