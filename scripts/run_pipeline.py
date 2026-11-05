"""Run the LangGraph pipeline and save every agent's report to ``runs/<run_id>/``.

Built so far: Agent A (Data Monitoring) -> Agent B (Drift Detection) -> the
routing decision for Agent C.

Writes:
    monitoring.json    Agent A's MonitoringReport
    drift.json         Agent B's DriftReport
    drift_whole.csv    whole-period drift with bands (one row per comparison)
    drift_segments.csv period x category drift with bands
    route.txt          the routing decision after Agent B

Usage (from the repository root):
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_pipeline.py` without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.corpus import load_corpus  # noqa: E402
from driftmon.graph import build_graph  # noqa: E402

SHOW = ["segment_id", "times_ordinary_embedding", "times_ordinary_lexical", "embedding_band",
        "lexical_band", "kind", "flagged", "emerging_terms"]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    run_dir = resolve_path(config["paths"]["runs"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs, vectors = load_corpus(config)
    state = build_graph().invoke({"config": config, "docs": docs, "vectors": vectors})

    mon, drift = state["monitoring"], state["drift"]
    (run_dir / "monitoring.json").write_text(mon.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "drift.json").write_text(drift.model_dump_json(indent=2), encoding="utf-8")
    whole = pd.DataFrame([s.model_dump() for s in drift.whole_period])
    segs = pd.DataFrame([s.model_dump() for s in drift.segments])
    whole.to_csv(run_dir / "drift_whole.csv", index=False)
    segs.to_csv(run_dir / "drift_segments.csv", index=False)
    (run_dir / "route.txt").write_text(state["next_step"] + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Agent A: periods ===")
    print(pd.DataFrame(mon.period_summary)[["period", "documents", "median_words", "categories",
                                           "new_sections", "new_section_share", "mix_cramers_v",
                                           "new_term_rate"]].round(3).to_string(index=False))
    elig = [s for s in mon.segments if s["eligible"]]
    print(f"\nEligible period x category segments: {len(elig)} of {len(mon.segments)}")
    print("\n=== Agent B: whole periods (bands vs ordinary change inside the baseline) ===")
    print(whole[SHOW].round(1).to_string(index=False))
    print(f"\n=== Agent B: segments ({len(segs)}) ===")
    print(segs[SHOW].round(1).to_string(index=False))
    print(f"\nFlagged (vs baseline): {len(drift.drifted_segments)}; controls: "
          f"{len(drift.control_segments)}; next step: {state['next_step']}")
    print(f"Reports in {run_dir} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
