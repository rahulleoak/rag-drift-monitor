"""Agent A: Data Monitoring.

Reads: the cleaned documents of every period (and, once the simulator
exists, the query stream).
Does: fixes the baseline snapshot; tracks volumes, text lengths, category
mix (chi-square with Cramér's V), new sections, renamed sections, new-term
rate and data quality from the cleaning step; marks which
period x category segments have enough data to analyse.
Writes: ``MonitoringReport``. Consumed by Agent B.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from driftmon import monitoring
from driftmon.config import resolve_path
from driftmon.drift.compare import eligible_segments, tokenize_documents
from driftmon.periods import study_periods
from driftmon.schemas import MonitoringReport, PipelineState


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def data_monitoring(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: build the MonitoringReport (and share the word tokens)."""
    config, docs = state["config"], state["docs"]
    periods = study_periods(config)
    baseline, later = periods[0].name, [p.name for p in periods[1:]]
    tokens = state.get("tokens") or tokenize_documents(docs)

    retention_csv = resolve_path(config["paths"]["runs"]) / "prepare" / "retention.csv"
    quality = pd.read_csv(retention_csv) if retention_csv.exists() else pd.DataFrame()
    segments = eligible_segments(docs, baseline, later, int(config["periods"]["min_segment_size"]))

    report = MonitoringReport(
        baseline=baseline, periods=later,
        period_summary=_records(monitoring.period_summary(docs, baseline, later, tokens)),
        data_quality=_records(quality), renamed_sections=_records(monitoring.renamed_sections(docs)),
        segments=_records(segments))
    return {"monitoring": report, "tokens": tokens}
