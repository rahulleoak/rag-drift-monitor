"""Typed contracts passed between agents.

Each agent writes exactly one report into the LangGraph state, and the next
agents read it. Keeping them here makes the dependencies between agents
explicit and testable. Tables are stored as lists of plain records so every
report can be saved as JSON.

Built so far: MonitoringReport (A), SegmentDrift and DriftReport (B).
Planned: RetrievalReport (C), ImpactReport (D), Recommendation (E).
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Band = Literal["low", "medium", "high"]
Kind = Literal["meaning", "wording", "both", "none"]


class MonitoringReport(BaseModel):
    """Agent A: what came in, compared with the baseline."""

    baseline: str
    periods: list[str]
    period_summary: list[dict[str, Any]]       # volume, length, mix change, new sections, new terms
    data_quality: list[dict[str, Any]]         # cleaning retention per period (empty descriptions etc.)
    renamed_sections: list[dict[str, Any]]     # raw labels merged into another name
    segments: list[dict[str, Any]]             # period x category, with eligibility and reason
    # per-period query volume, length and new-term rate; a note if no question set exists yet
    query_stream: list[dict[str, Any]] | str = "not available (no question set generated yet)"

    def eligible_segments(self) -> list[dict[str, Any]]:
        return [s for s in self.segments if s["eligible"]]


class SegmentDrift(BaseModel):
    """Agent B: drift of one segment (a whole period, or period x category)."""

    segment_id: str
    period: str
    category: str | None = None               # None = the whole period
    compared_with: str
    n_reference: int
    n_period: int
    mmd2: float
    mmd_q: float
    jsd_excess: float
    jsd_q: float
    centroid_shift: float
    novelty_mean: float
    novel_share: float
    new_term_rate: float | None = None
    times_ordinary_embedding: float            # mmd2 / median ordinary change
    times_ordinary_lexical: float              # jsd_excess / median ordinary change
    embedding_band: Band
    lexical_band: Band
    severity: Band
    kind: Kind
    significant: bool                           # both q-values <= alpha
    flagged: bool                               # significant and severity >= medium
    null_window: str                            # "month" or "quarter"
    null_n: int
    emerging_terms: list[str] = Field(default_factory=list)
    declining_terms: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class DriftReport(BaseModel):
    """Agent B: drift per segment, and which segments the next agents should test."""

    whole_period: list[SegmentDrift]
    segments: list[SegmentDrift]
    drifted_segments: list[str]                # flagged segment ids, most severe first
    control_segments: list[str]                # low-severity segments, for comparison in C and D
    reference: dict[str, Any]                  # ordinary-year row and band thresholds used
    any_flagged: bool
    topics: list[dict[str, Any]] = Field(default_factory=list)        # emerging topics (D12)
    term_shifts: list[dict[str, Any]] = Field(default_factory=list)   # term-context shifts (D12)
    metric_correlations: list[dict[str, Any]] = Field(default_factory=list)  # P6, across segments
    # each period's queries vs the baseline queries (driftmon.queries); a note if unavailable
    query_drift: list[dict[str, Any]] | str = "not available (no question set generated yet)"


class PipelineState(TypedDict, total=False):
    """The LangGraph state. Inputs first, then one report per agent."""

    config: dict[str, Any]
    docs: Any          # pandas DataFrame of cleaned documents
    vectors: Any       # numpy array aligned with docs
    tokens: list[list[str]]
    queries: Any       # pandas DataFrame: the query stream (simulated questions), or None
    query_vectors: Any  # numpy array aligned with queries
    monitoring: MonitoringReport
    drift: DriftReport
    next_step: str     # routing decision after B
