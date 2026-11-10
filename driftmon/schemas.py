"""Typed contracts passed between agents.

Each agent writes exactly one report into the LangGraph state, and the next
agents read it. Keeping them here makes the dependencies between agents
explicit and testable. Tables are stored as lists of plain records so every
report can be saved as JSON.

MonitoringReport (A), SegmentDrift and DriftReport (B), RetrievalReport (C),
ImpactReport (D), Recommendation and RecommendationReport (E).
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
    # each emerging / declining term's use per 1,000 words, baseline vs period (dashboard before/after chart)
    term_shares: list[dict[str, Any]] = Field(default_factory=list)
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
    # each period's queries vs the baseline queries (driftmon.data.queries); a note if unavailable
    query_drift: list[dict[str, Any]] | str = "not available (no question set generated yet)"


class RetrievalReport(BaseModel):
    """Agent C: stale vs refreshed retrieval, per period and drift status (D25)."""

    mode: str                                   # "full_evaluation" or "regression_check"
    questions: int                              # answerable questions evaluated
    questions_by_status: dict[str, int]         # flagged / control / other / baseline
    index_store: dict[str, Any]                 # where the indexes live, and whether they were reloaded
    known_item: list[dict[str, Any]]            # (b) MRR / Recall@k on each refreshed index, by who asks
    known_item_by_status: list[dict[str, Any]]  # (b) period questions by drift status
    ranking_change: list[dict[str, Any]]        # (c) baseline questions: overlap, newer share, displaced
    judged_by_period: list[dict[str, Any]] = Field(default_factory=list)   # (a) stale vs refreshed
    judged_by_status: list[dict[str, Any]] = Field(default_factory=list)   # (a) by period x drift status
    judged_by_type: list[dict[str, Any]] = Field(default_factory=list)     # (a) evergreen vs time-sensitive
    judge: dict[str, Any] = Field(default_factory=dict)                    # requests, problems, cost
    worst_queries: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ImpactReport(BaseModel):
    """Agent D: answer quality, stale vs refreshed, and which drift matters (D28)."""

    questions: int                                   # questions answered on both indexes
    sample: str = "all"                              # "all", or how the pilot sample was drawn
    threshold: float                                 # material loss in correct answers (fixed before results)
    by_period: list[dict[str, Any]] = Field(default_factory=list)
    by_status: list[dict[str, Any]] = Field(default_factory=list)       # flagged / control / other / baseline
    by_type: list[dict[str, Any]] = Field(default_factory=list)         # evergreen vs time-sensitive
    unanswerable: list[dict[str, Any]] = Field(default_factory=list)    # declined correctly, by index
    segments: list[dict[str, Any]] = Field(default_factory=list)        # class and likely cause per segment
    classes: dict[str, int] = Field(default_factory=dict)
    borderline: list[str] = Field(default_factory=list)                 # segments that may need more evidence
    evidence: list[dict[str, Any]] = Field(default_factory=list)        # the loop back to C: top 10 instead of top 5
    correlations: list[dict[str, Any]] = Field(default_factory=list)    # drift measure vs quality loss
    examples: list[dict[str, Any]] = Field(default_factory=list)        # answers that show the harm
    judge: dict[str, Any] = Field(default_factory=dict)                 # requests, failures, cost
    notes: list[str] = Field(default_factory=list)


Status = Literal["recommended", "not needed", "not checked", "rejected"]


class EvidenceItem(BaseModel):
    """One piece of evidence: where it comes from, and what it shows (with numbers)."""

    source: str        # the agent report or finding, e.g. "Agent D: answers by period"
    claim: str         # one sentence with the numbers


class Recommendation(BaseModel):
    """Agent E: one action from the brief, decided by a rule (D29)."""

    action: str                                  # the brief's wording
    key: str                                     # stable id, e.g. "reindex"
    status: Status
    rule: str                                    # when the rule fires, in words
    finding: str                                 # what the rule concluded, written by code
    scope: list[str] = Field(default_factory=list)       # sections or documents it applies to
    priority: int = 0                            # 1 = first; 0 = not recommended
    evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: str = ""                        # LLM-written (number-checked), or the finding
    explanation_check: str = ""                  # how the explanation was produced and checked


class RecommendationReport(BaseModel):
    """Agent E: every action considered, recommended ones first."""

    recommendations: list[Recommendation]
    summary: dict[str, int] = Field(default_factory=dict)   # count per status
    explanations: dict[str, Any] = Field(default_factory=dict)   # requests, rejected drafts, cost
    notes: list[str] = Field(default_factory=list)

    def recommended(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.status == "recommended"]


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
    questions: Any     # pandas DataFrame: every kept question (golden set), or None
    question_vectors: Any  # numpy array aligned with questions
    llm: Any           # LLMClient for the judged steps, or None to skip them
    retrieval: RetrievalReport
    retrieval_tables: dict[str, Any]  # per-question tables for Agent D and the run folder
    impact: ImpactReport
    impact_tables: dict[str, Any]     # per-answer and per-segment tables for Agent E and the run folder
    impact_sample: int | None         # pilot: answer only this many questions (None = all)
    deepen: list[str]                 # segments D sent back to C for more evidence (the loop)
    evidence_loops: int               # how many times D has asked C for more evidence
    longform: dict[str, Any]          # long-form evidence for E: changed facts, outdated test, chunking
    label_checks: Any                 # pandas DataFrame: blind hand labels vs the LLM (F19, F20), or None
    recommendations: RecommendationReport
