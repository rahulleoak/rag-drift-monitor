# UML diagrams

Three views of the final codebase, drawn from the code itself (imports in
`driftmon/`, the report classes in `driftmon/schemas.py`, the graph in
`driftmon/graph.py`). GitHub renders them; for the reasons behind the design,
see [`architecture.md`](architecture.md) and [`decisions.md`](decisions.md).

1. [Packages and dependencies](#1-packages-and-dependencies): which module uses which.
2. [Classes](#2-classes): the reports the agents pass to each other, and the core classes.
3. [Sequence of one run](#3-sequence-of-one-run): what happens, in order, including the evidence loop.

## 1. Packages and dependencies

An arrow means "imports from". Every package also reads `config.py`, and
every agent reads the study periods from `data/` (both omitted for clarity).
`data/`, `drift/` and `evaluation/` share helpers in both directions (for
example the drift statistics are reused to measure query drift). The rule the layout enforces: statistics and
decisions (`drift/`, `evaluation/`, `policy.py`) are plain Python;
the LLM is reached only through `llm/`, and only to write, judge and explain.

```mermaid
flowchart TB
    subgraph ENTRY["Entry points"]
        SCRIPTS["scripts/<br/>run_pipeline, export_demo,<br/>experiments (wording, reindex, length)"]
        APP["app/streamlit_app.py<br/>dashboard"]
    end
    subgraph ORCH["Orchestration"]
        GRAPH["graph.py<br/>LangGraph: nodes, routes, loop"]
    end
    subgraph AGENTS["agents/ (thin LangGraph nodes)"]
        A["data_monitoring (A)"]
        B["drift_detection (B)"]
        C["retrieval_eval (C)"]
        D["impact_assessment (D)"]
        E["recommendation (E)"]
    end
    subgraph LOGIC["Tested logic (no LLM decisions)"]
        DATA["data/<br/>records, cleaning, periods,<br/>monitoring, queries, simulate"]
        DRIFT["drift/<br/>embedding, lexical, severity,<br/>topics, relations, facts, length"]
        EVAL["evaluation/<br/>retrieval, answers, outdated,<br/>reindex, wording, labels, stats"]
        POLICY["policy.py<br/>one rule per action"]
    end
    subgraph INFRA["Infrastructure"]
        RAG["rag/<br/>embeddings, index (FAISS), fault"]
        LLM["llm/<br/>client (cache, batch, budget), prompts"]
        LONG["longform/<br/>Wikipedia, chunking, evidence"]
    end
    SCHEMAS["schemas.py<br/>typed reports + PipelineState"]
    DASH["dashboard/<br/>runs, story, live charts"]

    SCRIPTS --> GRAPH
    SCRIPTS --> DATA & DRIFT & EVAL & RAG & LLM & LONG & DASH
    APP --> DASH
    APP --> POLICY
    GRAPH --> A & B & C & D & E
    A --> DATA & DRIFT
    B --> DRIFT & DATA
    C --> EVAL & RAG
    D --> EVAL & LLM
    E --> POLICY & LLM
    DATA --> DRIFT & EVAL & RAG & LLM
    DRIFT --> DATA & EVAL & RAG
    EVAL --> DATA & DRIFT & RAG & LLM
    POLICY --> EVAL
    DASH --> POLICY & LONG
    GRAPH --> SCHEMAS
    AGENTS --> SCHEMAS
    DRIFT --> SCHEMAS
    POLICY --> SCHEMAS
    DASH --> SCHEMAS
```

## 2. Classes

The agents communicate only through these typed reports, held in one shared
state. Each agent is a function `(PipelineState) -> partial state` that reads
earlier reports and adds its own.

```mermaid
classDiagram
    direction LR

    class PipelineState {
        <<TypedDict>>
        config
        docs, vectors
        questions, question_vectors
        llm : LLMClient
        monitoring : MonitoringReport
        drift : DriftReport
        next_step : str
        retrieval : RetrievalReport
        impact : ImpactReport
        deepen : list~str~
        evidence_loops : int
        longform, label_checks
        recommendations : RecommendationReport
    }

    class MonitoringReport {
        baseline : str
        periods : list~str~
        period_summary
        data_quality
        renamed_sections
        segments
        query_stream
    }
    class DriftReport {
        whole_period : list~SegmentDrift~
        segments : list~SegmentDrift~
        drifted_segments : list~str~
        control_segments : list~str~
        any_flagged : bool
        topics
        term_shifts
        query_drift
    }
    class SegmentDrift {
        segment_id : str
        times_ordinary_embedding : float
        times_ordinary_lexical : float
        severity : low|medium|high
        flagged : bool
        emerging_terms, declining_terms
        examples
    }
    class RetrievalReport {
        mode : full_evaluation|regression_check
        known_item : MRR, Recall@10
        judged_by_period : Precision@5, nDCG@5
        ranking_change
        worst_queries
    }
    class ImpactReport {
        threshold : float
        by_period
        segments : class and cause
        classes
        borderline : list~str~
        evidence : the loop's conclusions
        correlations
    }
    class RecommendationReport {
        recommendations : list~Recommendation~
        summary
        explanations
    }
    class Recommendation {
        action : str
        status : recommended|not needed|not checked
        rule : str
        finding : str
        scope : list~str~
        evidence : list~EvidenceItem~
        explanation : str
    }
    class EvidenceItem {
        source : str
        claim : str
    }

    PipelineState o-- MonitoringReport : A writes
    PipelineState o-- DriftReport : B writes
    PipelineState o-- RetrievalReport : C writes
    PipelineState o-- ImpactReport : D writes
    PipelineState o-- RecommendationReport : E writes
    DriftReport *-- SegmentDrift
    RecommendationReport *-- Recommendation
    Recommendation *-- EvidenceItem

    class StudyIndexes {
        stale : VectorIndex
        refreshed : one VectorIndex per year
    }
    class VectorIndex {
        doc_ids
        search(queries, k)
    }
    class Embedder {
        embed_documents(texts)
        embed_queries(texts)
    }
    class LLMClient {
        budget, spent()
        estimate(requests)
        complete(request) LLMResult
        complete_many(requests, batch) list~LLMResult~
    }
    class LLMRequest {
        job, model, prompt
        schema : pydantic model
        key : cache key
    }
    class LLMResult {
        parsed
        cost_usd
        cached : bool
    }
    class Inputs {
        <<policy.py>>
        monitoring, drift, retrieval, impact
        longform, labels
        segments()
    }
    class Run {
        <<dashboard>>
        run_id, info
        monitoring, drift, retrieval, impact, recommendations
        flagged()
    }

    StudyIndexes *-- VectorIndex
    LLMClient ..> LLMRequest : sends
    LLMClient ..> LLMResult : returns (cached on disk)
    PipelineState ..> LLMClient
    Inputs ..> ImpactReport : rules read
    Inputs ..> DriftReport
    Inputs ..> RetrievalReport
    Run ..> RecommendationReport : loads a saved run
```

## 3. Sequence of one run

`python scripts/run_pipeline.py`: the graph runs A to E; after B it chooses
the full or a small check; after D a borderline section goes back to C once
(the evidence loop, D28; it fired only on the simulated failed re-index,
F26). The LLM is called only where shown; every reply is cached, so a rerun
costs nothing.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant S as run_pipeline.py
    participant G as LangGraph
    participant A as A. Data Monitoring
    participant B as B. Drift Detection
    participant C as C. Retrieval Eval
    participant D as D. Impact Assessment
    participant E as E. Recommendation
    participant L as LLM (Sonnet writes, Opus judges)
    participant R as runs/ folder

    U->>S: run
    S->>S: load documents, embeddings, questions, long-form evidence
    S->>G: invoke(state)
    G->>A: state
    A-->>G: MonitoringReport (mix, length, new sections, query stream)
    G->>B: state + MonitoringReport
    B-->>G: DriftReport (two techniques, severity, flagged + control sections)
    G->>G: route: any flagged? full evaluation : regression check
    G->>C: state + DriftReport
    C->>C: stale + refreshed FAISS indexes, top 10 per question
    C->>L: judge pooled top 5 (Opus, blind, shuffled)
    L-->>C: relevance grades
    C-->>G: RetrievalReport (Precision@5, nDCG@5, MRR, Recall@10)
    G->>D: state + RetrievalReport
    D->>L: answer from each index (Sonnet), grade each answer (Opus)
    L-->>D: answers + grades
    D->>D: loss per section with interval, class, cause, drift vs loss
    D-->>G: ImpactReport
    alt a section is borderline (loss past the threshold, interval reaches 0)
        G->>C: request evidence (once)
        C->>L: grade ranks 6-10
        C-->>G: deeper grades
        G->>D: answer from the top 10
        D->>L: answers + grades
        D-->>G: conclusion: confirmed / just outside the top 5 / escalate
    end
    G->>E: all reports + long-form evidence + hand-label checks
    E->>E: policy.py: one rule per action, evidence attached
    E->>L: explain each recommended action (Sonnet)
    L-->>E: explanations (numbers checked against the evidence)
    E-->>G: RecommendationReport
    G-->>S: final state
    S->>R: reports (JSON, CSV, Markdown), run_info.json
    Note over R: the dashboard reads saved runs only:<br/>no model or key needed to view them
```
