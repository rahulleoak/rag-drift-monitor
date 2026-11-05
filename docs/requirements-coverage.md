# Requirements Coverage

Every bullet in the original assignment (.docx), mapped to the part of the
system that meets it. Check this before submitting, and use it in the
walkthrough to show each requirement mapped to the system.

**Status key:** **Done** = built and tested · **Designed** = agreed in
`docs/architecture.md` / `docs/decisions.md`, not built yet · **To do** = a
deliverable not started · **Limitation** = cannot be met with this dataset;
documented instead.

Last checked: 2026-11-04 (gaps found and closed in D7).

## A. Data Monitoring Agent

| Requirement | How it is met | Status |
|---|---|---|
| Monitor incoming documents and user queries | Period-by-period article batches; simulated query stream summarised the same way | Documents **done**; queries wait for the LLM simulator |
| Maintain historical and recent data distributions | Fixed 2016–17 baseline vs each later year | **Done** (Agent A, F5) |
| Identify changes in vocabulary, topics and document characteristics | New-term rate, category mix (chi-square + Cramér's V), length, empty descriptions, new or renamed sections | **Done** (Agent A, F5) |
| Establish baseline data for comparison | 2016–17, chosen on evidence | **Done** (F2, D6) |

## B. Semantic Drift Detection Agent

| Requirement | How it is met | Status |
|---|---|---|
| Detect changes in semantic representations of historical and recent documents | Embedding drift per period and per category | **Done** (Agent B, F4, F5) |
| Compare embedding distributions using distance or similarity metrics | MMD with a permutation test; centroid distance; per-document novelty; dispersion (P3) | **Done** for whole periods (F4) |
| Identify emerging topics | Ranked emerging terms; topic clusters that grow vs the baseline, labelled by top terms (D12) | **Done** (F6) |
| Identify changing terminology | Jensen–Shannon divergence and log-odds emerging / declining terms (e.g. coronavirus → covid) | **Done** for whole periods (F4) |
| Identify shifts in semantic relationships | Term-context shift for key terms (D7, D12) | **Done** (F6) |
| Quantify drift; identify affected categories or topics | Per period × category scores, p-values with Benjamini–Hochberg, severity bands (P7); emerging topics | **Done** (F5, F6) |
| Generate explanations and supporting examples | LLM explains from the computed numbers; most novel articles and top terms as examples | Examples and terms **done**; LLM explanations to do |
| At least two complementary techniques | Embedding-based and lexical; correlation across segments shows how far they differ (P6) | **Done** (F4, F6: MMD vs word drift 0.68); chart with the dashboard |

## C. Retrieval Evaluation Agent

| Requirement | How it is met | Status |
|---|---|---|
| Evaluate retrieval quality across time periods | Per period, stale vs refreshed index; historical and recent questions | Designed |
| Identify changes in document relevance and retrieval rankings | Pooled LLM relevance judgements; top-K overlap, rank correlation, displaced articles (D7) | Designed |
| Precision@K, Recall@K, MRR or NDCG | All four (D2) | Designed |
| Identify queries with poor or irrelevant results | Worst queries per segment, with what was retrieved | Designed |

## D. RAG Impact Assessment Agent

| Requirement | How it is met | Status |
|---|---|---|
| Evaluate whether detected drift affects RAG performance | Stale vs refreshed index on the same questions (key experiment) | Designed |
| Compare answer quality across historical and recent evaluation datasets | Historical questions (from baseline articles) as control vs recent questions (D7) | Designed |
| Identify degradation in relevance, correctness and groundedness | LLM judge, checked against about 40 hand-labelled items (P8) | Designed |
| Relationships between drift scores and retrieval or answer quality | Which drift signal predicts quality loss across segments (P4) | Designed |
| Distinguish significant drift from drift that does not matter | Four-way classification: hurts / does not / quality loss without drift / healthy | Designed |

## E. Recommendation Agent

| Action in the brief | Evidence that triggers it (D7) | Status |
|---|---|---|
| Continue monitoring | Drift, but no material quality loss | Designed |
| Update or re-index the knowledge base | Quality recovers on the refreshed index | Designed |
| Improve document chunking or embedding strategies | Embedding: via model evaluation below. Chunking: not applicable | Designed / **Limitation** (1–2 sentence articles) |
| Update retrieval configurations | Right article usually found just outside the top K | Designed |
| Improve prompts or retrieval strategies | Retrieval fine, answers wrong or ungrounded | Designed |
| Trigger model evaluation or retraining | Evaluation is run by the system (second embedding model on affected segments) when re-indexing does not fix quality, or when the judge's agreement with hand labels is low. Retraining is recommended with evidence when neither model copes; not run by us | Designed |
| Escalate cases for human review | Quality loss without drift, or inconclusive evidence | Designed |
| Every recommendation has evidence and an explanation | Rules cite evidence; a check rejects recommendations without it | Designed |

## Dataset and experimental setup

| Requirement | How it is met | Status |
|---|---|---|
| Public dataset, relevance explained | HuffPost News Category; newsroom archive use case | **Done** (D3, F1) |
| Establish a historical baseline | 2016–17 | **Done** (D6) |
| Introduce later periods as new incoming data | 2018–2022, one year at a time; cleaned, with equal drift samples (F3) | Designed (data prepared) |
| Build a baseline RAG pipeline | bge-small + exact FAISS search; stale and refreshed indexes | Designed (model tested on GPU) |
| Simulate evolving documents and user queries | Period batches; LLM-generated query stream and golden set | Designed |
| Detect semantic drift | Agent B | Designed |
| Evaluate retrieval and answer quality across periods | Agents C and D | Designed |
| Document assumptions and evaluation methodology | `docs/decisions.md`, `docs/findings.md`, then the README | In progress |

## Deliverables

| Deliverable | Where | Status |
|---|---|---|
| Working application | LangGraph pipeline + Streamlit dashboard | To do |
| Source code with an appropriate structure | `driftmon/`, `scripts/`, `tests/` (D1) | In progress |
| Agent architecture diagram | `docs/architecture.md` (Mermaid) | **Done**, kept up to date |
| Drift analysis: metrics, visualisations, explanations, examples | Agent B outputs, charts, dashboard | To do |
| RAG evaluation across periods | Agent C and D outputs | To do |
| Impact assessment evidence | Agent D output, findings | To do |
| Recommendations with evidence | Agent E output | To do |
| README: setup, execution, technology choices, decisions, assumptions, limitations | `README.md`, built from the docs | To do |
| 15–20 minute walkthrough | Slides built from findings and decisions | To do |

## Success criteria

| Criterion | How it is met | Status |
|---|---|---|
| Working agentic system, not independent prompts | Agents depend on each other's reports; routing decided in code; two loops back to C (borderline impact, model evaluation) | Designed. Open: stakeholder question 2 on what counts as "agentic" |
| Meaningful collaboration between agents | Each agent's input is the previous agent's report (architecture §3) | Designed |
| Quantifiable and explainable drift detection | Numbers, significance tests and examples for every claim | Designed |
| Measurable retrieval and answer quality | Retrieval metrics and judged answer quality, with confidence intervals | Designed |
| Evidence-based impact assessment | Stale vs refreshed experiment; historical control set | Designed |
| Intelligent recommendations and decision-making | Evidence-triggered rules; model evaluation actually run | Designed |
| Sound engineering practices | Tested pure modules, config file, disk caches, documented decisions | In progress (57 tests) |
| Translate research into practice | MMD (Gretton et al.), Fightin' Words (Monroe et al.), TREC pooling, ideas from the author's paper (D5) | Designed |
