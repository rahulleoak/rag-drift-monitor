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
| Monitor incoming documents and user queries | Period-by-period article batches; simulated query stream summarised the same way | Documents **done**; queries: code **built** (D24), numbers after the full question set |
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
| Evaluate retrieval quality across time periods | Per period, stale vs refreshed index; historical and recent questions (D25) | **Done** (F15) |
| Identify changes in document relevance and retrieval rankings | Pooled LLM relevance judgements; top-K overlap, newer-article share, displaced source articles (D7, D25) | **Done** (F15) |
| Precision@K, Recall@K, MRR or NDCG | All four (D2) | **Done** (F15) |
| Identify queries with poor or irrelevant results | Worst queries per segment, with what was retrieved | **Done** (retrieval.json, dashboard) |

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
| Improve document chunking or embedding strategies | Embedding: via model evaluation below. Chunking: Wikipedia long-form set with a chunk-size experiment (D26, stakeholder request) | Designed |
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
| Working application | LangGraph pipeline + Streamlit dashboard | In progress (A, B and the dashboard skeleton run) |
| Source code with an appropriate structure | `driftmon/`, `scripts/`, `tests/` (D1) | In progress |
| Agent architecture diagram | `docs/architecture.md` (Mermaid) | **Done**, kept up to date |
| Drift analysis: metrics, visualisations, explanations, examples | Agent B outputs, charts, dashboard | Metrics, charts, examples **done** (F4–F6, `docs/figures/`); LLM explanations to do |
| RAG evaluation across periods | Agent C and D outputs | To do |
| Impact assessment evidence | Agent D output, findings | To do |
| Recommendations with evidence | Agent E output | To do |
| README: setup, execution, technology choices, decisions, assumptions, limitations | `README.md`, built from the docs | To do |
| 15–20 minute walkthrough | Slides built from findings and decisions | To do |

## Must be explicit in the README and walkthrough

Requirements the design meets but a reviewer could miss unless they are
named directly (checked against the assignment PDF, 2026-11-06):

| Brief wording | Where we meet it | What to say |
|---|---|---|
| "enterprise documents, including contracts, policies, procedures" / "solve a real enterprise AI problem" | News archive as a stand-in (HuffPost is the brief's first suggested dataset) | The mapping: newer reporting supersedes older ↔ new policies replace old ones; "coronavirus → covid" ↔ renamed business terms; "who holds this office" ↔ "which contract version is current" (presentation §12b) |
| "Existing documents become outdated or are replaced" | Time-sensitive questions (D20) on the stale vs refreshed archive | Name the requirement and show an outdated-answer example |
| "User queries evolve and introduce new terminology" / "Monitor incoming documents and user queries" | Query drift: the same two techniques on each period's incoming questions | Required, not optional (journal correction, 2026-11-06) |
| "continuously monitor" | Periods replayed as incoming batches | State it as an assumption; a deployment would schedule it (§12b) |
| "Improve document chunking" (stakeholder: long-form documents required) | Wikipedia long-form set, chunked; four-way chunking experiment (D26) | Explain the chunking strategy and how it changes what drift detection sees |

## Extensions, only if ahead of schedule

Not required by the brief; listed so they are not forgotten. **To revisit
together once the whole deliverable works end to end (pipeline and
dashboard), as agreed with the author on 2026-11-08.**

| Extension | Why | Source |
|---|---|---|
| Entity co-occurrence graph per period for a few key entities (networkx) | A visual complement to term-context shift: neighbours gained and lost (e.g. "Biden" with Obama in 2016–17, Congress in 2021) | D21 (colleague's knowledge-graph suggestion, lightweight version) |
| Keyword search (BM25) next to the dense retriever | Tests whether wording drift hurts keyword search but not meaning-based search | P2 |
| Fine-tuning experiment for the "recommend retraining" action | Evidence that retraining would help, with train and test from different periods | D7 |
| Baseline-candidates chart | Visual for the baseline decision | F2 |
| Newer-wording questions about older topics (e.g. "COVID" asked of an article that says "coronavirus") | Tests the one case where a change in wording itself could hurt search; F15 found drift level did not predict harm, and this would check that claim where it is most likely to fail. Needs a few extra LLM questions | F15 (author, 2026-11-06: do if time allows). **Done 2026-11-09 (D31, F23):** 18 of 370 questions had a natural newer wording; search lost the article once (rank, not the article, moves) |
| Larger Wikipedia sample (165 articles now, from about 1,000 vital articles) | Tightens the per-article rates (changed facts, chunking comparison); hours of fetching and re-running | Author, 2026-11-08 |
| Bonus: targeted re-index of the 90 articles with changed facts, then re-run the outdated-answer questions | The brief's "apply one fix and re-check"; Agent E already recommends it | Asked of Ani (email, 2026-11-07). **Done 2026-11-09 (D32, F24):** outdated answers 12.9% -> 3.0% for 64% of the work; changed facts 68% -> 0% |
| Second embedding model and the model-evaluation loop from E back to C | Designed (D7); not built because no section stays poor after re-indexing (D29). Could be shown by forcing it on one section | D7, D29 |
| More test questions per section | 43 period x section groups had too few questions (under 8) to classify | F20 |
| Test 128-token chunks for retrieval | They tracked edits best (0.89 vs 0.87); Agent E suggests testing them | F16, F22 |
| Changed-facts check for names (e.g. a new office holder), not only numbers and dates | A stated limitation of D27 | D27 |
| Larger hand-label samples; a second labeller for the answer sheet | Samples are 15-20 items per part; the answer sheet has one labeller | F19, F20 |
| The open "revisit" points in D17 (is the unanswerable set useful; do keep rates differ by section; did common knowledge make retrieval easier) and D2 (weight of known-item metrics) | Written down when the results did not exist yet | D17, D2, F11 |
| Let the LLM propose what to investigate (for example which terms to test), code tests it | More visibly agentic without the LLM deciding | D7 (not adopted yet) |

## Success criteria

| Criterion | How it is met | Status |
|---|---|---|
| Working agentic system, not independent prompts | Agents depend on each other's reports; routing decided in code; two loops back to C (borderline impact, model evaluation) | Built: A -> B -> C -> D (loop to C) -> E. The split (decisions in code, LLM for generation, judging and explanation) confirmed by the stakeholder, 2026-11-06 |
| Meaningful collaboration between agents | Each agent's input is the previous agent's report (architecture §3) | Designed |
| Quantifiable and explainable drift detection | Numbers, significance tests and examples for every claim | Designed |
| Measurable retrieval and answer quality | Retrieval metrics and judged answer quality, with confidence intervals | Designed |
| Evidence-based impact assessment | Stale vs refreshed experiment; historical control set | Designed |
| Intelligent recommendations and decision-making | Evidence-triggered rules; model evaluation actually run | Designed |
| Sound engineering practices | Tested pure modules, config file, disk caches, documented decisions | In progress (61 tests) |
| Translate research into practice | MMD (Gretton et al.), Fightin' Words (Monroe et al.), TREC pooling, ideas from the author's paper (D5) | Designed |
