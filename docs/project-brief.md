# Project Brief: Semantic Drift Detection & RAG Quality Assurance

**Timeline:** 1 week (deadline Tuesday, Nov 10, 2026) | **Language:** Python

# 1. Problem Statement

An enterprise organization operates a Retrieval-Augmented Generation
(RAG) based knowledge assistant that answers user queries using a
collection of enterprise documents, including contracts, policies,
procedures and business knowledge.

Over time, the knowledge base and user behavior evolve:

- New documents and information are continuously introduced.

- Business terminology and vocabulary change.

- Existing documents become outdated or are replaced.

- User queries evolve and introduce new terminology.

- Semantic relationships between concepts may change.

These changes can introduce semantic drift, potentially impacting
document retrieval relevance, answer accuracy and overall RAG system
performance.

The goal of this project is to design and develop an Agentic AI-based Semantic Drift
Detection and RAG Quality Assurance System.

The system should continuously monitor the RAG pipeline, detect and
explain semantic drift, evaluate its impact on retrieval and answer
quality, and recommend appropriate corrective actions.

# 2. Functional Requirements

## A. Data Monitoring Agent

- Monitor incoming documents and user queries.

- Maintain historical and recent data distributions.

- Identify changes in vocabulary, topics and document characteristics.

- Establish baseline data for comparison.

## B. Semantic Drift Detection Agent

- Detect changes in semantic representations of historical and recent
  documents.

- Compare embedding distributions using suitable distance or similarity
  metrics.

- Identify emerging topics, changing terminology and shifts in semantic
  relationships.

- Quantify drift and identify potentially affected document categories
  or topics.

- Generate explanations and supporting examples for detected drift.

Implement at least two complementary drift detection techniques.

## C. Retrieval Evaluation Agent

- Evaluate retrieval quality across different time periods.

- Identify changes in document relevance and retrieval rankings.

- Measure retrieval performance using suitable metrics such as
  Precision@K, Recall@K, MRR or NDCG.

- Identify queries for which the retrieval system is producing poor or
  irrelevant results.

## D. RAG Impact Assessment Agent

- Evaluate whether detected semantic drift actually affects RAG
  performance.

- Compare answer quality across historical and recent evaluation
  datasets.

- Identify potential degradation in answer relevance, correctness and
  groundedness.

- Establish relationships between drift scores and retrieval or answer
  quality.

- Distinguish between significant semantic drift and drift that does not
  materially impact system performance.

## E. Recommendation Agent

Based on the findings from other agents, recommend appropriate
corrective actions, such as:

- Continue monitoring without intervention.

- Update or re-index the knowledge base.

- Improve document chunking or embedding strategies.

- Update retrieval configurations.

- Improve prompts or retrieval strategies.

- Trigger model evaluation or retraining where appropriate.

- Escalate cases for human review.

Every recommendation should include supporting evidence and an
explanation.

# 3. Dataset and Experimental Setup

Publicly available datasets may be used to demonstrate the solution.

Suggested datasets:

1.  HuffPost News Category Dataset (2012–2022): Contains news headlines,
    descriptions, categories and publication timestamps, allowing
    temporal analysis of changing topics and terminology.

> <u>https://www.kaggle.com/datasets/rmisra/news-category-dataset</u>

2.  Arxiv research abstracts across different publication years.

3.  Amazon reviews across different time periods.

Another suitable dataset may be used, provided its relevance is
explained.

For the experiment:

- Establish a historical baseline.

- Introduce subsequent time periods as new incoming data.

- Build a baseline RAG pipeline.

- Simulate evolving documents and user queries.

- Detect semantic drift.

- Evaluate retrieval and answer quality across time periods.

Clearly document the assumptions and evaluation methodology.

# 4. Technical Expectations

The technology stack and architecture are open.

Suggested technologies:

| Component           | Suggested Technologies               |
|---------------------|--------------------------------------|
| Programming         | Python                               |
| Agent orchestration | LangGraph / AutoGen                  |
| LLM                 | OpenAI / Azure OpenAI or equivalent  |
| Embeddings          | Sentence Transformers / Hugging Face |
| RAG framework       | LangChain or custom implementation   |
| Vector database     | Qdrant / FAISS / Chroma              |
| Data processing     | Pandas / NumPy                       |
| Evaluation          | RAGAS or custom evaluation metrics   |
| Visualization       | Streamlit / React                    |
| Monitoring          | Custom dashboards / Evidently        |

Not every technology listed needs to be used.

# 5. Expected Deliverables

- Working Application: A functional prototype demonstrating semantic
  drift detection and RAG quality monitoring.

- Source Code: Complete, executable Python implementation with an
  appropriate project structure.

- Agent Architecture: Architecture diagram illustrating specialized
  agents, their responsibilities, communication and orchestration.

- Drift Analysis: Drift metrics, visualizations, explanations and
  examples of detected changes.

- RAG Evaluation: Retrieval and answer quality comparisons across
  different time periods.

- Impact Assessment: Evidence showing whether detected semantic drift
  actually affects RAG performance.

- Recommendations: Actionable recommendations generated by the system,
  with supporting evidence.

- Technical Documentation: README covering setup, execution, technology
  choices, design decisions, assumptions and limitations.

- Demonstration: A 15–20 minute stakeholder walkthrough of the working
  solution and its key technical decisions.

# 6. Success Criteria

The solution should be a working Agentic AI system rather than a
collection of independent LLM prompts. It should provide:

- Meaningful collaboration between specialized agents.

- Quantifiable and explainable semantic drift detection.

- Measurable RAG retrieval and answer quality evaluation.

- Evidence-based impact assessment.

- Intelligent recommendations and decision-making.

- Sound engineering practices and architectural decisions.

**Important:** The objective is not to maximize the number of agents or
achieve a predefined drift score. The priority is solving a real
enterprise AI problem with technical depth and sound engineering
decisions, translating AI research into a practical application.
