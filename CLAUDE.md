# Project: Semantic Drift Detection & RAG Quality Assurance

Full brief: `docs/project-brief.md`. Deadline: Tuesday, Nov 10, 2026,
followed by a 15–20 minute stakeholder walkthrough of the system and its
key technical decisions.

## How to work on this project
Act as a senior engineer who owns this system end to end:
- Understand the problem before writing code. Ask about anything ambiguous.
- Propose an approach with the main alternatives and tradeoffs, recommend one,
  and wait for approval before writing large amounts of code.
- Prefer simple, working, well-tested solutions over clever or sprawling ones.
  Build thin end to end first, then deepen.
- Push back when a request seems wrong, risky or over-engineered, and say why.
- Explain non-obvious decisions: every design choice must be defensible in a
  technical review.

## Goal
An agentic system that monitors a RAG pipeline, detects and explains semantic
drift in documents and queries, measures whether that drift hurts retrieval and
answer quality, and recommends corrective actions with evidence.

The core question every component should help answer: **which drift matters?**

## Use case (proposed, see docs/decisions.md)
A newsroom archive assistant: journalists and editors ask questions over the
company's own article archive, built from the HuffPost News Category Dataset
(2012–2022). Known drift to validate against: e.g. COVID vocabulary in 2020.

## Agents
A. Data Monitoring · B. Semantic Drift Detection (at least two complementary
techniques) · C. Retrieval Evaluation · D. RAG Impact Assessment · E. Recommendation.
Orchestrated with LangGraph. Agents must genuinely depend on each other's
outputs, not be independent prompts.

## Design principles
- Statistics, metrics and decisions live in plain, tested Python code.
  The LLM generates test queries, writes answers, judges answer quality and
  writes explanations. It never makes the final decision.
- Every drift claim needs a number, a significance check where possible,
  and concrete examples.
- Every recommendation carries its supporting evidence.
- Key experiment: an index built on the baseline period (stale) vs one
  refreshed with new data, to show causal impact of drift.
- Run everything locally. The only external service is an LLM API.
- Keep the data sample small while building; scale up at the end.

## Tech defaults
Python 3.11+, sentence-transformers (small model, e.g. bge-small or
all-MiniLM-L6-v2), FAISS or Chroma, pandas/numpy, LangGraph, Streamlit,
pytest. Cache all embeddings and LLM calls to disk.

## Conventions
- Small, focused modules; type hints; docstrings on public functions.
- Config (paths, periods, model names, thresholds) in one config file, not hard-coded.
- Secrets only via environment variables (`.env`, never committed).
- Large data files are gitignored; a small sample may be committed under `data/sample/`.

## Documentation as you go
These notes feed the README and the stakeholder walkthrough, so update them
as part of the work, not at the end. Write them in plain language.

- `docs/decisions.md`: one entry per significant choice, with date, context
  (the problem), options considered with their tradeoffs, the choice, why,
  and the evidence behind it (a measurement, a test, a failed attempt).
  Mark an entry "Revised" with the reason if a decision changes later.
- `docs/journal.md`: short dated entries, newest first. What was built,
  what was tried and abandoned and why, surprises, open questions.
- `docs/findings.md`: results worth showing, each with the exact numbers,
  the command or notebook that reproduces them, and the path to any chart.
  Note whether each finding supports or contradicts expectations.
- `docs/architecture.md`: keep it and its Mermaid diagram up to date whenever
  the agent design or data flow changes.

At the end of each working session, add a journal entry and update decisions
and findings before stopping.
