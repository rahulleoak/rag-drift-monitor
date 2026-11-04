# Design Decisions

One entry per significant choice, newest at the bottom, numbered so they can be
referenced (e.g. "see D3"). Written in plain language: each entry should be
defensible in a technical review on its own.

If a decision changes later, do not delete it. Mark it **Revised** with the
date and reason, and add a new entry that supersedes it.

## Entry template

```markdown
## D<n>. <Short title>
- **Date:** YYYY-MM-DD
- **Status:** Accepted | Revised (YYYY-MM-DD, see D<m>: reason)
- **Context:** the problem being solved and the constraints.
- **Options considered:**
  1. <Option A>: tradeoffs.
  2. <Option B>: tradeoffs.
- **Choice:** what we picked.
- **Why:** the reasoning.
- **Evidence:** a measurement, test, or failed attempt that supports it
  (with a link to the finding, test or commit).
```

## Decisions

## D1. Project structure: thin agents over pure, tested library modules
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04)
- **Context:** Five agents must genuinely depend on each other. Statistics,
  metrics and decisions must be testable, and the LLM must never make the
  final decision. Six days to the deadline, so the layout must be quick to
  navigate and explain in a 15-minute walkthrough.
- **Options considered:**
  1. *One script per agent with its logic inline.* Fastest to start, but the
     statistics can't be unit-tested without running the whole agent, and LLM
     calls end up scattered everywhere.
  2. *One module per agent that owns its own logic.* Readable, but the maths
     is tied to orchestration, and shared statistics (bootstrap, multiple-
     testing correction) get duplicated across agents.
  3. *Thin agents over pure library modules, with typed reports between them.*
     A little more structure up front.
- **Choice:** Option 3. A `driftmon/` package: `drift/`, `evaluation/` and
  `policy.py` hold pure functions; `agents/` holds thin LangGraph nodes;
  `schemas.py` defines the report each agent hands on; `llm.py` is the only
  code that calls an LLM; `simulate.py` (not an agent) plays the role of
  production traffic. A flat package (no `src/`) with a minimal
  `pyproject.toml` for `pip install -e .`; dependencies stay in
  `requirements.txt`.
- **Why:**
  - Statistics can be tested without LangGraph, models or an API key.
  - The "LLM never decides" rule can be checked by reading imports.
  - The typed reports make the dependencies between agents explicit.
  - Keeping the simulator outside the agents means they cannot peek at ground
    truth, except through the golden evaluation set, which is how a real team
    would maintain an eval suite.
- **Evidence:** the first real modules (`driftmon/data.py`,
  `scripts/explore_data.py`) are covered by 10 tests that run in under a second
  with no models or keys (`python -m pytest -q`).

## D2. Retrieval ground truth: known-item plus pooled LLM-judged relevance
- **Date:** 2026-11-04
- **Status:** Accepted for now; open to revision once results exist
  (docs/questions.md Q9)
- **Context:** A stale index cannot contain a document published after it was
  built, so "did we retrieve the question's source document?" scores zero on
  the stale index by construction. That alone would make the stale-vs-refreshed
  comparison trivially one-sided.
- **Options considered:**
  1. *Known-item only.* Cheap and exact, but meaningless for the stale index on
     new-period questions.
  2. *LLM-judged graded relevance of the pooled top-k from both indexes*
     (TREC-style pooling). A fair comparison, because an older document can
     still answer an evergreen question. Costs LLM calls; recall is relative to
     the pool.
  3. *Same category counts as relevant.* Free but far too lenient.
- **Choice:** 1 + 2. Known-item MRR and Recall@K on the refreshed index measure
  whether the retriever copes with new vocabulary. Pooled judgements give
  Precision@K and nDCG@K for the stale-vs-refreshed comparison.
- **Why:** each method answers a different question, and together they make
  "which drift matters" testable: segments with evergreen content should lose
  little on the stale index; segments with novel content should lose a lot.
- **Evidence:** reasoning only so far; to be checked against the first results.

## D3. Choose the dataset and periods only after exploring the candidates
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04). The choice itself is still open
  (docs/questions.md Q1, Q3).
- **Context:** The dataset decides the period boundaries, the sample size per
  period, and whether chunking can be demonstrated. HuffPost is the brief's
  first suggestion, but its documents are very short and, per its dataset card,
  most records predate mid-2018.
- **Options considered:**
  1. *Commit to HuffPost now.* Saves time, but risks discovering too late that
     the post-2018 periods are too thin.
  2. *Explore HuffPost and arXiv with one script, then decide.* About an hour
     of extra work.
  3. *Also explore Amazon reviews.* Its drift is mostly product mix and
     sentiment, a poor fit for a knowledge assistant, and it is a large
     download. Can be added if wanted.
- **Choice:** Option 2. Both datasets are mapped onto one record shape
  (`driftmon/data.py`), so the exploration script and later the pipeline treat
  them identically, and switching is a config change.
- **Decision criteria, written down before looking at the data:**
  1. Enough records in each later period for an equal-size sample
     (`periods.sample_per_period`, 2,000 while building).
  2. Known drift events inside the range to validate the detectors against.
  3. Category labels usable as segments: stable enough over time, with enough
     records per period × category.
  4. Document length: whether chunking can be meaningfully demonstrated.
  5. Cost: can it be embedded and evaluated on a CPU within the week?
- **Evidence:** pending. Exploration is blocked until the dataset files are
  available in this environment (see journal 2026-11-04).

## D4. Dependency versions: ranges now, exact lock later
- **Date:** 2026-11-04
- **Status:** Accepted
- **Context:** The full stack (sentence-transformers, FAISS, LangGraph) has not
  been installed or run together yet.
- **Options considered:**
  1. *Exact pins now.* Reproducible on paper, but untested combinations.
  2. *Lower bounds with major-version caps now, then `pip freeze` into a lock
     file once everything runs end to end.*
- **Choice:** Option 2.
- **Why:** exact pins are only worth something once that exact combination has
  been shown to work.
- **Evidence:** none yet. The lock file is created after the first end-to-end
  run.
