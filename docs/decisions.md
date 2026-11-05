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
  `scripts/explore_data.py`, since moved to `scripts/exploration/`) are covered by 10 tests that run in under a second
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
- **Status:** Accepted (approved 2026-11-04). Outcome: HuffPost, with the
  periods in D6.
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
- **Evidence (2026-11-04, F1):** HuffPost checked against the criteria above.
  1. *Records per later period:* partly met. 2019–2021 have about 2,000
     articles each and 2022 has 1,398, so the 2,000 sample size does not fit
     every period.
  2. *Known drift events:* met. COVID terms go from 0% to 18.4% of 2020
     articles; "coronavirus" gives way to "covid" in 2021.
  3. *Usable categories:* partly met. The labels were renamed twice, so they
     need a normalisation map, and from 2019 on only 5–9 categories per year
     reach 50 articles.
  4. *Document length:* not met. The median is 28 words, so chunking cannot
     be shown, as expected.
  5. *Cost:* met. The data loads in about 2 seconds; at most about 75k records
     are ever embedded.
  HuffPost passes on the criteria that matter most (2 and 5) and the gaps
  have workarounds, so arXiv was not downloaded. The dataset and period choice
  are set in D6.

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
- **Evidence:** 2026-11-04: the full stack installed together on Windows 11
  with Python 3.13 (torch 2.11 with CUDA 12.8, sentence-transformers 6.2,
  faiss-cpu 1.15, langgraph 1.2, pandas 3.1). `pip check` is clean and all
  19 tests pass. The exploration outputs are identical under pandas 2.3 and
  3.1. The exact lock file still waits for the first end-to-end run.

## D5. Ideas adopted from the author's paper (proposals P1–P8)
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04)
- **Context:** The author's paper "Semantic Drift in Paraphrasing" measures
  drift between pairs of sentences. This project measures drift between
  collections of documents over time. The ideas carry over; the numbers and
  thresholds do not. Full reasoning is in `docs/questions.md` Part 4.
- **Choice:**
  - **P1:** compare each period with the fixed baseline and with the previous
    period. Whether the two ever disagree enough to matter (slow drift that
    consecutive comparisons miss) is checked once results exist; if the
    difference is negligible, we say so rather than claim the effect.
  - **P2:** label each segment's drift as wording-only, meaning, both or
    neither, using the two drift techniques. Adding BM25 (keyword search) to
    test whether wording-only drift hurts keyword search but not
    meaning-based search is a stretch goal.
  - **P3:** track how spread out each period's embeddings are, and check that
    LLM-generated queries are not much less varied than their source
    documents.
  - **P4:** no fixed-weight composite score. Agent D measures which drift
    signal best predicts quality loss. *Changed from the proposal:* no
    composite for sorting the dashboard either. Segments are sorted by
    severity band, then by the signal D finds most predictive, so the order
    comes from evidence, not hand-picked weights.
  - **P5:** tiered cleaning with a retention table; thresholds set from the
    data as we go.
  - **P6:** a correlation heatmap of the drift metrics across segments. It
    changes no decision at run time. It checks two things: that the two
    techniques measure different things (the brief's "complementary"
    requirement), and how far the P4 analysis can be trusted, since strongly
    correlated signals cannot be told apart.
  - **P7:** low / medium / high severity bands with boundaries set from
    within-baseline variation (for example the 95th and 99th percentiles),
    not fixed constants.
  - **P8:** hand-label about 40 items to check the LLM judge, without seeing
    the judge's verdict first. Estimated about an hour.
- **Why:** each idea answers part of "which drift matters" at little extra
  cost, and the changes from the paper (calibrated bands, no fixed weights)
  avoid importing thresholds that were set for a different problem.
- **Evidence:** none yet; each item is checked as results come in.

## D6. Baseline 2016–17; later periods 2018 to 2022, one per year
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04), **with re-evaluation checks**
  (below). Settles Q3 and, with D3, Q1 (HuffPost).
- **Context:** the baseline is what the stale index is built on and what every
  later period is compared with. HuffPost's topic mix and section names
  change a lot over time (F1): politics goes from 0% of articles in 2012–13 to
  25–40% from 2016 on, lifestyle sections almost vanish after 2018, and
  sections were renamed. If the baseline's mix differs a lot from the later
  years, the drift detectors mostly measure that change in mix, not a change
  in meaning. Volume after 2018 is about 2,000 articles a year, so the later
  periods are fixed by the data: one per year, 2018 to 2022.
- **Options considered** (numbers from F2: mean mix difference with the later
  years, and how much the candidate differs between its own two halves):
  1. *2012–13* (83.1%, 12.5%). Rejected: no politics coverage at all, almost
     nothing in common with the later years.
  2. *2014–15, the earlier proposal* (50.9%, 27.5%). Rejected: half the
     articles would have to change category to match the later years, and the
     two years differ a lot from each other (2014 is when sections were
     renamed).
  3. *2015 only* (43.4%, 30.1% between half-years). The runner-up from the
     earlier discussion: it keeps the 2016 election as a test event and gives
     seven later periods. Rejected: worse mix match, and it is the least
     stable candidate internally.
  4. *2015–16* (36.8%, 17.3%). Rejected: worse on both measures than 2016–17
     for no gain.
  5. *2017 only* (29.5%, 5.6% between half-years). **Closest runner-up.** Same
     mix match as 2016–17 with half the articles (29,889, still far more than
     needed), and it keeps the election year out of "normal" variation.
     Rejected because a single year can only be split into half-years, so
     the normal-variation
     estimate used to set the severity bands (P7) would come from half-year
     differences, while the later periods are a year apart. That would
     probably understate normal variation, set the bands too tight and flag
     too much as drift.
  6. *2016–17* (30.2%, 13.5% between 2016 and 2017). **Chosen.**
- **Choice:** baseline 2016-01-01 to 2017-12-31 (61,987 articles); later
  periods 2018, 2019, 2020, 2021 and 2022 (ends 2022-09-23). Set in
  `config/config.yaml` under `periods`. Renamed sections are merged using
  `categories.normalise` for analysis; raw labels are kept.
- **Strengths:**
  - Closest topic mix to the later years among the two-year options.
  - Stable within itself, and its two halves are a year apart like the later
    periods, so "normal year-to-year variation" for the severity bands is
    measured at the right time scale.
  - Section names were already settled by 2016 (the 2014 renaming is outside
    it).
  - A realistic story: the archive assistant's index was built at the end of
    2017 and never refreshed.
  - Removes a distraction: "trump" is already in 17–24% of baseline articles,
    so the detectors flag what is new later on (COVID in 2020, the shift from
    "coronavirus" to "covid" and vaccines in 2021).
  - Plenty of data: 61,987 articles, far more than the per-period sample.
- **Weaknesses:**
  - The mix still differs by about 30%. About half of that is sections the
    baseline does not contain at all (mainly U.S. NEWS, new in 2018).
    Handled by measuring drift within each category as well as per period,
    and by reporting new sections as a structural change (Data Monitoring)
    rather than scoring them as semantic drift.
  - The 2016 election and #MeToo are inside the baseline, so they cannot be
    used as test events. COVID (2020) and the 2021 change in wording remain.
  - The election year is part of "normal" variation (2016 vs 2017), which
    could make the severity bands too loose.
  - Only five later periods, about 1,400–2,000 articles each; 2018 is a
    transition year (the site changed mid-year).
- **Re-evaluate the baseline if any of these happen** (checked once drift
  results exist; outcome recorded here):
  1. *Mix still dominates:* whole-period drift is high but drift within
     categories is low in most categories. Then the period-level numbers
     mostly show the change in mix. Response: reweight the baseline to each
     period's mix, or rely on the within-category numbers.
  2. *Bands too loose:* the bands set from 2016 vs 2017 are so wide that 2020
     (known COVID drift) is not flagged at least "medium". Response: consider
     2017 alone or a different estimate of normal variation.
  3. *2018 is dominated by the site change:* its drift comes mostly from
     structure (volume, new sections) rather than content.
  4. *U.S. NEWS is mostly politics in disguise* (checked with embeddings).
     Response: map it to POLITICS, which shrinks the unmatched share.
  5. *Length explains the drift* (added 2026-11-04, F3): baseline articles
     have a median of 24 words, 2019–2022 have 29–31. If embedding drift
     largely disappears when the baseline sample is matched to later
     lengths, length (format) rather than meaning is driving it. Response:
     length-matched baseline samples.
- **Evidence:** F2; `python scripts/exploration/compare_baselines.py`.

## D7. Close the gaps found by checking the design against the assignment
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04)
- **Context:** before building, every bullet in the original assignment
  (.docx) was mapped to the design (`docs/requirements-coverage.md`). Most
  were covered. Six were missing, under-specified, or had been set aside as
  stretch goals although the brief requires them.
- **Choices:**
  1. **Shifts in semantic relationships (Agent B) are required, not a
     stretch goal.** Term-context shift: for key terms ("mask", "remote",
     "vaccine", "zoom", plus the top emerging terms) measure how far the
     meaning of the articles containing them moves between periods, and show
     their neighbouring words before and after. Reuses the embeddings.
     Replaces the "stretch" status of Q7.
  2. **Query drift (Agents A and B).** B's techniques also run on the query
     stream. The simulator also writes some questions in newer wording about
     older topics (users' language ahead of the archive), so query drift is
     not just a copy of document drift.
  3. **Ranking changes (Agent C).** Top-K overlap and rank correlation
     between the stale and refreshed indexes, and which older articles were
     displaced by newer ones. Covers "changes in retrieval rankings".
  4. **Emerging topics across categories (Agent B).** Cluster each period's
     most novel articles and label each cluster with its top terms, so a topic
     such as COVID that spans several sections is reported as one topic.
  5. **Historical evaluation set (simulator, Agents C and D).** Questions
     are also generated from baseline articles. They are the control: they
     should score about the same on both indexes. Covers "compare answer
     quality across historical and recent evaluation datasets".
  6. **One rule per recommended action (Agent E, `policy.py`)**, so every
     action in the brief can be triggered by evidence:
     - *Continue monitoring:* drift, but no material quality loss.
     - *Update or re-index:* quality recovers on the refreshed index.
     - *Trigger model evaluation:* quality is still poor on the refreshed
       index (the articles are there but the model does not handle them), or
       the LLM judge's agreement with the hand labels is low. **The system
       carries this out**: it sends Retrieval Evaluation back once to re-run
       the affected segments with a second, ready-made embedding model, and
       the result becomes evidence. No model is built or trained.
     - *Recommend retraining:* that evaluation shows neither model handles the
       new vocabulary even with a fresh index. Output is a recommendation to
       fine-tune the embedding model on recent data, with the evidence. We do
       not run the retraining (a small fine-tune is a possible stretch goal,
       with training and test data from different periods to avoid leakage).
     - *Update retrieval configuration:* the right article is usually found
       just outside the top K.
     - *Improve prompts:* retrieval is fine but answers are wrong or not
       grounded in the retrieved articles.
     - *Escalate for human review:* quality drops without measurable drift,
       or the evidence is inconclusive.
     - *Improve chunking:* the rule exists but is documented as not
       applicable, since articles are one to two sentences.
- **Consequences:** the second embedding model (Q12) moves from stretch goal
  to required; it costs about a minute of extra embedding on the GPU. The
  architecture and diagram now include the model-evaluation loop from E back
  to C. Estimated extra build time: about half a day, mostly for items 2
  and 4.
- **Not adopted yet:** letting the LLM propose what to investigate (for
  example, which terms to test) while code tests it. This would make the
  agents more visibly "agentic" without letting the LLM decide. It waits
  for the stakeholder answer on what counts as agentic (questions.md,
  stakeholder question 2).
- **Correction:** questions.md Part 3, item 6 had paraphrased the requirement
  as "trigger model retraining", which dropped "evaluation"; it is now
  marked Revised.
- **Evidence:** `docs/requirements-coverage.md`.

## D8. Data cleaning rules, first version
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04) as a **first version, to be
  revisited** as later findings come in (author's request; triggers below).
- **Context:** proposal P5 (tiered cleaning with a retention table) needed
  thresholds. They were chosen by looking at the data per period (lengths,
  empty descriptions, exact and near-duplicates, hand-checked examples at
  several similarity levels) before writing the cleaning code.
- **Choice** (in `config/config.yaml`, `data_cleaning`; applied in order):
  1. **Drop empty descriptions.** *Revises Q2*, which proposed keeping them:
     all 1,034 are in the baseline and later years have none, so keeping
     them adds a baseline-only format. Their rate is still reported by Data
     Monitoring.
  2. **Drop records under 8 words** (title + description): 46 records with
     too little text to embed meaningfully (e.g. "Impeach! We can't wait
     until November 2020.").
  3. **Drop records over 80 words:** 34 records, all in the baseline; later
     years never exceed 53. Mostly "Sunday Roundup" newsletters.
  4. **Drop near-duplicates within a period at cosine ≥ 0.95,** keeping the
     earliest copy. Exact (not approximate) comparison against every kept
     document.
  5. **Equal drift samples of 1,398 per period** (all of 2022), random with
     the project seed. Used for drift statistics only; the indexes use
     every cleaned document.
- **Options considered for the near-duplicate threshold:** 0.97 (keeps most
  templated columns, e.g. "Daily Meditation" variants), 0.95 (chosen), 0.93
  (starts removing distinct articles about the same event, which are real
  content).
- **Why:** the rules remove formats that exist only in the baseline, so
  they cannot pass for drift, and they remove templated copies that would
  crowd retrieval results. Together they remove 1.8% overall.
- **Revisit the rules if:**
  1. the most novel or most drifted examples are dominated by a format
     (templates, boilerplate) rather than content: tighten the duplicate
     rule or add a template filter;
  2. the length gap explains a large part of embedding drift (D6, check 5):
     sample the baseline to match later lengths;
  3. retrieval results are crowded by near-identical templated articles:
     consider 0.93 with a hand check;
  4. 2022 being a partial year (January to September) shows up as a
     seasonal artefact.
- **Evidence:** F3; `python scripts/prepare_data.py`; tests in
  `tests/test_cleaning.py` and `tests/test_prepare_data.py`.

## D9. LLM provider and models: Sonnet 5.5 writes, Opus 5.5 judges
- **Date:** 2026-11-04
- **Status:** Accepted (approved 2026-11-04). The spending cap waits for the
  stakeholder answer (questions.md, stakeholder question 4).
- **Context:** the LLM writes test questions and reference answers, writes
  the RAG answers under test, judges retrieval relevance and answer quality,
  and writes explanations. An Anthropic API key is available (checked
  2026-11-04 with the free models endpoint). Q15 asked for a stronger model
  as judge than as generator.
- **Options considered** (per million tokens, input / output; rough full-run
  estimates with the Batch API):
  1. *Opus 5.5 ($4 / $20) for everything:* about $45–75. Simplest and
     strongest, but the judge grades its own model's writing (self-preference).
  2. *Sonnet 5.5 ($2 / $10) writes, Opus 5.5 judges:* about $25–40. A
     different and stronger judge.
  3. *Haiku 5.5 ($0.10 / $0.50) writes, Opus 5.5 judges:* cheapest; weaker
     test questions.
- **Choice:** option 2. `claude-sonnet-5-5` writes questions, reference
  answers and RAG answers; `claude-opus-5-5` judges and writes the
  explanations (few calls, where quality is most visible).
- **Why:** two different models for writing and judging reduce the risk of
  the judge favouring its own style, which matters because every quality
  number depends on the judge. The ~40 hand labels (P8) measure how far the
  judge can be trusted.
- **Technical consequences:**
  1. **No `temperature`:** current models return an error if it is sent.
     The old `temperature: 0.0` was removed from the config. Reproducibility
     comes from caching every response on disk, keyed on model, prompt
     version and input.
  2. **Structured outputs:** questions and judgements come back as JSON
     validated against a schema, not parsed from free text.
  3. **Refusals:** live calls use the API's automatic fallback to another
     model; batch jobs cannot, so declined requests are counted and reported,
     never silently dropped.
  4. **Batch API** for the bulk jobs (questions, answers, judgements): half
     price, results within hours rather than seconds.
  5. A **20-question pilot** measures the real cost per job before any full
     run.
- **Evidence:** price list checked 2026-11-04; cost figures are estimates
  until the pilot.

## D10. Indexes and first drift statistics: setup choices
- **Date:** 2026-11-05
- **Status:** Accepted as the thin first version; within-category
  comparisons and severity bands follow after review.
- **Choices:**
  1. **The refreshed index is cumulative:** for period P it holds the
     baseline plus every later period up to P, like an archive that is kept
     up to date. *Alternative:* baseline + P only, which is simpler but
     unrealistic (the 2021 archive would forget 2020).
  2. **One kernel width for all comparisons:** the RBF width comes from the
     baseline sample (median heuristic) and is reused, so MMD values are on
     one scale across periods.
  3. **Novelty** = 1 − similarity to the nearest article in the full stale
     index (all 60,591 baseline articles). "Novel" means above the 95th
     percentile of the baseline's own novelty (each baseline article against
     the others), so 5% is the expected rate with no drift.
  4. **Word drift reported as excess JSD** (JSD minus its average under
     shuffling). Raw JSD on word counts is mostly sampling noise at this
     size (2018 scored below the ordinary-year reference).
  5. **Emerging terms** by log-odds with an informative Dirichlet prior from
     the whole cleaned collection (prior strength 1,000 pseudo-counts;
     terms seen fewer than 5 times dropped).
  6. **500 permutations** for each test, then Benjamini–Hochberg across all
     comparisons.
  7. **Reference for "ordinary change":** 2017 vs 2016, both inside the
     baseline, at the same sample size.
- **Why:** each choice keeps numbers comparable across periods, and ties the
  measures to what retrieval experiences (the stale index).
- **Evidence:** F4; tests in `tests/test_drift.py` (including a direct check
  of the MMD formula and that shuffling removes the noise in JSD).
