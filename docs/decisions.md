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
  agents more visibly "agentic" without letting the LLM decide. The
  stakeholder has since confirmed the current split fits (2026-11-06,
  questions.md), so this stays an optional extension (parked list in
  `docs/requirements-coverage.md`).
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

## D11. Within-category comparisons and severity bands
- **Date:** 2026-11-05
- **Status:** Accepted (both recommendations approved 2026-11-05)
- **Context:** F4 showed every comparison is significant at this sample size,
  so severity must come from size relative to ordinary change. With only
  the 1,398 drift samples, just 7 baseline categories had 50 articles.
- **Choices:**
  1. **Within-category comparisons use every cleaned article** in each
     period × category group with at least 50 articles whose category also
     has 50 in the baseline (37 groups); both sides sampled to the same
     size, capped at 1,398. *Rejected:* the drift samples only (too few
     categories).
  2. **Ordinary change = the same month a year apart inside the baseline**
     (January 2016 vs January 2017, ...: 12 pairs), each pair re-sampled 5
     times at the size being judged, giving 60 values per measure. Medium
     = above the 95th percentile, high = above the 99th. *Rejected:*
     re-sampling 2017 vs 2016 only (sampling noise around one value; bands
     too tight).
  3. **Fallbacks:** if a category has too few articles per month, quarter
     pairs are used (4 pairs); if still too few, the sample size shrinks to
     what quarters hold; if no category null is possible, the pooled
     baseline null is used. Each segment records its window and size.
  4. **One null per category,** built at the smallest segment size in that
     category: a smaller sample gives a wider, more cautious null.
  5. **Flag rule:** both techniques significant after Benjamini–Hochberg
     *and* severity at least medium (Q11 option C). Severity is the higher of
     the two bands; the kind of drift (meaning / wording / both / none, P2)
     comes from which bands are above low.
  6. **Controls:** low-severity sections, kept for Agents C and D to compare
     against drifted ones.
  7. **"× ordinary"** = the measure divided by the median of its null; shown
     for readability, while the bands make the decisions.
  8. **Agent A** reports category-mix change with Cramér's V next to the
     chi-square p-value, for the same reason as above.
- **Why:** comparisons stay like-for-like (same section, same size), and
  "unusual" is defined by the archive's own history, not invented constants.
- **Evidence:** F5; `tests/test_agents.py` (null construction falls back from
  months to quarters; the graph flags a shifted synthetic period and routes
  to full evaluation, and routes to the regression check when nothing
  changed).

## D12. Emerging topics, term-context shift and the correlation check
- **Date:** 2026-11-05
- **Status:** Accepted (approved as part of finishing Agent B, 2026-11-05)
- **Choices:**
  1. **Emerging topics by growth, not novelty.** F4 showed COVID articles are
     not individually novel, so clustering only novel articles would miss
     them. Instead, k-means with 40 clusters over all drift samples
     (baseline and every later period, equal sizes); a topic is emerging
     when its share of a period is at least 3× its baseline share (add-one
     smoothing), with at least 15 articles and a significant one-sided
     Fisher test after Benjamini–Hochberg. Labelled by log-odds words;
     "sections" counts sections holding at least 10% of the topic.
     Clusters depend on the seed and k, so topics explain drift; they do not
     drive the flags.
  2. **Term-context shift:** centroid distance between the articles using a
     word in the baseline and in a later period, with a permutation
     p-value; "× ordinary" against the same word's 2016-vs-2017 shift with
     the second group sampled to the later period's size (20 draws). Terms
     are a curated list in the config (needing 20 articles on both sides);
     context words by log-odds, excluding the word itself.
  3. **Correlation check (P6):** Spearman correlation of six measures across
     all baseline comparisons; returned as a table (a chart will follow with
     the dashboard).
  4. **Query drift** waits for the LLM query stream; the report says so.
- **Corrections made while building:**
  - The log-odds prior now sums over the whole vocabulary, as in Monroe et
    al.; it had summed only over the reported terms, which broke in a
    degenerate test case. Real-data term lists were unchanged.
  - The first term-shift ratio compared a small later group against large
    baseline halves, so small groups looked shifted by noise ("home" 2020:
    9.6× before, 4.6× after matching).
- **Evidence:** F6; `tests/test_topics_relations.py` (a topic that grows
  from 5% to 60% is found and spans two sections; a word whose contexts
  change is detected while a stable one is not; correlations need
  variation).
- **Revised 2026-11-09 (emerging and declining words per segment):** the
  dashboard's new before/after word chart showed that 5 of 460 listed
  words moved the wrong way or barely moved, all in small segments (54-82
  articles, about 800 words). Example: "trump" listed as emerging in 2018
  TECH, where it is not used at all that year. Cause: the log-odds prior
  (1,000 pseudo-words, shaped like the whole archive) is as large as such
  a segment, so words common across the archive inherit a high rate.
  Choice: keep the log-odds ranking, but list a word as emerging only if
  its observed rate rose by at least 20%, and as declining only if it fell
  by as much (otherwise "recipes", 12.2 -> 12.0 per 1,000 words, was listed
  as fading in 2020 FOOD & DRINK). Whole-period
  lists (e.g. coronavirus in 2020) are unchanged. Test:
  `tests/test_dashboard.py::test_listed_words_move_the_way_their_label_says`.

## D13. Plan: detecting quality loss without drift
- **Date:** 2026-11-05
- **Status:** Proposed (built with Agents C and D, after the stakeholder
  answers)
- **Context:** F5 found politics is never flagged as drifted, yet questions
  about the Biden administration or January 6th cannot be answered from a
  2017 archive. Harm without measurable drift must still be caught.
- **Plan:**
  1. **Always evaluate some undrifted segments.** Agent C tests control
     segments (low drift) next to drifted ones, and the regression check
     runs even when nothing is flagged. Without this, quality loss without
     drift is invisible by construction.
  2. **Historical control questions** (from baseline articles) should score
     the same on both indexes; a drop there points to the pipeline itself
     (model, prompt, retrieval settings, the judge), not to drift.
  3. **Classify every evaluated segment** in Agent D on two axes: drift
     flagged (from B) × material quality loss (confidence interval excludes
     zero and the loss exceeds a threshold set before results). "Quality
     loss without drift" is one of the four classes.
  4. **Diagnose the cause** with checks that already exist in the design:
     refreshed index recovers quality → the content changed but the drift
     measures missed it (review that section's calibration, as with
     politics); retrieval fine but answers wrong → prompt or generation;
     both indexes poor → embedding model (trigger model evaluation);
     historical questions also drop → pipeline or judge; time-sensitive
     questions only → facts were superseded (who holds an office), which
     no distribution measure can see.
  5. **Agent E** maps the class and diagnosis to an action, by default
     "escalate to a human" with the evidence.
- **First test case:** politics, 2020–2022.

## D14. Visual style: the Rivvun AI brand, validated for colour-blind readers
- **Date:** 2026-11-05
- **Status:** Accepted (author's request, 2026-11-05)
- **Context:** the charts and dashboard will be shown to Rivvun AI
  stakeholders; the author asked for a colour scheme similar to Rivvun's.
- **Source:** the design tokens in rivvun.ai's stylesheet (read 2026-11-05):
  dark navy surfaces (`#070A11`, `#0D1725`, `#161B24`), a light cream theme
  (`#F6F4EF`, `#ECE7DC`), navy / off-white ink, slate and sand lines, one
  burnt-orange accent (`#C14100`), and the Inter typeface (Segoe UI is used
  here because Inter is not installed; it is in Rivvun's own fallback list).
- **Choice:**
  - The brand accent is series 1 and the hue of the severity ramp. The brand
    has a single accent, so two supporting series hues were chosen (a blue
    and a teal / green) and checked with the dataviz palette validator
    against each theme's own surface:
    - light (cream): `#C14100 #2F6CB0 #1E9A8A`, worst colour-blind ΔE 13.7,
      all at least 3:1 contrast; severity ramp `#E39A73 #C14100 #7A2900`.
    - dark (navy): `#D9581E #6383E0 #1F9E70`, worst colour-blind ΔE 9.8,
      all at least 3:1; severity ramp `#8A3000 #C9500F #F29A6B`.
    - Rejected along the way: a violet third colour (indistinguishable from
      the blue), and two teals that were too close to the blue on navy.
  - Correlations use a blue ↔ orange diverging scale with a neutral
    midpoint.
  - PNGs in both themes: light in `docs/figures/` (README, print), dark in
    `docs/figures/dark/` (slides on Rivvun's navy). The dashboard uses
    Rivvun's dark site theme (`.streamlit/config.toml`).
- **Why:** on-brand for the audience without giving up legibility: every
  palette passes the colour-blindness and contrast checks, labels sit in ink
  colours, and the text inside coloured cells switches between white and
  navy by fill brightness.

## D15. The LLM layer: budget, cache, structured output, refusals
- **Date:** 2026-11-05
- **Status:** Accepted (budget set by the author, 2026-11-05)
- **Context:** the LLM jobs (D9) spend real money on the author's key, and
  every quality number depends on them. Budget: **$150 in total**; $20 of
  credit is loaded now, so the first stretch is planned to spend under about
  $10. The stakeholder answer on spending (question 4) was not needed to
  start.
- **Choices** (all in `driftmon/llm.py`, the only module that calls an LLM):
  1. **Budget enforced in code, before spending:** each request (or batch)
     gets a cautious cost estimate; if the running total in the cost ledger
     (`runs/llm_costs.csv`) plus the estimate would pass `llm.budget_usd`,
     nothing is sent. Every billed reply is recorded with its real tokens and
     cost, at published prices (in the config).
  2. **Disk cache** keyed on model, prompt version, system and user prompt,
     schema and settings: reruns are free and repeatable. A prompt change
     bumps its version, so stale answers are never reused.
  3. **Structured output:** every reply must match a JSON schema generated
     from a pydantic model (strict: all fields required, no extras) and is
     validated again in Python.
  4. **Refusals:** live calls use the API's server-side fallback
     (`fallbacks: "default"`). The Batch API has none, so a batch refusal is
     retried live with fallback. Anything still refused is returned as
     refused and counted, never dropped silently.
  5. **Broken replies** (cut off at the output cap, or not matching the
     schema) are recorded in the ledger but not cached, so a rerun retries
     them.
  6. **Thinking effort per job** (config): "medium" for question writing and
     review to begin with; tuned after the pilot measures cost and quality.
- **Why:** money and reproducibility are controlled in one tested place, and
  failures are visible rather than silent.
- **Evidence:** `tests/test_llm.py` (7 tests with a fake API: strict
  schemas, caching across restarts, the budget refusing before any call,
  live and batch refusals, broken replies retried, batch at half price).

## D16. How test questions are written and accepted
- **Date:** 2026-11-05
- **Status:** Accepted (first version; the pilot decides what to tune)
- **Context:** every retrieval and answer-quality number rests on the test
  questions. The author's concern: do not build on low-quality questions.
- **Choices:**
  1. **Source articles:** sampled per period, baseline included (the
     control). Excluded: templated columns ("HuffPost Rise", "Funniest
     Tweets", "Daily Meditation", "What to watch", ...; list in the config)
     and records under 15 words, which rarely state an answerable fact.
     72,288 of 77,840 cleaned articles remain eligible.
  2. **Writer (Sonnet 5.5):** one question per article, as a journalist or
     editor would ask the archive assistant: specific, self-contained,
     paraphrased (not the headline), 6–30 words; a reference answer from the
     article only; evergreen or time-sensitive; and an honest "answerable
     from the article" flag.
  3. **Checks in code, before review:** length; no run of 4 consecutive
     headline words; at most 70% of content words from the headline;
     the writer's own "not answerable" flag; near-duplicate questions
     (embedding cosine ≥ 0.95).
  4. **Reviewer (Opus 5.5):** five yes/no checks per surviving question:
     answerable from the article alone, unambiguous, needs the article,
     reference answer correct, natural. **Code keeps a question only if
     every check is yes**; the reviewer gives a short reason either way.
  5. **Everything is kept for inspection:** every draft, its check results
     and the reason it was rejected.
  6. **Diversity check (P3):** the spread of kept questions vs the spread of
     their source articles, to catch generic, collapsed phrasing.
- **Why:** a stronger, different model checks the cheaper model's work
  (D9), and the decision stays in code. Rejections are visible, so the
  prompt can be tuned on evidence.
- **Evidence:** `tests/test_simulate.py`; the pilot (F7) will measure the
  keep rate, the reasons and the cost.

## D17. Question writing, version 2 and 3: stricter questions, an unanswerable set, measured bias
- **Date:** 2026-11-05
- **Status:** Accepted (author, 2026-11-05). **To check against results:**
  the author asked to revisit whether these choices were right once the
  retrieval and answer results exist (see "Revisit" below).
- **Context:** the first pilot (F7) found no time-sensitive questions,
  copying from the description, a lenient reviewer, outside facts in
  questions and two-part questions. Looking at one rejected teaser ("Who Is
  Responsible For Debt When You Get A Divorce Or Break Up?") raised three
  more points: the reference answer described the article instead of
  answering; 20% of sampled articles are teasers with no stated fact; and
  rejecting teasers may bias the test set away from the lifestyle and
  service sections where COVID drift was strongest (F5).
- **Choices:**
  1. **One fact per question,** and up to two questions per article: a
     second only for a time-sensitive fact.
  2. **Direct reference answers,** never "the article says…" (also checked
     in code).
  3. **Sharper time-sensitive rule:** counts still being reported, current
     rules, who holds a role now, ongoing situations.
  4. **Copy check extended to the description** (7 consecutive words; 5
     rejected proper names such as "8-year-old Malayan tiger named Eko").
  5. **Stricter reviewer:** eight checks, including single question and the
     type label, and it must say false whenever its reason names a problem.
  6. **Common knowledge allowed, invented event details not** (version 3,
     after version 2 rejected "Justin Trudeau", "Canada" and "Pennsylvania
     senator"): realistic for journalists, at the cost of a little of the
     writer's own knowledge entering the test. Stated as a limitation.
  7. **An unanswerable set** (teaser articles): kept on purpose to test
     whether the assistant admits the archive cannot answer instead of
     guessing (groundedness).
  8. **Keep rates reported per section and period** in every run, so any
     bias in what survives is visible; sampling will draw more from the
     flagged sections and the controls (politics).
- **Revised 2026-11-05 (r-v4, after the third pilot):** the reviewer no longer
  grades the writer's type label (it proved ambiguous, F8); it assigns its
  own label without seeing the writer's, and that label is final.
  Disagreement is recorded as a measure of ambiguity (17% in the pilot),
  not a rejection. An unanswerable question is kept if the reviewer agrees
  it is unanswerable and it asks one thing ("natural" and "unambiguous" no
  longer apply: such questions are broad and restate the headline by
  nature). Keep rate rose to 60%.
- **Revised 2026-11-06 (r-v5, after F11):** "natural" no longer treats reused
  names, places, numbers or facts as copying (copy-rejected questions were
  not easier to retrieve); verbatim copying stays caught in code. Oversample
  keep rate 35% → 58% (F12). Prompt tuning ends here.
- **Revisit when results exist:** (a) is the unanswerable set useful in the
  groundedness results? (b) do keep rates differ by section enough to bias
  the drift-vs-quality analysis? (c) did allowing common knowledge make
  retrieval easier than it should be?
- **Evidence:** F8.

## D18. The judge cannot search, and does not learn; its built-in knowledge is a stated limitation
- **Date:** 2026-11-05
- **Status:** Accepted (raised by the author, 2026-11-05)
- **Context:** the reviewer (and later the answer judge) must judge from the
  article or the retrieved context, not from elsewhere.
- **Facts, checked in code:**
  1. **No outside search.** A Claude model can search the web only if a
     request declares a search tool; no request in this project declares
     any tool (`driftmon/llm.py`, `_params`). Each request contains the
     instructions, one article and the question; a cached reply in
     `cache/llm/question_review/` shows exactly what was sent.
  2. **No learning across calls.** Every request is independent; the model
     has no memory of earlier articles and is not changed by our calls
     (under Anthropic's commercial terms, API data is not used for training
     by default). The only memory is our own disk cache of replies.
- **Limitation:** the model's built-in knowledge from training cannot be
  switched off, only constrained by instructions and checked. Evidence it
  holds the line: the reviewer rejected questions for facts it surely knows
  but the articles do not state ("the Paris deal", "blood-testing startup",
  the name of a primary winner). Using general knowledge for the
  evergreen / time-sensitive label is intended (how news evolves, not facts
  about the story).
- **Where it matters next:** the answer judge (Agent D) could accept an
  answer that is true in the world but not supported by the retrieved
  articles. Safeguards: the groundedness check, and the hand-labelled
  sample (P8) that measures how often the judge departs from the text.
  To be stated in the README.

## D19. Review each question twice; three label groups
- **Date:** 2026-11-06
- **Status:** Accepted (author, 2026-11-06), after the test in F9.
- **Context:** F8/F9: reviewer decisions can differ between runs, and the
  writer and reviewer often disagree on evergreen vs time-sensitive.
- **Choices:**
  1. **Two independent reviews per question** (`simulation.reviews_per_question`);
     kept only if both accept it. Independence comes from a replicate
     number in the LLM cache key (`LLMRequest.replicate`); replicate 0 keeps
     the original key, so earlier reviews were reused, not re-bought.
  2. **Label groups:** "evergreen" or "time_sensitive" when the writer and
     every review agree, "disputed" otherwise. Analyses report the clear
     groups separately and never treat one model's label as truth. A single
     convenience label is the majority of the three.
  3. **Reviewer consistency is reported in every run** (same decision, same
     label).
- **Why:** cheap protection against random lapses, and a measured
  reliability number for the walkthrough. It does not catch systematic
  errors (F9: the "milestone" question passed twice); the hand labels do.
- **Evidence:** F9; `tests/test_simulate.py` (two disagreeing reviews
  reject a question; label groups).

## D20. Oversample articles likely to state a revisable fact
- **Date:** 2026-11-06
- **Status:** Accepted (author, 2026-11-06)
- **Why we did it:** the experiment needs time-sensitive questions, where a
  stale archive gives an outdated answer (who holds an office, a toll still
  being counted). Random sampling yields very few: 1 clear time-sensitive
  question of 13 kept (F9). At that rate the ~180-question set would hold
  about 15, too few to say anything about outdated answers.
- **Options considered:**
  1. *Accept the natural rate:* honest, but too thin to analyse.
  2. *Ask the writer to always produce a time-sensitive question:* forces
     unnatural questions from articles that have no revisable fact.
  3. *Oversample a pool of likely-revisable articles* (chosen): a word
     filter on headline + description ("at least", "so far", "missing",
     "death toll", "still", "ongoing", "cases", "new rules", ...; list in
     `simulation.oversample_patterns`).
- **How it is reported:** every question carries `sample_pool` ("random"
  or "oversample"). Oversampled questions are **never mixed into
  archive-wide rates**; they are analysed as their own group (outdated
  answers), and the README says so.
- **Known bias:** the pool is 9% of eligible articles and leans towards
  hard news (politics 32%, world news 19%); it says nothing about lifestyle
  sections' time-sensitive facts.
- **Evidence:** F10.

## D21. Knowledge graph / ontology (OWL, RDF, Neo4j, Protégé): considered, not adopted
- **Date:** 2026-11-06
- **Status:** Not adopted for this project; recorded as future work.
- **Context:** a colleague suggested modelling how relationships between
  entities change with an ontology (OWL/RDF), stored in a graph database
  (Neo4j), edited and visualised in Protégé.
- **What is right about it:** the brief asks for "shifts in semantic
  relationships", and many time-sensitive facts are relationship changes
  ("Biden — holds office — President"). A temporal knowledge graph would
  make such changes explicit and easy to explain, and is an established
  approach.
- **Why not here:**
  1. **Extraction is the hard part, not storage.** Turning 78k one-to-two
     sentence descriptions into entities and relations needs an LLM or a
     relation-extraction model: cost, time, and extraction errors that would
     themselves look like drift.
  2. **OWL/RDF buys formal reasoning we do not use.** The system measures
     change; it does not infer new facts. Designing and defending an
     ontology is days of work, with four days left.
  3. **Infrastructure for little gain.** Neo4j is another service for
     reviewers to install; a few thousand relations fit in memory
     (pandas / networkx). Protégé is a manual editor for hand-built
     ontologies, not for automated monitoring.
  4. **The RAG under test is vector-based;** a graph would not change
     retrieval unless we also built GraphRAG (scope creep).
  5. **Relationship shift is already measured** distributionally, with a
     significance test and size-matched baseline (term-context shift, F6:
     "cases", "test", "home", "virus").
- **Lightweight middle ground, if time allows:** an entity co-occurrence
  graph per period for a few key entities (networkx), showing neighbours
  gained and lost (e.g. "Biden" with Obama and "vice" in 2016–17; with
  Congress and "administration" in 2021). A visual complement to F6, no new
  infrastructure.
- **Future work:** temporal knowledge graphs for relationship drift, and
  GraphRAG for questions about who holds which role.

## D22. Vector store revisited against the brief: FAISS stays, indexes saved to disk with Agent C
- **Date:** 2026-11-06
- **Status:** Accepted (author, 2026-11-06). Adds to Q13.
- **Context:** the author asked whether the brief expects a vector database.
  The brief's technology table lists "Vector database: Qdrant / FAISS /
  Chroma", and says the stack is open and not every listed technology
  must be used. FAISS is one of the three named options.
- **What FAISS is, precisely:** a vector search library, not a database
  server. Here it stores the vectors and does exact nearest-neighbour
  search; the article metadata (date, section, text) sits next to it in
  `data/processed/documents.parquet`, joined by document id. Together they
  do the job of a vector database for a single-user analysis system.
- **Why not Qdrant or Chroma:**
  1. **Exact search keeps the experiment clean.** Their default indexes are
     approximate (HNSW), which adds small random retrieval errors; a quality
     drop could then be drift or search noise.
  2. **Six indexes are cheap with FAISS** (one stale, one refreshed per
     later year): a few lines and about a second to build.
  3. **Nothing to install** for reviewers reproducing the results.
- **Gap closed with Agent C:** indexes are currently rebuilt in memory on
  every run from the embedding cache. Agent C will save each index to disk
  (`faiss.write_index`, under `indexes/`, gitignored) with a small manifest
  (model, document ids, build date), and reload them when the embeddings
  have not changed. This makes the store persistent, lets the dashboard and
  demos load instantly, and records exactly which documents each index
  held.
- **Walkthrough answer:** "FAISS is listed in the brief; we chose exact
  search because approximate search would blur the very effect we measure."

## D23. The full question set: four tagged sources, one batch
- **Date:** 2026-11-06
- **Status:** Accepted (sampling built; the run waits for the author's go).
- **Context:** the evaluation needs questions where drift was flagged, where
  it was not (controls), from the archive as it is, and time-sensitive ones
  (D20), within a $20 credit.
- **Choice** (`simulation.full_set` in the config; `simulate.full_sample`):
  8 articles per flagged section (14 sections), 3 per control segment (23),
  40 random baseline articles (the control questions) and 20 random per
  later year, and 34 per period from the likely-revisable pool. No article
  is drawn twice; each carries `sample_pool` and `sample_segment`, so every
  analysis can keep the groups apart. Flagged and control segments come
  from the latest drift run, so the sample follows what Agent B found.
- **Dry run:** 525 articles (flagged 112, control 69, random 140,
  oversample 204). Estimated $11.55 as a batch (the estimate has run about
  2–3× high; expected about $5–8).
- **Known imbalance:** 39 of the 69 control articles are from 2018, because
  most low-drift segments are 2018 sections; 2020 has 48 flagged articles.
  Reported, not corrected: it reflects where drift was and was not found.

## D24. Query drift: the simulated questions as the query stream
- **Date:** 2026-11-06
- **Status:** Accepted (approved by the author); code built and tested, numbers
  wait for the full question set.
- **Context:** the brief asks to monitor incoming queries and says user queries
  evolve and bring new terminology. There is no real query log.
- **Options considered:**
  1. *All generated questions as the stream.* Simple, but the flagged and
     control articles were picked *because* of document drift, so their
     questions would inflate query drift by design.
  2. *Only the pools drawn the same way in every period* (random and
     oversample). Smaller (about 80–130 questions a period), but a fair
     comparison between periods.
  3. *A separately generated query log* with personas. Closer to real users,
     but extra cost and still written from the same articles.
- **Choice:** 2. Each period's stream questions are compared with the
  baseline's using the same two techniques as for documents: MMD with a
  permutation test and centroid shift on query embeddings; excess JSD,
  log-odds emerging terms and new-term rate on question words (question
  wording such as "did", "according" is ignored). Benjamini–Hochberg across
  periods. Unanswerable questions count: real users ask those too. Agent A
  summarises the stream (volume, length, new-term rate, unanswerable share).
- **Scale:** the 2017-vs-2016 questions give a rough "ordinary year" row.
  The document severity bands (D11) need far more data than the stream has,
  so query drift reports significance and times-ordinary, not bands.
- **Limitation (README):** the questions are written from each period's
  articles, so query drift partly mirrors document drift by construction; a
  real query log can also drift on its own (new users, new tasks).
- **Code:** `driftmon/queries.py`; Agent A and B nodes; `query_drift.csv` in
  each run; tests in `tests/test_queries.py`.

## D25. Agent C: three retrieval comparisons, one judge call per question
- **Date:** 2026-11-06
- **Status:** Accepted (approved by the author); built and tested, numbers wait
  for the full question set.
- **Context:** the brief asks Agent C to evaluate retrieval across periods and
  find changes in relevance and rankings, and the key experiment is stale vs
  refreshed. A stale index cannot contain a later article, so known-item
  scores alone are one-sided (D2), and they are inflated anyway (F11).
- **Choice:** three comparisons, each answering one question:
  1. *(a) Did not re-indexing hurt?* Each later period's questions on the
     stale index and that period's refreshed index. Opus grades the pooled
     top 5 of both (0 = not relevant, 1 = on topic, 2 = answers). Metrics:
     answer@5 (the context the answer step reads contains an answer),
     precision@5, nDCG@5 against the pooled ideal; paired difference per
     question with a bootstrap interval; by period, by drift status from
     Agent B (flagged / control / other) and by question type. Baseline
     questions are judged on stale vs the latest refreshed index: the
     historical control.
  2. *(b) Does the retriever cope with drifted content?* On the same
     refreshed index, each period's questions vs the baseline's: MRR and
     Recall@10 of the source article. One index for both groups, so index
     size is not the explanation. Weighted down (F11).
  3. *(c) Are old questions crowded out?* Baseline questions on stale vs each
     refreshed index: top-10 overlap, share of the top 10 from newer periods,
     and whether the source article fell out. Free (no LLM). This replaces
     the rank correlation listed earlier: with mostly different documents
     in two top-10 lists, overlap and displacement say more.
- **Judge design:**
  - One request per question, grading every pooled article together, in a
    shuffled order, never saying which index found which article (tested).
    Cheaper than one call per article and blind to the index.
  - Top 5, not 10: about half the cost, and it is what the answer step reads.
  - The judge sees the reference answer but is told other articles may
    answer equally well, so an older article that still answers gets credit.
  - An article the judge skips is left ungraded and the question is dropped
    from the judged metrics, never silently scored 0 (tested). Counts are
    reported.
  - Unanswerable questions are not used here (no relevant article exists by
    design); they are for Agent D's abstention check.
- **Regression-check route** (no drift flagged): only random-pool questions.
- **Indexes (D22):** saved to `indexes/` with a manifest (model, fingerprint
  of document ids and vectors, periods and document ids per index); reloaded
  only when the fingerprint matches, otherwise rebuilt (tested). 650 MB for
  the six indexes; stale holds 60k articles, refreshed 2022 77.8k.
- **Cost:** about 1 cent per question as a batch; `run_pipeline.py --estimate`
  prints the exact estimate before any spend.
- **Code:** `driftmon/evaluation/retrieval.py`, `retrieval_metrics.py`,
  `agents/retrieval_eval.py`, prompt `rel-v1`; tests in
  `tests/test_retrieval.py`; dashboard page "Retrieval".

## D26. Long-form documents: Wikipedia articles at yearly snapshots, chunked
- **Date:** 2026-11-06
- **Status:** Accepted (approved by the author); not built yet.
- **Context:** the stakeholder (Ani, 2026-11-06, `docs/questions.md`) asked
  for long-form documents where chunking applies, next to HuffPost, and for
  an explanation of the chunking strategy and how it affects drift
  detection. HuffPost articles are one or two sentences, so chunking was a
  stated limitation until now.
- **Options considered:**
  1. *Wikipedia articles captured at the end of each year (2017–2022).* Same
     years as HuffPost; the same document is edited over time ("documents
     become outdated or replaced"); fits the newsroom use case (reference
     material); clear sections. Cons: page markup needs cleaning; the LLM
     has read Wikipedia, so answers may come from memory (handled by
     answering only from the retrieved text and judging groundedness); the
     API asks for contact info in the User-Agent (a project URL, not an
     email).
  2. *Company annual reports (SEC 10-K risk sections, EDGAR-CORPUS).* Most
     enterprise-like and the hardest chunking case (long, mostly repeated
     text with small new paragraphs). Cons: the dataset appears to end in
     2020; a second use case beside the newsroom; the SEC wants an email if
     we fetch filings ourselves.
  3. *arXiv papers.* Clear topic drift, but free abstracts are too short to
     chunk and full text means PDF parsing; papers are never updated.
- **Choice:** 1. The author would revisit 2 only if a test showed it to be
  better.
- **Plan:**
  - About 150–200 articles sampled from Wikipedia's "vital articles" list
    (stratified by topic, so the choice is not hand-picked), plus a small,
    labelled set of articles tied to known events (e.g. pandemic, vaccine)
    as the known-drift check.
  - Each article as it stood at the last revision before the end of 2017
    (baseline), 2018, … 2022.
  - Chunks split at section headings, then about 256 tokens with about 32
    tokens of overlap (bge-small reads at most 512 tokens).
  - **Chunking experiment:** four ways to read each article: first 512
    tokens only; one averaged vector; chunks of about 128; chunks of about
    256 tokens. The text added between two snapshots (from the revision
    diff) is the ground truth, so we can count how many known changes each
    one detects. Expectation: truncation misses changes deep in the
    article, and averaging dilutes small changes.
  - Agents A–C run on the chunks; stale vs refreshed then shows the stale
    index returning the outdated version of the same article.
- **Cost:** about $3–5 for questions and judging, after the credit top-up.

## D27. Address the changed-numbers blind spot: detect, measure harm, act
- **Date:** 2026-11-06
- **Status:** Accepted (approved by the author). Detection built (F18; precision 85% by two labellers, F19); harm measured: flagged edits change the answer 50% of the time vs 2% for edits that kept their numbers (F21); action (Agent E) to follow.
- **Context:** F17 measured a blind spot of embedding-based drift: when an
  updated document changes only a number, its chunk vector barely moves
  (median change 0.0008 against 0.0127 for other edits, about 16 times
  less; 62% of such chunks are among the least-changed quarter). Examples:
  "the Black Death … killed over 75 million people" became "over 20
  million"; "from 1347 to 1453" became "1347 to 1353" (no measurable
  change). These are exactly the edits that make an answer outdated
  (tolls, counts, dates), so drift scores alone would miss the drift that
  matters most for answers, and the brief names "existing documents become
  outdated" as a core problem.
- **Options considered:**
  1. *Accept and document it as a limitation.* No work, but leaves the
     most answer-relevant change undetected.
  2. *Lower the embedding thresholds.* Would flood alerts: number edits
     sit among ordinary edits, not above them.
  3. *Add a plain text check for changed facts, then test whether it
     matters, then act on it.* Small, testable, no LLM for detection.
- **Choice:** 3, in all three stages:
  - **Detect (Agents A/B):** a "changed facts" check for updated documents.
    Each changed sentence is matched with the most similar sentence of the
    previous version in the same section; if their numbers differ (after
    ignoring formatting such as "2217" vs "2,217"), it is reported as a
    changed fact, with both sentences as evidence. A third signal next to
    the embedding and word techniques.
  - **Measure harm (Agent D):** for documents with changed facts, ask the
    same question of the old and new version and check whether the answer
    changes.
  - **Act (Agent E):** a rule: changed facts in an indexed document →
    re-index that document (a targeted refresh), citing the sentences.
- **Why:** the check costs little, runs without an LLM and catches what
  the embeddings cannot see; the answer test then shows whether it
  matters.
- **Validation (stated honestly):** the check uses the same idea that found
  the blind spot, so catching those cases proves nothing on its own. It is
  validated in two ways: a hand-checked sample of flagged changes (17/20 real, F19)
  (precision: are they real fact changes?), and Agent D (do flagged
  documents give different answers more often than unflagged ones?). If
  Agent D shows no difference, the check is reported as not useful.
- **Limits:** applies only to documents that are edited (Wikipedia), not to
  HuffPost, where each period brings new articles; there, the
  time-sensitive questions in Agent D cover the same risk. Changed names
  (e.g. a new office holder) are not caught by a number check; left out
  unless time allows.

## D28. Agent D: answer quality, stale vs refreshed, and the outdated-answer test
- **Date:** 2026-11-07
- **Status:** Accepted (approved by the author). Spend: a 20-question pilot
  first; the full run after the credit top-up, with approval.
- **Context:** Agent C measures whether the right article is *found*. The
  brief asks whether drift hurts *answers* (correctness, groundedness,
  relevance), how drift scores relate to quality, and which drift matters.
  Three things only answers can show: whether the system invents an answer
  when nothing useful is found; whether a found article becomes a correct
  answer; and, for edited documents, whether the stale index's *old version
  of the right article* gives an outdated answer (D27's harm step), which
  relevance grading cannot see.
- **Choices:**
  1. **HuffPost answers.** For every test question (423 answerable, 15
     unanswerable), Sonnet answers from the top 5 of the stale and of the
     refreshed index (Agent C's saved retrieval), only from those articles.
  2. **Opus judges each answer in its own call** (Q10 rubric): correct
     against the reference answer (0/1/2), grounded in the retrieved
     articles, and whether it declined. It is not told which index the
     answer came from. Whether a decline was *right* is decided in code from
     Agent C's relevance grades (did the top 5 contain an answer?).
  3. **Material harm** (fixed before any results): the share of correct
     answers drops by at least 10 percentage points from refreshed to
     stale, and the bootstrap interval excludes zero. Each segment
     (period x section) with enough questions is classed as drift that
     hurts / drift that does not / harm without drift / healthy, with a
     likely cause (D13). Drift scores vs quality loss: rank correlation
     across segments.
  4. **Wikipedia outdated-answer test.** 60 flagged changed facts and 60
     edited passages without one: Sonnet writes a question from the new
     version; answers come from the old and the new version's chunk indexes
     (local retrieval); Opus judges whether the two answers state a
     different fact. The check is useful only if flagged passages change
     answers much more often.
  5. **Loop back to Agent C:** a borderline segment (drop at least the
     threshold but the interval includes zero) goes back to C once to grade
     ranks 6-10; D re-answers those questions from the top 10. Either the
     answer was just outside the top 5 (evidence for "update retrieval
     configuration") or the case goes to a human.
  6. **Judge check:** 15 answer items added to the blind hand-label sheet.
- **Options considered for the judge:** one call grading both answers of a
  question (as Agent C's relevance judge does) vs one call per answer. One
  call saves only about 15-20% (each answer brings its own five articles),
  while it invites comparing the two answers instead of grading each,
  adds order bias, weakens blindness (article dates reveal the newer index)
  and differs from how the hand labels grade. Separate calls; identical
  answers (both indexes found the same articles) reuse one cached grade.
- **Cost:** about $6-9 at batch prices (estimate; the last estimate was
  about twice the real cost).

## D29. Agent E: one rule per action in the brief, evidence attached, explanations number-checked
- **Date:** 2026-11-07
- **Status:** Accepted (approved by the author, 2026-11-07).
- **Context:** the brief asks Agent E to recommend corrective actions
  ("continue monitoring", "update or re-index", "improve chunking or
  embedding strategies", "update retrieval configurations", "improve prompts
  or retrieval strategies", "trigger model evaluation or retraining",
  "escalate for human review"), each with supporting evidence and an
  explanation. Agent D found that every news section with enough questions
  loses answers on the stale index, flagged by drift or not (F20), and the
  Wikipedia test found that changed facts change answers (F21).
- **Options considered:**
  - *An LLM reads all reports and writes recommendations.* Flexible, but
    the decision would be the LLM's, numbers can be invented, and runs are
    not repeatable. Contradicts D9 ("LLM writes, code decides").
  - *Rules only, with fixed text.* Repeatable and testable, but the text
    cannot explain the specific case well.
  - *Rules decide, the LLM explains, code checks the explanation*
    (chosen).
- **Choices:**
  1. **One rule per action in the brief** (`driftmon/policy.py`), each a
     plain function of the reports from Agents B, C and D, plus the
     long-form evidence (changed facts, outdated-answer test, chunking
     experiment) and the hand-label checks. Thresholds live in the config.
  2. **Every rule reports**, fired or not: "recommended" with its evidence;
     "not needed" with the number that ruled it out; "not checked" if its
     input is missing. This shows every action in the brief was considered.
  3. **Every recommendation carries evidence:** items with a source (agent
     report or finding) and a claim with the numbers. A recommendation
     without evidence is rejected in code and listed as rejected.
  4. **Explanations:** Sonnet writes 2-3 sentences for each recommended
     action from its evidence only. Code rejects an explanation if it has
     a number that is not in the evidence; it retries once with the
     offending numbers named, then falls back to the rule's own text. The
     check result is saved with each recommendation.
  5. **Order:** by kind of evidence first (measured answer loss, then
     escalation, then maintenance), then by size within a kind (loss x
     questions affected for re-indexing).
  6. **Escalation with a reason:** "harm without drift" segments go to a
     human with the lesson attached (drift alerts missed them; rely on the
     schedule), not a bare "please review".
- **Kept small on purpose:**
  - *Re-index and re-check (the bonus):* the refreshed index is the
    re-indexed system, so E cites that comparison rather than re-running it.
  - *The second embedding model loop (D7):* not built, because no segment
    is poor on the refreshed index. The rule exists and reports "not
    needed" with the number. If a later run needs it, it becomes a loop
    back to Agent C, as D7 describes.
- **Graph:** A -> B -> C -> D (loop to C) -> E -> end.
- **Cost:** a few cents (about ten short Sonnet calls, cached).

## D30. Quote the pipeline's drift multiples, not F4's
- **Date:** 2026-11-08
- **Status:** Accepted (approved by the author 2026-11-08)
- **Context:** Two "× ordinary" scales were in circulation for the same raw
  drift (MMD²). F4, an early script, divided by one comparison, 2017 vs 2016
  (MMD² 0.0021). The pipeline (Agent B, D11) divides by the median of 12
  same-month-a-year-apart pairs inside the baseline (MMD² 0.0047). So F4's
  multiples are about 2.3× the pipeline's: 2020 vs baseline 5.7× vs 2.5×,
  2022 vs 2021 1.2× vs 0.5×, 2022 vs baseline 4.7× vs 2.1× (run
  `20261107-202242`). The design summary sent to Ani and the presentation
  notes used F4's.
- **Options:** (a) F4's: one easy sentence ("5.7× an ordinary year") but
  rests on a single comparison, and if that year was quiet every multiple is
  inflated; (b) the pipeline's: steadier reference, and the same scale that
  sets the severity bands, the flags, the dashboard and the recommendations.
- **Choice:** (b) everywhere from now on. F4 stays as the record of the
  early result, with a note pointing here.
- **Why:** what is said in the walkthrough must match what the dashboard
  shows. Rankings and conclusions are identical under both (2020 largest;
  year-on-year comparison hides gradual drift), so nothing else changes.
- **If asked about 5.7×:** same measurement, smaller denominator (0.0021
  vs 0.0047); D11 already rejected the single-pair reference for the bands.

## D31. Test questions in newer wording about older articles
- **Date:** 2026-11-09
- **Status:** Accepted (approved by the author 2026-11-09); thresholds set
  before any result
- **Context:** F15 found that how far a section drifted did not change how
  much search lost: the drift that matters is new content the index lacks.
  But every test question was written from its own article (F11), so it
  shares the article's wording. The one case where a change of wording on
  its own could hurt search is untested: an older article asked about in
  later words ("COVID" for an article that says "coronavirus"). Parked as an
  extension since 2026-11-06 (F15, requirements coverage).
- **Options:**
  (a) rewrite questions in later wording and compare with the original
  wording: simple, but any rewording loses a little against the article's
  own words, so the drop would mix "reworded" with "newer";
  (b) two rewrites per question, one in the article's era and one that
  changes only the words a later reader would say differently, and compare
  the two: isolates the wording;
  (c) substitute words by a list (coronavirus -> covid): clean, but only
  covers the pairs we think of, and the swaps are often unnatural.
- **Choice:** (b). Questions from 2016-17 to 2021, up to 50 per drift
  status (flagged, control, other, baseline), at random. Sonnet writes both
  versions from the question and the article's date only (not the article),
  or says no natural newer wording exists (skipped). Code requires the newer
  version to add a word significantly more typical of 2022 than of the
  article's period (log-odds z >= 1.96 and the rate up 20%, as in D12).
  Opus checks that both versions ask the same thing as the original, shown
  in shuffled order. Search: the 2022 refreshed index (every article);
  found = the source article in the top 5. No LLM in the scoring.
- **Thresholds (config `evaluation.newer_wording`):** wording drift hurts
  search if newer wording finds the article at least 10 points less often
  than same-era wording, exact one-sided McNemar p < 0.05. Drift level
  matters if that drop is at least 10 points bigger in flagged sections than
  in control sections, one-sided permutation test p < 0.05.
- **Limitations:** the "newer word" check is a floor, not proof: for
  2016-17 it accepts ordinary words whose rate happened to rise (about
  1,700 words qualify). The writer and the meaning check do the real
  filtering. The questions name specific people and events, which carry
  search; this tests wording around them, not unfamiliar names.
- **Reproduce:** `python scripts/newer_wording.py` (`runs/wording/`).
