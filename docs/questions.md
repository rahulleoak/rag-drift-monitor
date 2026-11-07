# Clarifying Questions

**Status:** open. Written 2026-11-04 after reading `CLAUDE.md`,
`docs/project-brief.md` and the original assignment (.docx, same content as the
brief except the brief adds the Nov 10 deadline). As questions are answered,
the outcome moves into `docs/decisions.md`.

**Updates (2026-11-04):**
- Project structure approved (decision D1).
- Q1 and Q3: decided 2026-11-04 after exploration: HuffPost (D3), baseline
  2016–17 with yearly periods 2018–2022 (D6).
- Q9: accepted for now (D2).
- Other Part 2 items: recommendations are working assumptions, not yet
  confirmed.
- Part 4 added: ideas taken from the paper "Semantic Drift in Paraphrasing".
  P1–P8 approved 2026-11-04 and recorded as D5. One change: no
  composite score for sorting the dashboard either (see D5, P4).

Facts about the dataset used below come from the Kaggle dataset card. They
were **verified on 2026-11-04** (see `docs/findings.md` F1): 209,527 records;
2019–2022 hold 7,523 of them; descriptions are short (median 28 words with the
headline). F1 also found category renames and a large shift in category mix
that the card does not mention.

- About 210k records with fields `headline`, `short_description`, `category`
  (42 labels), `authors`, `date`, `link`.
- About 200k records fall between 2012 and May 2018, but only about 10k between
  May 2018 and 2022 (HuffPost changed its website). The COVID year, 2020, is in
  the sparse part.
- No article bodies. A "document" is a headline plus a one to two sentence
  description.

---

## Stakeholder answers (Ani, received 2026-11-06)

The questions actually sent were a shortened version of Part 1. Answers,
quoted closely:

- **Corpus / chunking:** "Please include longer documents as well, since
  chunking is an important part of what we want to see. You can use HuffPost
  for the time-based drift, but add a set of long-form documents (e.g. arXiv
  papers, Wikipedia articles or public reports) where chunking applies.
  Please explain your chunking strategy and how it affects drift detection."
  -> **New requirement.** HuffPost stays; a long-form corpus is added (D26).
- **Agentic:** stats and decisions in tested code, LLMs for generation,
  judging and explanation, LangGraph routing: "that fits ... the right
  split." -> D9/D15 confirmed.
- **Recommend or apply fixes:** "Recommending fixes is enough. If you have
  time, applying one fix (e.g. re-index) and re-checking would be a bonus,
  not a requirement."
- **Simulated queries and labels:** "LLM-generated queries with a
  hand-checked sample is acceptable. Please document how you validated
  them." -> P8 hand labels plus the reviewer checks (D16-D19) must be
  written up as the validation method.
- **LLM key:** "Please use your own key." (Whether reviewers will run the
  code without a key was not answered: ship precomputed results so the
  dashboard works without one.)
- **Deadline:** "You try completing it by Friday next week" -> **Friday
  2026-11-13** (was Tuesday 2026-11-10).
- **Checkpoint:** a short design summary around Wednesday/Thursday is
  welcome; quick feedback promised.

---

## Part 1. Ask stakeholders (top 5)

1. **Is the corpus framing acceptable?**
   The brief describes enterprise documents (contracts, policies, procedures)
   that get versioned and replaced. We propose a newsroom archive assistant over
   HuffPost headlines and short descriptions. Consequences: documents are one to
   two sentences, so chunking does not apply, and "documents become outdated or
   replaced" maps to "newer reporting supersedes older reporting" (for example,
   changing COVID guidance during 2020). Is this an acceptable stand-in, or do
   you expect long, enterprise-style documents where chunking matters (which
   would point to a dataset such as arXiv abstracts)?
   *Why it matters:* it fixes the dataset, and decides whether the "improve
   chunking" recommendation can be demonstrated at all.

2. **What counts as "agentic" for you?**
   Our plan keeps statistics, metrics and final decisions in tested Python code.
   The LLM generates queries, writes answers, judges quality and writes
   explanations. LangGraph routes between agents based on intermediate results
   (for example, skip the expensive answer evaluation if nothing drifted, or go
   back for more evidence when a result is borderline). Does that meet "a working
   agentic system rather than a collection of independent prompts", or do you
   expect the LLM itself to plan and choose actions?
   *Why it matters:* the success criteria and our rule "the LLM never makes the
   final decision" could be read as in tension. We think auditable decisions in
   code are the right enterprise choice, but want to confirm it is valued.

3. **Is simulated evaluation data acceptable?**
   There are no real user queries or human relevance labels. We plan to
   (a) replay history as periodic batches to simulate continuous monitoring,
   (b) generate user questions and reference answers with an LLM from sampled
   articles, and (c) use an LLM judge for relevance, correctness and
   groundedness, checked against a small hand-labelled subset (30 to 50 items).
   Is that acceptable, or do you have a query log, a labelled set, or a required
   evaluation framework (for example RAGAS)?
   *Why it matters:* questions generated from documents make query drift partly
   mirror document drift. We will state it as a limitation, but it limits how
   much weight the query-drift results can carry.

4. **What are the demo and run logistics?**
   Live run or pre-computed results in a dashboard? Will you run the code
   yourselves, and if so on what hardware (CPU only?) and with whose LLM API key?
   Is there a spending cap or a required LLM provider (the brief suggests
   OpenAI / Azure OpenAI "or equivalent")?
   *Why it matters:* decides whether the pipeline must finish in minutes on a
   laptop, how large the data sample can be, and which LLM client to build first.

5. **If time runs short, what matters most?**
   With about six days, depth has to be prioritised. Please rank:
   (a) rigour of the drift statistics, (b) the causal stale-vs-refreshed index
   experiment, (c) agent architecture and orchestration, (d) dashboard polish.
   Also: what should be handed in (repository + README, slides, a recording)?
   *Why it matters:* we plan to build the full loop thinly first, then deepen
   (b) and (a). We would change that if (c) or (d) carries more weight.

---

## Part 2. Decide myself

Each item lists options with tradeoffs and a recommendation. Approved items
become entries in `docs/decisions.md`.

### Q1. Dataset and use case
*Status: decided: A, HuffPost (D3, F1).*
- **A. HuffPost News Category (newsroom archive assistant).** Suggested first in
  the brief, small, timestamped, has categories to segment by, and has well-known
  drift events to validate against (2016 election, #MeToo in late 2017, COVID in
  2020). Downsides: short documents, volume cliff after mid-2018.
- **B. arXiv abstracts (research knowledge base).** Longer documents (chunking
  becomes meaningful), steady volume, clear terminology drift ("transformer",
  "LLM"). Downsides: multi-GB raw file, slower to embed, drift events less crisp
  to validate, harder for non-experts to follow in a demo.
- **C. Amazon reviews.** Large and timestamped, but drift is mostly product mix
  and sentiment: a poor fit for a knowledge assistant.
- **Recommendation: A**, subject to stakeholder question 1. Known drift (COVID
  vocabulary must be detected, or the detector is broken) is worth more than
  document length in a one-week project. State the short-document limitation
  openly.

### Q2. What counts as a "document"
- **A. Headline only.** Too little text; many headlines are teasers.
- **B. Headline + short description, one record = one chunk.** Uses all the text
  we have, no chunking parameters to justify. Some descriptions are empty.
- **C. Fetch full articles from the links.** Scraping, licensing and network
  problems; not reproducible. Rejected.
- **Recommendation: B.** Keep records with empty descriptions, and report their
  rate as a document-quality metric in Agent A.
  *Revised 2026-11-04 (D8): drop them. All 1,034 are in the baseline and later
  years have none, so keeping them adds a baseline-only format. Agent A still
  reports the rate.*

### Q3. Baseline and later periods
*Status: decided: B, with baseline 2016–17 instead of 2014–15 (D6, F2).*
- **A. Dense era only (2012 to mid-2018), yearly periods.** Plenty of data, but
  misses COVID, the clearest validation case.
- **B. Baseline from the dense era (for example 2014–2015, to confirm in step 5),
  then yearly periods through 2022, with an equal-size random sample per period.**
  Covers 2016, 2017 and 2020. Equal sizes keep drift statistics comparable
  across periods, because MMD values, classifier AUCs and p-values all depend on
  sample size. Downside: the sample size is capped by the sparsest year.
- **C. Quarterly or monthly windows.** Closer to "continuous", but too sparse
  after 2018, so statistics get noisy.
- **Recommendation: B.** Also compare two halves of the baseline against each
  other to measure how much drift is normal year to year, and use that to
  calibrate drift thresholds instead of inventing them. Final years chosen after
  step 5.

### Q4. Unit of analysis for "which drift matters"
- **A. Period level only.** With 6 to 8 periods, a drift-vs-quality correlation
  has about 7 points: no statistical power.
- **B. Period × category segments** (categories normalised, minimum sample size).
  Tens of data points; labels come with the data, so no clustering instability;
  easy to explain ("POLITICS 2020 drifted and retrieval dropped; FOOD & DRINK
  barely drifted and did not"). Downside: category labels are editorial and
  changed over time.
- **C. Period × embedding cluster (topics).** Catches topics that cut across
  categories (COVID touches many). Downside: cluster IDs are unstable between
  runs and harder to explain.
- **Recommendation: B** as the primary unit; add C later for emerging topics
  that span categories.

### Q5. Drift technique 1: embedding space
- **A. Centroid cosine distance.** Simple, but misses changes in spread or shape,
  and has no significance test.
- **B. MMD (kernel two-sample test, Gretton et al. 2012) with a permutation test,
  plus a per-document novelty score** (1 − highest cosine similarity to any
  baseline document). MMD gives a distribution-level number with a p-value.
  Novelty gives concrete example documents and links straight to retrieval: a
  novel document has no close neighbour in a stale index. Downsides: quadratic
  cost (fine for a few thousand documents per segment); p-values shrink at large
  sample sizes, so an effect-size threshold is also needed.
- **C. Domain classifier** (logistic regression trained to tell baseline from
  new; AUC as the drift score). Interpretable (0.5 = no drift) and its most
  confident predictions are examples. Downsides: needs cross-validation and
  largely overlaps with B.
- **Recommendation: B**, reporting A alongside as the intuitive number. C is a
  possible cross-check later.

### Q6. Drift technique 2: lexical
- **A. Jensen–Shannon divergence on term distributions.** One number per
  segment, but does not say which terms changed.
- **B. Log-odds ratio with an informative Dirichlet prior** ("Fightin' Words",
  Monroe et al. 2008). A z-score per term gives ranked emerging and declining
  terms with significance, and handles rare terms well.
- **C. Out-of-vocabulary rate against the baseline vocabulary.** Very simple
  but noisy (names, typos).
- **Recommendation: A + B** together as the second technique: a number per
  segment plus term-level explanations. It complements embeddings: a small
  embedding model may place "COVID" near "flu" (low embedding drift) while
  lexical drift is large; a topic shift in familiar words shows up in embeddings
  but not in vocabulary. C goes into Agent A's descriptive statistics.

### Q7. Shifts in semantic relationships (brief, Agent B)
- **A. Term-context shift.** For a term such as "mask", "zoom", "remote" or
  "corona", embed the documents containing it in each period, measure how far
  their centroid moved, and list its top co-occurring words per period. Cheap,
  reuses existing embeddings, very clear in a demo. Needs enough occurrences per
  period.
- **B. Train word2vec per period and align them** (Hamilton et al. 2016).
  Standard in research, but there is too little text per period for stable
  vectors.
- **C. Skip it.** Leaves a brief requirement unmet.
- **Recommendation: A**, with candidate terms taken from technique 2's top
  emerging terms plus a few known ones. Build it after the thin end-to-end loop
  works. *Revised 2026-11-04 (D7): required by the brief, not a stretch goal.*

### Q8. Simulated user queries
- **A. LLM writes a question and reference answer from a sampled article,**
  phrased as a journalist would ask. Gives ground truth (source document +
  reference answer). Downsides: wording echoes the source (retrieval looks
  better than reality), and query drift mirrors document drift.
- **B. Headlines as queries, descriptions as targets.** No LLM cost, thousands
  of queries, but not realistic questions.
- **C. LLM writes period-aware questions without seeing an article.** Less echo,
  but no ground truth, and the LLM's own world knowledge leaks in.
- **Recommendation: A**, with a mix of evergreen and time-sensitive questions,
  a prompt that asks for paraphrased wording to reduce echo, and the circularity
  stated as a limitation. Split the output into an unlabelled **query stream**
  (monitored by Agents A and B, like production traffic) and a labelled
  **golden evaluation set** (used by Agents C and D, like a maintained eval
  suite).

### Q9. Retrieval ground truth (an important subtlety)
*Status: accepted for now (D2).*
A stale index cannot contain a 2020 article, so "did we retrieve the source
article?" is zero by construction on the stale index. That alone would make the
stale-vs-refreshed comparison trivial.
- **A. Known-item only** (the source article is the one relevant document).
  Cheap and exact, but meaningless for the stale index on new-period questions.
- **B. LLM-judged graded relevance of the pooled top-k from both indexes**
  (TREC-style pooling). A fair comparison, because an older article can still
  answer an evergreen question. Costs LLM calls; recall is relative to the pool.
- **C. Same category = relevant.** Free, but far too lenient (an Ebola story
  "matches" a COVID question).
- **Recommendation: A + B.** Known-item MRR and Recall@K on the refreshed index
  test whether the retriever copes with new vocabulary. LLM-judged Precision@K
  and nDCG@K on pooled results measure the stale-vs-refreshed impact. This is
  also what makes "which drift matters" testable: evergreen segments should
  lose little on the stale index, novel segments should lose a lot.

### Q10. Answer quality evaluation
- **A. RAGAS.** Recognised, but a heavy dependency with frequent API changes,
  opaque prompts, and extra LLM calls.
- **B. Custom LLM-judge rubrics:** correctness against the reference answer,
  groundedness against the retrieved context, relevance to the question. Fixed
  prompts, temperature 0, cached, and judged by a different (ideally stronger)
  model than the one writing answers, to reduce self-preference. Transparent and
  cheap, but the judge must be shown to be trustworthy.
- **C. Lexical overlap (ROUGE, token F1).** Deterministic, but a poor proxy for
  correctness.
- **Recommendation: B**, plus a 30 to 50 item hand-labelled check with the
  agreement rate reported.

### Q11. Deciding "significant" drift and "material" impact
- **A. Fixed thresholds only.** Simple but arbitrary.
- **B. Statistical tests only** (bootstrap confidence interval excludes zero,
  permutation p < α). Principled, but at large sample sizes trivial effects
  become "significant".
- **C. Both:** statistically significant and larger than a practical threshold
  written in the config before looking at results; drift thresholds calibrated
  against within-baseline variation (Q3).
- **Recommendation: C.** This produces the 2×2 at the heart of the project:
  drift + impact → act; drift + no impact → benign, keep monitoring;
  no drift + impact → another cause, escalate; neither → healthy. Control for
  testing many segments at once with Benjamini–Hochberg.

### Q12. Embedding model
- **A. all-MiniLM-L6-v2.** Very fast and widely used; weaker retrieval quality
  and older training data.
- **B. bge-small-en-v1.5.** Same size class (384 dimensions), generally stronger
  on retrieval benchmarks. Needs a query instruction prefix; slightly slower.
- **Recommendation: B** as the production retriever. Keep A as a second
  model so an "evaluate a different embedding model" recommendation can be
  backed by evidence. *Revised 2026-11-04 (D7): required, not a stretch goal.*

### Q13. Vector store
- **A. FAISS exact search (IndexFlatIP).** Exact and deterministic (no
  approximate-search noise in the experiment), and holding two indexes (stale,
  refreshed) is trivial. No built-in metadata store, so metadata lives in a
  parquet side table.
- **B. Chroma.** Persistence and metadata filters, but heavier, and its
  approximate (HNSW) search adds noise to comparisons.
- **C. Plain numpy matrix product.** No dependency, but reinvents FAISS.
- **Recommendation: A.** At up to about 50k × 384 vectors, exact search takes
  milliseconds.

### Q14. Orchestration and agentic behaviour
- **A. Linear LangGraph pipeline A→B→C→D→E.** Simple, but reads as a script.
- **B. Typed shared state plus conditional edges decided in Python.** For
  example: when nothing drifted, skip full answer evaluation but still run a
  small regression check to catch silent degradation; when an impact estimate
  is borderline, loop back once for more evaluation queries. Real dependencies,
  explainable, testable and bounded. Slightly more wiring.
- **C. An LLM supervisor that picks the next agent.** Looks the most "agentic",
  but non-deterministic, hard to test, and conflicts with our decision rule.
- **Recommendation: B**, subject to stakeholder question 2.

### Q15. LLM provider
*Status: decided 2026-11-04 (D9): Anthropic; Sonnet 5.5 writes, Opus 5.5 judges.*
- **A. OpenAI / Azure OpenAI** (named in the brief). Not reachable from this
  cloud development environment as currently configured.
- **B. Anthropic.** Reachable from this environment.
- **C. A local model (for example via Ollama).** Fully local, but slower, a
  weaker judge, and extra setup for anyone running the code.
- **Recommendation:** a provider-agnostic wrapper (one completion function and a
  disk cache keyed on model, prompt and parameters), so the provider is a config
  value. Default to whatever stakeholders provide or prefer (question 4). Use a
  small model for bulk generation and a stronger one as judge.

### Q16. Dashboard
- **A. Streamlit reading saved run outputs** (JSON/parquet and charts). The demo
  cannot fail on a live LLM call, and it is fast. Not "live".
- **B. Streamlit triggers the full pipeline.** Live, but slow and fragile in a
  15-minute demo.
- **Recommendation: A**, with a button to re-run on the small sample if time
  allows.

---

## Part 3. Ambiguities and conflicts in the brief

1. **Enterprise documents vs news** (stakeholder question 1). News is not
   replaced; it is superseded. Time-sensitive questions ("what is the current
   mask guidance?") are the right analogue: a stale index can produce an answer
   that is fluent, grounded in its context, and wrong. Retrieval metrics miss
   this; answer correctness catches it.
2. **"Continuously monitor" vs a historical dataset.** We replay periods as
   batches. Window size (yearly) is a design choice; this is not a real-time
   system.
3. **"Monitor user queries"** with no query log. Queries are simulated (Q8), so
   query drift partly mirrors document drift.
4. **Agents A and B overlap** ("vocabulary, topics" vs "emerging topics,
   changing terminology"). Proposed split: A does cheap descriptive monitoring
   and data quality (volumes, lengths, category mix with a chi-square test,
   new-term rate, empty descriptions, which segments have enough data); B does
   embedding-based and statistically tested drift with explanations, only on
   segments A marked as having enough data.
5. **"Improve document chunking"** cannot be meaningfully shown on one to two
   sentence documents. We will say so rather than fake it.
6. **"Trigger model evaluation or retraining":** **Revised 2026-11-04 (D7).**
   This item first read "trigger model retraining", which dropped the
   "evaluation" half of the requirement. Now: the system triggers and runs a
   model evaluation (a second, ready-made embedding model on the affected
   segments) when re-indexing does not fix quality, and recommends
   retraining, with evidence, when neither model copes. We do not train
   models ourselves.
7. **"Relationship between drift scores and quality"** needs enough data points;
   period level gives about 7 (Q4).
8. **Recall@K** needs complete relevance judgements, which do not exist.
   Known-item plus pooled judgements is the standard workaround (Q9).
9. **"Agentic system" (success criteria) vs "the LLM never makes the final
   decision" (CLAUDE.md):** see stakeholder question 2.
10. **Evidently and RAGAS** are suggested in the brief. We plan small, tested
    implementations of the specific statistics instead, for transparency and
    fewer heavy dependencies. Worth one sentence in the README.
11. **Volume cliff after May 2018** (dataset card): the most interesting period,
    2020, is the sparsest. *Verified (F1): about 2,000 articles a year from 2019.*
12. **Category labels change over time** (the dataset reportedly has
    near-duplicates such as WORLDPOST / THE WORLDPOST). Drift to report, or noise
    to normalise? Proposed: normalise for segment analysis using a mapping in the
    config, and report the raw label changes as a Data Monitoring finding.
    *Verified and done (F1, D6): 9 renamed sections merged via
    `categories.normalise`.*

---

## Part 4. Ideas from "Semantic Drift in Paraphrasing" (unpublished draft)

The paper measures drift between individual text pairs (a source and its
paraphrase). This project measures drift between collections of documents over
time. Most findings carry over as design ideas; the paper's numbers and
thresholds do not. Each idea below needs your approval.

### P1. Compare each period with the baseline and with the previous period
- **From the paper:** the "Boiling Frog" effect. Each paraphrase step looked
  safe (mean SDS about 0.38–0.41), while cumulative drift reached 2.27 times
  the threshold.
- **Transfer:** measure every later period against the fixed baseline (what
  the stale index was built on) and against the previous period. Monitoring
  only consecutive windows, which is common in practice, can miss slow drift.
- **Recommendation: adopt.** Impact decisions use cumulative drift, because that
  is what a stale index actually experiences; period-over-period drift flags
  sudden events. Expectation to test, not a result: consecutive years look
  modest while 2020 vs the baseline is large.

### P2. Classify drift as lexical, semantic, or both
- **From the paper:** the "Distance Effect Gap" (lexical metrics report high
  drift while embedding metrics report little) and the Type 4 "stealth" vs
  Type 5 "decay" signatures.
- **Transfer:** place each segment on two axes, lexical drift (technique 2) and
  embedding drift (technique 1). That gives four kinds: surface drift (new
  words, same meaning, such as renamed terminology), semantic drift (new
  meaning or topics), both, or neither.
- **Hypothesis for "which drift matters":** a dense retriever tolerates surface
  drift but not semantic drift; a keyword retriever (BM25) is hurt by both.
- **Recommendation: adopt the four kinds now** (no extra cost: both signals are
  already planned). Adding BM25 to test the hypothesis is a stretch goal (one
  small dependency).

### P3. Track how spread out each period is, not just where it sits
- **From the paper:** "drift deceleration": later iterations had narrower,
  left-shifted distributions, read as model collapse.
- **Transfer, two uses:**
  1. Track the dispersion of each period's embeddings. A period dominated by
     one story (2020) should be narrower, which a centroid-based measure misses.
  2. A quality gate on LLM-generated queries. Generated questions can collapse
     into generic phrasing, so compare their diversity with the documents they
     came from and regenerate if it is too low.
- **Recommendation: adopt both.** Cheap.

### P4. No fixed-weight composite score for decisions
- **From the paper:** SDS, a weighted sum of seven metrics with a 0.35
  threshold taken from ParaScore.
- **Recommendation: do not base decisions on a fixed-weight composite here.**
  1. The 0.35 threshold was calibrated for sentence pairs, not for
     distributions of documents. We calibrate against within-baseline variation
     instead (Q3, Q11).
  2. Fixed weights are hard to defend in a review without something to fit
     them against.
- **What our setting adds:** we do have something to fit against, measured
  quality loss. Agent D can test which drift signal, or combination, best
  predicts quality loss across segments. That turns the paper's
  metric-correlation analysis into a direct answer to "which drift matters".
  A composite can still be shown in the dashboard for sorting.
- **Worth preparing for if the paper comes up in the walkthrough:** the
  r = 0.92 correlation between SBERT distance and SDS is partly built in,
  because SBERT is a 35%-weight component of SDS.

### P5. Tiered data cleaning with a retention table
- **From the paper:** Table 1 (length filter, near-duplicate removal by cosine,
  noise filter, with counts removed at each step).
- **Recommendation: adopt** in `scripts/prepare_data.py`: length bounds,
  near-duplicate removal by embedding cosine, and the retention table logged in
  `docs/findings.md`. Thresholds are chosen from the exploration results.

### P6. Correlation heatmap of the drift metrics
- **From the paper:** Figure 9.
- **Recommendation: adopt.** Show how our drift metrics correlate across
  segments. Low correlation supports the claim that the two techniques are
  complementary; high correlation would mean one is redundant, which is itself
  worth reporting.

### P7. Low / medium / high severity bands
- **From the paper:** SDS bands at 0.35 and 0.45.
- **Recommendation: adopt the bands, but set their boundaries from the
  within-baseline null** (for example, medium above the 95th percentile of
  normal year-to-year drift, high above the 99th) rather than fixed constants.
  They map onto actions: low → monitor, medium → investigate, high → act if
  impact is confirmed.

### P8. Human validation
- **From the paper:** future work, section 7.3.
- Already planned for the LLM judge (Q10, 30 to 50 hand-labelled items).
