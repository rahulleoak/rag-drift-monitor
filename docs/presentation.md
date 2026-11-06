# Presentation Notes

Raw material for the 15–20 minute stakeholder walkthrough: the thought
process behind the project, the ideas worth showing, and the numbers that
back them. Not the slides themselves. Sections follow a possible talk order;
each point links to the decision (D*n*) or finding (F*n*) that holds the
detail.

**Status markers:** ✅ done and measured · 🔨 designed, not built yet ·
❓ open question.

Last updated: 2026-11-05 (charts and dashboard skeleton added).

---

## 1. The problem in one sentence

A RAG assistant's documents and users change over time. Some of that drift
breaks retrieval and answers; most of it does not. **The system's job is to
answer: which drift matters?**

- Talking point: detecting drift is easy, since something always changes. The
  hard and useful part is telling harmful drift from harmless drift, and
  recommending what to do about it with evidence.
- The whole design hangs off a four-way classification (🔨):
  drift **that hurts** · drift **that does not** · quality loss **without**
  drift · **healthy**.

## 2. Use case and dataset

**Use case:** a newsroom archive assistant. Journalists ask questions over the
company's own article archive.

**Dataset:** HuffPost News Category (2012–2022), the first suggestion in the
brief. ✅ (D3, F1)

- *Why not arXiv:* longer documents (chunking would matter), but a multi-GB
  download, drift events less crisp, and harder for a non-expert audience to
  follow.
- *Why not Amazon reviews:* drift there is mostly product mix and sentiment,
  a poor fit for a knowledge assistant.
- *How it was decided:* the selection criteria were **written down before
  looking at the data** (records per period, known drift events, usable
  categories, document length, cost), then each was checked (D3).
- *Provenance verified:* the local file is byte-for-byte the single file
  Kaggle lists (87,295,572 bytes), so it is the complete dataset, not a
  sample. Fingerprint recorded; the pipeline warns if the file changes. ✅
- *Honest limitation:* "documents" are a headline plus a one-to-two-sentence
  description (median 28 words), so chunking cannot be shown. We say so
  rather than fake it.

## 3. What the data showed before any modelling (F1) ✅

1. **The known drift is clearly there.** "coronavirus" appears in 0% of
   articles every year up to 2019, then **18.4% in 2020**. In 2021 it falls to
   3.0% while "covid" stays at 10.3%: the same topic under a new word. This is
   the test the detectors must pass; if they miss COVID, they are broken.
2. **Volume falls off a cliff.** About 30–35k articles a year until 2017, then
   about 2,000 a year from 2019 (the dataset card confirms HuffPost changed its
   site).
3. **The archive changes shape, not just content.** Sections were renamed
   (WELLNESS → HEALTHY LIVING → WELLNESS), and politics grows from 10% of
   articles (2014) to 39% (2020).

- Talking point: point 3 is the trap. A naive drift detector would mostly
  measure "more politics, less lifestyle" and call it semantic drift.
- Chart idea: watch-term rates per year (`runs/explore/huffpost/watch_term_rates.csv`).

## 4. Choosing the baseline: an evidence-based decision (F2, D6) ✅

The baseline is what the "stale" index is built on and what every later
year is compared against.

- **Method:** for six candidate baselines, measure how different their topic
  mix is from each later year (share of articles that would have to change
  category; 0% = same mix).

  | Candidate | Mix difference vs 2018–2022 | Own halves differ by |
  |---|---|---|
  | 2014–15 (first proposal) | 50.9% | 27.5% |
  | 2015 | 43.4% | 30.1% |
  | **2016–17 (chosen)** | **30.2%** | **13.5%** |
  | 2017 (runner-up) | 29.5% | 5.6% (half-years) |

- **Talking point: the first proposal was rejected by the data.** 2014–15
  looked reasonable until measured: half its articles would have to change
  category to match the later years.
- **Why not the runner-up (2017 alone):** same mix match, but one year can only
  be split into half-years. Severity thresholds are calibrated on normal
  year-to-year variation, so a half-year estimate would set them too tight
  and flag too much.
- **What the baseline does *not* fix, said openly:** about half the remaining
  gap is sections that did not exist before 2018 (mainly U.S. NEWS, 23% of
  2019). Design response: compare within each category too, and report new
  sections as a structural change, not drift.
- **Built-in humility:** D6 lists five checks that would reopen the choice,
  e.g. "if 2020's COVID drift is not flagged at least medium, the thresholds
  are too loose."
- Chart ideas: one bar per candidate (mean mix difference, chosen one
  highlighted); stacked category shares per period
  (`runs/explore/huffpost/baselines/category_shares.csv`).

## 5. Cleaning: making periods comparable (F3, D8) ✅

| Period | Raw | Kept |
|---|---|---|
| 2016–17 | 61,987 | 60,591 (97.7%) |
| 2018 | 9,734 | 9,728 |
| 2019–2022 | 1,398–2,066 each | all but 0–1 each |

- **Talking point: every cleaning rule falls almost entirely on the
  baseline.** Headline-only records (1,034), very short records and long
  newsletters exist only in 2016–17. Left in, they are a baseline-only format
  that a drift detector could mistake for drift.
- **Near-duplicates are mostly templated columns,** not news: 134 of 290 are
  "HuffPost Rise: What You Need To Know On <date>". The threshold (cosine
  0.95) was chosen by hand-checking pairs at several levels: just below it are
  different articles about the same event, which must be kept.
- **The rules are a first version** with written triggers to revisit them (D8).
- **Equal samples:** 1,398 articles per period for drift statistics (all of
  2022), because drift tests are sensitive to sample size. The indexes use
  every cleaned article.
- **Validation:** a throwaway simulation and the real cleaning code, written
  separately, gave identical counts for every rule and period.
- ❓ **Length gap:** baseline articles have a median of 24 words, 2019–2022
  have 29–31. This might inflate embedding drift; D6 check 5 tests it.

## 6. Architecture (🔨, diagram in `docs/architecture.md`)

Three parts:
1. **Simulator** (not an agent): releases each year as "new documents", and
   uses an LLM to write user questions from sampled articles.
2. **The RAG assistant under test,** with two indexes:
   **stale** (baseline only) and **refreshed** (baseline + new year).
3. **Five agents** in LangGraph, each reading the previous agents' reports:
   A Data Monitoring → B Drift Detection → C Retrieval Evaluation →
   D Impact Assessment → E Recommendation.

**The key experiment:** ask both indexes the same questions. If quality drops
on the stale index and recovers on the refreshed one, the new content was
missing (re-index). If it stays poor on both, the model does not handle the
new content (trigger model evaluation). This turns "does drift matter?" into
a measurement, not an opinion.

**Talking points on "agentic":**
- The agents depend on each other: C cannot choose what to test without B's
  drifted segments; D needs B's scores to relate drift to quality; E's rules
  take D's classification. Remove one and the next breaks.
- Routing is decided in code: skip the expensive evaluation if nothing
  drifted (but still run a small check for silent degradation); go back for
  more evidence when a result is borderline; go back to test a second model
  when re-indexing does not fix quality.
- **The LLM never makes the final decision.** It writes questions, answers,
  judgements and explanations; statistics and decisions live in tested
  Python. Auditable and repeatable, which is what an enterprise needs.
- ❓ Whether this matches the reviewers' idea of "agentic" is stakeholder
  question 2. A possible extension: the LLM *proposes* what to investigate,
  code *tests* it.

## 7. Detecting drift: two complementary techniques (🔨)

- **Embedding drift:** MMD (a kernel two-sample test, Gretton et al. 2012)
  with a permutation test for significance; per-article novelty (distance to
  the nearest baseline article) for concrete examples; centroid distance as
  the intuitive number; spread of each period.
- **Word drift:** Jensen–Shannon divergence for one number per segment;
  log-odds with an informative prior ("Fightin' Words", Monroe et al. 2008)
  for ranked emerging and declining terms.
- **Why both:** a small embedding model may put "COVID" close to "flu" (little
  embedding drift) while the vocabulary changed completely; a topic shift in
  familiar words shows up in embeddings but not vocabulary.
- **Four kinds of drift** (from the author's paper, P2): wording-only, meaning,
  both, neither. Hypothesis to test: keyword search suffers from wording
  drift, meaning-based search mostly does not.
- **Also measured:** query drift; how the context around key terms changes
  ("mask", "remote", "vaccine"); emerging topics that cut across sections
  (D7).
- **Significance and severity:** correction for testing many segments at once
  (Benjamini–Hochberg); low / medium / high bands set from normal 2016-vs-2017
  variation, not invented constants (P7).

### 7b. First drift results (F4) ✅

- **COVID is the biggest shift:** 2020 drifts **5.7× more than an ordinary
  year** (2017 vs 2016) from the baseline, and has the biggest single-year
  jump (4.2×). Its articles are also the least spread out: one story
  dominated the year.
- **The Boiling Frog, measured:** 2022 vs 2021 looks almost ordinary (1.2×),
  but 2022 vs the baseline is 4.7×. Year-on-year monitoring alone would miss
  it. (Direct link to the author's paper, P1.)
- **The two techniques disagree, usefully:** embedding drift peaks in 2020
  and eases; word drift keeps rising. Evidence they are complementary.
- **Emerging terms read like a news timeline:** 2020 coronavirus, covid-19,
  biden; 2021 vaccine, shot, jan (6th); 2022 russia, putin, invasion,
  abortion.
- **Headline surprise for the talk:** to the embedding model, **COVID is not
  new.** COVID articles are no more "novel" than ordinary baseline articles
  (5.0%), because they land next to older stories on bird flu, Zika and
  Trump's health policy ("China's Coronavirus Infections Top 40,000" →
  "H7N9 Bird Flu Epidemic In China"). The most novel 2020 articles are
  quirky one-offs. **Why it matters:** a stale index will confidently
  retrieve related but wrong articles for COVID questions: fluent,
  grounded-looking, and wrong. That is harmful drift the embedding space
  does not flag, and exactly what the impact experiment will test. 🔨
- **"Everything is significant":** at 1,398 articles per side even 2017 vs
  2016 is significant. So the system judges drift by size relative to
  ordinary change, not by p-values.

### 7c. Severity and drift by section (F5, D11) ✅

- **How "unusual" is defined:** by the archive's own history, not invented
  thresholds. Ordinary change = the same month a year apart inside the
  baseline (January 2016 vs January 2017, ...). Medium = beyond the 95th
  percentile of that, high = beyond the 99th.
- **Whole years:** 2020 flagged high (meaning drift), 2021 medium (wording:
  vaccine, covid), 2022 medium on both. 2018 and 2019 low. No single year is
  unusual next to the previous one; the archive drifts away gradually.
- **Where COVID hit hardest: lifestyle journalism, not politics.** WELLNESS,
  FOOD & DRINK, STYLE & BEAUTY and PARENTING in 2020 are the most drifted
  sections of all ("Is It Safe To Go To The Grocery Store?", "How To Make
  Coronavirus Face Masks"). Practical, time-sensitive advice a 2017 archive
  cannot give: the best test cases for the impact experiment. 🔨
- **Sensible secondary signals:** world news in 2020 (outbreak) and 2022
  (Ukraine invasion); sports in Olympic years; late-night comedy on
  impeachment in 2019.
- **The politics paradox:** politics is *never* flagged, despite Trump ->
  Biden, January 6th and Ukraine, because 2016-17 politics (an election,
  then a new administration) already changed so much month to month. Good
  talking point: drift is relative to what is normal for that section. And
  the honest follow-up: if politics questions still lose quality on the
  stale index, that is "quality loss without drift", and the system has a
  path for it (regression check, escalate to a human).
- **Agent A in one line:** the topic mix moved (Cramér's V 0.2-0.5), and
  13-26% of later articles sit in sections the baseline does not have,
  including sections that existed before 2016 and *came back* (MONEY,
  HOME & LIVING, WEDDINGS).

### 7d. Topics and words that changed meaning (F6) ✅

- **Emerging topics fix the novelty blind spot:** grouping all articles into
  topics and looking for *growth* finds three COVID topics in 2020 with no
  baseline counterpart (political, lifestyle, world outbreak), then
  vaccines (2021) and the Ukraine war (2022). Lesson for the talk: "new" is
  not the same as "unlike anything before"; COVID was new by volume, not by
  vocabulary neighbourhood.
- **Everyday words changed meaning** (best slide material):
  - "cases": court cases → COVID cases
  - "test": North Korean missile tests → positive coronavirus tests
  - "home": refugees and family → hair, products, staying home
  - "virus": Zika → coronavirus; "outbreak": Ebola → pandemic
  - "russia": election probe → Ukraine war; "capitol": Capitol Hill →
    January 6th
- **Why it matters for RAG:** a 2017 archive will answer "how many cases
  this week?" with court cases. Same word, different meaning: a retrieval
  failure no keyword filter would catch. 🔨 (to be tested in Agent C)
- **Three families of drift, measured (P6):** distribution shift (MMD and
  centroid, 0.93: near-redundant), word drift (0.68 with MMD: related but
  distinct) and article novelty (0.07 with MMD: unrelated). Supports
  "complementary techniques" with a number.

### 7e. Plan: catching harm that drift measures miss (D13) 🔨

- Politics is never flagged, yet a 2017 archive cannot answer questions
  about the Biden administration. So the system always evaluates some
  undrifted sections and historical control questions, classifies every
  section on drift × quality loss, and diagnoses the cause (missed drift,
  prompt, model, pipeline, or superseded facts) before escalating.

## 8. Measuring quality honestly (🔨)

- **A subtle trap (D2):** a stale index cannot contain a 2020 article, so
  "did we retrieve the source article?" is zero by construction. Solution:
  known-item metrics on the refreshed index, plus LLM-judged relevance of the
  pooled results from both indexes (the TREC pooling method). An older
  article can still answer an evergreen question, which is exactly what
  separates harmless drift from harmful drift.
- **Control group:** questions from baseline articles should score the same on
  both indexes. If they do not, something other than drift is wrong (D7).
- **Two different models (D9):** Sonnet 5.5 writes questions and answers, Opus
  5.5 judges, so the judge is not grading its own style.
- **Blind judging:** the judge never sees which index produced an answer.
- **Trust in the judge is measured:** about 40 hand-labelled items, agreement
  reported (P8).
- **Cost control:** every LLM response cached to disk, a 20-question pilot
  before full runs, Batch API at half price.

## 9. Recommendations with evidence (🔨, D7)

One rule per action in the brief, each triggered by evidence:

| Action | Evidence that triggers it |
|---|---|
| Keep monitoring | Drift, but no quality loss |
| Re-index | Quality recovers on the refreshed index |
| Trigger model evaluation | Quality still poor on the refreshed index; **the system runs it** with a second embedding model |
| Recommend retraining | Neither model copes even with fresh data (recommended, not run) |
| Change retrieval settings | The right article sits just outside the top K |
| Improve prompts | Retrieval fine, answers wrong or ungrounded |
| Escalate to a human | Quality loss without drift, or inconclusive evidence |
| Improve chunking | Rule exists; not applicable to 1–2 sentence articles |

- Talking point: every recommendation carries its numbers; a check rejects
  any recommendation without cited evidence.

## 10. Research into practice: the author's paper

From "Semantic Drift in Paraphrasing" (measures drift between sentence
pairs; this project measures drift between collections over time). Approved
as D5.

- **Boiling Frog (P1):** each paraphrase step looked safe, but cumulative drift
  was 2.27× the threshold. Here: compare each year with the fixed baseline as
  well as the previous year. ❓ Whether the two disagree enough to matter
  will be reported either way.
- **Distance Effect Gap (P2):** lexical and semantic metrics disagree; here
  that becomes the four kinds of drift.
- **Drift deceleration (P3):** narrowing distributions; here, tracking the
  spread of each period, and checking generated questions are not collapsing
  into generic phrasing.
- **What deliberately did not transfer (P4, P7):** the fixed-weight composite
  score and its 0.35 threshold were calibrated for sentence pairs. Here the
  thresholds come from the data, and measured quality loss decides which
  drift signal matters.
- Be ready for: the paper's r = 0.92 between SBERT distance and the composite
  is partly built in, since SBERT is 35% of the composite.

## 11. Engineering practices worth showing

- **Evidence before decisions:** criteria written before looking at data (D3);
  thresholds hand-checked on examples (D8); baseline chosen by measurement
  (F2).
- **Every number reproduces:** a confidence check re-ran everything from
  scratch; 14 of 14 documented numbers matched.
- **Tests:** 36, covering every cleaning rule, the embedding cache, the
  baseline comparison and an end-to-end run on synthetic data. They run in
  about a second, with no models or keys.
- **Config, not hard-coding:** periods, thresholds, models and the category
  mapping all live in one file.
- **Caching:** embeddings cached (a rerun takes 14 s instead of 80 s); every
  LLM response cached too, so reruns cost nothing.
- **Money is controlled in code:** a $150 budget enforced *before* any
  request is sent, a ledger of every billed call, a 20-question pilot before
  any full run, and half-price batches for bulk jobs (D15).
- **Decisions are logged** with options, reasons and evidence (D1–D9); when one
  changes, it is marked "Revised", not deleted (e.g. Q2, Q7, Q12).
- **Requirements traceability:** `docs/requirements-coverage.md` maps every
  bullet of the assignment to the system.

## 12. Thought process: course corrections worth telling

Honest moments that show judgement, good for a "what we learned" slide:

- **The first baseline proposal was wrong,** and the data showed it (section 4).
- **A requirement had been misread:** "trigger model evaluation *or*
  retraining" had been paraphrased as "retraining" and set aside as out of
  scope. Re-reading the original brief line by line caught it; the
  evaluation half is now something the system actually does (D7).
- **Two brief requirements had been marked as stretch goals** (shifts in
  semantic relationships; a second embedding model). Both were made required
  after the requirements check.
- **An earlier assumption reversed:** "keep empty descriptions" (Q2) was
  overturned once the data showed they exist only in the baseline (D8).
- **The runner-up changed on measurement:** 2015 alone looked like the
  runner-up until its internal instability (30%) was measured; 2017 alone
  replaced it.
- **A cost-free alternative to temperature:** current models reject the
  temperature setting, so reproducibility comes from caching every response.
- **Expected COVID to be the most novel content; it was not** (F4). Instead of
  hiding it, it became the central hypothesis for the impact experiment.
- **Tuning the test-question writer taught us what "good" means** (F7, F8):
  version 1 kept 75% but had no time-sensitive questions and copied the
  articles; version 2 fixed that but rejected ordinary knowledge like
  "Justin Trudeau"; version 3 showed that even Opus cannot consistently say
  whether "what police said at the time" will go out of date. A strong model
  as a reviewer is useful, but its yes/no checks need the same scrutiny as
  any other measurement. Version 4 let the reviewer assign the label itself:
  60% kept, and the two models agree on the label 83% of the time, the
  ambiguity we now state with any time-sensitive result.
- **Oversampling on purpose, and saying so** (D20, F10): random sampling gave
  1 clear time-sensitive question in 13; a word-filtered pool of
  likely-revisable articles gave 4 in 9. Those questions are always
  reported as their own group, never mixed into archive-wide rates.
- **Correcting our own reading of the data:** after v4 I attributed three
  flipped decisions to the reviewer's randomness; a controlled test (same
  prompt, two reviews) showed most of it was the prompt change, and the
  reviewer agrees with itself 92% of the time (F9). Good example of testing
  an explanation before acting on it.
- **Expected politics to be the most drifted section; it never was** (F5),
  which exposed how the election year inflates "normal" for that section.
- **Two of our own measures needed correcting** (D12): a word-shift ratio
  inflated for small groups, and a prior summed over the wrong vocabulary.
  Both caught by tests and checks before any result was reported.
- **A metric that looked fine was mostly noise:** raw word-distribution
  divergence scored 2018 below an ordinary year. Subtracting what random
  shuffling gives ("excess") fixed it (D10).

## 12b. From prototype to production (README section and one slide)

- **Local by design** (CLAUDE.md rule; reviewers can clone, install, run;
  public data only; 220 MB fits a laptop). A take-home that needs cloud
  accounts risks never being run.
- **What a company deployment would change:** shared vector database
  (Qdrant, pgvector or managed) instead of in-memory FAISS; shared storage
  with backups; logins, permissions and audit logs; a secrets manager;
  scheduled monitoring with alerts; run history.
- **Why the move is cheap:** storage sits behind a few small interfaces
  (`VectorIndex`, the document loader, the LLM cache); agents and
  statistics do not change. Qdrant with exact search could replace FAISS.
- **Framed for Rivvun** (enterprise AI for contracts and commercial
  obligations): confidential customer documents (access control, audit,
  separation per customer, data location); accountability (decisions in
  code, evidence on every recommendation, cost ledger); continuous
  monitoring. The method transfers: new clause language, superseded
  policies and renamed terms are the contract-archive version of
  "coronavirus → covid". Do not guess Rivvun's tech stack; keep it generic.

## 13. Limitations to state up front

- Documents are short; chunking cannot be demonstrated.
- No real user queries or relevance labels: questions are LLM-generated from
  articles, so query drift partly mirrors document drift (mitigated by
  questions in newer wording about older topics).
- After 2018 only about 2,000 articles a year; 2022 is a partial year.
- The 2016 election and #MeToo fall inside the baseline, so cannot be used as
  test events.
- Roughly 30–35 period × category segments: enough to relate drift to
  quality, but not by a wide margin.
- The LLM judge is a model, not a human; its agreement with hand labels is
  reported.

## 14. Likely reviewer questions (with short answers)

- **"Why not just use a drift score threshold?"** Thresholds from other
  settings do not transfer; ours come from normal variation within the
  baseline, and measured quality loss decides which drift matters.
- **"Isn't this a pipeline, not agents?"** See section 6: real dependencies,
  routing on intermediate results, loops back for more evidence; decisions in
  code by design.
- **"How do you know the LLM judge is right?"** Hand labels, agreement
  reported; a different model from the one being judged; blind to the index.
- **"Is the LLM reviewer consistent?"** Measured: reviewing the same question
  twice gives the same keep/reject decision 92% of the time and the same
  label 100% of the time (F9). Its errors are mostly systematic (it
  consistently accepted one paraphrase that added a detail), which repeat
  reviews cannot catch; that is what the hand labels are for.
- **"Could the judge just look things up or use what it already knows?"** It
  cannot search: no request declares a search tool, and each call holds
  only the instructions and one article. It does not learn from our calls.
  Its built-in knowledge cannot be removed, only constrained and checked;
  evidence it holds the line: it rejected questions for facts it surely
  knows but the article does not state ("the Paris deal"). (D18)
- **"Why HuffPost and not enterprise documents?"** Public, timestamped, with a
  known drift event to validate against; limitations stated.
- **"Didn't the brief ask for a vector database?"** It lists "Qdrant / FAISS /
  Chroma" as options; we use FAISS with exact search, because approximate
  search would add retrieval noise to the very effect we measure (D22).
- **"Why not a knowledge graph?"** Considered (D21): it would make relationship
  changes explicit, but extraction from one-sentence descriptions is the hard
  part, OWL reasoning and a graph database add days and infrastructure for
  no change to retrieval, and relationship shift is already measured with a
  significance test (F6). Future work: temporal knowledge graphs, GraphRAG.
- **"What would you do with more time?"** Keyword search (BM25) to test the
  wording-vs-meaning hypothesis; fine-tuning experiment for the retraining
  recommendation; longer documents to show chunking.

## 15. Charts

Ready-made PNGs in two themes based on the Rivvun AI brand (D14): light
(cream) in `docs/figures/`, dark (navy, matching rivvun.ai) in
`docs/figures/dark/`, which suits dark slides. Regenerate both with
`python scripts/make_charts.py` after a pipeline run. Every palette was
checked for colour-blind safety and contrast with the dataviz validator.

| Chart | File | Use it for |
|---|---|---|
| COVID words per year | `pandemic_words.png` | Section 3: the known drift is in the data |
| Drift per year, vs baseline and vs previous year | `drift_per_year.png` | Sections 7b–7c: 2020 peak, Boiling Frog |
| Severity by section × year | `severity_by_section.png` | Section 7c: lifestyle sections, the politics paradox |
| Emerging topics | `emerging_topics.png` | Section 7d: found by growth, not novelty |
| Words that changed meaning | `words_that_changed_meaning.png` | Section 7d: cases, test, home, virus |
| How the drift measures relate | `measure_correlations.png` | Section 7d: three families of drift (P6) |

Still to make: baseline candidates (F2, from
`runs/explore/huffpost/baselines/mix_difference.csv`); stale vs refreshed
quality per segment, drift vs quality loss, and the four-way classification
(after Agents C and D).

**Live demo:** `streamlit run app/streamlit_app.py` (Rivvun dark theme): Overview (stat tiles,
interactive drift charts, most drifted segments), Data monitoring (Agent A),
Drift (interactive severity grid with hover, a drill-down into any segment,
topics, shifted words, correlations), and placeholder pages for Agents C–E.
