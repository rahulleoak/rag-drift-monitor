# Presentation Notes

Raw material for the 15–20 minute stakeholder walkthrough: the thought
process behind the project, the ideas worth showing, and the numbers that
back them. Not the slides themselves. Sections follow a possible talk order;
each point links to the decision (D*n*) or finding (F*n*) that holds the
detail.

**Status markers:** ✅ done and measured · 🔨 designed, not built yet ·
❓ open question.

Last updated: 2026-11-09 (all agents built and run; Wikipedia results; the
newer-wording test and the applied re-index; next steps in section 16).

The slide-by-slide brief (titles, bullets, speaker notes, screenshots) is
kept on the `presentation` branch until after the walkthrough. Numbers here
match the demo run `demo/runs/20261109-221754`.

---

## 1. The problem in one sentence

A RAG assistant's documents and users change over time. Some of that drift
breaks retrieval and answers; most of it does not. **The system's job is to
answer: which drift matters?**

- Talking point: detecting drift is easy, since something always changes. The
  hard and useful part is telling harmful drift from harmless drift, and
  recommending what to do about it with evidence.
- The whole design hangs off a four-way classification (✅, D13, F20):
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
- *Limitation, and the fix:* "documents" are a headline plus a one-to-two-sentence
  description (median 28 words), so chunking cannot be shown on HuffPost.
  After the stakeholder asked for long-form documents (2026-11-06), a second
  set was added: 165 Wikipedia articles at yearly snapshots (D26, section 8c).

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
- ❓ **Length gap, not tested:** baseline articles have a median of 24
  words, 2019–2022 have 29–31. This might inflate embedding drift. D6 check
  5 planned a re-run on length-matched baseline samples; it was not run (a
  next step, section 16). If asked: the finding that matters (drift does
  not predict harm) does not depend on the size of the drift scores.

## 6. Architecture (✅, diagram in `docs/architecture.md`)

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
  more evidence when a result is borderline (D to C, built; it did not fire
  because no section was borderline); go back to test a second model when
  re-indexing does not fix quality (designed; Agent E's rule shows it is not
  needed on this data, with the number).
- **The LLM never makes the final decision.** It writes questions, answers,
  judgements and explanations; statistics and decisions live in tested
  Python. Auditable and repeatable, which is what an enterprise needs.
- ✅ Confirmed by the stakeholder (Ani, 2026-11-06): this split "fits ...
  the right split". A possible extension: the LLM *proposes* what to
  investigate, code *tests* it (parked).

## 7. Detecting drift: two complementary techniques (✅)

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
  both, neither. Hypothesis not yet tested: keyword search suffers from
  wording drift, meaning-based search mostly does not (BM25 is a next step;
  for meaning-based search, F23 found newer wording barely hurts).
- **Also measured:** query drift; how the context around key terms changes
  ("mask", "remote", "vaccine"); emerging topics that cut across sections
  (D7).
- **Significance and severity:** correction for testing many segments at once
  (Benjamini–Hochberg); low / medium / high bands set from normal 2016-vs-2017
  variation, not invented constants (P7).

### 7b. First drift results (F4) ✅

> **Resolved (D30, 2026-11-08): quote the pipeline's numbers below.** F4's
> early figures (5.7×, 4.2×, 1.2×, 4.7×) divided by a single comparison
> (2017 vs 2016). The pipeline divides by 12 same-month pairs inside the
> baseline, the same scale the dashboard shows. If a reviewer quotes 5.7×
> from the design summary: same measurement, smaller denominator (MMD²
> 0.0021 vs 0.0047); rankings and conclusions are identical.

- **COVID is the biggest shift:** 2020 drifts **2.5× ordinary change** from
  the baseline (the only "high" year), and has the biggest single-year jump
  (1.9× from 2019). Its articles are also the least spread out: one story
  dominated the year.
- **The Boiling Frog, measured:** 2022 vs 2021 looks ordinary (0.5×), but
  2022 vs the baseline is 2.1× (medium, flagged). Year-on-year monitoring alone would miss
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
  does not flag. ✅ Confirmed by Agent C (F15): the COVID takeout question
  gets 2016 hepatitis and E. coli stories from the stale index.
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
  cannot give. ✅ They did lose answers on the stale index, but so did
  sections that were never flagged (F20).
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
  failure no keyword filter would catch. ✅ Measured: the stale index fails
  on new questions whether or not their section drifted (F15), so the cause
  is missing content more than shifted meaning.
- **Three families of drift, measured (P6):** distribution shift (MMD and
  centroid, 0.93: near-redundant), word drift (0.68 with MMD: related but
  distinct) and article novelty (0.07 with MMD: unrelated). Supports
  "complementary techniques" with a number.

### 7e. Catching harm that drift measures miss (D13) ✅

- Politics is never flagged, yet a 2017 archive cannot answer questions
  about the Biden administration. So the system always evaluates some
  undrifted sections and historical control questions, classifies every
  section on drift × quality loss, and diagnoses the cause before
  escalating. ✅ It paid off: 12 of the 19 sections that lost answers were
  never flagged (F20), and Agent E sends them to a person.

## 8. Measuring quality honestly (✅)

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
- **Trust in the judge is measured (F19, F20):** 50 blind items labelled by
  two people independently. On "answers or not" the relevance judge
  matches a person 14/15 and 11/15, while the two people match each other
  12/15; it is almost always the stricter one. Changed-facts precision
  80-85%. Answer judge 11/15 with one person, all differences borderline.
  Talking point: I report the first blind pass, not the numbers after my
  re-check, because the re-check was prompted by the same model family as
  the judge.
- **Cost control:** every LLM response cached to disk, a 20-question pilot
  before full runs, Batch API at half price.

### 8b. Results: retrieval and answers, stale vs refreshed (F15, F20) ✅

- **Not re-indexing loses almost every new question.** Stale index: 0-1%
  of later questions have an answer in the top 5; refreshed: 96-100%. Same
  at the answer level: 0-1% correct vs 96-100% (2018-2022). Baseline
  questions: 100% on both, so re-indexing breaks nothing old.
- **Silent, not wrong:** on the stale index the assistant declined 98-100%
  of the time when the top 5 held no answer; 1 of 438 stale answers added
  a claim not in the articles.
- **The headline: drift scores do not predict where the harm is.** All 19
  sections with enough questions lose answers; 7 were flagged ("drift that
  hurts"), 12 were not ("harm without drift"), 4 of them U.S. NEWS, a
  section new since the baseline that drift detection cannot score. No
  drift measure predicts the size of the loss (Spearman -0.19 to 0.41, all
  p > 0.1).
- Talking point: on a news archive, **the drift that matters is new
  content the index does not have**, not how far a section's meaning moved.
  Drift scores explain *what* changed; they do not decide *whether* to
  re-index.

### 8c. Long-form documents: chunking and changed facts (Wikipedia, D26-D28) ✅

- **Why it exists:** the stakeholder asked for long-form documents
  (2026-11-06). 165 Wikipedia articles (median about 8,000 tokens) at the
  end of each year 2016-2022: documents that are *edited*, the brief's
  "documents become outdated" case.
- **Chunking (F16):** a whole-article embedding reads only the first 512
  tokens (91% of the text unseen) and follows real edits poorly (Spearman
  0.51); 256-token chunks follow them at 0.87 (128-token: 0.89, twice the
  vectors). Agent E recommends chunking with these numbers.
- **The blind spot (F17, F18):** a changed number barely moves the vectors.
  About one article in five changes a fact each year, and 95% of articles
  with a changed fact had a "low" drift score. So a cheap text check finds
  replaced numbers and dates (D27): 391 since the index was built, in 90
  of 165 articles; precision 80-85% by blind hand labels.
- **Does it matter? (F21):** a flagged edit changes the answer 50% of the
  time, vs 2% for an edit that kept its numbers (Fisher p = 1.5e-10). The
  stale index still retrieves the edited passage 92% of the time, so it
  serves the outdated fact with confidence.

### 8d. Applying the fix and re-checking (D32, F24) ✅

The brief's "apply one fix and re-check": Agent E's targeted re-index of
the 90 flagged articles, applied to a copy of the index.

| Edited sentences (2017 -> 2022) | Outdated, 2017 index | Outdated, targeted fix |
|---|---|---|
| Changed facts, flagged articles | 68% | **0%** |
| Other edits, flagged articles | 12% | 0% |
| Edits in the other 75 articles | 10% | 10% |
| All, weighted | 12.9% | **3.0%** |

- For 64% of the re-embedding, it removes 77% of what a full re-index
  removes: **short of the 80% set in advance**. All 6 misses are "cannot
  answer", never a wrong fact: 3 are detail added later to an unflagged
  article (e.g. "14 August 1703" for "August 1703"), 3 are noise.
- Talking point: lead with "changed facts 68% -> 0%". Then: "it missed our
  own bar, so the recommendation became: re-index the flagged documents now,
  plus a scheduled full re-index." Shows the thresholds were not tuned.
- Do not quote the exploratory 86% (noise removed after seeing results) as
  the result; the interval on 77% is wide (50-92%).

### 8e. Questions in newer wording (D31, F23) ✅, backup slide

- **The objection it answers:** "your questions copy the article's words;
  what if users ask in today's words, COVID instead of coronavirus?"
- Of 370 questions about 2016-2021 articles, only **18 (5%)** have a
  natural newer wording (11 are coronavirus -> COVID). Search still found
  the article in the top 5 for 17 of them; the rank slipped for 5
  (mostly 1st -> 2nd), never improved.
- By the rule set in advance: wording drift does not hurt search; flagged
  vs control could not be told apart (8 and 5 questions).
- Say "no sign of harm", not "proof": 18 is small. The names in a question
  carry search, and names rarely change in five years.

## 9. Recommendations with evidence (✅, D29, F22)

One rule per action in the brief, in code; every rule reports, fired or not.
On our data (run `recommendations.md`):

| # | Action | Status | The evidence in one line |
|---|---|---|---|
| 1 | Re-index the archive **on a schedule** | recommended | 0-1% vs 96-100% correct; 12 of 19 losing sections not flagged by drift; no drift measure predicts the loss |
| 2 | Re-index the 90 articles with changed facts, plus a scheduled full re-index | recommended | 391 changed numbers or dates; flagged edits change answers 50% vs 2%; 95% had a "low" drift score; applied: outdated answers 13% -> 3% (section 8d) |
| 3 | Escalate 12 sections to a person | recommended | lost answers without a drift flag: 8 scored low, 4 are a section new since the baseline |
| 4 | Chunking for long documents | recommended | 92% of an article past 512 tokens; chunks follow edits 0.87-0.89 vs 0.51 |
| 5 | Keep monitoring old questions | recommended | 100% vs 100%; compare with the baseline (2022 is 0.5x vs 2021 but 2.1x vs baseline) |
| - | Retrieval settings, prompts, model evaluation, retraining | not needed | refreshed top 5 holds an answer 96-100%; 0.4% invented claims; every section recovers |

- **Talking point: rules decide, the LLM explains, code checks.** Sonnet
  writes each explanation from the evidence only; any number not in the
  evidence is rejected (one retry, then the rule's own text). All five
  passed on the first try; under $0.01 per run.
- **Talking point: "not needed" is a result.** The model-evaluation rule
  shows why no second embedding model was tried: every section recovers
  on the refreshed index. That's evidence, not a skipped step.
- **Course correction found by E:** 4 of the 12 "harm without drift"
  sections are U.S. NEWS, which did not exist in the baseline, so drift
  detection could not score it at all. E's escalation says so, with a
  different lesson (a new section needs its own baseline).
- **Likely question: "Why not let the LLM decide?"** Repeatable, testable,
  and no invented numbers (D9, D29). The LLM's value is the plain-language
  explanation, and that is checked.

## 10. Research into practice: the author's paper

From "Semantic Drift in Paraphrasing" (measures drift between sentence
pairs; this project measures drift between collections over time). Approved
as D5.

- **Boiling Frog (P1):** each paraphrase step looked safe, but cumulative drift
  was 2.27× the threshold. Here: compare each year with the fixed baseline as
  well as the previous year. ✅ They disagree: 2022 vs 2021 is 0.5× an
  ordinary year, but 2022 vs the baseline is 2.1× (flagged).
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
- **Tests:** 192, covering the cleaning rules, the drift statistics, the
  LLM layer (with a fake API), every agent, the policy rules, the dashboard
  on any run (including one without LLM results) and every script. They run
  in about 30 seconds, with no models or keys.
- **Thresholds written before results:** every experiment's pass/fail rule
  is in the config and the decision log before the run (D28, D31, D32), and
  one of them failed (F24) and was reported as failed.
- **Config, not hard-coding:** periods, thresholds, models and the category
  mapping all live in one file.
- **Caching:** embeddings cached (a rerun takes 14 s instead of 80 s); every
  LLM response cached too, so reruns cost nothing.
- **Money is controlled in code:** a $150 budget enforced *before* any
  request is sent, a ledger of every billed call, a 20-question pilot before
  any full run, and half-price batches for bulk jobs (D15).
- **Decisions are logged** with options, reasons and evidence (D1–D32); when one
  changes, it is marked "Revised", not deleted (e.g. D12, Q2, Q7, Q12).
- **Total LLM spend:** about $17 for the whole project, every call in a
  ledger (`runs/llm_costs.csv`).
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
- **Expected drifted sections to lose more; they did not** (F15, F20). The
  project's main finding came from a hypothesis that failed.
- **Two "× ordinary" scales were in circulation** (D30): the early figure
  (5.7× for 2020) used a single comparison; the pipeline's (2.5×) is
  steadier and is what the dashboard shows. Fixed everywhere, and the old
  number explained rather than hidden.
- **Our own fix missed our own bar** (F24): the targeted re-index reached
  77% against 80%, reported as a miss, and turned into a better
  recommendation.

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

- HuffPost documents are short, so chunking is shown on the Wikipedia set
  instead (165 articles, a small sample).
- No real user queries or relevance labels: questions are LLM-generated from
  articles, so query drift partly mirrors document drift, and questions
  name the people and events they ask about. The newer-wording test (F23)
  checks the most likely failure; a real query log could hold vaguer
  questions.
- After 2018 only about 2,000 articles a year; 2022 is a partial year.
- The 2016 election and #MeToo fall inside the baseline, so cannot be used as
  test events.
- 19 period × section groups have enough questions (8 or more) to
  classify; 43 more are reported but not classified.
- The LLM judge is a model, not a human; its agreement with hand labels is
  reported (samples of 15-20 items per part; the answer sheet has one
  labeller).
- The changed-facts check sees replaced numbers and dates only, not names
  or added numbers (D27, F24).

## 14. Likely reviewer questions (with short answers)

- **"Why not just use a drift score threshold?"** Thresholds from other
  settings do not transfer; ours come from normal variation within the
  baseline, and measured quality loss decides which drift matters.
- **"Isn't this a pipeline, not agents?"** See section 6: real dependencies,
  routing on intermediate results, loops back for more evidence; decisions in
  code by design.
- **"How do you know the LLM judge is right?"** Two people labelled blind:
  the judge agrees with them as often as they agree with each other on the
  answers cut, and where it disagrees it is almost always stricter (F19); a different model from the one being judged; blind to the index.
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
- **"What would you do with more time?"** See section 16; lead with the
  changed-facts check for names and added numbers, and BM25.
- **"Isn't it obvious a stale index fails on new questions?"** Yes, and
  that is the point: the system shows the obvious fix (re-index on a
  schedule) beats the sophisticated trigger (re-index when drift is
  flagged), because drift flags missed 12 of 19 losing sections.
- **"Why did your own fix fail its test?"** It did not fail at fixing
  changed facts (68% -> 0%); it missed detail added to articles it did not
  list, so it reached 77% of a full re-index against an 80% bar set before
  the run. Hence "targeted now, full on a schedule" (F24).

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

The dashboard has live versions of these plus the retrieval, answer and
Wikipedia charts; dark screenshots of each section for slides are on the
`presentation` branch. Not made: a baseline-candidates chart (F2, from
`runs/explore/huffpost/baselines/mix_difference.csv`).

**Live demo:** `streamlit run app/streamlit_app.py` (dark theme). Six pages
in the left bar: Overview, Data Monitoring (A), Drift (B), Retrieval (C),
Impact (D) and Recommendations (E). Every "in this run" sentence is built
from the run's numbers. Suggested path: Overview -> Impact (drift score vs
lost answers) -> Drift (one group in detail) -> Recommendations (expand the
evidence on action 2).

## 16. Next steps (what we would do with more time)

Not done, in the order I would do them; each has the reason it would help.

1. **Changed-facts check for names and added numbers.** The two gaps F24
   measured: a new office holder, or "August 1703" -> "14 August 1703".
   Needs care so added citation years are not flagged (D27).
2. **Keyword search (BM25) next to meaning-based search.** Tests the
   paper's hypothesis that wording drift hurts keyword search more (P2).
3. **More test questions per section.** 43 groups had too few (under 8) to
   classify; more questions would tighten the drift-vs-harm comparison (F20).
4. **A larger Wikipedia sample.** 165 of about 1,000 vital articles; would
   narrow the intervals on changed facts, chunking and the re-index check
   (F24's 50-92%).
5. **128-token chunks for retrieval.** They followed edits best (0.89 vs
   0.87) at twice the vectors; Agent E suggests testing them (F16).
6. **Larger hand-label samples and a second labeller for the answer
   sheet** (F19, F20).
7. **A second embedding model and the loop from E back to C.** Designed
   (D7); not built because every section recovers after re-indexing (D29).
   Could be shown by forcing it on one section.
8. **Let the LLM propose what to investigate** (for example which terms to
   test), with code testing it: more visibly agentic without the LLM
   deciding (D7; raised with the stakeholder, 2026-11-06).
9. **Revisit the open points in D17 and D2** (is the unanswerable set
   useful; weight of known-item metrics), written before results existed.
10. **Entity graph per period** for a few key entities (D21), and a
    **fine-tuning experiment** for the retraining action (D7): larger
    pieces of work, lower priority.
11. **Length-matched drift check** (D6, check 5): re-run document drift on
    baseline samples matched to later article lengths, to see how much of
    the embedding drift is length.
12. **Production:** scheduled runs with alerts, a shared vector database,
    access control and audit (section 12b).
