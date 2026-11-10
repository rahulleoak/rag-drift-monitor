# Presentation brief: RAG Drift Monitor

Single source for the slides. Upload this file and the images in this folder
(`00_architecture.png` to `18_applied_fix.png`). Every number below comes
from the committed demo run `20261109-221754` and the long-form results saved
with it; nothing else in `docs/` should be used for numbers (older notes
contain superseded figures). Updated 2026-11-10: chunking and the applied fix
added as slides, slides 6 and 7 merged, new backup questions, numbers updated.

## Instructions for the slide tool

- **Theme:** dark. Background #0D1725 (deep navy), text #F3F1EB (off-white),
  secondary text #ACACA8, accent #D9581E (burnt orange) for emphasis,
  blue #6383E0 for "before / stale / baseline". The images already use this
  palette.
- **One message per slide.** The slide title states the message as a
  sentence; at most 3 short bullets; one image where listed.
- **Use numbers exactly as written here.** Do not round differently, do not
  add numbers, do not invent examples.
- **Images:** place each image at a readable size (it is a screenshot of the
  real dashboard); do not crop away its heading.
- **Speaker notes:** copy the "Say" text into the notes, not onto the slide.
- **Length:** 14 main slides plus backup slides. Target about 17 minutes, then a
  3 minute live demo.

## The story in one paragraph

A RAG assistant answers questions from an archive whose content keeps
changing. Drift detectors can tell you *that* the content changed and *where*.
This project asks the question that matters to whoever runs the assistant:
**which drift actually breaks the answers?** Five agents check each new batch,
measure drift two ways, test search and answers on a stale and a refreshed
index, and recommend actions with their evidence. The main finding: without
re-indexing the assistant answers almost nothing about new events, and it
fails silently; but drift scores do not predict where that happens, so the
right policy is to re-index on a schedule, not on drift alerts.

## Timing

| # | Slide | Minutes |
|---|---|---|
| 1 | Title | 0.5 |
| 2 | The question | 1 |
| 3 | The setting | 1 |
| 4 | How it works | 1.5 |
| 5 | The key experiment | 1 |
| 6 | What drifted, and where | 1.5 |
| 7 | Without re-indexing, answers fail, silently | 1.5 |
| 8 | Drift scores do not predict where answers break | 1.5 |
| 9 | Long documents: chunking | 1 |
| 10 | Changed facts: the subtle failure | 1.5 |
| 11 | We applied the fix | 1.5 |
| 12 | What to do, with evidence | 1.5 |
| 13 | Can the judges be trusted? | 1 |
| 14 | Design decisions, limits, next steps | 1.5 |
| | Live demo | 3 |
| | **Total** | **about 20** (if running long, say slide 9 in one sentence and move on) |

---

## Slide 1: Title

- **Title:** RAG Drift Monitor
- **Subtitle:** Which drift actually breaks a RAG assistant's answers?
- **Small text:** Rahul · Take-home for Rivvun AI · November 2026
- **Image:** none (or a faded `01_overview.png` as background)
- **Say:** "A RAG assistant's archive keeps changing. Most drift monitoring
  tells you that something changed. I built a system that tells you which
  changes actually break the answers, and what to do about it."

## Slide 2: The question

- **Title:** Drift detectors say *what* changed. The operator needs to know *what broke*.
- **Bullets:**
  - Content drifts: new events, new sections, words that change meaning.
  - Not all drift matters; alerts on every shift are noise.
  - The question: which drift hurts retrieval and answers, and what should be done?
- **Image:** none
- **Say:** "Every comparison in this data is statistically significant: with
  thousands of articles, any difference is detectable. So significance alone
  says nothing. What matters is whether answers get worse."

## Slide 3: The setting

- **Title:** A newsroom archive assistant, replayed year by year
- **Bullets:**
  - HuffPost news archive: 77,840 articles after cleaning (of 79,244), 2016–2022.
  - Baseline 2016–17; each later year (2018–2022) arrives as a new batch.
  - Plus 165 long Wikipedia articles captured at each year end: long, edited documents where chunking matters.
- **Image:** none
- **Say:** "Journalists ask questions; the assistant answers from the
  archive. Replaying years gives real, dated drift with a known event, COVID
  in 2020. Wikipedia covers what news items cannot: long documents that are
  edited rather than replaced. There is no real query log, so an LLM wrote
  test questions from sampled articles, reviewed twice."

## Slide 4: How it works

- **Title:** Five agents, each building on the last
- **Image:** `00_architecture.png`
- **Bullets (under the image, short):**
  - A: what came in · B: what changed · C: does search still work · D: do answers suffer · E: what to do
  - LangGraph; routing decided in code (full check vs quick check; one evidence loop)
  - The LLM writes and judges; rules in code decide
- **Say:** "Each agent reads the reports of the ones before it, so they
  genuinely depend on each other. Routing is decided in code: if drift is
  found, run the full answer check; if a section's result is borderline, go
  back for more evidence once. Claude Sonnet 5.5 writes questions, answers and
  explanations; Claude Opus 5.5 judges. But no LLM decides what to recommend:
  rules in tested Python do, and every explanation is checked number by
  number against the evidence."

## Slide 5: The key experiment

- **Title:** Same questions, two indexes: the difference is the cost of not keeping up
- **Bullets:**
  - Stale index: baseline articles only (the assistant if nobody re-indexed).
  - Refreshed index: rebuilt with each year's articles.
  - 2016–17 questions are the historical control: they should score the same on both.
- **Image:** none (or a simple two-box graphic: blue "stale", orange "refreshed")
- **Say:** "This turns 'drift exists' into 'drift costs this much', and the
  control tells us re-indexing does not break old questions."

## Slide 6: What drifted, and where

- **Title:** 2020 moved furthest, and COVID reached sections you would not expect
- **Image:** `02_drift_per_year.png`
- **Bullets:**
  - 2020: 2.5× ordinary change from the baseline (high); 6 of 7 analysable sections flagged.
  - 2022 looks ordinary next to 2021 (0.5×) but has moved 2.1× from the baseline.
  - Questions drift too: their vocabulary moved 3–6× ordinary from 2018 on (coronavirus 2020; capitol, covid 2021).
- **Say:** "Drift is measured two ways, meaning (embeddings) and vocabulary,
  against the archive's own ordinary change inside the baseline, the same
  month a year apart. 1.0 is normal. A monitor that only compares with last
  year would say 2022 is fine; against the baseline it has clearly moved.
  COVID reached sections you would not expect: Food & Drink moved 17×
  ordinary in 2020, as delivery, pantry and coronavirus came in and summer,
  wine and meal faded. The users' questions drift as well, which the brief
  asks us to monitor."

## Slide 7: Without re-indexing, answers fail, silently

- **Title:** Without re-indexing, the assistant answers 0.3% of questions about new events
- **Image:** `11_correct_answers.png`
- **Bullets:**
  - 374 questions about 2018–2022: 0.3% correct on the stale index, 99% after re-indexing.
  - The control (2016–17 questions): 100% on both: re-indexing does not hurt old questions.
  - It fails silently: it declines (99% of cases) rather than inventing; 1 of 374 answers added an outside claim.
- **Say:** "Search tells the same story: an answer is in the top 5 for 0–1%
  of new questions on the stale index, 96–100% after re-indexing. And the
  failure is silent: the user gets 'the archive does not cover this', not a
  wrong answer that someone would notice."

## Slide 8: Drift scores do not predict where answers break

- **Title:** Every section lost answers, and drift detection flagged only 7 of 19
- **Image:** `12_drift_vs_loss.png`
- **Bullets:**
  - All 19 sections with enough questions lost 89–100% of their correct answers.
  - 12 were never flagged: 8 scored "low", 4 are in a section new since the baseline that drift cannot score.
  - Drift measures vs size of loss: rank correlation −0.19 to 0.41, none significant.
- **Say:** "This is the main finding. If drift predicted harm, the dots
  would rise from left to right. They are all at the top. Politics in 2020
  reads like politics in 2017 to an embedding, but every fact is new. So
  drift scores tell you where content changed, not where answers break: the
  policy has to be re-index on a schedule, not on alerts."

## Slide 9: Long documents: chunking

- **Title:** A whole-article embedding sees only the start of a long document
- **Image:** `17_chunking.png`
- **Bullets:**
  - 165 Wikipedia articles, median about 8,000 tokens; the model reads 512, so 91% of the text is never seen.
  - Following real edits: first 512 tokens 0.51 · averaged 0.78 · 256-token chunks 0.87.
  - Edited documents need each version compared with its own earlier version.
- **Say:** "The stakeholder asked for long documents, and this is why
  chunking is one of the recommendations: without it, drift detection is
  blind to most of the text. 128-token chunks did slightly better, 0.89, but
  need twice the vectors; Agent E suggests testing them for retrieval."

## Slide 10: Changed facts: the subtle failure

- **Title:** A changed number changes the answer half the time, and drift scores miss it
- **Image:** `14_edited_documents.png`
- **Bullets:**
  - Edits that replaced a number or date: the answer changed 50% of the time; rewordings 2% (Fisher p = 1.5e-10).
  - The old index still finds the edited passage 92% of the time, so it serves the outdated fact with confidence.
  - 95% of articles with a changed fact had a "low" drift score; a changed-facts check catches them (391 changes in 90 of 165 articles).
- **Say:** "On long documents, embeddings barely move when only a number
  changes, even with chunks. The reliable signal is a text check that
  matches each edited sentence to its earlier version and looks for a
  replaced number or date."

## Slide 11: We applied the fix

- **Title:** We applied the fix: changed-fact answers went from 68% outdated to 0%
- **Image:** `18_applied_fix.png`
- **Bullets:**
  - Re-indexing only the 90 flagged articles: outdated answers 12.9% → 3.0%, for 64% of the re-embedding.
  - It reached 77% of a full re-index's gain, short of the 80% we set in advance.
  - Every miss was "can't answer", never a wrong fact: detail added to articles it did not list.
- **Say:** "This is the brief's apply-a-fix-and-re-check. We copied the
  index, applied Agent E's recommendation, and asked 140 questions about
  edited sentences. It missed our own bar, and we report that. So the
  recommendation became: re-index the flagged documents now, and keep a
  scheduled full re-index."

## Slide 12: What to do, with evidence

- **Title:** Five actions recommended, four ruled out, each with its numbers
- **Image:** `15_recommendations.png`
- **Bullets:**
  1. Re-index on a schedule, not only on drift alerts.
  2. Re-index the 90 articles with changed facts now; full re-index on a schedule.
  3. Send the 12 unflagged sections that lost answers to a person.
  4. Section-aware chunking for long documents.
  5. Keep monitoring old questions; compare with the baseline, not just last year.
- **Small text:** Ruled out: retrieval settings, prompts, model evaluation, retraining (`16_not_recommended.png` as backup).
- **Say:** "Nine actions from the brief, one rule each. Sonnet explains each
  recommendation, and code rejects any explanation with a number that is not
  in the evidence; all five passed first time. The ruled-out actions matter
  too: re-indexing alone fixes every section, so retraining is not needed."

## Slide 13: Can the judges be trusted?

- **Title:** The LLM judges agree with people, and are stricter when they differ
- **Bullets (or a small table):**
  - Relevance judge vs two people: 14/15 and 11/15 (people vs each other: 12/15).
  - Changed-facts check, precision vs two people: 17/20 and 16/20 (people: 17/20).
  - Answer judge vs one person: 11/15.
- **Small text:** Blind hand labels, first pass. 196 automated tests. All LLM calls cost $17.20 in total (5,556 calls, cached); a full rerun costs nothing and takes about 2 minutes.
- **Say:** "Where the judge and a person disagree, the judge is almost always
  the stricter one, so the losses are not inflated."

## Slide 14: Design decisions, limits, next steps

- **Title:** Built to be trusted, honest about its limits
- **Bullets (decisions):**
  - The LLM writes; code decides (repeatable, testable, auditable).
  - Severity against the archive's own ordinary change, not p-values.
  - Stale vs refreshed on the same questions: harm measured, not assumed.
- **Bullets (limits → next):**
  - Questions are simulated → connect a real query log.
  - The changed-facts check sees replaced numbers and dates only → add names and added numbers.
  - Wording vs meaning untested for keyword search → add BM25 next to the dense retriever.
  - Small hand-label samples → grow them.
- **Say:** "If I had another week, the first thing would be a real query log:
  it is the one input that is simulated here."

## Live demo (3 minutes)

Run `streamlit run app/streamlit_app.py` before the call; it reads the saved
run, so it needs no key or network. Click path:

1. **Overview:** read the three main findings in the box; point at the top bar (archive, years, models, cost).
2. **Impact:** the grouped bars (hover the 2020 refreshed bar: 96% vs 0%), then the scatter (hover 2018 Queer Voices: drift "low", lost 100%).
3. **Drift:** the heatmap; then "One group in detail" on 2020 Food & Drink (words before and after: coronavirus 0 → 24 per 1,000 words).
4. **Recommendations:** open the evidence under action 1.

Fallback if anything fails: the README GIF, and the screenshots in these slides.

---

## Backup slides: likely questions

| Question | Short answer |
|---|---|
| Isn't it obvious a stale index can't answer new questions? | Yes, and that is the control for the method. The non-obvious parts: drift scores did not predict which sections broke (12 of 19 never flagged), old questions were unaffected, the failure is silent, and changed facts slip past drift scores. |
| Why not an off-the-shelf tool such as Evidently? | They flag drift column by column with fixed statistical thresholds. They do not test whether drift hurts answers, which is the core here. Some of their views inspired dashboard charts. |
| Why simulated questions? | No real query log exists for this archive. Questions were written from sampled articles, reviewed twice by a different model, and spot-checked by hand. Connecting a real log is the first next step. |
| Why is the LLM not deciding? | Decisions must be repeatable and testable. The LLM writes and judges; rules in code decide; explanations are rejected if a number is not in the evidence. |
| Why 2016–17 as the baseline? | Closest topic mix to the later years among the candidate windows (checked before choosing). |
| What does "× ordinary change" mean? | Drift measured against change inside the baseline itself (the same month a year apart); 1.0 is normal, 2.5 is two and a half times that. At this size every comparison is "significant", so the multiple is what separates drift from noise. |
| An early summary said 2020 was 5.7×; now it is 2.5×? | Same measurement, different yardstick: the early figure divided by one comparison (2017 vs 2016); the pipeline divides by the median of 12 same-month pairs inside the baseline. Rankings and conclusions are identical. |
| How much did it cost? | $17.20 for the whole project (5,556 LLM calls, Batch API where possible). Reruns hit the cache and cost nothing. |
| Would it work on our data? | The dashboard reads any run; the pipeline needs a loader for the document format, dates and sections, and a question set. |
| What if users ask in today's words, COVID instead of coronavirus? | Tested: only 5% of questions (18 of 370) had a natural newer wording; the article was still found for 17 of 18, sometimes a place lower. Names carry search, and names rarely change in five years. |
| Isn't some of the drift just article length? | Checked with length-matched samples: about a tenth (11%); no severity flag changed. |
| Why did your own fix miss its test? | It fixed changed facts completely (68% → 0%); it missed detail added to articles it did not list, reaching 77% against an 80% bar set before the run. Hence "targeted now, full on a schedule". |
| How does this fit contracts and policies? | Superseded policies and renamed terms are the contract version of "coronavirus → covid"; changed facts are "which contract version is current". Monitoring, evidence and the stale-vs-refreshed check carry over unchanged. |

## Backup image slides

One image per slide, title as given:

| # | Title | Image |
|---|---|---|
| B3 | Four actions checked and ruled out, each with its reason | `16_not_recommended.png` |
| B4 | Search tells the same story: an answer in the top 5 for 0–1% stale, 96–100% refreshed | `09_retrieval_bars.png` |
| B5 | The stale index declines; the refreshed index answers | `13_answer_example.png` |
| B6 | Where it drifted: 2020 flagged 6 of 7 analysable sections (86%); other years 24–40% | `04_drift_heatmap.png` |
| B7 | Most new topics existed before and grew: 2022 Ukraine/Russia, 135 articles against 8 | `07_topics.png` |
| B8 | Everyday words changed meaning: "home" in 2020 moved 4.6× ordinary change | `08_words_meaning.png` |
| B9 | 2020 Food & Drink: meaning 17.3×, words 7.1× | `06_one_group_words.png` |
| B10 | The dashboard overview: the run at a glance | `01_overview.png` |

## Numbers sheet (the only figures to use)

| What | Value |
|---|---|
| Articles after cleaning | 77,840 of 79,244 (98.2%) |
| Baseline / later years | 2016–17 / 2018, 2019, 2020, 2021, 2022 |
| Questions | 423 tested in search; 438 answered (423 + 15 built to be unanswerable); 374 about 2018–2022 |
| Correct answers, new events | 0.3% stale, 99% refreshed |
| Answer in top 5, new events | 0–1% stale, 96–100% refreshed |
| Control (2016–17 questions) | 100% on both indexes |
| Silent failure | declines 99% of cases; 1 of 374 added an outside claim |
| Unanswerable questions | refreshed index declined 93% (14 of 15), invented none |
| Sections lost answers | 19 of 19 (89–100% of correct answers) |
| Never flagged | 12 (8 scored low, 4 new section) |
| Drift vs loss correlation | −0.19 to 0.41, none significant (15 scored sections) |
| Biggest drift | 2020: 2.5× ordinary change (high) |
| Gradual drift | 2022: 0.5× vs 2021, 2.1× vs baseline |
| Share of sections flagged | 2020: 6 of 7 (86%); other years 24–40% |
| Most drifted group | 2020 Food & Drink: meaning 17.3×, words 7.1× |
| Largest topic | 2022 Ukraine/Russia: 135 vs 8 articles |
| Word that changed meaning most | "home" 2020: 4.6× ordinary |
| Old questions after re-indexing | top-10 overlap 93% → 86%; newer articles 7% → 14%; source article never pushed out |
| Changed facts | 391 changes in 90 of 165 Wikipedia articles |
| Answer changed | 50% (number/date edits) vs 2% (rewordings); Fisher p = 1.5e-10 |
| Old index finds edited passage | 92% |
| Changed-fact articles with "low" drift | 95% |
| Chunking (rank correlation with real edits) | first 512 tokens 0.51 · averaged vector 0.78 · 256-token chunks 0.87 · 128-token chunks 0.89 |
| Long articles | median about 8,000 tokens; 91% of the text past the model's 512 |
| Applied fix (targeted re-index) | outdated answers 12.9% → 3.0%; changed facts 68% → 0%; 77% of a full re-index's gain (bar 80%); 64% of the re-embedding; 140 edited sentences |
| Newer wording | 18 of 370 questions (5%) had one; article found for 17 of 18 |
| Length check | about 11% of embedding drift is length; no severity band changes |
| Query drift | question vocabulary 3–6× ordinary from 2018 (rough scale: one comparison, 2017 vs 2016 questions) |
| Recommendations | 5 recommended, 4 ruled out; 5 of 5 explanations passed the number check first time |
| Judges vs people | relevance 14/15 and 11/15 (people 12/15); changed facts 17/20 and 16/20 (people 17/20); answers 11/15 |
| LLM cost | $17.20 total, 5,556 calls; reruns $0 |
| Tests | 196 |

## Do not use (superseded)

- 2020 at "5.7×", 2022 at "1.2× / 4.7×" (early yardstick; use 2.5× / 0.5× / 2.1×).
- Correlation "up to 0.40" (it is 0.41).
- "$15.60 / 4,558 calls" and "175 tests" (now $17.20 / 5,556 and 196).
- "86%" for the applied fix (exploratory, after removing noise; the result is 77%).
- Slide numbers from the first draft: old slides 6 and 7 are merged; chunking and the applied fix are new slides 9 and 11.
- Any "tab" names (the dashboard now uses pages: Overview, Data Monitoring, Drift, Retrieval, Impact, Recommendations).
