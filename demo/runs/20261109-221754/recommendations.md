# Recommendations

Decided by rules in code (`driftmon/policy.py`, D29); explanations written by an LLM from the evidence only, and rejected if they use a number that is not in the evidence.

Summary: 5 recommended, 4 not needed

## 1. Update or re-index the knowledge base

Re-index the archive on a schedule with every new batch of articles, not only when drift is flagged. On questions about 2018 to 2022, the stale index answered 0-1% correctly versus 96-100% after a refresh, and 12 of the 19 sections that lost answers were not flagged by drift detection, with no drift measure predicting the size of the loss. Without scheduled re-indexing, questions about new events will keep failing silently, with the assistant declining to answer in 99% of those cases.

**Applies to (19):** 2020 POLITICS, 2019 WORLD NEWS, 2019 U.S. NEWS, 2020 WORLD NEWS, 2021 POLITICS, 2022 POLITICS, 2021 U.S. NEWS, 2018 POLITICS, 2019 POLITICS, 2022 U.S. NEWS, and 9 more

**Evidence:**

- *Agent D: answers by period:* Questions about 2018 to 2022: 0-1% answered correctly from the stale index vs 96-100% from the refreshed index (374 questions).
- *Agent D: sections:* 19 of 19 sections with enough questions lose a material share of answers (correct answers drop by 89-100 percentage points), and the refreshed index answers 89-100% of their questions correctly.
- *Agent D: historical control:* Questions about the baseline (2016-17): 100% correct on the stale index and 100% on the refreshed one, so re-indexing does not hurt older questions.
- *Agent D: how the stale index fails:* It fails silently: when nothing relevant is found it declines (99% of those cases); 1 of 374 stale answers added a claim not in the articles.
- *Agent D: which drift matters:* 12 of the 19 sections that lose answers were not flagged by drift detection (for example 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS).
- *Agent D: drift scores vs loss:* No drift measure predicts the size of the loss across sections (Spearman -0.19 to 0.41, all p > 0.05).
- *Agent D: example:* "How many children and teachers in Uvalde are being mourned through a series of funerals and burials after the Texas school shooting?" Stale index: "The archive does not say. None of the articles cover the Uvalde school shooting or its funerals; they concern other Texas shootings, such as the 2017 ..." Refreshed index: "The Uvalde community is mourning 19 children and two teachers killed in the shooting, with visitations, funerals and burials taking place over roughly two and ..."

*Rule:* a section loses at least the material share of correct answers on the stale index, and the refreshed index answers it well. *Explanation:* LLM, numbers checked (attempt 1).

## 2. Re-index specific documents (changed facts)

Re-index the 90 articles whose numbers or dates changed and check the flagged sentences, and pair this with a scheduled full re-index. Of the 165 articles checked, 90 had changed facts, questions on flagged edits got a different answer from the old and new version 50% of the time, and the old index still retrieves the edited passage 92% of the time, while 95% of these articles had a "low" drift score. If this is not done, the assistant will keep giving outdated answers with confidence and drift alerts will not catch them.

**Applies to (90):** Germany, Israel, Hong Kong, Delhi, Ukraine, Mumbai, Climate change, Jerusalem, Malaria, Virus, and 80 more

**Evidence:**

- *Wikipedia: changed-facts check:* 391 changed numbers or dates in 90 of 165 articles since the index was built (2018 to 2022).
- *Wikipedia: outdated-answer test:* Questions on flagged edits got a different answer from the old and the new version 50% of the time, vs 2% for edits that kept their numbers (Fisher p = 1.5e-10).
- *Wikipedia: changed facts vs drift scores:* 95% of articles with a changed fact had a "low" drift score, so drift alerts alone would miss them.
- *Wikipedia: outdated-answer test:* The old index still retrieves the edited passage 92% of the time, so it serves the outdated fact with confidence.
- *Hand labels:* Blind hand labels put the check's precision at 80% and 85% (author (first pass), second labeller).
- *Wikipedia: example:* Sumer: "2525 BC on a stele called the Stele of the Vultures." became "2450 BC on a stele called the Stele of the Vultures.".
- *Wikipedia: targeted re-index applied:* Re-indexing only these articles took outdated answers from 13% to 3% (edits that changed a fact: 68% to 0%), re-embedding 64% of the passages a full re-index would.

*Rule:* the changed-facts check flags documents, and the outdated-answer test shows flagged edits change answers. *Explanation:* LLM, numbers checked (attempt 1).

## 3. Escalate cases for human review

Send these 12 sections to a person for review: correct answers dropped by 89-100 percentage points in each, but drift detection did not flag them. 8 were scored with embedding drift of 0.4-2.1x ordinary change and severity low, and 4 (2019 U.S. NEWS, 2021 U.S. NEWS, 2022 U.S. NEWS, 2020 U.S. NEWS) are not in the baseline and could not be scored. If this is not done, these answer losses will keep going unnoticed because drift alerts do not catch them.

**Applies to (12):** 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS, 2022 POLITICS, 2021 U.S. NEWS, 2018 POLITICS, 2019 POLITICS, 2018 WORLD NEWS, 2022 U.S. NEWS, 2021 WORLD NEWS, and 2 more

**Evidence:**

- *Agent D: harm without drift:* In 12 sections correct answers drop by 89-100 percentage points, but drift detection did not flag them: 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS, 2022 POLITICS, 2021 U.S. NEWS and others.
- *Agent B: drift scores:* 8 of them were scored: embedding drift 0.4-2.1x ordinary change, severity low.
- *Agent A: data monitoring:* 4 of them could not be scored for drift (not in baseline): 2019 U.S. NEWS, 2021 U.S. NEWS, 2022 U.S. NEWS, 2020 U.S. NEWS. Agent A reports these as new sections.

*Rule:* a section loses answers without a drift flag, the evidence loop is unclear, or the judge disagrees with the hand labels. *Explanation:* LLM, numbers checked (attempt 1).

## 4. Improve document chunking or embedding strategies

For long documents such as the Wikipedia set, keep section-aware chunks of about 256 tokens and measure drift per chunk and per document against its own earlier version; chunks of 128 tokens are also worth testing for retrieval. The evidence is that 92% of a typical article's text lies past the 512 tokens the model reads, and in the experiment the first 512 tokens tracked real edits at only 0.51 and one averaged vector at 0.78, versus 0.89 for chunks of 128 tokens and 0.87 for chunks of 256 tokens. If this is not done, most edits to long documents will stay hidden from drift monitoring, and this does not apply to the news archive, where articles are short (median 30 words).

**Applies to (1):** long-form documents (Wikipedia set)

**Evidence:**

- *Wikipedia: long documents:* 92% of a typical Wikipedia article's text lies past the 512 tokens the embedding model reads (median 8541 tokens, 2022).
- *Wikipedia: chunking experiment:* How well each way of reading an article follows its real edits (Spearman): first 512 tokens 0.51, one averaged vector 0.78, chunks of 128 tokens 0.89, chunks of 256 tokens 0.87.
- *Agent A: data monitoring:* News articles are short (median 30 words per article), well within the model's limit.

*Rule:* a typical document is far longer than the embedding model reads, and chunks follow real edits clearly better than the first 512 tokens. *Explanation:* LLM, numbers checked (attempt 1).

## 5. Continue monitoring without intervention

Keep monitoring without intervention for questions about 2016-17, because answers there are unaffected: they were 100% correct before and 100% after re-indexing, even though newer articles make up 14% of the top 10 by 2022. Keep comparing each new batch with the baseline, not only with the previous year, since 2022 vs 2021 drift is 0.5x (low) while 2022 vs 2016-17 is 2.1x (medium). If that baseline comparison is skipped, gradual drift goes unnoticed.

**Applies to (1):** questions about the baseline (2016-17)

**Evidence:**

- *Agent D: historical control:* Questions about 2016-17: 100% correct before and 100% after re-indexing.
- *Agent C: ranking change:* By 2022, newer articles make up 14% of the top 10 for those questions.
- *Agent B: drift, whole periods:* 2022 vs 2021 is 0.5x ordinary change in embedding drift (low), but 2022 vs 2016-17 is 2.1x (medium).

*Rule:* drift is detected but answers do not suffer. *Explanation:* LLM, numbers checked (attempt 1).

## Checked, not recommended

| Action | Status | Why |
|---|---|---|
| Update retrieval configurations | not needed | Not needed: the refreshed index already puts an answer in the top 5, and no section had answers just outside it. On the refreshed index the top 5 holds an answering article for 96-100% of new questions. 0 sections were borderline, so the loop back to Agent C did not run. |
| Improve prompts or retrieval strategies | not needed | Not needed: where the right articles are found, the answers are correct and stay within them. On the refreshed index 96-100% of new questions are answered correctly. 4 of 846 answers added a claim not in the retrieved articles (0.5%). |
| Trigger model evaluation | not needed | Not needed: re-indexing fixes every section, so the embedding model handles the new content. The refreshed index answers 89-100% of questions correctly in every one of the 19 sections with enough questions (threshold 60%). |
| Recommend retraining | not needed | Not needed: model evaluation was not triggered, and retraining only follows a failed evaluation. Model evaluation: not needed. |
