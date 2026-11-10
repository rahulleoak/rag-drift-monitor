# Recommendations

Decided by rules in code (`driftmon/policy.py`, D29); explanations written by an LLM from the evidence only, and rejected if they use a number that is not in the evidence.

Summary: 6 recommended, 2 not needed, 1 not checked

## 1. Update or re-index the knowledge base

Re-index the archive on a schedule, with every new batch of articles, rather than only when drift is flagged. Questions about 2018 to 2022 were answered correctly 0-1% of the time from the stale index vs 76-100% from the refreshed one, and 12 of the 18 sections that lose answers were not flagged by drift detection, while no drift measure predicted the size of the loss. Without scheduled re-indexing, questions about new events keep losing almost every answer, and the stale index fails silently by declining rather than warning anyone.

**Applies to (18):** 2019 WORLD NEWS, 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS, 2021 U.S. NEWS, 2022 POLITICS, 2018 POLITICS, 2019 POLITICS, 2022 U.S. NEWS, 2018 SPORTS, and 8 more

**Evidence:**

- *Agent D: answers by period:* Questions about 2018 to 2022: 0-1% answered correctly from the stale index vs 76-100% from the refreshed index (374 questions).
- *Agent D: sections:* 18 of 19 sections with enough questions lose a material share of answers (correct answers drop by 89-100 percentage points), and the refreshed index answers 89-100% of their questions correctly.
- *Agent D: historical control:* Questions about the baseline (2016-17): 100% correct on the stale index and 100% on the refreshed one, so re-indexing does not hurt older questions.
- *Agent D: how the stale index fails:* It fails silently: when nothing relevant is found it declines (99% of those cases); 1 of 374 stale answers added a claim not in the articles.
- *Agent D: which drift matters:* 12 of the 18 sections that lose answers were not flagged by drift detection (for example 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS).
- *Agent D: drift scores vs loss:* No drift measure predicts the size of the loss across sections (Spearman -0.25 to 0.34, all p > 0.05).
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

## 3. Trigger model evaluation

Run an evaluation of a second embedding model on the 2020 WORLD NEWS section, where re-indexing did not help. On the refreshed index, only 15% of answers were correct and an answer appeared in the top 5 for just 15% of questions. If this is not done, journalists will keep getting mostly incorrect answers from this section.

**Applies to (1):** 2020 WORLD NEWS

**Evidence:**

- *Agent D: sections:* 2020 WORLD NEWS: 15% correct and an answer in the top 5 for 15% of questions on the refreshed index.

*Rule:* a section stays poor on the refreshed index, and its top 5 rarely holds an answer although the source article is in the index. *Explanation:* LLM, numbers checked (attempt 1).

## 4. Escalate cases for human review

Send these cases to a person for review: 12 sections lost correct answers by 89-100 percentage points, yet drift detection did not flag them, and the evidence loop for 2020 WORLD NEWS was unclear (loss 15% with the top 5 and 15% with the top 10, interval 0% to 30%). Of the 12, 8 were scored with embedding drift 0.4-2.1x ordinary change and severity low, and 4 (2019 U.S. NEWS, 2021 U.S. NEWS, 2022 U.S. NEWS, 2020 U.S. NEWS) are not in the baseline and could not be scored. Without a human review, these answer losses will keep going unflagged because the drift alerts do not catch them.

**Applies to (13):** 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS, 2022 POLITICS, 2021 U.S. NEWS, 2018 POLITICS, 2019 POLITICS, 2018 WORLD NEWS, 2022 U.S. NEWS, 2021 WORLD NEWS, and 3 more

**Evidence:**

- *Agent D: harm without drift:* In 12 sections correct answers drop by 89-100 percentage points, but drift detection did not flag them: 2020 POLITICS, 2019 U.S. NEWS, 2021 POLITICS, 2022 POLITICS, 2021 U.S. NEWS and others.
- *Agent B: drift scores:* 8 of them were scored: embedding drift 0.4-2.1x ordinary change, severity low.
- *Agent A: data monitoring:* 4 of them could not be scored for drift (not in baseline): 2019 U.S. NEWS, 2021 U.S. NEWS, 2022 U.S. NEWS, 2020 U.S. NEWS. Agent A reports these as new sections.
- *Agent D: evidence loop:* 2020 WORLD NEWS: loss 15% with the top 5 and 15% with the top 10, interval 0% to 30%.

*Rule:* a section loses answers without a drift flag, the evidence loop is unclear, or the judge disagrees with the hand labels. *Explanation:* LLM, numbers checked (attempt 1).

## 5. Improve document chunking or embedding strategies

For long documents such as the Wikipedia set, keep section-aware chunks of about 256 tokens and measure drift per chunk and per document against its own earlier version; chunks of 128 tokens are also worth testing for retrieval. The evidence is that 92% of a typical article's text lies past the 512 tokens the model reads, and in the experiment the first 512 tokens tracked real edits at only 0.51 and one averaged vector at 0.78, versus 0.89 for chunks of 128 tokens and 0.87 for chunks of 256 tokens. If this is not done, most edits to long documents will stay hidden from drift monitoring, and this does not apply to the news archive, where articles are short (median 30 words).

**Applies to (1):** long-form documents (Wikipedia set)

**Evidence:**

- *Wikipedia: long documents:* 92% of a typical Wikipedia article's text lies past the 512 tokens the embedding model reads (median 8541 tokens, 2022).
- *Wikipedia: chunking experiment:* How well each way of reading an article follows its real edits (Spearman): first 512 tokens 0.51, one averaged vector 0.78, chunks of 128 tokens 0.89, chunks of 256 tokens 0.87.
- *Agent A: data monitoring:* News articles are short (median 30 words per article), well within the model's limit.

*Rule:* a typical document is far longer than the embedding model reads, and chunks follow real edits clearly better than the first 512 tokens. *Explanation:* LLM, numbers checked (attempt 1).

## 6. Continue monitoring without intervention

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
| Update retrieval configurations | not needed | Not needed: the refreshed index already puts an answer in the top 5, and no section had answers just outside it. On the refreshed index the top 5 holds an answering article for 76-100% of new questions. 1 sections were borderline, so the loop back to Agent C ran. |
| Improve prompts or retrieval strategies | not needed | Not needed: where the right articles are found, the answers are correct and stay within them. On the refreshed index 76-100% of new questions are answered correctly. 3 of 846 answers added a claim not in the retrieved articles (0.4%). |
| Recommend retraining | not checked | Decide after the model evaluation: retrain only if the second model is also poor. Model evaluation: recommended. |
