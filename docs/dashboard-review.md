# Dashboard review: changes to make (temporary)

Working list from the dashboard walkthrough (started 2026-11-09). Each item is
ticked when done; the file is deleted once the list is finished.

**Status 2026-11-09:** all items done in the dashboard rewrite (commits d8e537c,
f90b28a) except re-recording the GIF. "Continue monitoring" is clarified by its new
explanation text, which now opens with what it covers (questions about the
baseline, unaffected).

## Whole dashboard

- [x] **Requirement: user experience.** The dashboard should look pleasing and
      be meaningful: every element earns its place by telling the viewer
      something, in plain words, with a consistent visual style. Applies to
      every item below.
- [x] **Remove the link icon next to every heading.** Streamlit adds an anchor
      link to headings for sharing a link to that section; in a single page
      with tabs it does nothing useful (a link to a heading in another tab does
      not open that tab), so it reads as a broken copy button. Turn anchors
      off on every heading.
- [x] **Rule: no results typed into the dashboard.** "How to read" text is
      fixed (true for any run); "what this run shows" text is built by code from
      the run's numbers, with wording that depends on the result, and is
      tested. Today one caption hard-codes a result ("2016–17: both indexes
      answer 100%") and some captions hard-code the baseline years ("2016–17")
      instead of reading them from the run.
- [x] **Show the models the run actually used.** Each run saves
      `run_info.json` (models per job, date, cost, question set); the dashboard
      reads model names from it (falling back to the config for older runs)
      instead of the hard-coded "Sonnet answers…; Opus grades…". Also gives
      every run its provenance. Re-export the demo run afterwards.
- [x] **Say what each tab shows.** A short "what this shows" line at the top of
      every tab. A first-time viewer could not tell what they were looking at.
- [x] **Say how to read each chart.** A one-line caption under each chart: what
      the axes and colours mean, and what to conclude.
- [x] **Plain names, not jargon or code names.** Column headers such as
      `times_ordinary_embedding`, `emerging_terms`; tooltip fields such as
      `lexical_band`, `embedding_band`, `kind`.
- [x] **No internal references.** The Retrieval tab caption cites "(F11)", a
      finding number that means nothing to a viewer.
- [x] **Hide Streamlit's "Deploy" button** (one setting).
- [x] **Re-record the README GIF** once the above is done.

## Overview

- [x] **State the main finding under the numbers:** drift detection shows where
      things changed, but the harm is everywhere, including sections that were
      never flagged.
- [x] **"Control segments"** → e.g. "Unchanged groups (for comparison)".
- [x] **"Next step: full evaluation"** is the pipeline's routing decision; say so
      (e.g. "Drift found → ran the full answer check"). It is also cut off
      ("full evalua…") on narrower screens.
- [x] **Line charts:** explain "× ordinary change" (1.0 = normal change inside
      2016–17), orange vs blue, and the medium/high lines; point out the 2020
      peak and that year-on-year (blue) hides how far the archive has moved
      (orange).
- [x] **"Most drifted segments" table:** readable column names; explain
      severity and kind (meaning / wording / both).

## Data monitoring (Agent A)

- [x] **State the takeaways at the top:** far fewer articles per year after
      2018 (a property of the dataset, not drift); the section mix reshuffles
      heavily (Cramér's V 0.2 → 0.5); up to a quarter of a year's articles are in
      sections new since the baseline; new vocabulary peaks in 2020; only 37 of
      134 year × section groups are big enough to analyse.
- [x] **Add a small chart** (articles per year, share in new sections) instead
      of leading with a wide table.
- [x] **Top table is too wide:** columns are cut off (mix_chi2, mix_p_value,
      mix_dof). Drop or hide the chi-square columns (always significant at
      this size) and keep Cramér's V, labelled as "how much the section mix
      changed (0–1)".
- [x] **Readable column names** in every table (documents → Articles,
      new_section_share → Share in new sections, new_term_rate → New words,
      n_period / n_baseline → Articles this year / in baseline, ...).
- [x] **Data quality:** one line saying 98.2% kept (77,840 of 79,244), and why
      the rest were removed.
- [x] **Renamed sections:** one line explaining why they are merged (a rename
      should not look like drift).
- [x] **Segments table (134 rows):** summarise in one line (37 analysable; 81
      too few articles; 16 not in the baseline, so drift cannot be scored) and
      put the full table in a collapsible box.
- [x] **Query stream:** replace "random and oversample pools" with plain words;
      explain the unanswerable share (questions built to test whether the
      assistant admits it does not know).

## Drift (Agent B)

- [x] **Intro line:** two techniques, meaning (embeddings) and vocabulary
      (word counts), both as "× ordinary change"; plus a pointer that these
      scores do not predict where answers break (see Impact).
- [x] **Bug: heatmap legend not visible.** Severity colours (dark brown = low,
      orange = medium, light peach = high) have no key on screen; lighter =
      more severe is not obvious on a dark background.
- [x] **Bug: "kind" labels on flagged cells do not render** (text layer of the
      heatmap shows nothing).
- [x] **Heatmap caption:** empty cells = too few articles to analyse; 2020 hits
      lifestyle sections (Wellness, Parenting, Style & Beauty, Food & Drink),
      while Politics and Entertainment stay low.
- [x] **Drill-down:** explain "Novel articles" (unlike anything in the
      baseline; low means familiar content recombined) and the segment ID
      format ("2020|FOOD & DRINK|vs 2016-17" → readable label).
- [x] **Charts hidden behind "Table" boxes without captions:** add a
      one-line takeaway to each (emerging topics grow rather than appear; words
      keep their spelling but change meaning, e.g. "test": nuclear → positive;
      the correlation grid shows the two meaning measures are redundant (0.93)
      while the word measure adds information (0.56–0.68)).
- [x] **Correlation grid** needs a "how to read" line (1 = always agree,
      0 = unrelated) or could move to a collapsible "methods" box.
- [x] **Query drift table:** readable columns (drop p-values and raw MMD into a
      collapsible box; keep "× ordinary" and new words); keep the caveat that
      the questions are generated from the articles.

## Retrieval (Agent C)

- [x] **Intro line:** this tests only the search step (does the top 5 contain
      an answer?), judged by Opus without knowing which index found the
      article.
- [x] **Top tiles are redundant:** "Questions 423" and "Judged questions 423"
      repeat; "Mode: full evaluation" repeats the Overview. Replace with
      something informative (e.g. answer found: stale vs refreshed).
- [x] **Explain 423 here vs 438 on Impact** (the unanswerable questions are
      only used in the answer step).
- [x] **Dumbbell caption:** stale 0–1% vs refreshed 96–100% for 2018–22; the
      2016–17 control is 100% on both (dots overlap), so re-indexing does not
      hurt old questions.
- [x] **(b) Known item:** replace "weighted down: F11" with plain words (the
      question was written from that article, so finding it is easy; kept as
      a sanity check only). Readable columns (reciprocal_rank → "Rank of the
      source article (1 = first)").
- [x] **(c) Old questions:** caption (overlap 93% → 86%; newer articles 7% →
      14% of old questions' top 10; the source article is never pushed out).
      Readable columns.
- [x] **Tables (b) and (c) are cut off** on the right in two narrow columns;
      stack them or trim columns.
- [x] **Bring the best example forward:** "Questions that lost most" is the
      most convincing part (e.g. the quinoa → fonio question) but is
      collapsed; show one or two examples open.

## Impact (Agent D)

- [x] **Intro line:** the full assistant runs (writer answers from each index's
      top 5; judge grades correct / grounded / declined, blind to the index).
      This is where "which drift matters?" is answered.
- [x] **Tiles:** "Material loss threshold 10%" is a setting, not a result
      (move to a caption: at least 10 points and the 95% interval excludes
      zero, fixed before results). "Unanswerable: declined (stale) 100%" is
      expected; show the refreshed figure (93%, 14 of 15, none invented),
      which is the meaningful one. Say 438 = 423 + 15 unanswerable.
- [x] **Dumbbell caption is the hard-coded result** ("both indexes answer
      100%"): build it from the run. Keep the silent-failure point (declines
      rather than invents; 1 of 374 added an outside claim).
- [x] **Scatter is the key chart; explain it:** if drift predicted harm, dots
      would rise left to right; instead every section sits at 89–100% loss.
      Orange = flagged and harmed (7); blue = harmed but never flagged (12: 8
      scored low, 4 in a section new since the baseline, listed below).
      Takeaway: drift scores show where content changed, not where answers
      break → re-index on a schedule.
- [x] **Scatter legend shows empty classes** ("drift that does not hurt",
      "healthy" have no sections); show only classes present, or say "none".
- [x] **Scatter x-axis says "× an ordinary year";** use "× ordinary change"
      (D30 wording).
- [x] **Correlation table:** one-line reading (all weak, −0.19 to 0.41, none
      significant: drift does not predict the size of the loss). Readable
      measure names.
- [x] **README says Spearman up to 0.40; the data says 0.41.** Fix the README.
- [x] **Evidence loop caption:** say why "did not run" is good news (every
      result was clear-cut) and what it would have done (top 10 instead of
      top 5, answered again).
- [x] **Examples box collapsed:** show one example open.
- [x] **Wikipedia section:** lead with what it means (a changed number or date
      changed the answer 50% of the time vs 2% for rewordings), and surface
      the hidden point now only in a tooltip: the old index still retrieved
      the edited passage 92% of the time, so it serves the outdated fact
      confidently, and 95% of those articles had "low" drift scores. Replace
      the statistics-report caption ("Difference 48% (needed 20%)…").
- [x] **Tab order / flow:** the most important tab is fifth of six; consider
      surfacing its two findings on the Overview (see Overview items).

## Recommendations (Agent E)

- [x] **Intro: how decisions are made,** in plain words: nine actions from the
      brief; a rule in code decides each; the LLM only explains, and an
      explanation with a number not in the evidence is rejected.
- [x] **Explain "Not checked"** (actions that could not be evaluated because
      an input was missing; 0 = every action was tested).
- [x] **Explain the order:** measured answer loss first, then cases for a
      person, then maintenance.
- [x] **"Continue monitoring" is confusing:** say plainly that old (baseline)
      questions are fine (100% on both indexes, though newer articles are 14%
      of their results), so no action, plus the lesson to compare with the
      baseline, not only the previous year (0.5× vs 2.1×).
- [x] **Evidence labels use internal document numbers** ("D27: changed-facts
      check", "F19: hand labels", "F16: chunking experiment", "F21: example");
      use plain source names (e.g. "Wikipedia: changed-facts check", "Hand
      labels"). Source: `driftmon/policy.py` evidence sources.
- [x] **Footer jargon:** "Explanation: LLM, numbers checked (attempt 1)" →
      e.g. "Explanation written by Claude Sonnet 5.5; every number checked
      against the evidence (passed first time)". Model name from run_info.
- [x] **"Checked, not recommended" reads awkwardly:** "(not needed): Not
      needed: …" repeats, and evidence sentences run together. One clean
      sentence each, with the deciding number.

## Round 2: the author's final comments (2026-11-09)

- [x] Title and page names in title case ("RAG Drift Monitor", "Data Monitoring").
- [x] Pages in the left bar; run facts in a top bar of chips; the run picker
      at the top right, level with the title.
- [x] Less text: "how to read" moved into each heading's ⓘ icon; this run's
      finding stated under the heading; numbers bold white; "stale" and
      "refreshed" carry their chart colour as a dot.
- [x] Lists of article titles as bullets (stale and refreshed side by side).
- [x] Subtitle: "Tracks how a document archive drifts over time, and whether
      that drift breaks the answers of a RAG (retrieval-augmented generation)
      assistant." Archive and years moved to the top bar.
- [x] Question headings replaced by short headings plus a statement.
- [x] Captions rewritten as full sentences, larger and lighter.
- [x] Hover (ⓘ) details on headings, tiles and table columns.
- [x] Live charts instead of images (topics, words that changed meaning,
      method check).
- [x] Section mix as labelled lines per section (validated palette, new
      sections dashed) instead of a squeezed stacked bar.
- [x] Stale vs refreshed as grouped bars (Retrieval, Impact).
- [x] Drift-vs-loss scatter zoomed to the data.
- [x] Evidence grouped by source, without double colons.
