# Dashboard review: changes to make (temporary)

Working list from the dashboard walkthrough (started 2026-11-09). Each item is
ticked when done; the file is deleted once the list is finished.

## Whole dashboard

- [ ] **Rule: no results typed into the dashboard.** "How to read" text is
      fixed (true for any run); "what this run shows" text is built by code from
      the run's numbers, with wording that depends on the result, and is
      tested. Today one caption hard-codes a result ("2016–17: both indexes
      answer 100%") and some captions hard-code the baseline years ("2016–17")
      instead of reading them from the run.
- [ ] **Show the models the run actually used.** Each run saves
      `run_info.json` (models per job, date, cost, question set); the dashboard
      reads model names from it (falling back to the config for older runs)
      instead of the hard-coded "Sonnet answers…; Opus grades…". Also gives
      every run its provenance. Re-export the demo run afterwards.
- [ ] **Say what each tab shows.** A short "what this shows" line at the top of
      every tab. A first-time viewer could not tell what they were looking at.
- [ ] **Say how to read each chart.** A one-line caption under each chart: what
      the axes and colours mean, and what to conclude.
- [ ] **Plain names, not jargon or code names.** Column headers such as
      `times_ordinary_embedding`, `emerging_terms`; tooltip fields such as
      `lexical_band`, `embedding_band`, `kind`.
- [ ] **No internal references.** The Retrieval tab caption cites "(F11)", a
      finding number that means nothing to a viewer.
- [ ] **Hide Streamlit's "Deploy" button** (one setting).
- [ ] **Re-record the README GIF** once the above is done.

## Overview

- [ ] **State the main finding under the numbers:** drift detection shows where
      things changed, but the harm is everywhere, including sections that were
      never flagged.
- [ ] **"Control segments"** → e.g. "Unchanged groups (for comparison)".
- [ ] **"Next step: full evaluation"** is the pipeline's routing decision; say so
      (e.g. "Drift found → ran the full answer check"). It is also cut off
      ("full evalua…") on narrower screens.
- [ ] **Line charts:** explain "× ordinary change" (1.0 = normal change inside
      2016–17), orange vs blue, and the medium/high lines; point out the 2020
      peak and that year-on-year (blue) hides how far the archive has moved
      (orange).
- [ ] **"Most drifted segments" table:** readable column names; explain
      severity and kind (meaning / wording / both).

## Data monitoring (Agent A)

- [ ] **State the takeaways at the top:** far fewer articles per year after
      2018 (a property of the dataset, not drift); the section mix reshuffles
      heavily (Cramér's V 0.2 → 0.5); up to a quarter of a year's articles are in
      sections new since the baseline; new vocabulary peaks in 2020; only 37 of
      134 year × section groups are big enough to analyse.
- [ ] **Add a small chart** (articles per year, share in new sections) instead
      of leading with a wide table.
- [ ] **Top table is too wide:** columns are cut off (mix_chi2, mix_p_value,
      mix_dof). Drop or hide the chi-square columns (always significant at
      this size) and keep Cramér's V, labelled as "how much the section mix
      changed (0–1)".
- [ ] **Readable column names** in every table (documents → Articles,
      new_section_share → Share in new sections, new_term_rate → New words,
      n_period / n_baseline → Articles this year / in baseline, ...).
- [ ] **Data quality:** one line saying 98.2% kept (77,840 of 79,244), and why
      the rest were removed.
- [ ] **Renamed sections:** one line explaining why they are merged (a rename
      should not look like drift).
- [ ] **Segments table (134 rows):** summarise in one line (37 analysable; 81
      too few articles; 16 not in the baseline, so drift cannot be scored) and
      put the full table in a collapsible box.
- [ ] **Query stream:** replace "random and oversample pools" with plain words;
      explain the unanswerable share (questions built to test whether the
      assistant admits it does not know).

## Drift (Agent B)

- [ ] **Intro line:** two techniques, meaning (embeddings) and vocabulary
      (word counts), both as "× ordinary change"; plus a pointer that these
      scores do not predict where answers break (see Impact).
- [ ] **Bug: heatmap legend not visible.** Severity colours (dark brown = low,
      orange = medium, light peach = high) have no key on screen; lighter =
      more severe is not obvious on a dark background.
- [ ] **Bug: "kind" labels on flagged cells do not render** (text layer of the
      heatmap shows nothing).
- [ ] **Heatmap caption:** empty cells = too few articles to analyse; 2020 hits
      lifestyle sections (Wellness, Parenting, Style & Beauty, Food & Drink),
      while Politics and Entertainment stay low.
- [ ] **Drill-down:** explain "Novel articles" (unlike anything in the
      baseline; low means familiar content recombined) and the segment ID
      format ("2020|FOOD & DRINK|vs 2016-17" → readable label).
- [ ] **Charts hidden behind "Table" boxes without captions:** add a
      one-line takeaway to each (emerging topics grow rather than appear; words
      keep their spelling but change meaning, e.g. "test": nuclear → positive;
      the correlation grid shows the two meaning measures are redundant (0.93)
      while the word measure adds information (0.56–0.68)).
- [ ] **Correlation grid** needs a "how to read" line (1 = always agree,
      0 = unrelated) or could move to a collapsible "methods" box.
- [ ] **Query drift table:** readable columns (drop p-values and raw MMD into a
      collapsible box; keep "× ordinary" and new words); keep the caveat that
      the questions are generated from the articles.

## Retrieval (Agent C)

- [ ] **Intro line:** this tests only the search step (does the top 5 contain
      an answer?), judged by Opus without knowing which index found the
      article.
- [ ] **Top tiles are redundant:** "Questions 423" and "Judged questions 423"
      repeat; "Mode: full evaluation" repeats the Overview. Replace with
      something informative (e.g. answer found: stale vs refreshed).
- [ ] **Explain 423 here vs 438 on Impact** (the unanswerable questions are
      only used in the answer step).
- [ ] **Dumbbell caption:** stale 0–1% vs refreshed 96–100% for 2018–22; the
      2016–17 control is 100% on both (dots overlap), so re-indexing does not
      hurt old questions.
- [ ] **(b) Known item:** replace "weighted down: F11" with plain words (the
      question was written from that article, so finding it is easy; kept as
      a sanity check only). Readable columns (reciprocal_rank → "Rank of the
      source article (1 = first)").
- [ ] **(c) Old questions:** caption (overlap 93% → 86%; newer articles 7% →
      14% of old questions' top 10; the source article is never pushed out).
      Readable columns.
- [ ] **Tables (b) and (c) are cut off** on the right in two narrow columns;
      stack them or trim columns.
- [ ] **Bring the best example forward:** "Questions that lost most" is the
      most convincing part (e.g. the quinoa → fonio question) but is
      collapsed; show one or two examples open.
