# Findings

Results worth showing in the README and stakeholder walkthrough. Each finding
gives the exact numbers, the command that reproduces them, the path to any
chart, and whether it supports or contradicts what we expected.

## Entry template

```markdown
## F<n>. <Short title>
- **Date:** YYYY-MM-DD
- **Result:** the finding, with exact numbers.
- **Reproduce:** `the exact command or notebook`
- **Chart:** path/to/chart.png (or "none")
- **Expectation:** what we expected beforehand.
- **Verdict:** Supports | Contradicts | Mixed, with one line on why.
- **Implication:** what this changes in the design or the experiment.
```

## Findings

## F1. HuffPost exploration: known drift is clearly present, but volume and labels change sharply after 2017
- **Date:** 2026-11-04
- **Result:**
  - **Provenance (checked 2026-11-04):** the dataset linked in the assignment
    (`kaggle.com/datasets/rmisra/news-category-dataset`, "News Category
    Dataset" by rmisra, CC BY 4.0). Kaggle's API lists one file,
    `News_Category_Dataset_v3.json`, 87,295,572 bytes. The local copy has
    exactly that size, so it is the full dataset, not a sample. SHA-256
    `2cfc51fa4a82a96ecbe251fa383737aa1dea2e0a517f61dece633fcc25a4a065`
    (also in the config).
  - 209,527 records, 0 malformed, 2012-01-28 to 2022-09-23, 42 categories.
    Median 28 words per record (headline + description); 19,712 records
    (9.41%) have an empty description.
  - **Volume cliff, as the dataset card said:** about 30–35k articles a year
    from 2012 to 2017, then 9,734 in 2018, 2,005 in 2019, 2,054 in 2020,
    2,066 in 2021 and 1,398 in 2022 (data ends in September).
  - **Known drift is easy to see** (% of each year's records mentioning a term):
    "coronavirus" 0.0% every year up to 2019, then 18.4% in 2020, 3.0% in 2021;
    "covid" 0.0% up to 2019, then 10.6% (2020) and 10.3% (2021); "vaccine"
    0.0–0.3% up to 2019, then 0.8% (2020) and 7.0% (2021); "trump" 0.0–0.1% in
    2012–2014, 1.7% in 2015, 16.7% in 2016, 22–27% from 2017 to 2020. The
    shift from "coronavirus" (2020) to "covid" (2021) is a clean example of
    terminology changing while the topic stays the same.
  - **The category labels themselves drift.** Lifestyle labels used in
    2012–2013 disappear after 2014 and come back after 2018 under the old
    names: WELLNESS → HEALTHY LIVING, PARENTING → PARENTS, FOOD & DRINK →
    TASTE, STYLE & BEAUTY → STYLE (2014 has both). World news moves from
    WORLDPOST (2014–2015) to THE WORLDPOST (2015–2017) to WORLD NEWS (2017
    onwards). U.S. NEWS first appears in 2018.
  - **The category mix shifts heavily.** POLITICS is 10% of 2014 but 31% of
    2016, 35% of 2018 and 39% of 2020. From 2019 on, only POLITICS,
    ENTERTAINMENT, WORLD NEWS and U.S. NEWS have more than about 150 articles
    a year.
- **Reproduce:** `python scripts/exploration/explore_data.py --dataset huffpost` (tables in
  `runs/explore/huffpost/`; the term rates are `watch_term_rates.csv`, with
  terms set in `config/config.yaml` under `exploration.watch_terms`).
- **Chart:** none yet.
- **Expectation:** a volume cliff after mid-2018 (from the dataset card) and
  visible COVID vocabulary in 2020.
- **Verdict:** Supports both expectations. The label changes and the size of
  the category-mix shift were not expected.
- **Implication:**
  1. Periods after 2018 hold about 2,000 articles each, so `sample_per_period`
     (2,000) cannot be met for 2022, and later periods will mostly be the full
     year, not a sample.
  2. A baseline from 2014–2015 (the earlier proposal) would mix two problems:
     real drift, and a jump in the politics share from 10% to almost 40%.
     Raw embedding drift would mostly measure the mix change. The baseline
     should be closer in mix to the later periods, or the comparison should be
     made within category, or both.
  3. Category segments need a label-normalisation map (above) before they can
     be compared across years, and from 2019 on there are only 5–9 segments
     per year above the 50-article minimum.
  4. The detectors have clear known events to be checked against: COVID in
     2020 and the 2020 → 2021 terminology shift.

## F2. Choosing the baseline: 2016–17 has the closest topic mix to the later years and is stable within itself
- **Date:** 2026-11-04
- **Question:** which baseline period lets the drift detectors measure changes
  in meaning, rather than changes in the mix of topics (F1)?
- **Method:** for each candidate baseline, compare its category mix with each
  later year (2018–2022) after merging renamed sections
  (`categories.normalise` in the config). The measure is the share of
  articles that would have to change category to make the two mixes equal
  (total variation distance): 0% = same mix, 100% = nothing in common. Also
  measured: how much each candidate changes between its own first and second
  half, and what share of each later year falls in categories the candidate
  does not contain at all.
- **Result:**

  | Candidate | vs 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Articles | Own halves differ by |
  |---|---|---|---|---|---|---|---|---|
  | 2012–13 | 86.3 | 85.5 | 75.3 | 84.4 | 83.7 | **83.1** | 65,938 | 12.5 (year vs year) |
  | 2014–15 | 45.3 | 53.8 | 50.4 | 53.8 | 51.0 | **50.9** | 64,345 | 27.5 (year vs year) |
  | 2015 | 34.4 | 48.6 | 45.1 | 46.5 | 42.4 | **43.4** | 32,006 | 30.1 (half-year) |
  | 2015–16 | 27.5 | 41.5 | 39.2 | 39.9 | 35.7 | **36.8** | 64,104 | 17.3 (year vs year) |
  | **2016–17** | 17.9 | 38.8 | 30.6 | 30.7 | 33.1 | **30.2** | 61,987 | **13.5 (year vs year)** |
  | 2017 | 18.2 | 37.3 | 28.1 | 29.3 | 34.5 | **29.5** | 29,889 | 5.6 (half-year) |

  - **About half of the gap that remains with 2016–17 comes from sections the
    baseline does not have, mainly U.S. NEWS.** U.S. NEWS does not exist
    before 2018 and makes up 23.3% of 2019, 9.6% of 2020, 14.7% of 2021 and
    17.2% of 2022. With 2016–17 as the baseline, 1.9% / 25.8% / 13.3% / 16.3% /
    17.5% of 2018–2022 is in sections the baseline does not contain. That is
    two-thirds of the 2019 gap (25.8 of 38.8) and 43–53% for 2020–2022. No
    choice of baseline can fix this.
  - **Other shifts against 2016–17:** POLITICS is 33.2% of the baseline and
    25–40% of the later years; ENTERTAINMENT goes from 10.8% to 17–19% (except
    2019); WORLD NEWS jumps to 17.5% in 2019.
  - The year-vs-year and half-year stability numbers are not directly
    comparable: half-year splits hide some of the yearly variation.
- **Reproduce:** `python scripts/exploration/compare_baselines.py` (tables in
  `runs/explore/huffpost/baselines/`; candidates in `exploration.baseline_candidates`).
- **Chart:** none yet. Suggested for the slides: one bar per candidate showing
  its mean mix difference, with the chosen baseline highlighted; and a stacked
  bar of category shares per period (`category_shares.csv`).
- **Expectation:** later baselines would match the later years better, with
  2014–15 (the earlier proposal) clearly worse.
- **Verdict:** Supports. Not expected: how unstable 2015 is within itself
  (30.1% between its halves), and how much of the remaining gap one new
  section (U.S. NEWS) explains (about half).
- **Implication:** baseline set to 2016–17 (D6). Drift must also be measured
  within each category, because about 30% of the mix still differs. New
  sections such as U.S. NEWS are reported by Data Monitoring as a change in
  the archive's structure, not scored as semantic drift.
- **For the walkthrough:**
  - One-line message: "Before measuring drift, we checked that the baseline
    would not mistake a change in topic mix for a change in meaning."
  - The earlier proposal (2014–15) was rejected on evidence: about half of
    its articles would have to change category to match the later years.
  - We say openly what the baseline does not solve (the U.S. NEWS section,
    the shift in mix) and how the design handles it (within-category
    comparisons, structural change reported separately).

## F3. Cleaning removes 2.3% of the baseline and almost nothing from later years
- **Date:** 2026-11-04
- **Result:** retention table (rules applied in this order; each sees only
  what the previous ones kept):

  | Period | Raw | Empty description | Under 8 words | Over 80 words | Near-duplicate (cosine ≥ 0.95) | **Kept** | Kept % | Drift sample |
  |---|---|---|---|---|---|---|---|---|
  | 2016–17 | 61,987 | 1,034 | 44 | 34 | 284 | **60,591** | 97.7 | 1,398 |
  | 2018 | 9,734 | 0 | 2 | 0 | 4 | **9,728** | 99.9 | 1,398 |
  | 2019 | 2,005 | 0 | 0 | 0 | 1 | **2,004** | 100.0 | 1,398 |
  | 2020 | 2,054 | 0 | 0 | 0 | 1 | **2,053** | 100.0 | 1,398 |
  | 2021 | 2,066 | 0 | 0 | 0 | 0 | **2,066** | 100.0 | 1,398 |
  | 2022 | 1,398 | 0 | 0 | 0 | 0 | **1,398** | 100.0 | 1,398 |
  | Total | 79,244 | 1,034 | 46 | 34 | 290 | **77,840** | 98.2 | 8,388 |

  - **Every rule mostly affects the baseline.** Empty descriptions, very
    short and very long records all exist (almost) only in 2016–17; later
    years have none. Left in, they would be an artificial difference between
    baseline and later years that the drift detector could flag.
  - **Near-duplicates are templated columns and reposts.** The most removed
    are "The 20 Funniest Tweets From Women This Week" (46), "HuffPost Rise:
    What You Need To Know On <date>" (134 of the 290) and "The
    Funniest Tweets From Parents This Week" (18). Hand-checked pairs: at
    cosine ≥ 0.97 almost all are identical reposts or templates; 0.95–0.97 is
    mostly templates; 0.93–0.95 is genuinely different articles about the same
    event (e.g. two reports of Hong Kong airport reopening), which are kept.
  - **The length gap survives cleaning:** median 24 words in the baseline,
    25 in 2018, 29–31 in 2019–2022 (same in the drift samples). This is a
    real change in document characteristics and may inflate embedding drift
    (D6, check 5).
  - Renamed sections merged: 41 raw categories become 32.
  - The scratch simulation of these rules, written separately before the
    real code, gave identical counts for every rule and period.
- **Reproduce:** `python scripts/prepare_data.py` (tables in
  `runs/prepare/retention.csv`; every removed document and its rule in
  `runs/prepare/removed.csv`; output `data/processed/documents.parquet`).
  The first run embeds 78,130 texts (about 78,000 distinct) and takes about
  80 s in total on the GPU; reruns take about 14 s, using the cache.
- **Chart:** none.
- **Expectation:** small removals; exact duplicates rare (F1 found 103).
- **Verdict:** Supports. Not expected: that every rule falls almost entirely
  on the baseline, and that templated columns make up most near-duplicates.
- **Implication:** the cleaned periods are comparable in format. The drift
  sample is 1,398 per period (all of 2022). Templated columns that survive
  (e.g. "Daily Meditation", "Headline Quiz") should be excluded when
  generating test questions, because they have no single right answer.
