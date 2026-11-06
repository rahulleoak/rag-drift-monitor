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
- **Chart:** `docs/figures/pandemic_words.png` (coronavirus, covid, vaccine per year).
- **Expectation:** a volume cliff after mid-2018 (from the dataset card) and
  visible COVID vocabulary in 2020.
- **Verdict:** Supports both expectations. The label changes and the size of
  the category-mix shift were not expected.
- **Implication:**
  1. Periods after 2018 hold about 2,000 articles each, so `sample_per_period`
     (2,000) cannot be met for 2022, and later periods will mostly be the full
     year, not a sample. *(Resolved in F3: the sample is 1,398.)*
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

## F4. Whole-period drift: COVID is the biggest shift, but not where expected
- **Date:** 2026-11-05
- **Setup:** drift samples of 1,398 articles per period; each later year
  compared with the 2016–17 baseline and with the previous year; 500
  permutations per test; a reference row compares 2017 with 2016 (one
  ordinary year of change inside the baseline). Settings in D10.
- **Result:**

  | Period | vs | MMD² | × ordinary year | Word drift (excess JSD) | New-term rate | Spread (dispersion) | Novel articles |
  |---|---|---|---|---|---|---|---|
  | 2017 (reference) | 2016 | 0.0021 | 1.0 | 0.031 | – | 0.545 | 5.0% (by definition) |
  | 2018 | baseline | 0.0037 | 1.8 | 0.035 | 3.0% | 0.561 | 8.2% |
  | 2019 | baseline | 0.0071 | 3.4 | 0.067 | 3.7% | 0.561 | 9.5% |
  | **2020** | **baseline** | **0.0119** | **5.7** | 0.073 | **5.5%** | **0.539** | 8.9% |
  | 2021 | baseline | 0.0091 | 4.3 | 0.077 | 5.1% | 0.554 | 10.9% |
  | 2022 | baseline | 0.0099 | 4.7 | 0.077 | 4.8% | 0.561 | 10.6% |
  | 2019 | 2018 | 0.0036 | 1.7 | 0.048 | | | |
  | **2020** | **2019** | **0.0088** | **4.2** | 0.072 | | | |
  | 2021 | 2020 | 0.0044 | 2.1 | 0.047 | | | |
  | 2022 | 2021 | 0.0025 | 1.2 | 0.038 | | | |

  1. **Every comparison is "statistically significant",** including 2017 vs
     2016 (all q = 0.002, the smallest possible with 500 permutations). At
     1,398 articles per side any difference is detectable, so significance
     alone cannot separate drift from ordinary change. Effect size against
     an ordinary year is what matters (supports Q11 and P7).
  2. **2020 is the largest embedding shift:** 5.7× an ordinary year against
     the baseline, and the largest single-year jump (4.2× from 2019). Its
     articles are also the least spread out (0.539), consistent with one
     story dominating the year (P3's prediction).
  3. **The Boiling Frog pattern appears (P1):** 2022 vs 2021 looks almost
     ordinary (1.2×), yet 2022 vs the baseline is 4.7×. Monitoring only
     year-on-year change would miss how far the archive has moved.
  4. **The two techniques disagree after 2020:** embedding drift peaks in 2020
     and eases, while word drift keeps rising (0.073 → 0.077). They measure
     different things (supports "complementary"; P6 will quantify it).
  5. **Emerging terms match real events** (log-odds z-scores vs the baseline):
     2020: biden, coronavirus, virus, covid-19; 2021: vaccine, shot, jan,
     gop; 2022: russia, putin, invasion, abortion. Declining: clinton,
     hillary, then trump (2021–22). "said" rises every year: a shift towards
     wire-style reporting.
  6. **Surprise: COVID articles are not "novel" to the embedding model.**
     Only 5.0% of 2020's 539 COVID articles have no close baseline match,
     exactly the baseline's own rate, against 10.1% for other 2020
     articles. Their nearest baseline articles are about Trump (146), health
     (55), flu (30) and Zika (27), e.g. "China's Coronavirus Infections Top
     40,000" → "H7N9 Bird Flu Epidemic In China Worst Since 2013". The most
     novel 2020 articles are one-off human-interest stories.
- **Reproduce:** `python scripts/run_drift.py` (tables in `runs/drift/`;
  about 10 s after the first run).
- **Chart:** `docs/figures/drift_per_year.png` (vs baseline and vs previous year, with the medium/high thresholds).
- **Expectation:** 2020 the largest drift; COVID articles among the most
  novel.
- **Verdict:** Mixed. 2020 is the largest shift as expected, but per-article
  novelty does not flag COVID at all.
- **Implication:**
  - Drift will be judged by effect size relative to ordinary change, with
    bands calibrated on the baseline (next step), not by p-values.
  - **Key hypothesis for the impact experiment:** COVID questions on the
    stale index will retrieve related but wrong articles (bird flu, Zika,
    Trump health policy): harmful drift that the embedding model does not
    register as new. Word drift and distribution-level MMD catch it;
    article-level novelty does not. This is the clearest "which drift
    matters" case to test with Agents C and D.

## F5. Severity bands and drift by section: COVID hit lifestyle journalism hardest; politics never looks unusual
- **Date:** 2026-11-05
- **Setup:** the A → B pipeline in LangGraph. Bands from ordinary change
  inside the baseline (same month a year apart, D11). "× ordinary" here is
  relative to the *median month-pair change*, which is larger than the
  single 2017-vs-2016 value used in F4, so the multiples are smaller than
  F4's; the ranking is the same.
- **Whole periods vs the baseline:**

  | Period | Embedding (× ordinary) | Words (× ordinary) | Bands (embedding / words) | Kind | Flagged |
  |---|---|---|---|---|---|
  | 2018 | 0.8 | 0.6 | low / low | none | no |
  | 2019 | 1.5 | 1.1 | low / low | none | no |
  | **2020** | **2.5** | 1.2 | **high** / low | meaning | **yes** |
  | 2021 | 1.9 | 1.2 | low / **medium** | wording | **yes** |
  | 2022 | 2.1 | 1.2 | medium / medium | both | **yes** |

  Every year-on-year comparison is "low", including 2020 vs 2019 (1.9×):
  measured against month-to-month news swings, no single year is
  extraordinary on its own; the archive moves away gradually (P1).
  **D6 check 2 passes:** 2020 is flagged high.
- **Agent A:** category mix vs the baseline has Cramér's V of 0.21 (2018),
  0.51 (2019), 0.37 (2020), 0.41 (2021), 0.42 (2022). Sections absent from
  the baseline make up 1.9% of 2018 and 13–26% of later years; besides U.S.
  NEWS, they include MONEY, HOME & LIVING and WEDDINGS, sections that existed
  before 2016 and *returned* after 2018.
- **Sections (37 of 134 period × category groups are large enough):** 14
  flagged vs the baseline. Highlights:
  - **2020 lifestyle sections are the most drifted of all,** with both
    techniques high: WELLNESS (52 of 67 articles COVID-related, e.g.
    "Social Distancing: Places And Events To Avoid"), FOOD & DRINK (28 of 54:
    "Is It Safe To Go To The Grocery Store?"), STYLE & BEAUTY (35 of 75: "How
    To Make Coronavirus Face Masks"), PARENTING (21 of 58; 11 are tweet or
    shopping roundups). Also WORLD NEWS 2020 (outbreak, virus).
  - **2022 WORLD NEWS:** ukraine, russian, invasion (both high).
  - **SPORTS in Olympic years** (2018 Winter, 2021 Tokyo): wording drift.
  - **2019 COMEDY:** late-night hosts on the shutdown and impeachment.
  - **POLITICS is never flagged** (0.4–1.0× ordinary, 2018–2022) despite
    Trump → Biden, the Capitol attack and Ukraine. The baseline's own
    political coverage (election year, then a new administration) changes
    so much month to month that later change looks ordinary. This is D6's
    weakness 2 (the election year inside "normal" variation), seen for one
    section.
- **Caveats:** multiples for small sections (54–75 articles) are imprecise
  (FOOD & DRINK 2020: 17×); the bands, built at the segment's size, are what
  drive decisions. 97 groups are too small or have no baseline counterpart
  (reported by Agent A instead).
- **Reproduce:** `python scripts/run_pipeline.py` (about 60 s; reports in
  `runs/<run_id>/`: `monitoring.json`, `drift.json`, `drift_whole.csv`,
  `drift_segments.csv`, `route.txt`).
- **Chart:** `docs/figures/severity_by_section.png` (section × year severity grid).
- **Expectation:** 2020 flagged; politics among the most drifted sections.
- **Verdict:** Mixed. 2020 flagged as expected; politics not flagged at all.
- **Implication:**
  - The 2020 lifestyle sections are the best test cases for the impact
    experiment: practical, time-sensitive questions ("is it safe to…") that
    a 2017 archive cannot answer.
  - **Politics is the key test of "quality loss without drift":** if
    politics questions lose quality on the stale index although no drift
    was flagged, the system must still catch it (the regression check and
    the "escalate" rule exist for this).

## F6. Emerging topics, changing word meanings, and how the drift measures relate
- **Date:** 2026-11-05
- **Result:**
  1. **Emerging topics find COVID where article novelty could not (F4).**
     Clustering all drift samples together (40 topics) and looking for
     *growth* gives 17 emerging topic-years. In 2020, three COVID topics with
     almost no baseline counterpart: political COVID (117 articles vs 0),
     lifestyle COVID across wellness, style and food (104 vs 0, 3 sections)
     and the world outbreak (89 vs 5). Later: vaccines (2021, 79 vs 8),
     the Ukraine war (2022, 135 vs 8), the Hong Kong protests (2019), Biden
     (2020–21), the royals (2021). One "topic" is a format, not news: weekly
     funny-tweets roundups (25 in 2020 vs 1).
  2. **Words that changed meaning** (centroid shift of the articles using a
     word, vs the same word's shift between 2016 and 2017 at matched group
     size; 44 term-years for 17 terms):

     | Term | Year | × ordinary | Context before (baseline) | Context after |
     |---|---|---|---|---|
     | home | 2020 | 4.6 | refugees, world, state | hair, coronavirus, products, clothes |
     | virus | 2020 | 3.5 | zika, florida, congress | china, coronavirus, covid-19 |
     | cases | 2020 | 2.9 | sexual, assault, supreme (court) | coronavirus, covid-19, spread |
     | biden | 2022 | 2.9 | joe, vice, obama | russian, administration, ukraine |
     | russia | 2022 | 2.5 | trump, election, campaign | ukraine, war, europe, aid |
     | test | 2020 | 2.4 | north korea, nuclear, missile | positive, coronavirus |
     | outbreak | 2020 | 2.3 | ebola, zika | coronavirus, china, pandemic |

     "mask" changes clearly (wore, campaign → face, coronavirus, wearing) but
     has too few baseline articles per half (under 10) for an ordinary
     estimate, so it is reported without a ratio. All 44 shifts are
     "significant", the same large-sample effect as in F4, so the ratio is
     what matters.
  3. **Correlation of the drift measures across 42 comparisons with the baseline (P6,
     Spearman):** MMD and centroid shift 0.93 (two views of the same
     embedding shift, near-redundant); MMD and word drift (excess JSD) 0.68
     (related but distinct); MMD and article novelty 0.07 (unrelated);
     novelty and new-term rate 0.68.
- **Reproduce:** `python scripts/run_pipeline.py` (`runs/<run_id>/topics.csv`,
  `term_shifts.csv`, `correlations.csv`).
- **Chart:** `docs/figures/emerging_topics.png`, `docs/figures/words_that_changed_meaning.png`, `docs/figures/measure_correlations.png`.
- **Expectation:** COVID visible as an emerging topic; several words with
  shifted meaning; the two techniques only moderately correlated.
- **Verdict:** Supports all three. Not expected: how cleanly ordinary words
  ("home", "cases", "test") change meaning, and that article novelty is
  almost unrelated to distribution-level drift.
- **Implication:**
  - Three families of drift signal, not two: distribution shift (MMD,
    centroid), word drift (JSD, new terms) and article novelty. Agent D will
    test which family predicts quality loss (P4).
  - Centroid shift adds little beyond MMD; it stays as the intuitive number.
  - Shifted words make concrete retrieval tests: "how many COVID cases" on a
    stale index should retrieve court cases. 🔨
  - A first version of the term-shift ratio was inflated for small groups
    (noisy centroids); fixed by matching group sizes (D12).

## F7. Question pilot: good questions, cheap, and five things we had not accounted for
- **Date:** 2026-11-05
- **Result:** 20 articles (5 baseline, 3 per later year) → **15 questions
  kept**. Cost **$0.18** in total: Sonnet writing $0.064 (20 calls, about
  $0.003 each), Opus reviewing $0.112 (15 calls, about $0.0075 each); the
  advance estimate ($0.51) was about 3× too high (output assumed 900 tokens;
  actual about 130–185). No refusals, no broken replies. Questions are as
  varied as their source articles (diversity ratio 1.03, P3: no collapse).
  - **Rejected (5):** 4 by the writer's own "not answerable" flag, all
    teasers with no stated fact ("Hopes And Fears In The GOP
    Establishment", "Dad's Instagram Post Sums Up The Struggle Of
    Weekends"); 1 for reusing the headline ("How many FCC complaints… 60
    Minutes interview"). The checks worked as designed.
  - **Kept questions are specific and paraphrased**, e.g. "How much in U.S.
    goods did China target with new tariff increases in August 2019?" from
    "Trade War Escalates With China Announcing Tariff Hikes On $75 Billion".
- **Not accounted for:**
  1. **No time-sensitive questions at all** (15 of 15 evergreen). These are
     the questions where a stale archive gives a superseded answer (D13), so
     the impact experiment needs them.
  2. **Questions echo the description**, not the headline: the reviewer
     noted "borrows heavily from the description" twice, but the copy check
     only compares with the headline. Copied wording makes retrieval look
     easier than it is.
  3. **The reviewer passed all 15**, even when its own reason named a
     problem ("'Pennsylvania' is unstated in the article but harmless",
     "partly echoes its own answer"). Its yes/no checks are lenient; the
     code only sees the yes/no.
  4. **Questions add outside facts** (Pennsylvania, the Paris agreement,
     "blood-testing"). Realistic for a journalist, but it lets the writer's
     own knowledge leak into the test.
  5. **A third are two-part questions** ("How many…, and did police…?"),
     which are harder to judge as right or wrong.
- **Reproduce:** `python scripts/generate_questions.py` (cached: a rerun
  costs nothing); every draft and decision in
  `runs/questions/pilot/questions.csv`; costs in `runs/llm_costs.csv`.
- **Expectation:** a keep rate around 70–80%; some time-sensitive questions.
- **Verdict:** Mixed. Quality and cost are better than expected; the
  question mix and the checks need tightening before scaling up.
- **Implication:** a version-2 prompt and checks (see journal), then a
  second pilot before generating the ~180-question set. At about $0.01 per
  kept question, the small set costs about $1–2, and the full plan is far
  inside the budget.

## F8. Question pilots 2 and 3: stricter checks halve the keep rate; type labels are ambiguous even for Opus
- **Date:** 2026-11-05
- **Setup:** the same 20 articles as the first pilot (F7), so versions compare
  directly.
- **Result:**

  | Version | Drafts | Kept | Keep rate | Time-sensitive kept | Unanswerable kept | Cost |
  |---|---|---|---|---|---|---|
  | v1 (F7) | 20 | 15 | 75% | 0 | n/a | $0.18 |
  | v2 | 24 | 9 | 38% | 3 | 0 | $0.28 |
  | v3 | 25 | 9 | 36% | 2 | 0 | $0.33 |
  | **v4** (reviewer labels the type; looser unanswerable rule) | 25 | **13 + 2** | **60%** | 2 | **2** | $0.25 |

  Total spent so far: $1.03 of the $150 budget. v4 reused the v3 writer
  replies from the cache and reran only the reviews.
  - **v4:** writer and reviewer agree on the type label for 20 of 24
    reviewed questions (83%). The 4 disagreements are exactly the ambiguous
    cases: the Brooklyn death toll and gang-link questions, the NATO chief's
    warning, a candidate's background. The unanswerable set now holds the
    divorce-debt question and "What did a dad's Instagram post… capture
    about weekends for parents?" (borderline: the headline half-answers it).
  - **v2 fixed what it targeted:** two-part questions gone; time-sensitive
    questions appear ("How many people have Russian investigators said died
    in the Moscow plane fire so far?"); the reviewer catches questions that
    give away their own answer and wrong type labels.
  - **v2 overshot:** "no outside facts" rejected common knowledge ("Justin
    Trudeau", "Canada", "Pennsylvania senator"); the 5-word description copy
    check rejected proper names; no unanswerable question survived, partly
    because the writer treated the description as the whole article.
  - **v3's remaining rejections are about half correct** (invented details:
    "after the US decision on the Paris deal"; a winner the article never
    names; "on Sunday" with no date) **and half noise:**
    - **The type label is unstable even for Opus:** the same question ("Did
      police see any initial indication that the Brooklyn shootings were
      linked to gangs?") was accepted as time-sensitive in v2 and rejected
      as "should be evergreen" in v3. Whether "what police said at the
      time" is time-sensitive is genuinely ambiguous.
    - **"Natural" rejects unanswerable questions for restating the
      headline,** which they inherently do.
  - Diversity stayed healthy (ratio 1.03–1.05: no collapse into generic
    phrasing).
- **Reproduce:** `python scripts/generate_questions.py --name pilot_v2` /
  `--name pilot_v3` with the matching prompt versions (cached: reruns are
  free); tables in `runs/questions/pilot_v2/` and `pilot_v3/`.
- **Expectation:** v3 would keep 60–70%.
- **Verdict:** Contradicts. Stricter checking found real writer errors, but
  two checks (type label, "natural" for unanswerable questions) add noise.
- **Implication (done in v4):** the reviewer assigns the type label instead
  of judging the writer's (disagreement recorded, not a rejection), and an
  unanswerable question needs only the reviewer's agreement that it is
  unanswerable and a single ask. Keep rate 60%. Prompt tuning stops here:
  at about $0.016 per kept question the remaining rejections are mostly
  right and affordable. The type label carries an ambiguity of about 17%
  (the disagreement rate), to be stated with any evergreen vs
  time-sensitive result; a cleaner subset uses only questions where both
  models agree.

## F9. How reliable is the question reviewer? Mostly consistent; its errors are systematic, not random
- **Date:** 2026-11-06
- **Two observations from comparing v3 and v4 (F8), the reason for this test:**
  1. 8 of 25 decisions flipped between v3 and v4. 5 were the intended
     effect; 3 had nothing to do with the change. One of those let a
     question with an invented detail through: "What did the Pretty Little
     Liars cast members do together to mark *a milestone*?" (v3 rejected it;
     the article never says "milestone").
  2. Opus labels fewer questions time-sensitive than the writer (3 vs 6 of
     the reviewed drafts), so with Opus's label final, time-sensitive
     questions became scarcer.
- **Test (v5):** the same 20 articles and the same r-v4 prompt, each
  question reviewed **twice independently**; a question is kept only if both
  reviews accept it. Labels are grouped: "evergreen" or "time_sensitive" if
  the writer and both reviews agree, otherwise "disputed".
- **Result:**
  - **Same keep/reject decision in 22 of 24 double-reviewed questions
    (92%); identical type label in 24 of 24 (100%).**
  - **Correction to observation 1:** the 3 "unexplained" v3→v4 flips were
    not mainly random. The reviewer prompt also changed between v3 and v4
    (r-v3 → r-v4), so those flips mix the prompt change with randomness.
    Measured with the *same* prompt, the reviewer disagrees with itself on
    about 8% of decisions.
  - **The "milestone" question passed both reviews** ("'milestone'
    reasonably paraphrases freedom from plot holes"). It is a consistent
    judgement under r-v4, not a random lapse, so reviewing twice cannot
    catch it. Systematic errors like this are what the hand labels (P8)
    are for.
  - **Only 1 question changed** (15 → 14 kept): "What did a dad's Instagram
    post… capture about weekends for parents?" left the unanswerable set
    (review 2: the headline answers it, which is defensible). The other
    split decision (the Uttarakhand missing count) was rejected by review 1
    for a hedged reference answer ("about 140" vs the article's 140).
  - **Labels:** among the 13 kept answerable questions, 9 clear evergreen,
    **1 clear time-sensitive**, 3 disputed. The two reviews never disagree
    on the label: the disagreement is between writer and reviewer, a
    systematic difference in interpretation, not noise.
  - Cost: $0.25 for the second reviews; $1.27 spent in total.
- **Reproduce:** `python scripts/generate_questions.py --name pilot_v5`
  (cached); `runs/questions/pilot_v5/questions.csv`.
- **Expectation:** double review would catch random lapses such as the
  "milestone" question, and lower the keep rate noticeably.
- **Verdict:** Mixed. The reviewer is more consistent than feared (92%);
  double review changed one decision and could not catch the systematic
  miss.
- **Implication:**
  - Double review stays for the full set: it is cheap (about $1.50 more for
    ~300 drafts) and gives a measured reliability figure; it protects
    against random lapses, not systematic ones.
  - Hand labels must be drawn to include kept questions, so systematic
    reviewer errors are measured.
  - **Clear time-sensitive questions are very scarce (1 of 13).** An
    analysis of "outdated answers" needs an explicit oversample of articles
    likely to contain revisable facts (tolls, counts still being reported,
    ongoing situations), stated as such.

## F10. The oversample quadruples clear time-sensitive questions, at a lower keep rate
- **Date:** 2026-11-06
- **Setup:** 20 articles from the likely-revisable pool (D20), same period
  counts as the random pilots, same prompts, double review (D19).
- **Result:**

  | Sample | Articles | Kept answerable | Clear time-sensitive | Disputed label | Keep rate | Writer–reviewer label agreement | Cost |
  |---|---|---|---|---|---|---|---|
  | Random (v5, F9) | 20 | 13 | 1 (8%) | 3 | 56% | 83% | $0.25 (2nd reviews only) |
  | **Oversample** | 20 | 9 | **4 (44%)** | 1 | 35% | 91% | $0.60 |

  - Kept time-sensitive questions are what the experiment needs: "How many
    people had died in the New Zealand mosque shootings as of March 17,
    2019?"; "How many U.S. states had detected the Omicron variant as of
    early December 2021?"; "How many COVID-19 cases did North Paulding High
    School in Georgia report…?"
  - Labels are less ambiguous here (agreement 91% vs 83%).
  - The keep rate is lower mainly because both reviewers reject questions
    that reuse the description's wording (19 "not natural" votes): breaking
    news is full of specific names and places ("Carr fire in northern
    California… on Saturday"), which are hard to paraphrase. Some of these
    flags treat proper names as copying; the strictness also keeps
    retrieval from being artificially easy.
  - Reviewer consistency 96% (decisions), 100% (labels). Total spent $1.87.
- **Reproduce:** `python scripts/generate_questions.py --name oversample_pilot --oversample`.
- **Expectation:** more time-sensitive questions.
- **Verdict:** Supports; the keep rate cost is acceptable (about $0.07 per
  kept question).
- **Implication:** about 200 oversampled articles would give roughly 40
  clear time-sensitive questions for about $6 (live) or $3 (batch); no
  further prompt tuning.

## F11. Rejecting "copied" wording does not make retrieval harder; every question finds its article easily
- **Date:** 2026-11-06
- **Question:** the reviewers reject questions that reuse the article's
  wording as "not natural" (F10). Is that justified, i.e. do those questions
  make retrieval artificially easy?
- **Method (no LLM cost):** for every drafted question from the random v5
  pilot and the oversample pilot (51 questions), embed the question and
  search the *whole* archive (77,840 articles); record where its source
  article ranks, and the share of its content words found in the article.
- **Result:**

  | Group | Questions | Source ranked 1st | In top 10 | Word overlap |
  |---|---|---|---|---|
  | Kept | 23 | 91% | 100% | 0.53 |
  | Rejected as "not natural" (copying) | 14 | 79% | 86% | 0.62 |
  | Rejected for other reasons | 14 | 79% | 100% | 0.56 |

  1. **Copy-rejected questions are not easier to retrieve**, in fact a
     little harder, despite slightly more shared words. The rejection does
     not protect the evaluation from easy matches. (Small groups: the
     differences are not significant; the point is that there is no sign of
     the feared effect.)
  2. **Every group is easy:** a question written from one article finds that
     article first among 77,840 about 80–90% of the time. This is the
     circularity expected when questions are written from their own source
     (Q8): known-item retrieval on the refreshed index will be near its
     ceiling and say little.
- **Reproduce:** the analysis snippet in the journal entry of 2026-11-06
  (uses `driftmon.index.VectorIndex`, the cached question embeddings and
  `data/processed/questions_{pilot_v5,oversample_pilot}.parquet`).
- **Expectation:** copy-rejected questions would retrieve their source more
  easily.
- **Verdict:** Contradicts.
- **Implication:**
  - The "natural" check should stop counting reused names and facts as a
    problem: verbatim copying is already caught by the code checks (4
    headline words, 7 description words). It still rejects awkward
    phrasing. Proposed as reviewer r-v5.
  - The informative retrieval comparison is stale vs refreshed (where the
    stale index lacks the source), plus harder questions in newer wording
    about older topics (D7, item 2), not known-item rank on the refreshed
    index. Revisit D2's weighting of known-item metrics when Agent C runs.

## F12. Reviewer r-v5 (names and facts are not copying): oversample keep rate 35% → 58%
- **Date:** 2026-11-06
- **Change:** "natural" no longer counts reused names, places, numbers or
  facts as copying; only awkward phrasing or a whole copied sentence fails
  it (F11 showed copy-rejected questions were not easier to retrieve).
  Writer unchanged; reviews rerun twice on both pilots.
- **Result:**

  | Pilot | Kept before (r-v4) | Kept after (r-v5) | Clear time-sensitive | Reviewer consistency |
  |---|---|---|---|---|
  | Random (20 articles) | 13 + 1 unanswerable | 14 + 1 | 1 | 92% |
  | Oversample (20 articles) | 9 | **15** | **6** (was 4) | 87% |

  - **Correctly rescued:** "How many boxes of documents from the probe into
    violations at the former Rocky Flats weapons plant have gone missing?",
    "How many victims have California's wildfires claimed so far, as of
    early August 2018?", the New Zealand and SEC questions.
  - **Correctly rejected now:** "What happened to the Navy SEAL who died
    during a Fleet Week demonstration?" (half-answers itself).
  - **Still lenient:** "Who was killed by the Carr fire… on Saturday?" (no
    date; a similar "on Sunday" question was rejected earlier); a yes/no
    question about Bernie Sanders that common knowledge answers; the
    "milestone" question (F9) still passes. These are what the hand labels
    (P8) must measure.
  - Spent so far: $2.82 of $150.
- **Reproduce:** `python scripts/generate_questions.py --name pilot_v6` and
  `--name oversample_v2 --oversample` (cached).
- **Expectation:** more kept questions, especially in the oversample.
- **Verdict:** Supports. Prompt tuning ends here; the remaining errors are
  reviewer leniency, measured by hand labels rather than more prompts.

## F15. Stale vs refreshed: not re-indexing loses almost every new question; drift level does not change that
- **Date:** 2026-11-06
- **Setup:** Agent C (D25) on the full question set (423 answerable
  questions). Opus grades the pooled top 5 of the stale and refreshed index
  in one blind, shuffled call per question. 423 of 423 judged, no failures,
  no skipped articles; cost $1.65 as a batch (the estimate was $3.17).
- **Result (a), judged, "answer" = an article in the top 5 answers the question:**

  | Period | Questions | Stale answer | Refreshed answer | Stale nDCG@5 | Refreshed nDCG@5 | Δ nDCG (95% CI) |
  |---|---|---|---|---|---|---|
  | 2016-17 (baseline questions) | 49 | 1.00 | 1.00 | 0.968 | 0.969 | +0.001 (−0.006 to 0.008) |
  | 2018 | 99 | 0.01 | 1.00 | 0.081 | 0.966 | +0.885 (0.848 to 0.921) |
  | 2019 | 70 | 0.00 | 1.00 | 0.022 | 0.970 | +0.948 |
  | 2020 | 83 | 0.00 | 0.964 | 0.010 | 0.944 | +0.933 |
  | 2021 | 69 | 0.00 | 0.986 | 0.012 | 0.968 | +0.955 |
  | 2022 | 53 | 0.00 | 0.981 | 0.018 | 0.969 | +0.951 |

  1. **Without re-indexing, questions about new events fail almost
     completely** (0–1% answered), in every year and for every question
     type, including "evergreen" ones: they are about specific events the
     stale index never saw. The stale index returns related but wrong
     articles, as F4 predicted: "Which food delivery company released data on
     what takeout items have grown more popular during the coronavirus
     pandemic?" gets 2016 hepatitis A and E. coli outbreak stories (all
     graded 0); the refreshed index ranks "The Delivery Foods Americans Are
     Ordering More During Lockdown" first (grade 2).
  2. **Re-indexing does not hurt old questions:** baseline questions score
     the same on both (Δ nDCG +0.001), although by 2022 14% of their top 10
     are newer articles (c) and no source article is pushed out.
  3. **Drift level does not change the picture.** Later-period questions,
     by Agent B's drift status:

     | Status | Questions | Stale answer | Refreshed answer | Refreshed nDCG@5 |
     |---|---|---|---|---|
     | Flagged sections | 123 | 0.000 | 0.992 | 0.960 |
     | Control sections | 166 | 0.006 | 0.982 | 0.958 |
     | Other | 85 | 0.000 | 0.988 | 0.976 |

     Flagged sections lose as much as control sections without
     re-indexing, and retrieve as well as them after it. On this archive,
     **the drift that matters is new content the index does not have**, not
     how far a section's meaning moved. 2020 is the hardest year on the
     refreshed index (answer 0.964, nDCG 0.944; MRR 0.904 in (b)), but the
     intervals overlap the other years.
- **Caveats:** questions are written from their source article, so they
  are easy to retrieve (F11); a stale index cannot hold a later article, so
  (a) is one-sided by construction (D2). The judge is checked against hand
  labels (P8) before these numbers are final.
- **Reproduce:** `python scripts/run_pipeline.py` (cached; `runs/20261106-113106/`:
  `retrieval.json`, `retrieval_judged.csv`, `retrieval_grades.csv`).
- **Expectation:** stale worse on new questions (yes); drifted sections hit
  harder than controls (no).
- **Verdict:** Mixed: supports the re-indexing recommendation; contradicts
  "more drift, more damage". Agent D (answer quality) and the Wikipedia set,
  where the stale index holds an outdated version of the *same* article,
  test the subtler case.

## F16. Chunking: chunk-level comparison follows real edits far better than whole-article vectors
- **Date:** 2026-11-06
- **Setup:** the Wikipedia set (D26): 165 articles at the end of 2017
  (baseline) and of each year 2018–2022; 815 article × year pairs with a
  baseline version (2 articles are new since the baseline). Ground truth:
  the share of the later version's words in sentences that are new or
  rewritten since the baseline version. Each way of reading an article
  gives a change score; we measure how well it follows the ground truth
  (Spearman rank correlation, and AUC for a "big change" = at least 20% of
  the text new or rewritten, threshold set before looking).
- **Result:**

  | How the article is read | Follows real change (Spearman) | Spots big changes (AUC) |
  |---|---|---|
  | First 512 tokens only (whole-article embedding) | 0.51 | 0.77 |
  | One averaged vector of all chunks | 0.78 | 0.89 |
  | **Chunks of ~256 tokens** (our default) | **0.87** | **0.93** |
  | Chunks of ~128 tokens | 0.89 | 0.95 |
  | Chunks, largest single change instead of average | 0.55–0.56 | 0.80–0.82 |

  1. **A whole-article embedding sees only the start.** Every article is
     longer than 512 tokens (median about 8,000); 91% of the text is never
     read. Edits are spread through articles, so the start still moves a
     little, but the score follows real change poorly (0.51).
  2. **Averaging dilutes.** It ranks articles reasonably (0.78), but the
     scores are tiny (0.006 on average in 2022, when 54% of the text had
     changed), so any alert threshold would sit in the noise.
  3. **Chunks work best.** 128-token chunks are slightly better than 256
     (0.89 vs 0.87) but need nearly twice as many vectors (96,819 vs
     52,058); 256 stays the default.
  4. **Known-drift check passes:** the event set (COVID-19, vaccine,
     Ukraine, …) changes more than the random sample (89% vs 54% of the
     text by 2022) and its chunk scores are twice as high, rising sharply
     in 2020.
  5. **Lesson for drift detection: edited documents need a paired test.**
     The archive-level MMD test (Agent B's technique) is built for new,
     independent documents each period, as in HuffPost. Here the same
     articles appear on both sides, so the test breaks: on article vectors
     MMD² is negative with p = 1.0 in every year; on chunks it is mostly
     not significant, because about half of the chunks are unchanged
     copies. For documents that are edited rather than replaced, drift has
     to be measured per document against its own earlier version, as
     above.
- **Reproduce:** `python scripts/fetch_wikipedia.py` then
  `python scripts/longform_drift.py` (tables in `runs/longform/`;
  about 4 minutes the first time, for 74k embeddings).
- **Expectation:** truncation misses changes and averaging dilutes them
  (yes, both). Not expected: the archive-level test does not apply to
  edited documents.
- **Verdict:** Supports chunking at about 256 tokens and per-document
  comparison for updated documents.
- **Follow-up:** ordinary-year bands and the number-edit blind spot in F17.

## F17. Wikipedia against an ordinary year, and the blind spot: changed numbers barely move the vectors
- **Date:** 2026-11-06
- **Setup:** each article also captured at the end of 2016, so 2016 → 2017
  is the "ordinary year" of editing (as D11 does for HuffPost). Each year
  is compared with the year before (chunks of about 256 tokens). Bands:
  above the 95th / 99th percentile of the ordinary year's article scores =
  medium / high. Blind spot: an edited chunk counts as "number only" when
  every new or rewritten sentence in it differs from an old sentence only
  in its numbers.
- **Result, year on year:**

  | | Ordinary year (2017) | 2018 | 2019 | **2020** | 2021 | 2022 |
  |---|---|---|---|---|---|---|
  | Random sample: × ordinary (median) | 1.0 | 0.9 | 0.8 | 1.0 | 1.0 | 0.9 |
  | Random sample: medium or high | 5% | 5% | 1% | 5% | 3% | 9% |
  | Event set: × ordinary (median) | 1.2 | 0.8 | 1.1 | **4.3** | 3.4 | 3.0 |
  | Event set: medium or high | 20% | 18% | 18% | **45%** | 17% | 17% |

  The random sample changes at an ordinary pace every year (about 5% above
  the medium line, as expected by construction). The event articles jump
  in 2020 (4.3× an ordinary year) and stay high: the known-drift check
  passes with the year-on-year view too. The amount of editing is steady
  (about 55% of chunks unchanged each year, 56% in the ordinary year);
  what changes is *what* gets edited.
- **Result, blind spot:**

  | Edited chunks | Chunks | Median change | Share in the least-changed quarter |
  |---|---|---|---|
  | Changed only by numbers | 100 | 0.0008 | 62% |
  | Other edits | 21,107 | 0.0127 | 25% |

  A chunk whose only edit is a number moves about 16 times less than a
  typical edited chunk. Examples: "the Black Death … killed over 75
  million people" → "over 20 million" (chunk change 0.006); "from 1347 to
  1453" → "1347 to 1353" (0.000); bitcoin "over 4,000 altcoins" → "over
  6,000" (0.007). Some number edits are only formatting ("2217" →
  "2,217", "80s" → "1980s") and should be ignored; the embedding cannot
  tell those apart from a changed fact.
- **Reproduce:** `python scripts/fetch_wikipedia.py` then
  `python scripts/longform_drift.py` (`runs/longform/year_on_year*.csv`,
  `number_edits.csv`, `number_edit_examples.csv`).
- **Expectation:** event articles above an ordinary year in 2020 (yes);
  number edits nearly invisible to embeddings (yes).
- **Verdict:** Supports. **Implication:** embedding drift cannot be the
  only guard for facts that change by number (tolls, counts, dates); the
  answer-level check in Agent D (does the answer change between the old
  and new version?) is needed, and a cheap text check for changed numbers
  could flag such edits for re-indexing.

## F18. The changed-facts check: about one article in five changes a fact each year, almost always unseen by embedding drift
- **Date:** 2026-11-06
- **Setup:** the D27 check on the Wikipedia set, each year vs the year
  before (2017 → 2022). A changed sentence is matched with the most
  similar sentence of the previous version in the same section (word
  overlap ≥ 0.6); it is a changed fact when a number was *replaced*.
  Ignored: numbers only added (new context) or only removed (dropped
  citations), formatting ("2217" vs "2,217"), citation lines, image
  captions, fragments, and cleaning leftovers.
- **First version failed the hand check:** of 12 sampled flags, about 2
  were real; the rest were dropped ISBNs, added context ("In 2016, a
  study…"), "seven" → "7" and markup leftovers. The rules above came from
  that check (tests in `tests/test_facts.py`).
- **Result:** 505 changed facts in 97 of 165 articles over five years.

  | | 2018 | 2019 | 2020 | 2021 | 2022 |
  |---|---|---|---|---|---|
  | Random sample: articles with a changed fact | 18% | 23% | 25% | 22% | 20% |
  | … of which the embedding band is "low" | 18% | 23% | 25% | 22% | 18% |
  | Event set: articles with a changed fact | 9% | 45% | 55% | 58% | 50% |
  | … of which the embedding band is "low" | 9% | 36% | 18% | 50% | 50% |

  In the random sample, **almost every article whose facts changed looks
  ordinary to embedding drift**. Event articles changed facts most from
  2019; in 2020 most were caught anyway because they also changed a lot in
  other ways, but in 2021–22 half had changed facts with a "low" band.
- **Precision (my reading of 25 random flags, to be confirmed by the
  author's blind labels):** about 23 of 25 are real changed facts, e.g.
  malaria "228 million cases … 405,000 deaths" → "229 million … 409,000";
  tuberculosis deaths 1.6 → 1.5 million; Paris Agreement signatories 194
  → 193; Germany's president Gauck (2012) → Steinmeier (2017); Catholic
  Church members 1.27 → 1.29 billion. Misses: a changed maths example
  ("1/5" → "2/5") and a rearranged sentence.
- **Reproduce:** `python scripts/longform_drift.py`
  (`runs/longform/fact_changes.csv`, `fact_changes_summary.csv`,
  `fact_changes_hand_check.csv`); 20 flags go into the hand-label sheet
  (`scripts/label_sheet.py make`, part "facts").
- **Expectation:** changed facts would mostly sit in low embedding bands
  (yes, almost all).
- **Verdict:** Supports D27's detection step. Usefulness is not yet shown:
  Agent D must show that flagged articles give changed answers.
