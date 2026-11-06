# Project Journal

Short dated entries, **newest first**. One entry per working session at
minimum, added before stopping. Plain language.

## Entry template

```markdown
## YYYY-MM-DD: <one-line summary>
- **Built:** what now exists or works.
- **Tried and abandoned:** what, and why.
- **Surprises:** anything unexpected in the data, tools or results.
- **Open questions:** what is unresolved and who or what can resolve it.
- **Next:** the first thing to do in the next session.
```

## Entries

## 2026-11-06 (later): Vector store checked against the brief
- **Checked:** the brief lists FAISS as a vector database option; D22
  records why FAISS (exact search) stays and what it is precisely.
- **Planned for Agent C:** save the FAISS indexes to `indexes/` with a
  manifest and reload them when embeddings are unchanged (currently rebuilt
  in memory each run). `indexes/` added to .gitignore.
- **Storage today:** raw data 84 MB, cleaned articles 9 MB (parquet),
  embeddings 120 MB (numpy cache), LLM replies 1 MB (JSON), runs 2.5 MB;
  all local and gitignored except `docs/figures/`.

## 2026-11-06 (later): Is rejecting "copied" wording justified? Measured: no
- **Checked:** F11. Copy-rejected questions retrieve their source article
  at rank 1 79% of the time, kept ones 91%: the rejection does not prevent
  easy matches. All questions find their own article easily (circularity,
  Q8).
- **Analysis snippet (reproduces F11):** embed each pilot question with
  `Embedder.embed_queries`, search a `VectorIndex` over all cleaned
  documents for the top 100, and record the source article's rank; group by
  kept / rejected "not natural" / rejected otherwise.
- **Recorded:** extensions to do only if ahead (requirements-coverage.md);
  storage choices stated in architecture.md (FAISS in memory, files; no
  database server).
- **Proposed next:** reviewer r-v5 where "natural" no longer counts reused
  names and facts as copying (code checks already catch verbatim copying);
  then the full question set.

## 2026-11-06 (later): Oversample pilot; knowledge-graph suggestion reviewed
- **Built:** `simulate.revisable_fact_pool` and `--oversample` (D20); every
  question tagged with its sample pool. 1 new test (76 in total).
- **Results:** F10. Clear time-sensitive questions 4 of 9 kept (vs 1 of 13
  random); keep rate 35%; $0.60; $1.87 spent in total.
- **Reviewed:** a colleague's suggestion to model relationship change with
  OWL/RDF, Neo4j and Protégé. Not adopted; reasons and a lightweight
  alternative recorded in D21.
- **Next:** generate the full question set (random + oversample) as a batch.

## 2026-11-06: Double review and label groups tested
- **Built:** independent repeat requests (`LLMRequest.replicate`, cache-safe),
  two reviews per question with "kept only if both accept", three label
  groups (clear evergreen, clear time-sensitive, disputed), reviewer
  consistency in the run summary (D19). 2 new tests (75 in total).
- **Results:** F9. Reviewer consistency 92% (decisions), 100% (labels).
  Only one decision changed. Clear time-sensitive questions: 1 of 13 kept.
  $1.27 spent in total.
- **Correction:** the v3→v4 flips were mostly the prompt change, not
  randomness; recorded in F9.
- **Open:** time-sensitive questions are too scarce for their own analysis;
  an oversample of "breaking news" articles is proposed.
- **Next:** decide the oversample, then generate the ~180-question set.

## 2026-11-05 (late night): Question reviewer version 4; prompt tuning stops
- **Built:** reviewer r-v4 assigns the type label itself (without seeing
  the writer's); disagreement recorded as `type_agreed`; unanswerable
  questions kept if the reviewer agrees and they ask one thing. Agreement
  rate shown in every run. Tests updated (73 pass).
- **Results:** F8 updated. 60% kept (13 answerable + 2 unanswerable); label
  agreement 83%; $0.25 for reviews only; $1.03 spent in total.
- **Recorded:** D18, the judge cannot search or learn; its built-in
  knowledge is a stated limitation (the author's question).
- **Next:** the ~180-question set, sampled with extra weight on flagged
  sections and controls, as a batch (about $3–6).

## 2026-11-05 (late night): Question writer versions 2 and 3
- **Built:** prompts q-v2/r-v2 then q-v3/r-v3 (D17): one fact per question,
  up to two per article, direct reference answers, sharper time-sensitive
  rule, description copy check, stricter reviewer with eight checks, an
  unanswerable set, keep rates per section and period. Tests updated
  (73 pass).
- **Results:** F8. Keep rate 75% → 38% → 36%; time-sensitive questions now
  appear; unanswerable set still empty. Spent $0.78 in total.
- **Surprises:** Opus is inconsistent on the evergreen / time-sensitive label
  for the same question across runs; "natural" rejects unanswerable
  questions for restating the headline.
- **Next (proposed to the author):** reviewer assigns the type label;
  relax the unanswerable acceptance; rerun only the reviews (about $0.20);
  then sample the ~180-question set weighted towards flagged and control
  sections.

## 2026-11-05 (night): First live LLM run: the question pilot (checkpoint 3 of 3)
- **Fixed first:** the shell defines `ANTHROPIC_API_KEY` as an empty
  string, which hid the `.env` key (python-dotenv never overrides an
  existing variable). `llm.api_key()` now falls back to `.env` when the
  environment value is empty, without changing the environment. Test added.
  Nothing was spent by the failed attempt.
- **Pilot:** F7. 15 of 20 questions kept for $0.18. First real spend.
- **Surprises:** zero time-sensitive questions; questions echo the
  description; the reviewer is lenient; outside facts leak into questions;
  many two-part questions.
- **Proposed next (to agree with the author):** prompt and checks version 2:
  one fact per question; time-sensitive questions where the article
  supports one; no facts that are not in the article; the copy check
  extended to the description; stricter reviewer checks (no outside facts,
  single question); then a second 20-article pilot (about $0.20) and the
  author's review of the questions.

## 2026-11-05 (night): Question generator built (checkpoint 2 of 3)
- **Built:** `driftmon/prompts.py` (writer and reviewer prompts with
  versions and schemas), `driftmon/simulate.py` (select, write, check,
  review, decide), `scripts/generate_questions.py` (with `--dry-run`),
  simulation settings in the config (D16). 4 new tests (72 in total).
- **Dry run:** 20 pilot articles across all six periods; estimated cost
  $0.51 (a cautious upper bound); nothing spent.
- **Next:** checkpoint 3, the live pilot, then review the questions with the
  author before generating more.

## 2026-11-05 (night): LLM layer built (checkpoint 1 of 3)
- **Context:** the author set a $150 total budget ($20 loaded now) and asked
  to start the LLM work on a small scale while waiting for the stakeholder
  answers: a discovery run to find what was not accounted for. Plan in three
  committed checkpoints so an interruption loses nothing: (1) LLM layer,
  (2) question generator, checks and Opus review, (3) the 20-question pilot.
- **Built:** `driftmon/llm.py` (D15) and its config (budget, prices,
  per-job effort). 7 new tests with a fake API (68 in total). No money
  spent yet.
- **Checked against current docs:** structured output via
  `output_config.format`, server-side fallback (`fallbacks: "default"`,
  beta `server-side-fallback-2026-07-01`; not available on the Batch API),
  no temperature, Batch API request shapes; the installed SDK (2.4.0)
  accepts all of them.
- **Next:** checkpoint 2, the question generator with its quality checks
  and the Opus review.

## 2026-11-05 (late night): Charts and dashboard in the Rivvun style
- **Checked:** the charts and dashboard commit was complete after the
  interruption (clean tree, 61 tests, all six charts present).
- **Built:** a light and a dark chart theme from rivvun.ai's design tokens
  (D14); both rendered for every chart; the dashboard now uses Rivvun's dark
  site theme (`.streamlit/config.toml`) and shows the dark charts. One more
  test (dark charts are made and found by the dashboard).
- **Process:** five palette candidates validated; two rejected (a violet
  too close to the blue; teals too close to the blue on navy). Every chart
  inspected in both themes; the light "low" severity step lightened so low
  severity recedes.
- **Note:** "Rivunn" turned out to be Rivvun AI (rivvun.ai), Seattle,
  enterprise AI for commercial obligations: the likely audience.
- **Next:** unchanged: the LLM parts once the stakeholder answers arrive.

## 2026-11-05 (night): Charts and dashboard skeleton
- **Built:** `driftmon/charts.py` and `scripts/make_charts.py` (six PNG
  charts from a saved run, copied to `docs/figures/`);
  `driftmon/dashboard_data.py` (loads saved runs) and
  `app/streamlit_app.py` (Overview, Data monitoring, Drift with drill-down,
  placeholder pages for C–E; interactive charts with hover). 4 new tests
  (61 in total), including the whole dashboard rendered headless on a run
  built from a synthetic archive.
- **Process:** colours validated with the dataviz palette validator
  (categorical slots all-pairs colour-blind safe; ordinal ramp passes);
  every chart rendered and inspected, which caught overlapping labels, a
  subtitle colliding with a legend, gridlines over bars and an empty row in
  the correlation grid. All fixed.
- **Fixed:** `make_charts.py` crashed on an empty results file (a run with
  no shifted words); found by the synthetic-run test.
- **Next:** wait for the stakeholder answers, then the LLM simulator and
  Agent C. Charts for C and D, and the baseline-candidates chart (F2), come
  later.

## 2026-11-05 (evening): Agent B finished (except query drift)
- **Built:** `driftmon/drift/topics.py` (emerging topics by growth),
  `driftmon/drift/relations.py` (term-context shift with size-matched
  ordinary shift), the P6 correlation table, all in the DriftReport and
  saved by `run_pipeline.py`. 3 new tests (57 in total); the suite also
  passes with runtime warnings treated as errors.
- **Results:** F6. Three COVID topics in 2020; everyday words change
  meaning (cases, test, home, virus); MMD vs word drift 0.68, MMD vs
  novelty 0.07.
- **Fixed:** log-odds prior summed over the wrong vocabulary (found via a
  warning in a degenerate test); term-shift ratio inflated for small groups
  ("home" 2020 9.6× → 4.6×).
- **Planned:** D13, how quality loss without drift is detected (answers
  the author's question; built with C and D).
- **Open questions:** stakeholder answers (LLM work waits); charts come
  with the dashboard.
- **Next:** PR for this branch; then the LLM simulator and Agent C once the
  answers arrive.

## 2026-11-05 (afternoon): Agents A and B running in LangGraph
- **Built:** `driftmon/monitoring.py` (Agent A statistics: volume, length,
  category mix with Cramér's V, new and renamed sections, new-term rate);
  `driftmon/drift/compare.py` (shared comparisons; `run_drift.py` now uses
  it and reproduces F4 exactly); `driftmon/drift/severity.py` (bands from
  same-month-a-year-apart pairs); `driftmon/drift/report.py` (flags,
  controls, terms, examples); typed reports in `schemas.py`; agent nodes A
  and B; `graph.py` with the routing decision; `scripts/run_pipeline.py`.
  7 new tests (54 in total), including the whole graph on a synthetic
  archive.
- **Results:** F5. 2020, 2021 and 2022 flagged; the 2020 lifestyle sections
  are the most drifted of all; politics is never flagged. The pipeline runs
  in about 60 s.
- **Surprises:** politics never looks unusual, because the baseline's own
  politics coverage swings so much (D6 weakness 2). Pandas 3 returns
  read-only arrays (an in-place `&=` failed).
- **Checked:** the flagged 2020 lifestyle sections are mostly COVID
  content, not templated columns, so D8's revisit trigger 1 does not fire.
- **Open questions:** LLM explanations for B (wait for the stakeholder
  answers); topic clusters, term-context shift and query drift (D7); the
  P6 correlation heatmap.
- **Next:** decide with the author what comes next: the remaining Agent B
  items, or the LLM simulator and Agent C once the stakeholder answers
  arrive.

## 2026-11-05 (later): Indexes and first drift results
- **Built:** `driftmon/index.py` (stale and cumulative refreshed indexes),
  `driftmon/corpus.py` (documents + cached vectors), the embedding drift
  measures (MMD with permutations, centroid shift, dispersion, novelty), the
  word drift measures (JSD with permutations and excess over shuffling,
  log-odds emerging terms, new-term rate), Benjamini–Hochberg, and
  `scripts/run_drift.py`. 11 new tests (47 in total).
- **Results:** F4. 2020 is the largest drift (5.7× an ordinary year); the
  Boiling Frog pattern shows in 2022; the techniques disagree after 2020.
- **Surprises:** COVID articles are not novel to the embedding model (their
  nearest baseline articles are about flu, Zika and Trump's health policy).
  Every comparison was significant, including 2017 vs 2016. Raw JSD was
  mostly noise; excess JSD fixed it.
- **Tried and abandoned:** raw JSD as the word-drift score.
- **Open questions (for review before building):** within-category
  comparisons (sample design), severity bands (how to build the null
  distribution), Agent A's monitoring statistics.
- **Next:** those three, then wiring Agents A and B in LangGraph.

## 2026-11-05: LLM setup chosen; presentation notes started
- **Built:** D9 (Sonnet 5.5 writes, Opus 5.5 judges; temperature removed from
  the config); `docs/presentation.md`, the running notes for the walkthrough,
  now part of the documentation list in `CLAUDE.md`.
- **Surprises:** this worktree had been switched to an old branch from Zed's
  branch picker, rolling the files on disk back to the early scaffold.
  Nothing was lost (all work was committed); switched back and re-tested.
  Two folders share the name `rag-drift-monitor`; the worktree is the one
  with `steady-ledge` in its path.
- **Open questions:** stakeholder answers (agentic definition, budget) before
  building the LLM jobs; `driftmon/llm.py` and `driftmon/prompts.py` are still
  stubs by design.
- **Next:** the two indexes and the Agent A/B drift statistics, which need no
  LLM.

## 2026-11-04 (late): Data cleaning built and run
- **Built:**
  - `driftmon/embeddings.py`: embeddings with a disk cache keyed on model and
    exact text; queries get the model's prefix; interrupted saves cannot
    misalign the cache.
  - `driftmon/cleaning.py`: the rules as plain functions; exact greedy
    near-duplicate removal that keeps the earliest copy.
  - `driftmon/periods.py`: periods from the config (now shared with the
    baseline comparison script, which had its own copy).
  - `scripts/prepare_data.py`: checks the raw file against the recorded
    fingerprint, applies the rules, writes `data/processed/documents.parquet`
    and the retention table.
  - 17 new tests (36 in total, all pass).
- **Checked:** the dataset is the one linked in the assignment and complete
  (one file, byte-for-byte the size Kaggle lists); fingerprint recorded in
  the config and F1.
- **Results:** F3. Cleaning keeps 98.2% overall; nearly all removals are in
  the baseline. Drift sample: 1,398 per period.
- **Surprises:** every rule falls almost entirely on the baseline; most
  near-duplicates are templated columns. The length gap (median 24 vs 29–31
  words) survives cleaning, so it is added as check 5 in D6.
- **Tried and abandoned:** a 0.93 duplicate threshold; it removes distinct
  articles about the same event.
- **Open questions:** rules are a first version (D8 lists when to revisit).
  Exclude surviving templated columns when generating test questions.
- **Next:** the stale and refreshed indexes (`driftmon/index.py`), then the
  first drift statistics on the drift samples.

## 2026-11-04 (night): Design checked against the assignment
- **Built:** `docs/requirements-coverage.md`, mapping every bullet of the
  original assignment to the part of the system that meets it, with a
  status. D7 records six additions; the architecture and its diagram are
  updated.
- **Surprises:**
  - The earlier notes had paraphrased "trigger model evaluation or
    retraining" as "trigger model retraining", and so set it aside. Spotted by
    the author. The evaluation half is now something the system actually
    does.
  - Two brief requirements had been marked stretch goals: shifts in semantic
    relationships (Q7) and the second embedding model (Q12). Both are now
    required.
- **Open questions:** whether to let the LLM propose what to investigate
  (waits for stakeholder question 2).
- **Next:** data cleaning (P5) on a new branch.

## 2026-11-04 (evening): Environment set up
- **Built:** `.venv` (Python 3.13) with the CUDA build of PyTorch and all of
  `requirements.txt`; the project installed in editable mode. `pip check` is
  clean and all 19 tests pass with pytest, including the 3 data-loading tests
  that could not run before. Setup steps added at the top of
  `requirements.txt`; D4 evidence updated.
- **Checked:**
  - The exploration CSVs are identical under pandas 3.1 (venv) and 2.3
    (system).
  - bge-small runs on the GPU: about 5 s to load and 1.3 s for 5,000
    headlines, so embedding the whole working set takes well under a minute.
  - A first check that retrieval works by meaning: a "covid vaccine uptake"
    query scores 0.74 against a coronavirus-vaccination headline and 0.43
    against a pasta headline.
- **Surprises:** `python` on this machine is the Microsoft Store build. The
  venv was made with the regular install (`py -3.13`) instead, because the
  Store build redirects file paths. Hugging Face's cache cannot use symlinks
  on Windows without Developer Mode; this only costs disk space.
- **Next:** data cleaning (P5) with the retention table, then set
  `sample_per_period`; then embeddings with a disk cache.

## 2026-11-04 (later): Baseline chosen: 2016–17
- **Built:**
  - `scripts/exploration/` folder for data exploration. `explore_data.py`
    moved into it; `compare_baselines.py` added. The new script compares
    candidate baselines by how closely their topic mix matches the later
    years, how stable each is internally, and how much of each later year is
    in sections the baseline does not have. 7 new tests (14 of 14 exploration
    tests pass with the stand-in runner).
  - Config: `active_dataset: huffpost`; baseline 2016–17 and later periods
    2018–2022; `categories.normalise` mapping 9 renamed sections; the
    baseline candidates under `exploration`. `sample_per_period` set to null
    until cleaning, because 2022 has only 1,398 articles.
  - F2 (the comparison, with walkthrough notes) and D6 (options, strengths,
    weaknesses, runner-ups and the checks that would reopen the choice).
    Architecture updated: Data Monitoring reports new sections as a
    structural change; Drift Detection also compares within categories.
- **Tried and abandoned:** 2014–15 (the earlier proposal), rejected on F2: a
  51% mix difference against the later years.
- **Surprises:**
  - 2015 differs by 30% between its own two halves, so the earlier runner-up
    (2015 only) was the weakest option on stability.
  - About half of the remaining gap with 2016–17 is one new section, U.S.
    NEWS (23% of 2019 articles).
- **Open questions:**
  - Re-evaluate the baseline if any check in D6 fails (author's request:
    keep an eye on its weaknesses).
  - Is U.S. NEWS mostly politics? To check once embeddings exist.
  - Go-ahead for the environment and package setup; stakeholder answers; LLM
    key (needed from about Thursday, not before).
- **Next:** after the go-ahead, set up the environment and run the full test
  suite with pytest; then cleaning (P5) and setting `sample_per_period`.

## 2026-11-04 (local machine): Exploration: HuffPost downloaded and explored
- **Built:**
  - Moved from the cloud environment to a local Windows machine (RTX 4060 Ti).
    Kaggle and Hugging Face are reachable here.
  - Downloaded HuffPost (v3, 209,527 records) into `data/raw/` with the Kaggle
    API and ran the exploration script (now `scripts/exploration/explore_data.py`).
    Results logged as F1.
  - Added a sixth exploration output: the share of each year's articles that
    mention a list of watch terms tied to known events, set in the config.
    It shows the known drift is present before any detector is built.
    2 new tests.
- **Tried and abandoned:** downloading arXiv. HuffPost passes the D3 criteria
  that matter most, and arXiv is a multi-GB file.
- **Surprises:**
  - Category labels were renamed twice (for example WELLNESS → HEALTHY
    LIVING in 2014, then back after 2018). This is label drift, not topic
    drift, and has to be normalised before categories are compared.
  - POLITICS goes from 10% of articles (2014) to 39% (2020). Against an
    early baseline, embedding drift would mostly measure this change in mix.
  - The Kaggle key was first saved into `.env.example`, which git tracks. It
    was moved into `.env` (gitignored) before any commit, and `.env.example`
    now has an empty `KAGGLE_API_KEY=` placeholder.
  - pytest and the ML packages are not installed yet. The exploration tests
    were run with a stand-in runner (7 of 7 pass); the full suite waits for
    the project environment.
- **Open questions:**
  - Baseline and period choice, given F1 (to be approved). Leading option: a
    2016–2017 baseline, whose topic mix and labels are closest to the later
    years, with 2018–2022 as the later periods, equal-size samples, and
    within-category comparisons alongside whole-period ones.
  - P1–P8 approved later the same day and recorded as D5. One change from
    the proposal: no composite score for sorting the dashboard either.
  - Stakeholder questions sent; waiting for answers. No LLM key yet.
- **Next:** once the periods are approved, set them in the config, write the
  category normalisation map, and create the project environment (needs the
  go-ahead).

## 2026-11-04 (later): Scaffold, exploration script, ideas from the paper
- **Built:**
  - The approved structure (D1): `driftmon/` package with module stubs whose
    docstrings state each module's job and inputs/outputs; `config/config.yaml`;
    `requirements.txt` (version ranges, D4); `pyproject.toml`; `.gitignore`;
    `.env.example`.
  - `docs/architecture.md` drafted with the Mermaid agent-flow diagram (checked
    to render with mermaid-cli).
  - `driftmon/data.py`: streaming loaders that map HuffPost and arXiv onto one
    record shape.
  - `scripts/explore_data.py`: prints records per year, per category per year,
    the top 20 categories and 5 seeded sample titles per year, and writes them
    as CSV to `runs/explore/<dataset>/`.
  - 10 tests on synthetic fixtures; all pass.
- **Deviation from "stubs only":** `data.py` and `config.py` contain real code
  because the exploration script needs them, and the same parsers will feed the
  pipeline. One parser, tested once, instead of a throwaway copy in the script.
- **Tried and abandoned:** looked for another reachable source for the
  datasets. Kaggle, Hugging Face, arxiv.org, Zenodo and the UCSD Amazon host
  are all blocked here; only package indexes are reachable. Third-party GitHub
  mirrors were not used: their integrity is unknown, and they are outside this
  session's repository scope.
- **Surprises:** `download.pytorch.org` is also blocked, so installing
  sentence-transformers here will pull the default (larger) torch wheel from
  PyPI. Worth knowing before the first heavy install.
- **Decisions recorded:** D1 structure, D2 retrieval ground truth (Q9),
  D3 explore HuffPost and arXiv before choosing (criteria written down first),
  D4 dependency versions.
- **Ideas from the author's paper** ("Semantic Drift in Paraphrasing") added as
  proposals P1–P8 in `docs/questions.md`. The strongest transfers:
  - P1: compare each period with the baseline as well as with the previous one.
  - P2: classify drift as lexical, semantic, or both.
  - P4: use measured quality loss to find which drift signal predicts harm,
    instead of a fixed-weight composite.
- **Open questions:** approval of P1–P8; stakeholder questions 1–5 still open.
- **Next:** once the files are in `data/raw/`, run
  `python scripts/explore_data.py --dataset huffpost` and `--dataset arxiv`.
  Log F1 in findings, then decide Q1/Q3 against the D3 criteria.

## 2026-11-04: Setup, brief review, clarifying questions
- **Built:** `docs/` folder with the brief moved into it; skeletons for
  decisions, journal, findings and architecture; `docs/questions.md` with five
  stakeholder questions, 16 design decisions with recommendations, and 12 flagged
  ambiguities. Project structure proposed in chat, awaiting approval. No
  application code yet, by design.
- **Tried and abandoned:** nothing yet.
- **Surprises:**
  - The session started on a stale copy of `CLAUDE.md` (no "Documentation as you
    go" section). The correct version was already on `main` (commit `1a9d5b0`),
    so the branch was fast-forwarded to it before any work. Nothing was lost:
    the branch had no commits of its own.
  - The original assignment (.docx) matches `docs/project-brief.md` except for
    the deadline line. It reads as a take-home assignment, so "stakeholders" are
    effectively the reviewers.
  - The dataset card says only about 10k of the 210k records are after May
    2018. If true, the COVID year is in the sparse part of the data. This shapes
    the choice of periods (equal-size samples per period).
  - The documents are headline + one to two sentence description, not full
    articles, so chunking does not apply.
  - In this cloud environment, `www.kaggle.com` and `huggingface.co` are
    blocked (no dataset download, no sentence-transformers models), no LLM API
    key is set, `api.anthropic.com` is reachable and `api.openai.com` is not.
- **Open questions:** see `docs/questions.md` (stakeholder questions 1–5;
  design decisions Q1–Q16 awaiting approval).
- **Next:** after approval, scaffold the structure (step 4), then write and run
  `scripts/explore_data.py` (step 5) once the dataset is available here.
