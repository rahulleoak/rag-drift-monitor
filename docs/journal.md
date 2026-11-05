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
