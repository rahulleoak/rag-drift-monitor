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
