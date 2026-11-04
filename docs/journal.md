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
