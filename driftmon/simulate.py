"""The simulator: stands in for production traffic (not an agent).

Writes the golden evaluation set: one question per sampled archive article,
with a reference answer and the article as its known source (D9, D16).

Flow for each article:
    1. **Select** articles from every period (baseline included, as the
       control), skipping templated columns ("HuffPost Rise", "Funniest
       Tweets", ...) and very short records, which have no single answer.
    2. **Write** (Sonnet): question, reference answer, evergreen or
       time-sensitive, answerable from the article or not.
    3. **Check in code:** length, copying the headline, the writer's own
       "not answerable" flag, near-duplicate questions.
    4. **Review** (Opus) the drafts that passed: yes/no checks
       (answerable, unambiguous, needs the article, reference correct,
       natural).
    5. **Decide in code:** keep a question only if every check passes.

Every decision and its reason is kept in the output table, so rejected
questions can be inspected.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from driftmon import cleaning
from driftmon.drift import embedding as emb
from driftmon.drift import lexical as lex
from driftmon.embeddings import Embedder
from driftmon.llm import LLMClient, LLMRequest
from driftmon.prompts import (QUESTION_SYSTEM, QUESTION_TEMPLATE, QUESTION_VERSION, REVIEW_SYSTEM,
                              REVIEW_TEMPLATE, REVIEW_VERSION, QuestionDraft, QuestionReview)

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


# ---------- 1. selection ----------

def eligible_articles(docs: pd.DataFrame, exclude_patterns: Sequence[str], min_words: int) -> pd.Series:
    """True for articles a question can be written from."""
    pattern = re.compile("|".join(exclude_patterns), re.IGNORECASE) if exclude_patterns else None
    templated = docs["title"].str.contains(pattern) if pattern else pd.Series(False, index=docs.index)
    return (~templated) & (docs["words"] >= min_words)


def sample_articles(docs: pd.DataFrame, counts: Mapping[str, int], seed: int) -> pd.DataFrame:
    """A random sample of ``counts[period]`` eligible articles from each period."""
    parts = []
    for k, (period, n) in enumerate(counts.items()):
        pool = docs[docs["period"] == period]
        if len(pool) < n:
            raise ValueError(f"period {period!r} has {len(pool)} eligible articles, fewer than {n}")
        parts.append(pool.sample(n=n, random_state=seed + k))
    return pd.concat(parts)


# ---------- 3. code checks ----------

def copies_headline(question: str, title: str, n: int = 4) -> bool:
    """True if the question repeats any run of ``n`` consecutive headline words."""
    q, t = _words(question), _words(title)
    grams = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    return any(tuple(q[i:i + n]) in grams for i in range(len(q) - n + 1))


def headline_overlap(question: str, title: str) -> float:
    """Share of the question's content words that also appear in the headline."""
    q = set(lex.tokenize(question))
    return len(q & set(lex.tokenize(title))) / len(q) if q else 0.0


def check_draft(draft: QuestionDraft, title: str, limits: Mapping[str, Any]) -> list[str]:
    """Reasons to reject a draft before review (empty list = passes)."""
    reasons = []
    n = len(draft.question.split())
    if n < limits["min_words"] or n > limits["max_words"]:
        reasons.append(f"length {n} words")
    if copies_headline(draft.question, title, int(limits["copy_ngram"])):
        reasons.append("copies the headline")
    if headline_overlap(draft.question, title) > float(limits["max_headline_overlap"]):
        reasons.append("mostly headline words")
    if not draft.answerable_from_article:
        reasons.append("writer: not answerable")
    return reasons


# ---------- requests ----------

def question_request(article: Mapping[str, Any], cfg: Mapping[str, Any]) -> LLMRequest:
    job = cfg["llm"]["jobs"]["questions"]
    return LLMRequest(job="questions", model=cfg["llm"]["generator_model"], system=QUESTION_SYSTEM,
                      prompt=QUESTION_TEMPLATE.format(**_fields(article)), schema=QuestionDraft,
                      prompt_version=QUESTION_VERSION, effort=job["effort"], max_tokens=int(job["max_tokens"]))


def review_request(article: Mapping[str, Any], draft: QuestionDraft, cfg: Mapping[str, Any]) -> LLMRequest:
    job = cfg["llm"]["jobs"]["question_review"]
    return LLMRequest(job="question_review", model=cfg["llm"]["judge_model"], system=REVIEW_SYSTEM,
                      prompt=REVIEW_TEMPLATE.format(**_fields(article), question=draft.question,
                                                    reference_answer=draft.reference_answer),
                      schema=QuestionReview, prompt_version=REVIEW_VERSION, effort=job["effort"],
                      max_tokens=int(job["max_tokens"]))


def _fields(a: Mapping[str, Any]) -> dict[str, str]:
    return {"date": str(a["date"]), "category": a["category"], "title": a["title"], "text": a["text"]}


# ---------- 2-5. the whole flow ----------

CHECKS = ("answerable", "unambiguous", "needs_article", "reference_correct", "natural")


def generate_questions(cfg: Mapping[str, Any], articles: pd.DataFrame, llm: LLMClient,
                       embedder: Embedder, batch: bool | None = None) -> pd.DataFrame:
    """Write, check, review and decide; one row per article with every decision."""
    limits = cfg["simulation"]["question_checks"]
    rows = articles.reset_index(drop=True).to_dict("records")
    drafts = llm.complete_many([question_request(a, cfg) for a in rows], batch=batch)

    reasons: list[list[str]] = []
    for a, d in zip(rows, drafts):
        reasons.append(check_draft(d.parsed, a["title"], limits) if d.ok else [f"writer: {d.error}"])

    # near-duplicate questions (embedding cosine), among drafts still in play
    live = [i for i, r in enumerate(reasons) if not r]
    if len(live) > 1:
        vecs = embedder.embed_queries([drafts[i].parsed.question for i in live])
        for i, dup in zip(live, cleaning.near_duplicates(vecs, float(limits["duplicate_cosine"]))):
            if dup:
                reasons[i].append("near-duplicate question")

    # review the survivors; code decides from the reviewer's yes/no checks
    live = [i for i, r in enumerate(reasons) if not r]
    reviews = dict(zip(live, llm.complete_many(
        [review_request(rows[i], drafts[i].parsed, cfg) for i in live], batch=batch)))
    out = []
    for i, (a, d) in enumerate(zip(rows, drafts)):
        rec = {"doc_id": a["doc_id"], "period": a["period"], "category": a["category"],
               "date": a["date"], "title": a["title"], "text": a["text"],
               "question": d.parsed.question if d.ok else "",
               "reference_answer": d.parsed.reference_answer if d.ok else "",
               "question_type": d.parsed.question_type if d.ok else "",
               **{c: None for c in CHECKS}, "review_reason": ""}
        rv = reviews.get(i)
        if rv is not None and not rv.ok:
            reasons[i].append(f"review: {rv.error}")
        elif rv is not None:
            rec.update({c: getattr(rv.parsed, c) for c in CHECKS}, review_reason=rv.parsed.reason)
            reasons[i] += [f"review: not {c}" for c in CHECKS if not getattr(rv.parsed, c)]
        rec["kept"] = not reasons[i]
        rec["rejected_by"] = "; ".join(reasons[i])
        out.append(rec)
    return pd.DataFrame(out)


def diversity(questions: Sequence[str], sources: np.ndarray, embedder: Embedder) -> dict[str, float]:
    """Spread of the kept questions vs the spread of their source articles (P3).

    Generated questions can collapse into generic phrasing; if the questions
    are much less spread out than the articles they came from, that shows.
    """
    if len(questions) < 2:
        return {"questions": float("nan"), "articles": float("nan"), "ratio": float("nan")}
    q = emb.dispersion(embedder.embed_queries(list(questions)))
    a = emb.dispersion(sources)
    return {"questions": q, "articles": a, "ratio": q / a if a else float("nan")}
