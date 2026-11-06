"""The simulator: stands in for production traffic (not an agent).

Writes the golden evaluation set from sampled archive articles: each
question has a reference answer and its article as the known source
(D9, D16, D17).

Flow:
    1. **Select** articles from every period (baseline included, as the
       control), skipping templated columns ("HuffPost Rise", "Funniest
       Tweets", ...) and very short records.
    2. **Write** (Sonnet): one or two single-fact questions per article (a
       second only for a time-sensitive fact), each with a direct reference
       answer and a type label; or, for a teaser with no stated fact, the
       question a journalist would naturally ask, flagged as not answerable.
    3. **Check in code:** length, copying the headline or the description,
       reference answers that describe the article, near-duplicates.
    4. **Review** (Opus): seven yes/no checks per question, and its own
       evergreen / time-sensitive label (final; disagreement with the writer
       is recorded, not a rejection).
    5. **Decide in code:**
       - answerable questions are kept only if all eight checks pass;
       - unanswerable questions become a separate **unanswerable set**
         (does the assistant admit the archive cannot answer, instead of
         guessing?) if the reviewer agrees they are not answerable and they
         are clear, single and natural.

Every question, check and reason is kept in the output table; keep rates
per section and period are reported so any bias in what survives is visible.
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
from driftmon.prompts import (QUESTION_SYSTEM, QUESTION_TEMPLATE, QUESTION_VERSION, REVIEW_CHECKS,
                              REVIEW_SYSTEM, REVIEW_TEMPLATE, REVIEW_VERSION, DraftQuestion,
                              QuestionDraft, QuestionReview)

_WORD = re.compile(r"[a-z0-9']+")
_META = re.compile(r"^\s*(the|this) (article|description|headline|story)\b|\bthe article (says|states)",
                   re.IGNORECASE)
UNANSWERABLE_REFERENCE = "Not stated in the article."
# checks an unanswerable question must pass (besides the reviewer agreeing it is not answerable)
# An unanswerable question is kept if the reviewer agrees the article does not answer it and it
# asks one thing. It may be broad ("how is debt split in a divorce?") and restate the headline:
# that is the point of the test, so "unambiguous" and "natural" are not required (D17).
UNANSWERABLE_CHECKS = ("single_question",)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


# ---------- 1. selection ----------

def eligible_articles(docs: pd.DataFrame, exclude_patterns: Sequence[str], min_words: int) -> pd.Series:
    """True for articles a question can be written from."""
    pattern = re.compile("|".join(exclude_patterns), re.IGNORECASE) if exclude_patterns else None
    templated = docs["title"].str.contains(pattern) if pattern else pd.Series(False, index=docs.index)
    return (~templated) & (docs["words"] >= min_words)


def revisable_fact_pool(docs: pd.DataFrame, patterns: Sequence[str]) -> pd.Series:
    """True for articles whose headline or description suggests a revisable fact
    (a toll still being counted, an ongoing situation, current rules), D20.

    Used to oversample time-sensitive questions, which random sampling rarely
    yields (F9). Such questions are always reported as their own group.
    """
    if not patterns:
        return pd.Series(False, index=docs.index)
    pattern = re.compile("|".join(patterns), re.IGNORECASE)
    return (docs["title"] + " " + docs["text"]).str.contains(pattern)


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

def copies_text(question: str, source: str, n: int) -> bool:
    """True if the question repeats any run of ``n`` consecutive words of ``source``."""
    q, t = _words(question), _words(source)
    grams = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    return any(tuple(q[i:i + n]) in grams for i in range(len(q) - n + 1))


def copies_headline(question: str, title: str, n: int = 4) -> bool:
    """True if the question repeats ``n`` consecutive headline words."""
    return copies_text(question, title, n)


def headline_overlap(question: str, title: str) -> float:
    """Share of the question's content words that also appear in the headline."""
    q = set(lex.tokenize(question))
    return len(q & set(lex.tokenize(title))) / len(q) if q else 0.0


def check_question(item: DraftQuestion, title: str, text: str, answerable: bool,
                   limits: Mapping[str, Any]) -> list[str]:
    """Reasons to reject one drafted question before review (empty list = passes)."""
    reasons = []
    n = len(item.question.split())
    if n < limits["min_words"] or n > limits["max_words"]:
        reasons.append(f"length {n} words")
    if copies_text(item.question, title, int(limits["copy_ngram"])):
        reasons.append("copies the headline")
    if copies_text(item.question, text, int(limits["copy_ngram_description"])):
        reasons.append("copies the description")
    if headline_overlap(item.question, title) > float(limits["max_headline_overlap"]):
        reasons.append("mostly headline words")
    if answerable and _META.search(item.reference_answer):
        reasons.append("reference describes the article")
    return reasons


# ---------- requests ----------

def _fields(a: Mapping[str, Any]) -> dict[str, str]:
    return {"date": str(a["date"]), "category": a["category"], "title": a["title"], "text": a["text"]}


def question_request(article: Mapping[str, Any], cfg: Mapping[str, Any]) -> LLMRequest:
    job = cfg["llm"]["jobs"]["questions"]
    return LLMRequest(job="questions", model=cfg["llm"]["generator_model"], system=QUESTION_SYSTEM,
                      prompt=QUESTION_TEMPLATE.format(**_fields(article)), schema=QuestionDraft,
                      prompt_version=QUESTION_VERSION, effort=job["effort"], max_tokens=int(job["max_tokens"]))


def review_request(article: Mapping[str, Any], item: DraftQuestion, cfg: Mapping[str, Any],
                   replicate: int = 0) -> LLMRequest:
    job = cfg["llm"]["jobs"]["question_review"]
    return LLMRequest(job="question_review", model=cfg["llm"]["judge_model"], system=REVIEW_SYSTEM,
                      prompt=REVIEW_TEMPLATE.format(**_fields(article), question=item.question,
                                                    reference_answer=item.reference_answer,
                                                    question_type=item.question_type),
                      schema=QuestionReview, prompt_version=REVIEW_VERSION, effort=job["effort"],
                      max_tokens=int(job["max_tokens"]), replicate=replicate)


def review_passes(review: QuestionReview, answerable: bool) -> list[str]:
    """Checks one review fails (empty list = this review accepts the question)."""
    if answerable:
        return [f"not {c}" for c in REVIEW_CHECKS if not getattr(review, c)]
    failed = [f"not {c}" for c in UNANSWERABLE_CHECKS if not getattr(review, c)]
    if review.answerable:
        failed.append("finds it answerable")
    return failed


def type_group(writer: str, reviewers: Sequence[str]) -> str:
    """'time_sensitive' or 'evergreen' if the writer and every review agree, else 'disputed' (D19)."""
    labels = {writer, *reviewers}
    return labels.pop() if len(labels) == 1 else "disputed"


# ---------- 2-5. the whole flow ----------

def generate_questions(cfg: Mapping[str, Any], articles: pd.DataFrame, llm: LLMClient,
                       embedder: Embedder, batch: bool | None = None) -> pd.DataFrame:
    """Write, check, review and decide.

    Returns one row per drafted question (or per article whose writing
    failed), with ``set`` = "answerable", "unanswerable" or "" (rejected) and
    the reasons for every rejection.
    """
    limits = cfg["simulation"]["question_checks"]
    max_q = int(cfg["simulation"].get("max_questions_per_article", 2))
    arts = articles.reset_index(drop=True).to_dict("records")
    drafts = llm.complete_many([question_request(a, cfg) for a in arts], batch=batch)

    # one row per drafted question
    rows: list[dict[str, Any]] = []
    for a, d in zip(arts, drafts):
        base = {"doc_id": a["doc_id"], "period": a["period"], "category": a["category"],
                "date": a["date"], "title": a["title"], "text": a["text"]}
        if not d.ok:
            rows.append({**base, "question_id": f"{a['doc_id']}-q0", "item": None, "answerable": None,
                         "reasons": [f"writer: {d.error}"]})
            continue
        items = d.parsed.questions[:max_q] or []
        if not items:
            rows.append({**base, "question_id": f"{a['doc_id']}-q0", "item": None, "answerable": None,
                         "reasons": ["writer: no question"]})
        answerable = d.parsed.answerable_from_article
        for k, item in enumerate(items[:1] if not answerable else items):
            rows.append({**base, "question_id": f"{a['doc_id']}-q{k}", "item": item, "answerable": answerable,
                         "reasons": check_question(item, a["title"], a["text"], answerable, limits)})

    # near-duplicate questions (embedding cosine), among questions still in play
    live = [i for i, r in enumerate(rows) if not r["reasons"]]
    if len(live) > 1:
        vecs = embedder.embed_queries([rows[i]["item"].question for i in live])
        for i, dup in zip(live, cleaning.near_duplicates(vecs, float(limits["duplicate_cosine"]))):
            if dup:
                rows[i]["reasons"].append("near-duplicate question")

    # review the survivors, independently n times; code decides from the yes/no checks (D19)
    n_rev = int(cfg["simulation"].get("reviews_per_question", 1))
    live = [i for i, r in enumerate(rows) if not r["reasons"]]
    pairs = [(i, k) for i in live for k in range(n_rev)]
    results = llm.complete_many([review_request(rows[i], rows[i]["item"], cfg, replicate=k)
                                 for i, k in pairs], batch=batch)
    reviews: dict[int, list] = {}
    for (i, _), res in zip(pairs, results):
        reviews.setdefault(i, []).append(res)

    out = []
    for i, r in enumerate(rows):
        item: DraftQuestion | None = r["item"]
        rec = {k: r[k] for k in ("question_id", "doc_id", "period", "category", "date", "title", "text")}
        rec.update(question=item.question if item else "", reference_answer=item.reference_answer if item else "",
                   question_type="", writer_type=item.question_type if item else "", review_types="",
                   type_group="", type_agreed=None, reviews_agree=None,
                   writer_answerable=r["answerable"], **{f"rv_{c}": None for c in REVIEW_CHECKS},
                   review_reason="")
        reasons = r["reasons"]
        kept_as = ""
        revs = reviews.get(i, [])
        if any(not rv.ok for rv in revs):
            reasons += [f"review: {rv.error}" for rv in revs if not rv.ok]
        elif revs:
            parsed: list[QuestionReview] = [rv.parsed for rv in revs]
            first = parsed[0]
            labels = [p.question_type for p in parsed]
            # label: majority of writer + reviewers (first reviewer breaks ties); the group says
            # whether everyone agreed (D19). Disagreement is recorded, never a rejection (D17).
            votes = [item.question_type, *labels]
            majority = max(("time_sensitive", "evergreen"), key=lambda t: (votes.count(t), t == labels[0]))
            fails = [review_passes(p, bool(r["answerable"])) for p in parsed]
            rec.update({f"rv_{c}": getattr(first, c) for c in REVIEW_CHECKS},
                       review_reason=" | ".join(p.reason for p in parsed),
                       question_type=majority, review_types=",".join(labels),
                       type_group=type_group(item.question_type, labels),
                       type_agreed=first.question_type == item.question_type,
                       reviews_agree=len({not f for f in fails}) == 1)
            if all(not f for f in fails):  # kept only if every review accepts it
                kept_as = "answerable" if r["answerable"] else "unanswerable"
            else:
                for k, f in enumerate(fails, start=1):
                    reasons += [f"review {k}: {x}" for x in f]
        rec["set"] = kept_as
        rec["rejected_by"] = "; ".join(reasons)
        out.append(rec)
    return pd.DataFrame(out)


def keep_rates(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group: articles, questions drafted, kept answerable, kept unanswerable, keep rate."""
    g = table.groupby(by)
    return pd.DataFrame({
        "articles": g["doc_id"].nunique(),
        "questions": g.size(),
        "answerable": g["set"].apply(lambda s: int((s == "answerable").sum())),
        "unanswerable": g["set"].apply(lambda s: int((s == "unanswerable").sum())),
        "keep_rate": g["set"].apply(lambda s: round(float((s != "").mean()), 2)),
    })


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
