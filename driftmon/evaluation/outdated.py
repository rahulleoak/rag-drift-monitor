"""The outdated-answer test on edited documents (Agent D; D27's harm step, D28).

Does a changed fact flagged by the D27 check actually change the answer a
user gets? Two groups of edited Wikipedia sentences, each with a close
earlier version:
    - flagged: the changed-facts check found a replaced number;
    - comparison: numbers unchanged, only the wording moved.
For each, Sonnet writes a question from the new sentence (blind to the
group). The question is answered twice, from the top passages of the old
and the new version's chunk index (local retrieval over every article).
Opus judges whether the two answers state the same fact, without knowing
which version gave which.

The check is useful if flagged sentences change answers clearly more often
than the comparison group: by at least ``min_difference``, with Fisher's
exact test below ``alpha`` (both fixed before results). Plain code decides.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from driftmon.drift import facts
from driftmon.evaluation.stats import bootstrap_mean_ci
from driftmon.llm.client import LLMRequest
from driftmon.llm.prompts import (ANSWER_CHANGE_SYSTEM, ANSWER_CHANGE_TEMPLATE, ANSWER_CHANGE_VERSION,
                              FACT_QUESTION_SYSTEM, FACT_QUESTION_TEMPLATE, FACT_QUESTION_VERSION,
                              WIKI_ANSWER_SYSTEM, WIKI_ANSWER_TEMPLATE, WIKI_ANSWER_VERSION, AnswerChange,
                              FactQuestion, RagAnswer)

FLAGGED, COMPARISON = "flagged", "comparison"
SIDES = ("old", "new")


def edit_items(snaps: pd.DataFrame, order: Sequence[str], min_similarity: float) -> pd.DataFrame:
    """Every edited sentence with a number, between consecutive versions of
    each article, in one of the two groups."""
    by_key = {(r.page_id, r.period): r for r in snaps.itertuples(index=False)}
    rows = []
    for pid in sorted({r.page_id for r in snaps.itertuples(index=False)}):
        for prev, cur in zip(order, order[1:]):
            a, b = by_key.get((pid, prev)), by_key.get((pid, cur))
            if a is None or b is None:
                continue
            old, new = json.loads(a.sections), json.loads(b.sections)
            base = {"page_id": pid, "title": b.title, "event": bool(b.event), "previous": prev, "period": cur}
            for c in facts.fact_changes(old, new, min_similarity):
                rows.append({**base, "group": FLAGGED, "section": c.section, "before": c.before, "after": c.after})
            for section, before, after, _ in facts.wording_edits(old, new, min_similarity):
                rows.append({**base, "group": COMPARISON, "section": section, "before": before, "after": after})
    return pd.DataFrame(rows)


def sample_items(items: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` items per group, at random; item ids are stable for a seed."""
    out = [g.sample(n=min(n, len(g)), random_state=seed) for _, g in items.groupby("group")]
    s = pd.concat(out, ignore_index=True)
    s.insert(0, "item_id", [f"{g[0].upper()}{i:03d}" for i, g in enumerate(s["group"])])
    return s


def question_requests(items: pd.DataFrame, cfg: Mapping[str, Any]) -> list[LLMRequest]:
    """One Sonnet request per item: a question from the new sentence. The
    writer never sees the old sentence or the group."""
    job = cfg["llm"]["jobs"]["fact_questions"]
    return [LLMRequest(job="fact_questions", model=cfg["llm"]["generator_model"], system=FACT_QUESTION_SYSTEM,
                       prompt=FACT_QUESTION_TEMPLATE.format(title=r["title"], section=r["section"],
                                                            sentence=r["after"]),
                       schema=FactQuestion, prompt_version=FACT_QUESTION_VERSION, effort=job["effort"],
                       max_tokens=int(job["max_tokens"])) for r in items.to_dict("records")]


def passage_listing(chunks: pd.DataFrame) -> str:
    """Retrieved passages as the assistant sees them: lettered, in rank order."""
    return "\n\n".join(f"[{ascii_uppercase[i]}] {r['title']} - {r['section']}: {r['text']}"
                       for i, r in enumerate(chunks.to_dict("records")))


def answer_requests(questions: Sequence[str], passages: Sequence[str], cfg: Mapping[str, Any]
                    ) -> list[LLMRequest]:
    """One Sonnet request per (question, version's passages)."""
    job = cfg["llm"]["jobs"]["wiki_answers"]
    return [LLMRequest(job="wiki_answers", model=cfg["llm"]["generator_model"], system=WIKI_ANSWER_SYSTEM,
                       prompt=WIKI_ANSWER_TEMPLATE.format(question=q, passages=p), schema=RagAnswer,
                       prompt_version=WIKI_ANSWER_VERSION, effort=job["effort"], max_tokens=int(job["max_tokens"]))
            for q, p in zip(questions, passages)]


def change_requests(item_ids: Sequence[str], questions: Sequence[str], old: Sequence[str], new: Sequence[str],
                    cfg: Mapping[str, Any]) -> list[LLMRequest]:
    """One Opus request per item: do the two answers state the same fact?
    The order of the two answers is shuffled per item; versions are not named."""
    job = cfg["llm"]["jobs"]["answer_change"]
    rng = np.random.default_rng(int(cfg["seed"]))
    flips = rng.random(len(item_ids)) < 0.5
    out = []
    for q, a, b, flip in zip(questions, old, new, flips):
        first, second = (b, a) if flip else (a, b)
        out.append(LLMRequest(job="answer_change", model=cfg["llm"]["judge_model"], system=ANSWER_CHANGE_SYSTEM,
                              prompt=ANSWER_CHANGE_TEMPLATE.format(question=q, answer_1=first, answer_2=second),
                              schema=AnswerChange, prompt_version=ANSWER_CHANGE_VERSION, effort=job["effort"],
                              max_tokens=int(job["max_tokens"])))
    return out


def contains_sentence(sentence: str, texts: Sequence[str]) -> bool:
    """Whether a sentence (whitespace aside) appears in any retrieved passage."""
    s = " ".join(sentence.split())
    return any(s in " ".join(t.split()) for t in texts)


def summarise(per_item: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per group: how often the answer changed, with an interval."""
    resamples, seed = int(cfg["impact"]["bootstrap_resamples"]), int(cfg["seed"])
    rows = []
    for group, g in per_item.dropna(subset=["verdict"]).groupby("group"):
        changed = (g["verdict"] == "different").astype(float).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(changed, resamples, seed=seed)
        rows.append({"group": group, "items": len(g), "answer_changed": mean, "changed_low": lo, "changed_high": hi,
                     "one_declined": (g["verdict"] == "one_declined").mean(),
                     "both_declined": (g["verdict"] == "both_declined").mean(),
                     "edit_retrieved_old": g["edit_retrieved_old"].mean(),
                     "edit_retrieved_new": g["edit_retrieved_new"].mean()})
    return pd.DataFrame(rows)


def verdict(per_item: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Is the changed-facts check useful? Decided by the thresholds fixed
    before results: a difference of at least ``min_difference`` in how often
    answers changed, and Fisher's exact test below ``alpha``."""
    o = cfg["longform"]["outdated"]
    g = per_item.dropna(subset=["verdict"])
    counts = {grp: (int((x["verdict"] == "different").sum()), int((x["verdict"] != "different").sum()))
              for grp, x in g.groupby("group")}
    if FLAGGED not in counts or COMPARISON not in counts:
        return {"useful": None, "reason": "both groups are needed"}
    (fc, fn), (cc, cn) = counts[FLAGGED], counts[COMPARISON]
    rate_f, rate_c = fc / max(fc + fn, 1), cc / max(cc + cn, 1)
    _, p = stats.fisher_exact([[fc, fn], [cc, cn]], alternative="greater")
    diff = rate_f - rate_c
    useful = bool(diff >= float(o["min_difference"]) and p < float(o["alpha"]))
    return {"useful": useful, "flagged_changed": rate_f, "comparison_changed": rate_c, "difference": diff,
            "p_value": float(p), "min_difference": float(o["min_difference"]), "alpha": float(o["alpha"])}
