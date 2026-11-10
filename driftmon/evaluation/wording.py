"""The newer-wording test: does a question in later words still find an older article? (D31)

F15 found that how far a section drifted did not change how much search
lost. Those questions were written from their own article, so they share its
wording. This test asks the one case where a change of wording on its own
could hurt search: an older article asked about in the words of a later year
("COVID" for an article that says "coronavirus").

For each sampled question, Sonnet writes two versions:
    - same_era: reworded in the words of the article's time;
    - newer: the same_era version with only the words a later reader would
      say differently changed (or none, and the question is skipped).
Code then checks that the newer version adds at least one word that became
significantly more common by the query period (the same log-odds ranking and
20% rule as Agent B, D12). Opus checks that both versions ask the same thing
as the original. Search is the query period's refreshed index (it holds every
article), scored as known item: is the source article in the top ``hit_k``.

Comparing newer with same_era isolates the wording: any rewording loses a
little against the article's own words. Plain code decides, by thresholds
fixed before results (config ``evaluation.newer_wording``).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from driftmon.drift import lexical as lex
from driftmon.evaluation.stats import bootstrap_mean_ci
from driftmon.llm.client import LLMRequest
from driftmon.llm.prompts import (MEANING_SYSTEM, MEANING_TEMPLATE, MEANING_VERSION, REWORDING_SYSTEM,
                                  REWORDING_TEMPLATE, REWORDING_VERSION, MeaningCheck, Rewording)

GROUPS = ("flagged", "control", "other", "baseline")
VERSIONS = ("original", "same_era", "newer")


def settings(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The test's settings (``evaluation.newer_wording``)."""
    return cfg["evaluation"]["newer_wording"]


def sample_questions(questions: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Up to ``per_group`` answerable questions per drift status, from the
    question periods, at random. Needs a ``drift_status`` column."""
    s = settings(cfg)
    q = questions[(questions["set"] == "answerable") & questions["period"].isin(s["question_periods"])]
    out = [g.sample(n=min(int(s["per_group"]), len(g)), random_state=int(cfg["seed"]))
           for status, g in q.groupby("drift_status") if status in GROUPS]
    if not out:
        return q.iloc[:0].assign(group=pd.Series(dtype=str))
    items = pd.concat(out).sort_values(["drift_status", "question_id"]).reset_index(drop=True)
    return items.rename(columns={"drift_status": "group"})


def rewording_requests(items: pd.DataFrame, cfg: Mapping[str, Any]) -> list[LLMRequest]:
    """One Sonnet request per question: the same-era and the newer version.
    The writer sees the question and the article's date, not the article,
    so neither version can borrow the article's wording."""
    job = cfg["llm"]["jobs"]["rewording"]
    system = REWORDING_SYSTEM.format(query_time=_query_time(cfg))
    return [LLMRequest(job="rewording", model=cfg["llm"]["generator_model"], system=system,
                       prompt=REWORDING_TEMPLATE.format(date=str(r["date"])[:10], question=r["question"]),
                       schema=Rewording, prompt_version=REWORDING_VERSION, effort=job["effort"],
                       max_tokens=int(job["max_tokens"])) for r in items.to_dict("records")]


def _query_time(cfg: Mapping[str, Any]) -> str:
    """How the prompt names the query period, e.g. 'late 2022'."""
    return f"late {settings(cfg)['query_period']}"


# ---------- which words are newer (code, no LLM) ----------

def period_counts(docs: pd.DataFrame) -> dict[str, Counter]:
    """Word counts per period over headline and description."""
    return {p: lex.term_counts(lex.tokenize(f"{t} {x}") for t, x in zip(g["title"], g["text"]))
            for p, g in docs.groupby("period")}


def newer_terms(counts: Mapping[str, Counter], period: str, query_period: str, min_z: float,
                min_rate_change: float) -> set[str]:
    """Words significantly more typical of the query period than of ``period``
    (log-odds z at least ``min_z``) whose rate rose at least ``min_rate_change``."""
    target, ref = counts[query_period], counts[period]
    table = lex.log_odds_dirichlet(target, ref, target + ref, min_count=3)
    nt, nr = max(sum(target.values()), 1), max(sum(ref.values()), 1)
    rate_t = table["count_target"] / nt
    rate_r = (table["count_reference"] + 0.5) / nr  # a word absent before still has a small rate
    keep = (table["z"] >= min_z) & (rate_t >= rate_r * min_rate_change)
    return set(table.index[keep])


def added_newer_words(newer: str, same_era: str, original: str, terms: set[str]) -> list[str]:
    """Words of the newer version that neither the same-era version nor the
    original uses, and that are newer by ``newer_terms``."""
    before = set(lex.tokenize(same_era)) | set(lex.tokenize(original))
    return sorted({t for t in lex.tokenize(newer) if t not in before and t in terms})


def _norm(text: str) -> str:
    return " ".join(lex.tokenize(text or ""))


def code_checks(items: pd.DataFrame, terms_by_period: Mapping[str, set[str]]) -> pd.DataFrame:
    """Adds ``newer_words`` and ``rejected_by``; empty ``rejected_by`` = kept so far."""
    words, reasons = [], []
    for r in items.to_dict("records"):
        why = []
        same, newer = r.get("same_era") or "", r.get("newer") or ""
        w = added_newer_words(newer, same, r["question"], terms_by_period.get(r["period"], set()))
        if not same.strip():
            why.append("no rewording")
        elif _norm(same) == _norm(r["question"]):
            why.append("same_era equals original")
        if not r.get("has_newer") or not newer.strip():
            why.append("no newer wording")
        elif _norm(newer) == _norm(same):
            why.append("newer equals same_era")
        elif not w:
            why.append("no newer word by the data")
        words.append(", ".join(w))
        reasons.append("; ".join(why))
    return items.assign(newer_words=words, rejected_by=reasons)


# ---------- the meaning check (Opus) ----------

def meaning_requests(items: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[LLMRequest], np.ndarray]:
    """One Opus request per question: do both rewordings ask the same thing?
    The two are shown in a shuffled order, not named; returns the flips
    (True = newer shown first)."""
    job = cfg["llm"]["jobs"]["meaning_check"]
    rng = np.random.default_rng(int(cfg["seed"]))
    flips = rng.random(len(items)) < 0.5
    reqs = []
    for r, flip in zip(items.to_dict("records"), flips):
        first, second = (r["newer"], r["same_era"]) if flip else (r["same_era"], r["newer"])
        reqs.append(LLMRequest(job="meaning_check", model=cfg["llm"]["judge_model"], system=MEANING_SYSTEM,
                               prompt=MEANING_TEMPLATE.format(original=r["question"], first=first, second=second),
                               schema=MeaningCheck, prompt_version=MEANING_VERSION, effort=job["effort"],
                               max_tokens=int(job["max_tokens"])))
    return reqs, flips


def unshuffle(checks: Sequence[MeaningCheck | None], flips: Sequence[bool]) -> pd.DataFrame:
    """(same_era_same, newer_same, meaning_reason) per item, in item order."""
    rows = []
    for c, flip in zip(checks, flips):
        if c is None:
            rows.append({"same_era_same": None, "newer_same": None, "meaning_reason": None})
            continue
        a, b = c.rewording_1_same, c.rewording_2_same
        same, newer = (b, a) if flip else (a, b)
        rows.append({"same_era_same": same, "newer_same": newer, "meaning_reason": c.reason})
    return pd.DataFrame(rows)


# ---------- search and the verdict (code) ----------

def rank_of(retrieved: Sequence[str], target: str) -> int | None:
    """1-based rank of ``target`` in ``retrieved``, or None if absent."""
    for i, d in enumerate(retrieved, 1):
        if d == target:
            return i
    return None


def add_ranks(items: pd.DataFrame, ranked: Mapping[str, Sequence[Sequence[str]]], hit_k: int) -> pd.DataFrame:
    """Per version: the source article's rank, whether it is in the top
    ``hit_k``, and its reciprocal rank (0 when absent)."""
    out = items.copy()
    for v in VERSIONS:
        ranks = [rank_of(ids, d) for ids, d in zip(ranked[v], out["doc_id"])]
        out[f"rank_{v}"] = ranks
        out[f"found_{v}"] = [r is not None and r <= hit_k for r in ranks]
        out[f"rr_{v}"] = [1.0 / r if r else 0.0 for r in ranks]
    return out


def summarise(kept: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per group and overall: how often each version finds the article, and
    the drop from same-era to newer wording with an interval."""
    resamples, seed = int(cfg["impact"]["bootstrap_resamples"]), int(cfg["seed"])
    rows = []
    parts = [(g, x) for g, x in kept.groupby("group")] + [("all", kept)]
    for group, g in parts:
        if not len(g):
            continue
        drop = (g["found_same_era"].astype(float) - g["found_newer"].astype(float)).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(drop, resamples, seed=seed)
        rows.append({"group": group, "questions": len(g),
                     **{f"found_{v}": g[f"found_{v}"].mean() for v in VERSIONS},
                     **{f"mrr_{v}": g[f"rr_{v}"].mean() for v in VERSIONS},
                     "drop": mean, "drop_low": lo, "drop_high": hi})
    return pd.DataFrame(rows)


def mcnemar_p(lost: int, gained: int) -> float:
    """One-sided exact McNemar: is losing the article (found with same_era,
    not with newer) more common than gaining it?"""
    n = lost + gained
    return 1.0 if n == 0 else float(stats.binomtest(lost, n, 0.5, alternative="greater").pvalue)


def group_gap_p(a: np.ndarray, b: np.ndarray, permutations: int, seed: int) -> float:
    """One-sided permutation test: is the mean drop in ``a`` bigger than in ``b``?"""
    if not len(a) or not len(b):
        return float("nan")
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        hits += pooled[:len(a)].mean() - pooled[len(a):].mean() >= observed - 1e-12
    return (hits + 1) / (permutations + 1)


def verdict(kept: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The two questions, by the thresholds fixed before results."""
    s = settings(cfg)
    same, newer = kept["found_same_era"].astype(bool), kept["found_newer"].astype(bool)
    lost, gained = int((same & ~newer).sum()), int((~same & newer).sum())
    drop = float(same.mean() - newer.mean()) if len(kept) else float("nan")
    p = mcnemar_p(lost, gained)
    hurts = bool(len(kept) and drop >= float(s["min_drop"]) and p < float(s["alpha"]))

    d = (kept["found_same_era"].astype(float) - kept["found_newer"].astype(float))
    fl, co = d[kept["group"] == "flagged"].to_numpy(), d[kept["group"] == "control"].to_numpy()
    gap = float(fl.mean() - co.mean()) if len(fl) and len(co) else float("nan")
    gp = group_gap_p(fl, co, int(s["permutations"]), int(cfg["seed"]))
    matters = (None if np.isnan(gap) else
               bool(gap >= float(s["min_group_gap"]) and gp < float(s["alpha"])))
    return {"questions": int(len(kept)), "found_same_era": float(same.mean()) if len(kept) else None,
            "found_newer": float(newer.mean()) if len(kept) else None, "drop": drop,
            "lost": lost, "gained": gained, "mcnemar_p": p, "wording_hurts_search": hurts,
            "flagged_questions": int(len(fl)), "control_questions": int(len(co)),
            "flagged_drop": float(fl.mean()) if len(fl) else None,
            "control_drop": float(co.mean()) if len(co) else None,
            "group_gap": gap, "group_gap_p": gp, "drift_level_matters": matters,
            "min_drop": float(s["min_drop"]), "min_group_gap": float(s["min_group_gap"]),
            "alpha": float(s["alpha"]), "hit_k": int(s["hit_k"])}
