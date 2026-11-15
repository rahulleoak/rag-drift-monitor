"""Keyword vs meaning-based search under drift (D40).

The hypothesis (P2): wording drift (new words for the same things) hurts
keyword search more than meaning-based search, because keyword search can
only match the words it sees. Tested two ways, without the LLM:

1. **By drift kind.** Every answerable question is searched on the index the
   judged comparison uses (its period's refreshed index; the latest one for
   baseline questions) with both searches. Does the source article land in
   the top ``hit_k``? Agent B labels each later section's drift as wording,
   meaning, both or none. If the hypothesis holds, keyword search falls
   further behind meaning search in sections with wording drift than in
   sections with none.
2. **Newer wording** (the D31 test's kept questions): the same question in
   same-era words and in later words, searched on the query period's
   refreshed index. Does keyword search lose the article more often?

Code decides, with thresholds set before the run (``keyword_search``).
Caveat built into the reading: each question was written from its article,
so the two share words, which flatters keyword search.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from driftmon.evaluation.wording import group_gap_p, mcnemar_p

SEARCHES = ("meaning", "keyword")
WORDING_DRIFT = ("wording", "both")   # Agent B's kinds with a vocabulary change


def ranks(ids: np.ndarray, sources: Sequence[str], scores: np.ndarray | None = None) -> list[int]:
    """1-based rank of each query's source article in its result list; 0 = not in it.
    With ``scores`` (keyword search), a result with score 0 shares no word with the
    query, so it does not count."""
    out = []
    for j, (row, src) in enumerate(zip(ids, sources)):
        ok = (row == src) if scores is None else (row == src) & (scores[j] > 0)
        hit = np.flatnonzero(ok)
        out.append(int(hit[0]) + 1 if len(hit) else 0)
    return out


def drift_kinds(segments: pd.DataFrame, baseline: str) -> dict[tuple[str, str], str]:
    """(period, section) -> Agent B's drift kind, against the baseline. Periods as text: a
    saved CSV reads "2018" back as a number."""
    s = segments[segments["compared_with"].astype(str) == baseline]
    return {(str(p), c): k for p, c, k in zip(s["period"], s["category"], s["kind"])}


def per_question(questions: pd.DataFrame, rank_meaning: Sequence[int], rank_keyword: Sequence[int],
                 kinds: Mapping[tuple[str, str], str], baseline: str, hit_k: int) -> pd.DataFrame:
    """One row per question: both ranks, found in the top ``hit_k``, reciprocal rank, drift kind."""
    q = questions[["question_id", "period", "category"]].copy().reset_index(drop=True)
    q["kind"] = ["baseline" if p == baseline else kinds.get((p, c), "unknown")
                 for p, c in zip(q["period"], q["category"])]
    for name, r in zip(SEARCHES, (rank_meaning, rank_keyword)):
        r = np.asarray(r)
        q[f"rank_{name}"] = r
        q[f"found_{name}"] = ((r > 0) & (r <= hit_k)).astype(float)
        q[f"rr_{name}"] = np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)
    q["shortfall"] = q["found_meaning"] - q["found_keyword"]   # +1: only meaning search found it
    return q


def summarise(pq: pd.DataFrame) -> pd.DataFrame:
    """Per drift kind and overall: questions, how often each search finds the article, MRR."""
    groups = [("all", pq), *[(k, g) for k, g in pq.groupby("kind", sort=True)]]
    rows = []
    for name, g in groups:
        row = {"kind": name, "questions": len(g)}
        for s in SEARCHES:
            row[f"found_{s}"] = g[f"found_{s}"].mean()
            row[f"mrr_{s}"] = g[f"rr_{s}"].mean()
        row["shortfall"] = g["shortfall"].mean()
        row["only_meaning"] = int((g["shortfall"] > 0).sum())
        row["only_keyword"] = int((g["shortfall"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(pq: pd.DataFrame, wording: pd.DataFrame | None, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The two questions, by the thresholds fixed before results."""
    s = cfg["keyword_search"]
    alpha, min_gap, min_drop = float(s["alpha"]), float(s["min_group_gap"]), float(s["min_drop"])
    only_m, only_k = int((pq["shortfall"] > 0).sum()), int((pq["shortfall"] < 0).sum())
    overall = {"questions": len(pq), "found_meaning": float(pq["found_meaning"].mean()),
               "found_keyword": float(pq["found_keyword"].mean()), "only_meaning": only_m,
               "only_keyword": only_k,
               "p_meaning_better": mcnemar_p(only_m, only_k), "p_keyword_better": mcnemar_p(only_k, only_m)}

    w = pq[pq["kind"].isin(WORDING_DRIFT)]["shortfall"].to_numpy()
    n = pq[pq["kind"] == "none"]["shortfall"].to_numpy()
    gap = float(w.mean() - n.mean()) if len(w) and len(n) else float("nan")
    p = group_gap_p(w, n, int(s["permutations"]), int(cfg["seed"]))
    by_kind = {"wording_drift_questions": len(w), "no_drift_questions": len(n),
               "shortfall_wording_drift": float(w.mean()) if len(w) else None,
               "shortfall_no_drift": float(n.mean()) if len(n) else None, "gap": gap, "p": p,
               "wording_drift_hurts_keyword_more": None if np.isnan(gap) else bool(gap >= min_gap and p < alpha)}

    out = {"thresholds": {"hit_k": int(s["hit_k"]), "min_group_gap": min_gap, "min_drop": min_drop,
                          "alpha": alpha},
           "overall": overall, "by_drift_kind": by_kind}
    if wording is not None and len(wording):
        res = {}
        for name in SEARCHES:
            same, newer = wording[f"found_same_era_{name}"].astype(bool), wording[f"found_newer_{name}"].astype(bool)
            lost, gained = int((same & ~newer).sum()), int((~same & newer).sum())
            drop = float(same.mean() - newer.mean())
            res[name] = {"found_same_era": float(same.mean()), "found_newer": float(newer.mean()), "drop": drop,
                         "lost": lost, "gained": gained, "p": mcnemar_p(lost, gained),
                         "hurts": bool(drop >= min_drop and mcnemar_p(lost, gained) < alpha)}
        res["questions"] = len(wording)
        res["keyword_hurt_more"] = bool(res["keyword"]["hurts"] and not res["meaning"]["hurts"])
        out["newer_wording"] = res
    return out
