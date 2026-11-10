"""Apply Agent E's targeted re-index and check it worked (D32).

Agent E recommends re-indexing the long-form articles whose numbers or dates
changed since the index was built. This test applies that fix and compares
three chunk indexes over every article:
    - stale: every article as it was when the index was built (baseline);
    - targeted: the listed articles at their latest version, the rest stale;
    - full: every article at its latest version (the reference).
Items are sentences edited between the baseline and the latest version, in
three strata: number edits in listed articles, wording edits in listed
articles, and any edit in unlisted articles (what the targeted fix can miss).
A question is written from the latest sentence, answered from each index, and
an Opus judge compares the stale and the targeted answer with the full one.

Plain code decides, by thresholds fixed before results
(``longform.reindex``): the targeted fix is enough if it fixes at least
``min_recovered`` of the outdated answers the full re-index fixes, weighted
by stratum size.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from driftmon.drift import facts
from driftmon.evaluation import outdated as od

STRATA = ("listed_number", "listed_wording", "unlisted")
INDEXES = ("stale", "targeted", "full")
OUTDATED = ("different", "one_declined")  # differs from the full re-index, or only one of them answers


def listed_pages(fact_changes: pd.DataFrame, baseline: str) -> set[int]:
    """Pages Agent E lists for re-indexing: a changed fact after the baseline."""
    return set(fact_changes.loc[fact_changes["period"] != baseline, "page_id"].astype(int))


def edit_strata(snaps: pd.DataFrame, baseline: str, latest: str, listed: set[int],
                min_similarity: float) -> pd.DataFrame:
    """Every sentence edited between the baseline and the latest version, with
    its stratum: changed facts (``facts.fact_changes``) and every other edit,
    with or without a number (``facts.other_edits``), so a new name in an
    unlisted article can be sampled too."""
    by_key = {(r.page_id, r.period): r for r in snaps.itertuples(index=False)}
    rows = []
    for pid in sorted({r.page_id for r in snaps.itertuples(index=False)}):
        a, b = by_key.get((pid, baseline)), by_key.get((pid, latest))
        if a is None or b is None:
            continue
        old, new = json.loads(a.sections), json.loads(b.sections)
        base = {"page_id": int(pid), "title": b.title, "previous": baseline, "period": latest}
        for c in facts.fact_changes(old, new, min_similarity):
            rows.append({**base, "group": od.FLAGGED, "section": c.section, "before": c.before, "after": c.after})
        for section, before, after, _ in facts.other_edits(old, new, min_similarity):
            rows.append({**base, "group": "other", "section": section, "before": before, "after": after})
    items = pd.DataFrame(rows)
    in_list = items["page_id"].isin(listed)
    stratum = np.where(~in_list, "unlisted",
                       np.where(items["group"] == od.FLAGGED, "listed_number", "listed_wording"))
    return items.assign(listed=in_list, stratum=stratum)


def sample_strata(items: pd.DataFrame, per_stratum: Mapping[str, int], seed: int) -> pd.DataFrame:
    """A random sample per stratum; item ids are stable for a seed."""
    parts = [g.sample(n=min(int(per_stratum[s]), len(g)), random_state=seed)
             for s, g in items.groupby("stratum") if s in per_stratum]
    out = pd.concat(parts, ignore_index=True)
    prefix = {"listed_number": "N", "listed_wording": "W", "unlisted": "U"}
    out.insert(0, "item_id", [f"{prefix[s]}{i:03d}" for i, s in enumerate(out["stratum"])])
    return out


def index_chunks(chunks: pd.DataFrame, baseline: str, latest: str, listed: Iterable[int]
                 ) -> dict[str, pd.DataFrame]:
    """The chunks each of the three indexes holds. An article missing from the
    baseline stays missing from the stale index (and from targeted, unless listed)."""
    listed = set(listed)
    old, new = chunks[chunks["period"] == baseline], chunks[chunks["period"] == latest]
    targeted = pd.concat([old[~old["page_id"].isin(listed)], new[new["page_id"].isin(listed)]])
    return {"stale": old.reset_index(drop=True), "targeted": targeted.reset_index(drop=True),
            "full": new.reset_index(drop=True)}


def work(index_tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """How much re-indexing each fix takes: passages re-embedded."""
    full = index_tables["full"]
    t = index_tables["targeted"]
    redone = int((t["period"] == full["period"].iloc[0]).sum()) if len(full) else 0
    return {"passages_full": int(len(full)), "passages_targeted": redone,
            "targeted_share": redone / len(full) if len(full) else float("nan")}


def outdated(verdicts: pd.Series) -> pd.Series:
    """True where an answer is outdated against the full re-index."""
    return verdicts.isin(OUTDATED)


def _weighted(df: pd.DataFrame, col: str, population: Mapping[str, int]) -> float:
    rates = df.groupby("stratum")[col].mean()
    total = sum(population[s] for s in rates.index)
    return float(sum(rates[s] * population[s] for s in rates.index) / total) if total else float("nan")


def recovered(df: pd.DataFrame, population: Mapping[str, int]) -> float:
    """Share of the stale index's outdated answers that the targeted fix also fixes (weighted)."""
    s, t = _weighted(df, "outdated_stale", population), _weighted(df, "outdated_targeted", population)
    return (s - t) / s if s > 0 else float("nan")


def summarise(items: pd.DataFrame, population: Mapping[str, int]) -> pd.DataFrame:
    """Per stratum and weighted overall: outdated answers on the stale and the targeted index."""
    df = items.dropna(subset=["verdict_stale", "verdict_targeted"])
    rows = []
    for s, g in df.groupby("stratum"):
        rows.append({"stratum": s, "population": population.get(s, 0), "items": len(g),
                     "outdated_stale": g["outdated_stale"].mean(), "outdated_targeted": g["outdated_targeted"].mean(),
                     "edit_retrieved_stale": g["edit_retrieved_stale"].mean(),
                     "edit_retrieved_full": g["edit_retrieved_full"].mean()})
    if len(df):
        rows.append({"stratum": "all (weighted)", "population": sum(population.values()), "items": len(df),
                     "outdated_stale": _weighted(df, "outdated_stale", population),
                     "outdated_targeted": _weighted(df, "outdated_targeted", population),
                     "edit_retrieved_stale": _weighted(df, "edit_retrieved_stale", population),
                     "edit_retrieved_full": _weighted(df, "edit_retrieved_full", population)})
    return pd.DataFrame(rows)


def verdict(items: pd.DataFrame, population: Mapping[str, int], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Is the targeted re-index enough? By the thresholds fixed before results,
    with a stratified bootstrap interval for the recovered share."""
    r = cfg["longform"]["reindex"]
    df = items.dropna(subset=["verdict_stale", "verdict_targeted"]).reset_index(drop=True)
    if not len(df):
        return {"enough": None, "reason": "no judged items"}
    rec = recovered(df, population)
    rng = np.random.default_rng(int(cfg["seed"]))
    groups = [g.index.to_numpy() for _, g in df.groupby("stratum")]
    boots = []
    for _ in range(int(cfg["impact"]["bootstrap_resamples"])):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = recovered(df.loc[idx], population)
        if not np.isnan(v):
            boots.append(v)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    listed = df[df["listed"]]
    agree = float((~listed["outdated_targeted"]).mean()) if len(listed) else float("nan")
    enough = bool(rec >= float(r["min_recovered"]) and agree >= float(r["min_listed_agreement"]))
    return {"enough": enough, "recovered": rec, "recovered_low": lo, "recovered_high": hi,
            "outdated_stale": _weighted(df, "outdated_stale", population),
            "outdated_targeted": _weighted(df, "outdated_targeted", population),
            "listed_agreement": agree, "items": int(len(df)),
            "min_recovered": float(r["min_recovered"]), "min_listed_agreement": float(r["min_listed_agreement"])}
