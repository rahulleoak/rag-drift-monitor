"""Evidence for Agent E that is produced outside the agent graph.

- Long-form documents (D26, D27): the changed-facts check, the
  outdated-answer test, the chunking experiment and the applied targeted
  re-index (D32), saved by ``scripts/longform_drift.py``,
  ``scripts/outdated_answers.py`` and ``scripts/reindex_check.py``.
- Hand-label checks (F19, F20): blind labels vs the LLM, saved by
  ``scripts/label_sheet.py score``.

Missing files are skipped; the rules that need them then report
"not checked".
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from driftmon.config import resolve_path


def load_longform(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The long-form results Agent E's rules read (empty dict if none)."""
    d = resolve_path(cfg["recommendation"]["longform_dir"])
    out: dict[str, Any] = {"baseline": cfg["periods"]["baseline"]["name"]}
    tables = {"fact_changes": "fact_changes.csv", "fact_summary": "fact_changes_summary.csv",
              "outdated_summary": "outdated_summary.csv", "outdated_items": "outdated_items.csv",
              "chunking_eval": "chunking_eval.csv", "longform_summary": "summary.csv"}
    for key, name in tables.items():
        if (d / name).exists():
            out[key] = pd.read_csv(d / name)
    for key in ("outdated_verdict", "reindex_verdict"):
        if (d / f"{key}.json").exists():
            out[key] = json.loads((d / f"{key}.json").read_text(encoding="utf-8"))
    if "longform_summary" in out:
        out["articles"] = int(out["longform_summary"]["articles"].iloc[-1])
    if "fact_summary" in out:
        out["low_band_share"] = low_band_share(out["fact_summary"])
    if "outdated_items" in out:
        out["example"] = outdated_example(out["outdated_items"])
    return out


def low_band_share(summary: pd.DataFrame) -> float:
    """Of the articles with a changed fact, the share whose drift band was low."""
    with_facts = (summary["with_changed_facts"] * summary["articles"]).sum()
    low = (summary["changed_facts_but_band_low"] * summary["articles"]).sum()
    return float(low / with_facts) if with_facts else float("nan")


def outdated_example(items: pd.DataFrame) -> dict[str, Any] | None:
    """A short flagged edit whose answer changed: the clearest example."""
    hit = items[(items["group"] == "flagged") & (items["verdict"] == "different")]
    if not len(hit):
        return None
    hit = hit.assign(length=hit["before"].str.len() + hit["after"].str.len()).sort_values(["length", "item_id"])
    r = hit.iloc[0]
    return {k: r[k] for k in ("item_id", "title", "period", "before", "after", "question", "answer_old",
                              "answer_new")}


def load_label_checks(cfg: Mapping[str, Any]) -> pd.DataFrame | None:
    """Hand labels vs the LLM, one row per (who, part); None if no files."""
    rows = []
    for item in cfg["recommendation"].get("label_checks", []):
        path = resolve_path(item["file"])
        if path.exists():
            rows.append(pd.read_csv(path).assign(who=item["who"], file=item["file"]))
    return pd.concat(rows, ignore_index=True) if rows else None


SNAPSHOT_FILES = ("outdated_summary.csv", "outdated_items.csv", "outdated_verdict.json", "chunking_eval.csv",
                  "summary.csv", "fact_changes_summary.csv", "reindex_verdict.json")


def save_longform_snapshot(cfg: Mapping[str, Any], dest) -> list[str]:
    """Copy the small long-form result files Agent E used into ``dest`` (a run's
    ``longform/`` folder), so a saved run is self-contained for the dashboard."""
    import shutil
    from pathlib import Path
    src = resolve_path(cfg["recommendation"]["longform_dir"])
    dest = Path(dest)
    copied = []
    for name in SNAPSHOT_FILES:
        if (src / name).exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dest / name)
            copied.append(name)
    return copied
