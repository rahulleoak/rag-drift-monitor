"""Dataset loading: turn each candidate dataset into one common record shape.

Both candidate datasets (docs/questions.md Q1) are mapped onto ``Record``
(doc_id, date, category, title, text), so exploration and the pipeline treat
them identically and switching datasets is a config change.

Files are streamed line by line (both are JSON Lines), because the arXiv
snapshot is several GB and must not be loaded into memory at once.

Period assignment lives in ``driftmon.data.periods``; cleaning and deduplication
in ``driftmon.data.cleaning``; category normalisation is applied by
``scripts/prepare_data.py`` using the config's ``categories.normalise``.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    """One document in the common shape used throughout the project."""

    doc_id: str
    date: date
    category: str
    title: str
    text: str


@dataclass
class LoadStats:
    """Counts collected while loading, reported so no data loss is silent."""

    lines_read: int = 0
    records: int = 0
    skipped: int = 0  # malformed JSON or a missing / unparseable required field


def _clean(value: Any) -> str:
    """Collapse all whitespace (arXiv abstracts contain hard line breaks)."""
    return " ".join(str(value or "").split())


def parse_huffpost(obj: dict[str, Any], line_no: int) -> Record:
    """Map one HuffPost News Category record to a ``Record``.

    Raw fields: link, headline, category, short_description, authors, date.
    The category label is kept exactly as published; merging near-duplicate
    labels is a later, documented decision.
    """
    category = _clean(obj["category"])
    if not category:
        raise ValueError("empty category")
    return Record(
        doc_id=f"hp-{line_no}",
        date=date.fromisoformat(str(obj["date"])[:10]),
        category=category,
        title=_clean(obj.get("headline")),
        text=_clean(obj.get("short_description")),
    )


def parse_arxiv(obj: dict[str, Any], line_no: int) -> Record:
    """Map one arXiv metadata record (Kaggle Cornell snapshot) to a ``Record``.

    The date is the first version's submission date. ``update_date`` is the
    last update, which would place old papers in recent years, so it is only
    a fallback. The category is the first listed (the primary) one.
    """
    versions = obj.get("versions") or []
    if versions and versions[0].get("created"):
        published = parsedate_to_datetime(versions[0]["created"]).date()
    else:
        published = date.fromisoformat(str(obj["update_date"])[:10])
    categories = _clean(obj.get("categories")).split()
    if not categories:
        raise ValueError("no categories")
    return Record(
        doc_id=f"arxiv-{obj['id']}",
        date=published,
        category=categories[0],
        title=_clean(obj.get("title")),
        text=_clean(obj.get("abstract")),
    )


PARSERS: dict[str, Callable[[dict[str, Any], int], Record]] = {
    "huffpost": parse_huffpost,
    "arxiv": parse_arxiv,
}


def iter_records(
    dataset: str,
    path: str | Path,
    stats: LoadStats | None = None,
    limit: int | None = None,
) -> Iterator[Record]:
    """Stream ``Record`` objects from a JSON Lines dataset file.

    Malformed lines are skipped and counted in ``stats`` rather than raising,
    so one bad line cannot stop a multi-GB load.

    Args:
        dataset: a key of ``PARSERS`` ("huffpost" or "arxiv").
        path: the raw dataset file.
        stats: optional counters, updated in place.
        limit: stop after this many records (for quick runs).
    """
    if dataset not in PARSERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(PARSERS)}")
    parse = PARSERS[dataset]
    stats = stats if stats is not None else LoadStats()
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if limit is not None and stats.records >= limit:
                return
            if not line.strip():
                continue
            stats.lines_read += 1
            try:
                record = parse(json.loads(line), line_no)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                stats.skipped += 1
                continue
            stats.records += 1
            yield record
