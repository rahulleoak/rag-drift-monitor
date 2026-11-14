"""Ask the archive: the RAG pipeline under test, live, for the dashboard.

The same pipeline the experiment measures (D28): the question is embedded
with bge-small, the top articles are retrieved from the stale index
(baseline only) and from a refreshed index, and Claude Sonnet answers from
each with the exact prompt Agent D uses. Shown side by side, so anyone can
see what not re-indexing costs for a question of their own.

Needs what a fresh clone does not have: the cleaned documents, the saved
indexes (``scripts/run_pipeline.py`` builds both) and an API key. Without
them the dashboard hides the page (``missing``). Answers are cached like
every other LLM call, so asking the same question twice is free.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import faiss
import pandas as pd

from driftmon.config import resolve_path
from driftmon.evaluation.answers import article_listing
from driftmon.llm.client import LLMClient, LLMRequest, api_key
from driftmon.llm.prompts import ANSWER_SYSTEM, ANSWER_TEMPLATE, ANSWER_VERSION, RagAnswer
from driftmon.rag.embeddings import Embedder
from driftmon.rag.index import VectorIndex


MAX_QUESTION = 300   # characters; a question, not a document


def missing(cfg: Mapping[str, Any]) -> list[str]:
    """What the page needs and this checkout lacks (empty = the page can run)."""
    out = []
    if not (resolve_path(cfg["paths"]["data_processed"]) / "documents.parquet").exists():
        out.append("the cleaned documents (scripts/prepare_data.py)")
    if not (resolve_path(cfg["paths"]["indexes"]) / "manifest.json").exists():
        out.append("the saved search indexes (scripts/run_pipeline.py)")
    if not api_key():
        out.append("an API key (.env)")
    return out


@dataclass
class Archive:
    """The documents, the embedding model and the saved indexes, loaded once."""

    docs: pd.DataFrame
    embedder: Embedder
    directory: Any
    manifest: dict[str, Any]
    _indexes: dict[str, VectorIndex] = field(default_factory=dict)

    @property
    def baseline(self) -> str:
        """The period(s) the stale index holds, e.g. '2016-17'."""
        return ", ".join(self.manifest["indexes"]["stale"].get("periods", ["the baseline"]))

    @property
    def years(self) -> list[str]:
        """Years a refreshed index exists for, oldest first."""
        return [n.removeprefix("refreshed_") for n in self.manifest["indexes"] if n.startswith("refreshed_")]

    def index(self, name: str) -> VectorIndex:
        """A saved index ("stale" or "refreshed_<year>"), read from disk on first use."""
        if name not in self._indexes:
            entry = self.manifest["indexes"][name]
            self._indexes[name] = VectorIndex.from_faiss(entry["doc_ids"],
                                                         faiss.read_index(str(self.directory / f"{name}.faiss")))
        return self._indexes[name]


def open_archive(cfg: Mapping[str, Any]) -> Archive:
    """Load the documents and the embedding model; indexes load on first use."""
    directory = resolve_path(cfg["paths"]["indexes"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    docs = pd.read_parquet(resolve_path(cfg["paths"]["data_processed"]) / "documents.parquet",
                           columns=["doc_id", "date", "period", "category", "title", "text"])
    return Archive(docs=docs, embedder=Embedder.from_config(cfg, model_name=manifest["embedding_model"]),
                   directory=directory, manifest=manifest)


def make_llm(cfg: Mapping[str, Any]) -> LLMClient:
    """The project's LLM client: same cache, cost ledger and budget as the pipeline."""
    return LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                     ledger=resolve_path(cfg["paths"]["runs"]) / "llm_costs.csv")


def retrieve(archive: Archive, question: str, index_name: str, k: int) -> list[str]:
    """Top ``k`` document ids for the question from one index."""
    ids, _ = archive.index(index_name).search(archive.embedder.embed_queries([question]), k)
    return [str(d) for d in ids[0]]


def answer_request(question: str, doc_ids: Sequence[str], docs: pd.DataFrame,
                   cfg: Mapping[str, Any]) -> LLMRequest:
    """The exact request Agent D sends for one question and one index's articles."""
    job = cfg["llm"]["jobs"]["answers"]
    return LLMRequest(job="answers", model=cfg["llm"]["generator_model"], system=ANSWER_SYSTEM,
                      prompt=ANSWER_TEMPLATE.format(question=question, articles=article_listing(doc_ids, docs)),
                      schema=RagAnswer, prompt_version=ANSWER_VERSION, effort=job["effort"],
                      max_tokens=int(job["max_tokens"]))


def ask(archive: Archive, question: str, year: str, llm: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Answer one question from the stale index and from the index refreshed up to ``year``.

    Returns, per side, the retrieved articles (rank, date, section, headline) and the answer
    (found, text, the letters it cites), plus the cost of this question.
    """
    k = int(cfg["impact"]["answer_k"])
    d = archive.docs.set_index("doc_id")
    before = llm.spent()
    out: dict[str, Any] = {"question": question, "year": year, "k": k}
    for side, name in (("stale", "stale"), ("refreshed", f"refreshed_{year}")):
        ids = retrieve(archive, question, name, k)
        res = llm.complete(answer_request(question, ids, archive.docs, cfg))
        out[side] = {
            "index": name,
            "articles": [{"letter": chr(65 + i), "doc_id": x, "date": str(d.at[x, "date"])[:10],
                          "section": d.at[x, "category"], "title": d.at[x, "title"]} for i, x in enumerate(ids)],
            "found": res.parsed.found if res.ok else None,
            "answer": res.parsed.answer if res.ok else f"(no answer: {res.error or res.stop_reason})",
            "sources": list(res.parsed.sources) if res.ok else [],
            "cached": res.cached,
        }
    out["cost_usd"] = round(llm.spent() - before, 4)
    return out
