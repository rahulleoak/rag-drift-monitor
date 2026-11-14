"""Ask the archive: the live RAG page's engine (no model, no API: fakes only)."""
import copy
import json

import faiss
import numpy as np
import pandas as pd

from driftmon.config import load_config
from driftmon.dashboard import ask
from driftmon.evaluation.answers import answer_requests
from driftmon.llm.client import LLMResult
from driftmon.llm.prompts import RagAnswer

DOCS = pd.DataFrame({
    "doc_id": ["b1", "b2", "n1"], "date": ["2016-03-01", "2017-05-01", "2020-04-01"],
    "period": ["2016-17", "2016-17", "2020"], "category": ["WELLNESS", "FOOD & DRINK", "WELLNESS"],
    "title": ["Flu season tips", "Summer wine guide", "How to make a face mask"],
    "text": ["Stay home when sick.", "Light reds for July.", "Cut two layers of cotton."]})
VECS = np.eye(3, dtype=np.float32)


class FakeEmbedder:
    def embed_queries(self, texts):
        return np.array([[0.1, 0.0, 1.0]], dtype=np.float32)   # closest to the 2020 article


class FakeLLM:
    def __init__(self):
        self.requests, self.total = [], 0.0

    def spent(self):
        return self.total

    def complete(self, req):
        self.requests.append(req)
        self.total += 0.01
        found = "How to make a face mask" in req.prompt   # the 2020 article was retrieved
        return LLMResult(key=req.key, parsed=RagAnswer(found=found, sources=["A"] if found else [],
                                                       answer="Cut two layers of cotton." if found else
                                                       "The archive does not say."),
                         text="", stop_reason="end_turn", model=req.model, input_tokens=0, output_tokens=0,
                         cost_usd=0.01)


def _archive(tmp_path):
    names = {"stale": ["b1", "b2"], "refreshed_2020": ["b1", "b2", "n1"]}
    for name, ids in names.items():
        idx = faiss.IndexFlatIP(3)
        idx.add(VECS[[list(DOCS.doc_id).index(i) for i in ids]])
        faiss.write_index(idx, str(tmp_path / f"{name}.faiss"))
    manifest = {"embedding_model": "fake", "indexes": {n: {"doc_ids": ids} for n, ids in names.items()}}
    return ask.Archive(docs=DOCS, embedder=FakeEmbedder(), directory=tmp_path, manifest=manifest)


def test_missing_names_what_the_page_needs(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["data_processed"], cfg["paths"]["indexes"] = str(tmp_path / "d"), str(tmp_path / "i")
    monkeypatch.setattr(ask, "api_key", lambda: None)
    assert len(ask.missing(cfg)) == 3
    (tmp_path / "d").mkdir(); (tmp_path / "d" / "documents.parquet").write_text("x")
    (tmp_path / "i").mkdir(); (tmp_path / "i" / "manifest.json").write_text("{}")
    monkeypatch.setattr(ask, "api_key", lambda: "k")
    assert ask.missing(cfg) == []


def test_ask_answers_from_both_indexes_side_by_side(tmp_path):
    cfg = load_config()
    archive, llm = _archive(tmp_path), FakeLLM()
    assert archive.years == ["2020"]
    out = ask.ask(archive, "How do I make a face mask at home?", "2020", llm, cfg)
    assert out["stale"]["found"] is False and out["refreshed"]["found"] is True
    assert out["refreshed"]["articles"][0]["title"] == "How to make a face mask"
    assert out["refreshed"]["articles"][0]["date"] == "2020-04-01"
    assert all(a["doc_id"] != "n1" for a in out["stale"]["articles"])   # the stale index never holds 2020
    assert out["cost_usd"] == 0.02 and len(llm.requests) == 2


def test_the_page_sends_exactly_agent_ds_request():
    cfg = load_config()
    q = pd.DataFrame({"question_id": ["q1"], "question": ["How do I make a face mask at home?"]})
    plan = pd.DataFrame({"question_id": ["q1"], "doc_ids": [["n1", "b1"]]})
    theirs = answer_requests(plan, q, DOCS, cfg)[0]
    ours = ask.answer_request("How do I make a face mask at home?", ["n1", "b1"], DOCS, cfg)
    assert ours.key == theirs.key   # same prompt, model and settings, so the same cache entry
