"""Tests for the question generator (driftmon.simulate) with a fake API."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from driftmon import simulate
from driftmon.config import load_config
from driftmon.embeddings import Embedder
from driftmon.llm import LLMClient
from driftmon.prompts import QuestionDraft
from tests.test_llm import FakeAPI, _msg

LIMITS = {"min_words": 6, "max_words": 30, "copy_ngram": 4, "max_headline_overlap": 0.7,
          "duplicate_cosine": 0.95}


def test_eligible_articles_skip_templates_and_short_records():
    docs = pd.DataFrame({"title": ["The 20 Funniest Tweets From Women This Week", "Senate passes budget bill",
                                   "HuffPost Rise: What You Need To Know On March 3", "Short one"],
                         "words": [30, 30, 30, 8]})
    ok = simulate.eligible_articles(docs, ["funniest tweets", "huffpost rise"], min_words=15)
    assert list(ok) == [False, True, False, False]


def test_headline_copy_and_overlap():
    title = "Trade War Escalates With China Announcing Tariff Hikes On $75 Billion Of U.S. Products"
    assert simulate.copies_headline("Why did China announcing tariff hikes on US goods matter?", title)
    assert not simulate.copies_headline("How much in American imports did Beijing target in 2019?", title)
    assert simulate.headline_overlap("China tariff hikes products", title) == 1.0


def test_check_draft_reasons():
    title = "Senate Passes Budget Bill After Long Night"
    good = QuestionDraft(question="Which chamber approved the spending plan after an overnight session?",
                         reference_answer="The Senate.", question_type="evergreen",
                         answerable_from_article=True)
    assert simulate.check_draft(good, title, LIMITS) == []
    bad = good.model_copy(update={"question": "Senate passes budget bill after long night?",
                                  "answerable_from_article": False})
    reasons = simulate.check_draft(bad, title, LIMITS)
    assert "copies the headline" in reasons and "writer: not answerable" in reasons


def _fake_embedder(tmp_path):
    def enc(batch):
        return np.array([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                         .normal(size=16) for t in batch])
    return Embedder("fake", tmp_path / "emb", encode_fn=enc)


def test_generate_questions_flow(tmp_path):
    articles = pd.DataFrame({
        "doc_id": ["a", "b", "c", "d"], "period": ["2016-17", "2020", "2020", "2020"],
        "category": ["POLITICS"] * 4, "date": ["2016-05-01", "2020-03-01", "2020-03-02", "2020-03-03"],
        "title": ["Senate Passes Budget Bill After Long Night", "Masks Required On City Buses",
                  "Opinion: Why We Need Hope", "Masks Required On City Buses Again"],
        "text": ["Lawmakers approved it 51-49.", "The rule starts Monday.", "A reflection.",
                 "The rule returns Monday."], "words": [20] * 4})
    writer = {
        "Senate": {"question": "By what margin did the upper chamber approve the spending plan?",
                   "reference_answer": "51-49.", "question_type": "evergreen", "answerable_from_article": True},
        "Opinion": {"question": "What does the author argue about hope in difficult times?",
                    "reference_answer": "That it is needed.", "question_type": "evergreen",
                    "answerable_from_article": False},
        "Masks Required On City Buses\n": None,
    }
    bus_q = {"question": "When does the face covering rule for public buses take effect in the city?",
             "reference_answer": "Monday.", "question_type": "time_sensitive", "answerable_from_article": True}

    def responder(kw, live):
        prompt = kw["messages"][0]["content"]
        if "Question:" in prompt:  # review
            ok = "Monday" in prompt
            body = {"answerable": True, "unambiguous": ok, "needs_article": True,
                    "reference_correct": True, "natural": True, "reason": "fine" if ok else "vague"}
        elif "Senate" in prompt:
            body = writer["Senate"]
        elif "Opinion" in prompt:
            body = writer["Opinion"]
        else:
            body = bus_q  # both bus articles give the same question: one is a duplicate
        return _msg(json.dumps(body), model=kw["model"])

    cfg = copy.deepcopy(load_config())
    cfg["simulation"]["question_checks"] = LIMITS
    cfg["llm"]["prices"] = {cfg["llm"]["generator_model"]: {"input": 2, "output": 10},
                            cfg["llm"]["judge_model"]: {"input": 4, "output": 20}}
    llm = LLMClient(config=cfg["llm"], cache_dir=tmp_path / "cache", ledger=tmp_path / "l.csv",
                    client=FakeAPI(responder), sleep=lambda s: None)
    table = simulate.generate_questions(cfg, articles, llm, _fake_embedder(tmp_path), batch=False).set_index("doc_id")

    assert table.loc["c", "rejected_by"] == "writer: not answerable"
    assert table.loc["d", "rejected_by"] == "near-duplicate question"
    assert bool(table.loc["b", "kept"]) and table.loc["b", "question_type"] == "time_sensitive"
    assert not table.loc["a", "kept"] and table.loc["a", "rejected_by"] == "review: not unambiguous"
    assert table.loc["a", "review_reason"] == "vague"
    # only the two drafts that passed the code checks were sent to the (judge) reviewer
    reviewed = [c for c in llm.client.live_calls if "Question:" in c["messages"][0]["content"]]
    assert len(reviewed) == 2 and all(c["model"] == cfg["llm"]["judge_model"] for c in reviewed)
    writes = [c for c in llm.client.live_calls if "Question:" not in c["messages"][0]["content"]]
    assert all(c["model"] == cfg["llm"]["generator_model"] for c in writes)
