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
from driftmon.prompts import REVIEW_CHECKS, DraftQuestion
from tests.test_llm import FakeAPI, _msg

LIMITS = {"min_words": 6, "max_words": 30, "copy_ngram": 4, "copy_ngram_description": 7,
          "max_headline_overlap": 0.7, "duplicate_cosine": 0.95}


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


def test_check_question_reasons():
    title, text = "Senate Passes Budget Bill After Long Night", "Lawmakers approved it by a vote of 51 to 49."
    good = DraftQuestion(question="Which chamber approved the spending plan after an overnight session?",
                         reference_answer="The Senate.", question_type="evergreen")
    assert simulate.check_question(good, title, text, True, LIMITS) == []
    copied = good.model_copy(update={"question": "Is it true lawmakers approved it by a vote of 51 to 49?"})
    assert "copies the description" in simulate.check_question(copied, title, text, True, LIMITS)
    meta = good.model_copy(update={"reference_answer": "The article says the Senate passed it."})
    assert simulate.check_question(meta, title, text, True, LIMITS) == ["reference describes the article"]
    # the meta check does not apply to unanswerable questions ("Not stated in the article.")
    assert simulate.check_question(meta, title, text, False, LIMITS) == []


def _fake_embedder(tmp_path):
    def enc(batch):
        return np.array([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                         .normal(size=16) for t in batch])
    return Embedder("fake", tmp_path / "emb", encode_fn=enc)


def _review(label="evergreen", **false):
    body = {c: True for c in REVIEW_CHECKS}
    body.update({k: False for k in false})
    body["question_type"] = label
    body["reason"] = ", ".join(false) or "fine"
    return body


def test_generate_questions_flow(tmp_path):
    articles = pd.DataFrame({
        "doc_id": ["a", "b", "c", "d"], "period": ["2016-17", "2020", "2020", "2020"],
        "category": ["POLITICS", "WELLNESS", "MONEY", "WELLNESS"],
        "date": ["2016-05-01", "2020-03-01", "2020-03-02", "2020-03-03"],
        "title": ["Senate Passes Budget Bill After Long Night", "Masks Required On City Buses",
                  "Who Pays The Debt After A Divorce?", "Masks Required On City Buses Again"],
        "text": ["Lawmakers approved it 51-49.", "The rule starts Monday; 3 riders were fined so far.",
                 "Here's how the debt gets split.", "The rule returns Monday."], "words": [20] * 4})
    budget_q = {"question": "By what margin did the upper chamber approve the spending plan?",
                "reference_answer": "51-49.", "question_type": "evergreen"}
    bus_q = {"question": "When does the face covering rule for public buses take effect in the city?",
             "reference_answer": "Monday.", "question_type": "evergreen"}
    fines_q = {"question": "How many bus riders have been fined under the face covering rule?",
               "reference_answer": "Three riders.", "question_type": "time_sensitive"}
    debt_q = {"question": "How is shared debt divided between spouses when they divorce?",
              "reference_answer": "Not stated in the article.", "question_type": "evergreen"}

    def responder(kw, live):
        prompt = kw["messages"][0]["content"]
        if "Question:" in prompt:  # review
            if "upper chamber" in prompt:
                body = _review(no_invented_details=False)  # strict reviewer rejects
            elif "divided between spouses" in prompt:
                body = _review(answerable=False, natural=False)  # agrees: not answerable (and broad)
            elif "fined" in prompt:
                body = _review(label="time_sensitive")
            else:
                body = _review(label="time_sensitive")  # the bus start date: reviewer disagrees
        elif "Senate" in prompt:
            body = {"questions": [budget_q], "answerable_from_article": True}
        elif "Divorce" in prompt:
            body = {"questions": [debt_q], "answerable_from_article": False}
        elif "Again" in prompt:
            body = {"questions": [bus_q], "answerable_from_article": True}  # duplicate of b's question
        else:
            body = {"questions": [bus_q, fines_q], "answerable_from_article": True}
        return _msg(json.dumps(body), model=kw["model"])

    cfg = copy.deepcopy(load_config())
    cfg["simulation"]["question_checks"] = LIMITS
    cfg["llm"]["prices"] = {cfg["llm"]["generator_model"]: {"input": 2, "output": 10},
                            cfg["llm"]["judge_model"]: {"input": 4, "output": 20}}
    llm = LLMClient(config=cfg["llm"], cache_dir=tmp_path / "cache", ledger=tmp_path / "l.csv",
                    client=FakeAPI(responder), sleep=lambda s: None)
    t = simulate.generate_questions(cfg, articles, llm, _fake_embedder(tmp_path), batch=False)
    t = t.set_index("question_id")

    assert t.loc["b-q0", "set"] == "answerable" and t.loc["b-q1", "set"] == "answerable"
    assert t.loc["b-q1", "question_type"] == "time_sensitive"          # second question per article
    # the reviewer's label is final; disagreement is recorded, not a rejection
    assert t.loc["b-q0", "writer_type"] == "evergreen" and t.loc["b-q0", "question_type"] == "time_sensitive"
    assert t.loc["b-q0", "type_agreed"] == False and t.loc["b-q1", "type_agreed"] == True  # noqa: E712
    assert t.loc["c-q0", "set"] == "unanswerable"                        # teaser kept for abstention tests
    assert t.loc["d-q0", "rejected_by"] == "near-duplicate question"
    assert t.loc["a-q0", "set"] == "" and t.loc["a-q0", "rejected_by"] == "review: not no_invented_details"
    rates = simulate.keep_rates(t.reset_index(), "category")
    assert rates.loc["WELLNESS", "questions"] == 3 and rates.loc["WELLNESS", "answerable"] == 2
    assert rates.loc["MONEY", "unanswerable"] == 1
    # only questions that passed the code checks went to the (judge) reviewer
    reviewed = [c for c in llm.client.live_calls if "Question:" in c["messages"][0]["content"]]
    assert len(reviewed) == 4 and all(c["model"] == cfg["llm"]["judge_model"] for c in reviewed)
