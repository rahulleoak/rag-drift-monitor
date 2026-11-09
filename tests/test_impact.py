"""Tests for Agent D: answers, judging, segment classes (D13, D28)."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from driftmon.agents.impact_assessment import impact_assessment, pilot_sample
from driftmon.evaluation import answers as an
from driftmon.evaluation import retrieval as rv
from driftmon.rag.index import load_or_build_study_indexes
from driftmon.llm.client import LLMClient
from tests.test_llm import FakeAPI, _msg
from tests.test_retrieval import _archive, _config, _judge, _questions

FLAGGED = "2020|WELLNESS|vs 2016-17"


def _responder(kw, live):
    """Relevance judge, RAG answerer and answer judge, told apart by their system prompt.
    The answerer answers only if an article names the question's topic; the judge
    marks an answer correct if it names the topic."""
    system = kw["system"][0]["text"]
    prompt = kw["messages"][0]["content"]
    if system.startswith("You judge search results"):
        return _judge(kw, live)
    token = re.search(r"in (topic\S+?x\d+)\?", prompt).group(1)
    if system.startswith("You are a newsroom archive assistant"):
        hit = re.search(rf"^\[([A-Z])\] .*\b{token}\b", prompt, flags=re.M)
        body = ({"found": True, "answer": f"It was {token}.", "sources": [hit.group(1)]} if hit else
                {"found": False, "answer": "The archive does not say.", "sources": []})
    else:
        answer = prompt.split("Assistant's answer:")[1]
        right = token in answer
        body = {"reason": "ok", "correctness": 2 if right else 0, "grounded": True, "declined": not right}
    msg = _msg(json.dumps(body), model=kw["model"])
    return msg if live else ("succeeded", msg)


def _state(tmp_path, per_period=8):
    cfg = _config(tmp_path)
    docs, vecs = _archive()
    q, qv = _questions(docs, vecs, per_period=per_period)
    idx, _ = load_or_build_study_indexes(docs, vecs, "2016-17", ["2020"], tmp_path / "idx", "m")
    api = FakeAPI(_responder)
    llm = LLMClient(config=cfg["llm"], cache_dir=tmp_path / "cache", ledger=tmp_path / "l.csv", client=api,
                    sleep=lambda s: None)
    tables = rv.evaluate(cfg, q, qv, docs, idx, flagged=[FLAGGED], controls=[], llm=llm)
    drift = SimpleNamespace(drifted_segments=[FLAGGED], control_segments=[], segments=[])
    return {"config": cfg, "docs": docs, "questions": q, "llm": llm, "drift": drift,
            "retrieval_tables": tables}, api


def test_agent_d_finds_drift_that_hurts_and_a_steady_historical_control(tmp_path):
    state, api = _state(tmp_path)
    out = impact_assessment(state)
    rep, seg = out["impact"], out["impact_tables"]["segments"].set_index("segment_id")

    # 2020 questions: the stale index cannot answer, the refreshed one can -> material harm on a flagged segment
    s = seg.loc[FLAGGED]
    assert s["stale_correct"] == 0 and s["refreshed_correct"] == 1 and s["material"]
    assert s["class"] == an.DRIFT_HURTS and s["cause"] == "missing content (re-index)"
    # baseline questions: the historical control does not move
    b = seg.loc["2016-17|WELLNESS|vs 2016-17"]
    assert b["class"] == "historical control" and b["loss_correct"] == 0
    # declining when nothing was found counts as the right response
    pq = out["impact_tables"]["paired"]
    assert (pq.loc[pq["period"] == "2020", "stale_declined_ok"] == 1).all()
    # the unanswerable question is answered on both indexes and reported separately
    assert rep.unanswerable and rep.unanswerable[0]["questions"] == 1
    assert rep.judge["failed_answers"] == 0 and rep.judge["failed_judgements"] == 0
    assert rep.examples and rep.examples[0]["period"] == "2020"

    # the judge is never told which index an answer came from
    judge_prompts = [c["params"]["messages"][0]["content"] for b_ in api.batches for c in b_
                     if c["params"]["system"][0]["text"].startswith("You check answers")]
    assert judge_prompts and not any(re.search("stale|refreshed|index", p, re.I) for p in judge_prompts)


def test_identical_contexts_are_answered_and_judged_once(tmp_path):
    state, api = _state(tmp_path)
    impact_assessment(state)
    sent = [c["params"]["messages"][0]["content"] for b in api.batches for c in b]
    assert len(sent) == len(set(sent))   # no request was paid for twice


def test_pilot_sample_spreads_over_periods():
    ans = pd.DataFrame({"question_id": [f"q{i}" for i in range(40)], "period": ["a", "b"] * 20})
    un = pd.DataFrame({"question_id": ["u1", "u2", "u3"], "period": ["a", "b", "a"]})
    a, u = pilot_sample(ans, un, 20, seed=0)
    assert len(a) + len(u) == 20 and len(u) == 2 and a["period"].value_counts().min() == 9


def _cfg():
    return {"seed": 0, "impact": {"min_answer_quality_drop": 0.1, "min_segment_questions": 8,
                                  "bootstrap_resamples": 300, "recovered_correct": 0.6, "retrieval_ok": 0.6}}


def test_classes_and_causes():
    assert an.classify(True, True) == an.DRIFT_HURTS and an.classify(False, True) == an.DRIFT_HARMLESS
    assert an.classify(True, False) == an.HARM_WITHOUT_DRIFT and an.classify(False, False) == an.HEALTHY
    cfg = _cfg()
    assert an.diagnose(True, 0.9, 0.9, cfg) == "missing content (re-index)"
    assert an.diagnose(False, 0.9, 0.9, cfg) == ""
    assert an.diagnose(True, 0.3, 0.9, cfg).startswith("answer step")
    assert an.diagnose(False, 0.3, 0.2, cfg).startswith("retrieval or embedding model")


def test_material_needs_size_and_an_interval_above_zero():
    cfg = _cfg()

    def seg(losses, flagged=True):
        n = len(losses)
        pq = pd.DataFrame({"question_id": [f"q{i}" for i in range(n)], "period": "2020", "category": "X",
                           "set": "answerable", "loss_correct": losses, "stale_correct": 0.0,
                           "refreshed_correct": 1.0, "stale_hallucinated": 0.0,
                           "refreshed_context_has_answer": 1.0})
        return an.segments(pq, "2016-17", ["2020|X|vs 2016-17"] if flagged else [], cfg).iloc[0]

    big = seg([1.0] * 10)
    assert big["material"] and not big["borderline"] and big["class"] == an.DRIFT_HURTS
    noisy = seg([1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # mean 0.1, interval spans 0
    assert not noisy["material"] and noisy["borderline"] and noisy["class"] == an.DRIFT_HARMLESS
    small = seg([0.05] * 10, flagged=False)                              # real but below the threshold
    assert not small["material"] and small["class"] == an.HEALTHY
    few = seg([1.0] * 5)
    assert few["class"] == "too few questions" and not few["material"]


def test_failed_judgements_are_left_out_not_scored_wrong():
    plan = pd.DataFrame({"question_id": ["q1", "q1"], "role": ["stale", "refreshed"], "index": ["s", "r"],
                         "doc_ids": [["d1"], ["d2"]]})
    q = pd.DataFrame({"question_id": ["q1"], "set": ["answerable"]})
    ok = SimpleNamespace(ok=True, parsed=SimpleNamespace(answer="x", found=True))
    good = SimpleNamespace(ok=True, parsed=SimpleNamespace(correctness=2, grounded=True, declined=False,
                                                           reason="r"))
    bad = SimpleNamespace(ok=False, parsed=None)
    t = an.per_answer(plan, [ok, ok], [good, bad], q, {("q1", "stale"): 1.0, ("q1", "refreshed"): 1.0})
    assert t.loc[0, "correct"] == 1 and t.loc[0, "declined_ok"] == 1
    assert np.isnan(t.loc[1, "correct"])


def test_drift_quality_correlation():
    seg = pd.DataFrame({"segment_id": [f"s{i}" for i in range(6)], "loss_correct": [0, .1, .2, .3, .4, .5],
                        "baseline": False, "class": an.HEALTHY})
    drift = pd.DataFrame({"segment_id": [f"s{i}" for i in range(6)], "mmd2": [1, 2, 3, 4, 5, 6],
                          "jsd_excess": [6, 5, 4, 3, 2, 1]})
    c = an.drift_quality_correlation(seg, drift).set_index("measure")
    assert c.at["mmd2", "spearman"] == 1 and c.at["jsd_excess", "spearman"] == -1


def test_evidence_loop_grades_ranks_6_to_10_and_concludes(tmp_path):
    from driftmon.agents.retrieval_eval import retrieval_evaluation
    from driftmon.graph import DONE, MORE_EVIDENCE, _request_evidence, route_after_impact
    from driftmon.schemas import RetrievalReport
    state, api = _state(tmp_path)
    state["config"]["retrieval"].update(top_k=5, judge_k=3)
    state.update(impact_assessment(state))
    assert route_after_impact(state) == DONE                      # nothing borderline: no loop
    state["impact"] = state["impact"].model_copy(update={"borderline": [FLAGGED]})
    assert route_after_impact(state) == MORE_EVIDENCE
    state.update(_request_evidence(state))
    state["retrieval"] = RetrievalReport(mode="full_evaluation", questions=0, questions_by_status={},
                                         index_store={}, known_item=[], known_item_by_status=[], ranking_change=[])
    state.update(retrieval_evaluation(state))                     # C, in deepen mode
    deep = state["retrieval_tables"]["deep"]
    assert deep["judge"]["requests"] > 0 and set(deep["answer_deep"]["question_id"]).issubset(
        set(state["retrieval_tables"]["questions"]["question_id"]))
    state.update(impact_assessment(state))                        # D, evidence round
    ev = state["impact"].evidence
    assert len(ev) == 1 and ev[0]["segment_id"] == FLAGGED and ev[0]["conclusion"] == an.CONFIRMED
    assert state["deepen"] == [] and route_after_impact(state) == DONE  # bounded: at most one loop


def test_evidence_conclusion():
    assert an.evidence_conclusion(0.0, True, 0.1) == an.CONFIRMED
    assert an.evidence_conclusion(0.3, False, 0.1) == an.JUST_OUTSIDE
    assert an.evidence_conclusion(0.0, False, 0.1) == an.UNCLEAR
