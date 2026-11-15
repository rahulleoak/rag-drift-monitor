"""The truth table (D39): one change at a time, same articles; code decides where losses land."""
import numpy as np
import pandas as pd

from driftmon.config import load_config
from driftmon.evaluation import answers as an
from driftmon.evaluation import truth_table as tt
from driftmon.llm.client import LLMResult
from driftmon.llm.prompts import ANSWER_SYSTEM, AnswerJudgement, RagAnswer

CFG = load_config()
N = 10   # questions whose source article is in both indexes


def _data():
    """N questions from 2020 articles plus one question whose article the stale index lacks."""
    qs = [{"question_id": f"q{i}", "set": "answerable", "period": "2020", "category": "POLITICS",
           "question": f"What happened {i}?", "reference_answer": f"Fact {i}."} for i in range(N + 1)]
    docs = pd.DataFrame([{"doc_id": f"d{i}", "title": f"Title {i}", "text": f"Text {i}.", "date": "2020-01-01"}
                         for i in range(N + 1)])
    runs, saved = [], []
    for i in range(N + 1):
        for role, index in (("stale", "stale"), ("refreshed", "refreshed_2020")):
            held = not (i == N and role == "stale")
            runs.append({"question_id": f"q{i}", "index": index, "source_in_index": held,
                         "doc_ids": f"[np.str_('d{i}'), np.str_('d{(i + 1) % (N + 1)}')]"})
            saved.append({"question_id": f"q{i}", "role": role, "index": index,
                          "context_has_answer": 1.0 if held else 0.0})
    saved.append({"question_id": "u0", "role": "stale", "index": "stale", "context_has_answer": 0.0})
    return pd.DataFrame(qs), docs, pd.DataFrame(runs), pd.DataFrame(saved)


def _result(parsed):
    return LLMResult(key="k", parsed=parsed, text="", stop_reason="", model="m", input_tokens=0, output_tokens=0,
                     cost_usd=0.0)


def _scored(plan, questions, runs, cell, wrong=()):
    """Answers judged right where the top k held the article, except the given question ids."""
    answers = [_result(RagAnswer(found=True, answer="An answer of five words.", sources=["A"])) for _ in range(len(plan))]
    judgements = []
    for r in plan.to_dict("records"):
        right = r["context_has_answer"] == 1 and r["question_id"] not in wrong
        judgements.append(_result(AnswerJudgement(reason="r", correctness=2 if right else 0, grounded=True,
                                                  declined=False)))
    return tt.score(plan, answers, judgements, questions, runs, cell)


def test_parse_ids():
    assert tt.parse_ids("[np.str_('hp-1'), np.str_('hp-2')]") == ["hp-1", "hp-2"]
    assert tt.parse_ids(["hp-1"]) == ["hp-1"]


def test_plan_keeps_answerable_rows_with_a_saved_top_k():
    questions, _, runs, saved = _data()
    plan = tt.load_plan(saved, runs, k=1)
    assert len(plan) == 2 * (N + 1) and "u0" not in set(plan["question_id"])
    assert plan.loc[0, "doc_ids"] == ["d0"]


def test_each_cell_changes_one_thing():
    questions, docs, runs, saved = _data()
    plan = tt.load_plan(saved, runs, k=2)
    base = an.answer_requests(plan, questions, docs, CFG)
    cells = {c.name: tt.answer_requests(plan, questions, docs, CFG, c) for c in tt.CELLS}
    assert [r.key for r in cells["baseline"]] == [r.key for r in base]   # the cache serves the baseline
    p, m, b = cells["prompt"][0], cells["model"][0], cells["both"][0]
    assert p.model == base[0].model and p.system != ANSWER_SYSTEM and "under 10 words" in p.system
    assert p.prompt == base[0].prompt and p.prompt_version != base[0].prompt_version
    assert m.model == CFG["truth_table"]["small_model"] and m.system == ANSWER_SYSTEM and m.prompt == base[0].prompt
    assert b.model == m.model and b.system == p.system


def test_a_worse_answer_step_lands_in_producing():
    questions, _, runs, saved = _data()
    plan = tt.load_plan(saved, runs, k=2)
    scored = pd.concat([_scored(plan, questions, runs, "baseline"),
                        _scored(plan, questions, runs, "prompt", wrong={f"q{i}" for i in range(6)}),
                        _scored(plan, questions, runs, "model")], ignore_index=True)
    s = tt.summarise(scored).set_index(["cell", "role"])
    assert s.loc[("baseline", "stale"), "knowledge"] == 1 and s.loc[("baseline", "stale"), "producing"] == 0
    assert s.loc[("prompt", "refreshed"), "producing"] == 6 and s.loc[("prompt", "stale"), "knowledge"] == 1
    assert s.loc[("prompt", "refreshed"), "producing_share"] == 6 / (N + 1)
    v = tt.verdict(scored, CFG)
    # stale and refreshed hand over the same top k for q0-q9: one paired answer each, not two
    assert v["prompt"]["paired_answers"] == N + 1 and v["prompt"]["lost"] == 6 and v["prompt"]["gained"] == 0
    assert v["prompt"]["isolated"] and v["prompt"]["producing_caught"] and v["prompt"]["p"] < 0.05
    assert v["prompt"]["reading"].startswith("caught")
    assert v["model"]["isolated"] and not v["model"]["producing_caught"] and v["model"]["reading"].startswith("no loss")
    assert "both" not in v


def test_too_few_losses_are_not_called_caught_and_moved_buckets_are_flagged():
    questions, _, runs, saved = _data()
    plan = tt.load_plan(saved, runs, k=2)
    base = _scored(plan, questions, runs, "baseline")
    few = _scored(plan, questions, runs, "prompt", wrong={"q0", "q1"})
    v = tt.verdict(pd.concat([base, few], ignore_index=True), CFG)["prompt"]
    assert not v["producing_caught"] and v["p"] > 0.05 and v["reading"].startswith("no reliable change")
    # an answer step cannot move knowledge: if it does, the split is not isolating the cause
    moved = few.copy()
    moved.loc[moved["question_id"] == "q2", ["in_index", "correct"]] = [False, 0.0]
    moved.loc[moved["question_id"] == "q2", "bucket"] = "knowledge"
    v = tt.verdict(pd.concat([base, moved], ignore_index=True), CFG)["prompt"]
    assert not v["isolated"] and v["reading"].startswith("not isolated")
    assert np.isclose(v["moves"]["refreshed"]["knowledge"], 1)
