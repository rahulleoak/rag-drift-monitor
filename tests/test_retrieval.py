"""Tests for Agent C: retrieval metrics, saved indexes and the three comparisons."""
import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from driftmon.config import load_config
from driftmon.evaluation import retrieval as rv
from driftmon.evaluation import retrieval_metrics as rm
from driftmon.evaluation.stats import bootstrap_mean_ci
from driftmon.rag.index import load_or_build_study_indexes
from driftmon.llm.client import LLMClient
from tests.test_llm import FakeAPI, _msg


# ---------- metrics, hand-computed ----------

def test_known_item_metrics():
    assert rm.reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert rm.reciprocal_rank(["a", "b"], "z") == 0.0
    assert rm.recall_at_k(["a", "b", "c"], "c", 2) == 0.0 and rm.recall_at_k(["a", "b", "c"], "c", 3) == 1.0


def test_graded_metrics():
    assert rm.precision_at_k([2, 0, 1, 0, 0], 5) == pytest.approx(0.4)
    assert rm.precision_at_k([2, 0, 1], 5, min_grade=2) == pytest.approx(0.2)  # missing ranks count as 0
    assert rm.answer_at_k([1, 1, 2], 2) == 0.0 and rm.answer_at_k([1, 1, 2], 3) == 1.0
    # DCG of [2, 0, 1] = 3/log2(2) + 0 + 1/log2(4) = 3.5; ideal from the pool [2, 1, 1] = 3 + 1/log2(3) + 0.5
    ideal = 3 + 1 / np.log2(3) + 0.5
    assert rm.ndcg_at_k([2, 0, 1], [2, 1, 1, 0], 3) == pytest.approx(3.5 / ideal)
    assert rm.ndcg_at_k([0, 0], [0, 0, 0], 2) == 0.0
    assert rm.overlap_at_k(["a", "b", "c"], ["c", "a", "x"], 3) == pytest.approx(2 / 3)


def test_bootstrap_interval_contains_the_mean():
    mean, lo, hi = bootstrap_mean_ci([0.0, 1.0] * 50, resamples=500)
    assert mean == 0.5 and lo < 0.5 < hi and hi - lo < 0.3
    assert np.isnan(bootstrap_mean_ci([])[0])


# ---------- a tiny archive ----------

def _config(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["periods"]["baseline"] = {"name": "2016-17", "start": "2016-01-01", "end": "2017-12-31"}
    cfg["periods"]["later"] = [{"name": "2020", "start": "2020-01-01", "end": "2020-12-31"}]
    cfg["paths"]["indexes"] = str(tmp_path / "indexes")
    cfg["impact"]["bootstrap_resamples"] = 200
    cfg["retrieval"].update(top_k=5, judge_k=3)
    cfg["llm"]["prices"] = {cfg["llm"]["generator_model"]: {"input": 2, "output": 10},
                            cfg["llm"]["judge_model"]: {"input": 4, "output": 20}}
    return cfg


def _archive(n=30, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p, date, cat in (("2016-17", "2016-06-01", "WELLNESS"), ("2020", "2020-06-01", "WELLNESS")):
        for i in range(n):
            rows.append({"doc_id": f"{p}-{i}", "period": p, "date": date, "category": cat,
                         "title": f"Story topic{p[:4]}x{i}", "text": f"About topic{p[:4]}x{i}."})
    docs = pd.DataFrame(rows)
    vecs = rng.normal(size=(len(docs), dim))
    return docs, vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _questions(docs, vecs, per_period=8, seed=1):
    rng = np.random.default_rng(seed)
    rows, qv = [], []
    for p in ("2016-17", "2020"):
        idx = np.where(docs["period"] == p)[0][:per_period]
        for i in idx:
            d = docs.iloc[i]
            token = re.search(r"topic\S+?x\d+", d["text"]).group(0)
            rows.append({"question_id": f"{d['doc_id']}-q0", "doc_id": d["doc_id"], "period": p,
                         "category": d["category"], "set": "answerable", "sample_pool": "random",
                         "type_group": "evergreen", "question": f"What happened in {token}?",
                         "reference_answer": "Something."})
            v = vecs[i] + 0.1 * rng.normal(size=vecs.shape[1])
            qv.append(v / np.linalg.norm(v))
    q = pd.DataFrame(rows)
    # an unanswerable question: never retrieved or judged here (it is for Agent D)
    q.loc[len(q)] = {**q.iloc[0].to_dict(), "question_id": "x-q0", "set": "unanswerable"}
    qv.append(qv[0])
    return q, np.array(qv)


def _judge(kw, live):
    """Grade 2 for the article that names the question's topic, else 0."""
    prompt = kw["messages"][0]["content"]
    token = re.search(r"in (topic\S+?x\d+)\?", prompt).group(1)
    grades = [{"article": letter, "grade": 2 if re.search(rf"\b{token}\b", line) else 0}
              for letter, line in re.findall(r"^\[([A-Z])\] (.*)$", prompt, flags=re.M)]
    msg = _msg(json.dumps({"grades": grades}), model=kw["model"])
    return msg if live else ("succeeded", msg)


def test_indexes_are_saved_and_reloaded_only_when_unchanged(tmp_path):
    docs, vecs = _archive()
    d = tmp_path / "idx"
    first, loaded = load_or_build_study_indexes(docs, vecs, "2016-17", ["2020"], d, "m")
    assert not loaded and (d / "stale.faiss").exists() and (d / "refreshed_2020.faiss").exists()
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["indexes"]["stale"]["documents"] == 30 and manifest["indexes"]["refreshed_2020"][
        "periods"] == ["2016-17", "2020"]
    again, loaded = load_or_build_study_indexes(docs, vecs, "2016-17", ["2020"], d, "m")
    assert loaded and len(again.refreshed["2020"]) == 60
    ids_a, _ = first.refreshed["2020"].search(vecs[:3], 2)
    ids_b, _ = again.refreshed["2020"].search(vecs[:3], 2)
    assert (ids_a == ids_b).all()
    _, loaded = load_or_build_study_indexes(docs, vecs * 0.5, "2016-17", ["2020"], d, "m")
    assert not loaded  # different vectors: rebuilt, never a stale saved index


def test_three_comparisons(tmp_path):
    cfg = _config(tmp_path)
    docs, vecs = _archive()
    q, qv = _questions(docs, vecs)
    idx, _ = load_or_build_study_indexes(docs, vecs, "2016-17", ["2020"], tmp_path / "idx", "m")
    api = FakeAPI(_judge)
    llm = LLMClient(config=cfg["llm"], cache_dir=tmp_path / "cache", ledger=tmp_path / "l.csv",
                    client=api, sleep=lambda s: None)
    t = rv.evaluate(cfg, q, qv, docs, idx, flagged=["2020|WELLNESS|vs 2016-17"], controls=[], llm=llm,
                    batch=True)

    assert "x-q0" not in set(t["questions"]["question_id"])            # unanswerable not evaluated
    assert set(t["questions"]["drift_status"]) == {"baseline", "flagged"}
    # the judge never learns which index found an article
    prompts = [r["params"]["messages"][0]["content"] for r in api.batches[0]]
    assert len(prompts) == 16 and not any(re.search("stale|refreshed|index", p, re.I) for p in prompts)

    by = t["judged_by_period"].set_index("period")
    # (a) 2020 questions: the stale index cannot hold the answer; the refreshed one finds it
    assert by.loc["2020", "stale_answer"] == 0 and by.loc["2020", "refreshed_answer"] == 1
    assert by.loc["2020", "delta_ndcg_low"] > 0
    # baseline questions (historical control): no change between stale and refreshed
    assert by.loc["2016-17", "delta_answer"] == 0
    assert t["judge"]["judged_questions"] == 16 and t["judge"]["missing_grades"] == 0

    # (b) on the same refreshed index, both groups find their source (random vectors: no drift)
    k = t["known_summary"].set_index("asked_by")
    assert k.loc["period questions", "recall"] == 1 and k.loc["baseline questions", "recall"] == 1
    # (c) baseline questions: their source stays; newer articles may enter the top-k
    rc = t["ranking_summary"].iloc[0]
    assert rc["index"] == "refreshed_2020" and rc["source_displaced"] == 0 and 0 <= rc["newer_share"] <= 1

    worst = rv.worst_queries(t, docs, n=3)
    assert len(worst) == 3 and all(w["period"] == "2020" for w in worst)
    assert worst[0]["refreshed_top3"][0].startswith("Story topic2020")


def test_a_skipped_article_is_not_scored_as_irrelevant(tmp_path):
    cfg = _config(tmp_path)
    docs, vecs = _archive()
    q, qv = _questions(docs, vecs, per_period=2)

    def lazy(kw, live):  # grades only article A
        msg = _msg(json.dumps({"grades": [{"article": "A", "grade": 2}, {"article": "Z", "grade": 1}]}),
                   model=kw["model"])
        return msg if live else ("succeeded", msg)
    idx, _ = load_or_build_study_indexes(docs, vecs, "2016-17", ["2020"], tmp_path / "idx", "m")
    llm = LLMClient(config=cfg["llm"], cache_dir=tmp_path / "cache", ledger=tmp_path / "l.csv",
                    client=FakeAPI(lazy), sleep=lambda s: None)
    t = rv.evaluate(cfg, q, qv, docs, idx, [], [], llm, batch=False)
    assert not t["judged"]["judged"].any() and t["judge"]["missing_grades"] > 0
    assert t["judge"]["unknown_letters"] == 4
    assert t["judged"]["stale_ndcg"].isna().all()
