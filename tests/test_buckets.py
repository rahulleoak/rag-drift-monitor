"""The four kinds of drift on the dashboard (D38): labels, sentences and pages."""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from driftmon import policy as pol
from driftmon.dashboard import buckets, story
from driftmon.dashboard.runs import load_run

DEMO = Path(__file__).resolve().parents[1] / "demo" / "runs"
NORMAL = "20261109-221754"
APP = str(DEMO.parents[1] / "app" / "streamlit_app.py")
SPLIT = [{"role": "refreshed", "answers": 423, "wrong": 5, "knowledge": 0, "retrieval": 5, "producing": 0,
          "unplaced": 0},
         {"role": "stale", "answers": 423, "wrong": 373, "knowledge": 373, "retrieval": 0, "producing": 0,
          "unplaced": 0}]


def test_every_cause_and_action_has_a_kind():
    assert set(buckets.ACTION) == set(pol.ACTIONS)
    assert set(buckets.CAUSE) == {pol.MISSING_CONTENT, pol.RETRIEVAL_OR_MODEL, pol.ANSWER_STEP}
    assert all(buckets.action_kinds(k) and buckets.action_note(k) for k in pol.ACTIONS)


def test_labels():
    assert buckets.cause_label(pol.MISSING_CONTENT) == "Knowledge · missing content (re-index)"
    assert buckets.cause_label(pol.ANSWER_STEP).startswith("Producing · ")
    assert buckets.cause_label(pol.RETRIEVAL_OR_MODEL).startswith("Retrieval · ")
    assert buckets.cause_label("") == "" and buckets.cause_label(float("nan")) == ""
    assert buckets.cause_label("something else") == "something else"
    assert buckets.action_kinds(pol.REINDEX) == "Knowledge"
    assert buckets.action_kinds(pol.PROMPTS) == "Query · Producing"
    assert buckets.action_kinds(pol.MODEL_EVAL) == "Retrieval"   # the embedding model: its rule fires on retrieval
    assert buckets.action_kinds(pol.MONITOR) == buckets.ALL
    assert "Fix: update the documents and re-index." in buckets.action_note(pol.REINDEX)


def _run(split):
    run = load_run(DEMO, NORMAL)
    return dataclasses.replace(run, impact=run.impact.model_copy(update={"attribution": split}))


def test_sentences_follow_the_numbers():
    run = _run(SPLIT)
    s = story.attribution_takeaway(run)
    assert s.startswith("On the stale index, all 373 wrong answers were about articles it never held (knowledge)")
    assert "all 5 wrong answers were search misses" in s and "answer step is not where quality is lost" in s
    o = story.overview_attribution(run)
    assert "all 373 on the stale index" in o and "all 5 on the refreshed index" in o and "producing: 0" in o
    assert o in story.overview_findings(run)
    mixed = [dict(SPLIT[0], wrong=7, retrieval=5, producing=2), SPLIT[1]]
    s = story.attribution_takeaway(_run(mixed))
    assert "of 7 wrong answers, 5 search misses" in s and "2 wrong with an answer in front of the model" in s
    assert "not where quality is lost" not in s
    assert "2 went wrong with the right article" in story.overview_attribution(_run(mixed))
    assert story.attribution_takeaway(_run([])) == "" and story.overview_attribution(_run([])) == ""


@pytest.fixture(scope="module")
def runs_with_split(tmp_path_factory):
    """The demo run, with the split saved as a new pipeline run would save it."""
    out = tmp_path_factory.mktemp("runs")
    shutil.copytree(DEMO / NORMAL, out / NORMAL)
    path = out / NORMAL / "impact.json"
    imp = json.loads(path.read_text(encoding="utf-8"))
    imp["attribution"] = SPLIT
    path.write_text(json.dumps(imp), encoding="utf-8")
    return out


def _app(runs, monkeypatch, page):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(runs))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    return at


def test_impact_page_names_the_kind(runs_with_split, monkeypatch):
    at = _app(runs_with_split, monkeypatch, "Impact")
    assert any(h.value == "Why answers went wrong" for h in at.subheader)
    tables = [d.value for d in at.dataframe]
    split = next(t for t in tables if "Producing" in t.columns)
    assert list(split["Knowledge"]) == [373, 0] and list(split["Retrieval"]) == [0, 5]
    causes = next(t for t in tables if "Likely cause" in t.columns)["Likely cause"]
    assert (causes == "Knowledge · missing content (re-index)").any()
    assert not causes.str.contains("^missing content", regex=True).any()   # every cause carries its kind


def test_recommendations_carry_a_kind_chip(runs_with_split, monkeypatch):
    at = _app(runs_with_split, monkeypatch, "Recommendations")
    text = " ".join(m.value for m in at.markdown)
    assert 'class="chip kind" title="Knowledge: the archive holds outdated content' in text
    assert ">All kinds</span>" in text and ">Query · Producing</span>" in text


def test_older_runs_without_the_split_still_render(monkeypatch):
    at = _app(DEMO, monkeypatch, "Impact")
    assert not any(h.value == "Why answers went wrong" for h in at.subheader)
