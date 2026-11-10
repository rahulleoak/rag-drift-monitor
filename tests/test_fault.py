"""The simulated failed re-index (D34) and how the dashboard shows the evidence loop."""
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftmon.dashboard import story
from driftmon.dashboard.runs import list_runs, load_run, run_label
from driftmon.rag import fault as ft
from driftmon.rag.index import build_study_indexes

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "runs"
FAULT = {"period": "2020", "category": "WORLD NEWS", "drop_share": 0.5}
LOOP_ROW = {"segment_id": "2020|WORLD NEWS|vs 2016-17", "questions": 20, "loss_top5": 0.15, "loss_top10": 0.2,
            "loss_top10_low": 0.0, "loss_top10_high": 0.4, "stale_correct_top5": 0.0, "stale_correct_top10": 0.0,
            "conclusion": "still unclear (escalate to a human)"}


def _docs():
    rows = [("b1", "2016-17", "WORLD NEWS"), ("b2", "2016-17", "POLITICS"),
            *[(f"w{i}", "2020", "WORLD NEWS") for i in range(4)], ("p1", "2020", "POLITICS"),
            ("x1", "2021", "WORLD NEWS")]
    docs = pd.DataFrame(rows, columns=["doc_id", "period", "category"])
    rng = np.random.default_rng(0)
    v = rng.normal(size=(len(docs), 8)).astype(np.float32)
    return docs, v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dropped_ids_come_from_the_one_section_and_share():
    docs, _ = _docs()
    drop = ft.dropped_doc_ids(docs, FAULT, seed=0)
    assert len(drop) == 2 and set(drop) <= {"w0", "w1", "w2", "w3"}
    assert drop == ft.dropped_doc_ids(docs, FAULT, seed=0)          # repeatable


def test_dropped_articles_leave_every_refreshed_index_but_not_the_stale_one():
    docs, v = _docs()
    idx = build_study_indexes(docs, v, "2016-17", ["2020", "2021"])
    drop = ft.dropped_doc_ids(docs, FAULT, seed=0)
    out = ft.without_documents(idx, docs, v, drop)
    assert out.stale is idx.stale
    for p in ("2020", "2021"):
        assert not set(drop) & set(out.refreshed[p].doc_ids)
        assert len(out.refreshed[p]) == len(idx.refreshed[p]) - 2
    ids, _ = out.refreshed["2020"].search(v[[docs.index[docs.doc_id == "p1"][0]]], 1)
    assert ids[0][0] == "p1"                                          # vectors still line up with their ids


@pytest.fixture(scope="module")
def run():
    return load_run(DEMO, list_runs(DEMO)[0])


def test_story_names_the_loop_and_the_failure(run):
    assert "evidence loop" not in story.route_sentence(run) and story.fault_banner(run) == ""
    looped = dataclasses.replace(run, impact=run.impact.model_copy(update={"evidence": [LOOP_ROW]}),
                                 info={**run.info, "fault_injection": FAULT})
    assert "1 borderline section went back to Agent C" in story.route_sentence(looped)
    text = story.evidence_loop_takeaway(looped)
    assert "2020 WORLD NEWS (20 questions)" in text and "15%" in text and "escalate" in text
    assert "50% of the 2020 WORLD NEWS articles" in story.fault_banner(looped)


def test_dashboard_shows_the_label_banner_and_loop(tmp_path, monkeypatch, run):
    from streamlit.testing.v1 import AppTest
    rid = run.run_id
    dest = tmp_path / "runs" / rid
    shutil.copytree(DEMO / rid, dest)
    info = json.loads((dest / "run_info.json").read_text(encoding="utf-8"))
    info.update(label="evidence loop demo", fault_injection=FAULT)
    (dest / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    imp = json.loads((dest / "impact.json").read_text(encoding="utf-8"))
    imp["evidence"] = [LOOP_ROW]
    (dest / "impact.json").write_text(json.dumps(imp), encoding="utf-8")
    assert run_label(tmp_path / "runs", rid).endswith("· evidence loop demo")

    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(tmp_path / "runs"))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Simulated failure" in w.value for w in at.warning)
    at.sidebar.radio[0].set_value("Impact").run()
    assert not at.exception
    assert any("Evidence loop" in s.value for s in at.subheader)
    assert any("escalate" in m.value for m in at.markdown)
