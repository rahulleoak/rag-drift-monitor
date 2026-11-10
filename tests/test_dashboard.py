"""Tests for the charts and the dashboard, on a run built from a synthetic archive."""
import os
from pathlib import Path

import pandas as pd
import pytest

from driftmon.dashboard.runs import list_runs, load_run, severity_grid, stale_vs_refreshed
from driftmon.schemas import RetrievalReport
from driftmon.graph import build_graph
from scripts.make_charts import latest_run, make_all
from tests.test_agents import _config, _synthetic_corpus

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A saved run in the same layout as scripts/run_pipeline.py writes."""
    runs = tmp_path_factory.mktemp("runs")
    docs, vecs = _synthetic_corpus()
    state = build_graph().invoke({"config": _config(), "docs": docs, "vectors": vecs})
    out = runs / "20260101-000000"
    out.mkdir()
    (out / "monitoring.json").write_text(state["monitoring"].model_dump_json(), encoding="utf-8")
    (out / "drift.json").write_text(state["drift"].model_dump_json(), encoding="utf-8")
    (out / "route.txt").write_text(state["next_step"] + "\n", encoding="utf-8")
    d = state["drift"]
    pd.DataFrame([s.model_dump() for s in d.whole_period]).to_csv(out / "drift_whole.csv", index=False)
    pd.DataFrame([s.model_dump() for s in d.segments]).to_csv(out / "drift_segments.csv", index=False)
    pd.DataFrame(d.topics).to_csv(out / "topics.csv", index=False)
    pd.DataFrame(d.term_shifts).to_csv(out / "term_shifts.csv", index=False)
    pd.DataFrame(d.metric_correlations).to_csv(out / "correlations.csv", index=False)
    judged = [{"period": p, "questions": 10, "stale_answer": s, "refreshed_answer": r, "delta_answer": r - s,
               "delta_answer_low": r - s - 0.1, "delta_answer_high": r - s + 0.1}
              for p, s, r in (("2016-17", 0.9, 0.9), ("2020", 0.3, 0.8))]
    report = RetrievalReport(
        mode="full_evaluation", questions=20, questions_by_status={"flagged": 10, "baseline": 10},
        index_store={"directory": "indexes", "reloaded": True}, known_item=[{"index": "refreshed_2020"}],
        known_item_by_status=[], ranking_change=[{"index": "refreshed_2020", "overlap": 0.8}],
        judged_by_period=judged, judge={"judged_questions": 20, "cost_usd": 0.12},
        worst_queries=[{"question": "Q?", "period": "2020", "category": "WELLNESS", "drift_status": "flagged",
                        "stale_ndcg": 0.0, "refreshed_ndcg": 1.0, "stale_top3": ["a"], "refreshed_top3": ["b"]}])
    (out / "retrieval.json").write_text(report.model_dump_json(), encoding="utf-8")
    (runs / "not-a-run").mkdir()  # ignored: no drift report
    return out


def test_list_and_load_run(run_dir):
    runs = run_dir.parent
    assert list_runs(runs) == ["20260101-000000"]
    assert latest_run(runs) == run_dir
    run = load_run(runs, "20260101-000000")
    assert run.baseline == "2016-17" and run.next_step == "full_evaluation"
    # in the synthetic archive POLITICS is the whole period, so both are flagged
    assert set(run.flagged()["segment_id"]) == {"2020|ALL|vs 2016-17", "2020|POLITICS|vs 2016-17"}
    grid = severity_grid(run)
    assert set(grid["category"]) == {"ALL SECTIONS", "POLITICS"}
    long = stale_vs_refreshed(run).set_index(["period", "index"])
    assert len(long) == 4 and long.loc[("2020", "Refreshed index"), "value"] == 0.8


def test_make_all_charts(run_dir, tmp_path):
    cfg = _config()
    names = make_all(cfg, run_dir, [run_dir / "charts", tmp_path / "copy"])
    assert {"drift_per_year.png", "severity_by_section.png"} <= set(names)
    for n in names:
        assert (run_dir / "charts" / n).stat().st_size > 5_000
        assert (tmp_path / "copy" / n).exists()
    dark = make_all(cfg, run_dir, [run_dir / "charts" / "dark"], theme="dark")
    assert dark == names and (run_dir / "charts" / "dark" / names[0]).exists()
    run = load_run(run_dir.parent, run_dir.name)
    assert f"dark/{names[0][:-4]}" in run.charts  # the dashboard finds the dark versions


def test_dashboard_renders_every_tab(run_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(run_dir.parent))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Data monitoring", "Drift", "Retrieval",
                                          "Impact", "Recommendations"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Groups flagged for drift"] == "2"
    assert "Questions tested" in metrics and "Found after re-indexing" in metrics
    assert any("ran the full answer check" in c.value for c in at.caption)   # the routing decision, in words
    segment = next(s for s in at.selectbox if s.label == "Group")
    segment.select("2020|POLITICS|vs 2016-17").run()
    assert not at.exception


def test_dashboard_without_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No pipeline runs" in w.value for w in at.warning)
    os.environ.pop("DRIFTMON_RUNS_DIR", None)


DEMO = Path(__file__).resolve().parents[1] / "demo" / "runs"


def test_the_bundled_demo_run_loads_with_every_agent():
    from driftmon.dashboard.runs import answers_stale_vs_refreshed, harm_vs_drift
    runs = list_runs(DEMO)
    assert len(runs) == 1, "demo/runs should hold exactly one run (scripts/export_demo.py)"
    run = load_run(DEMO, runs[0])
    assert run.impact is not None and run.recommendations is not None and run.retrieval is not None
    assert "outdated_summary" in run.longform and "outdated_verdict" in run.longform
    long = answers_stale_vs_refreshed(run)
    assert set(long["index"]) == {"Stale index (2016-17 only)", "Refreshed index"}
    seg = harm_vs_drift(run)
    assert len(seg) and not seg["class"].isin(["too few questions", "historical control"]).any()


def test_dashboard_renders_the_demo_run_on_every_tab(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(DEMO))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Correct without re-indexing", "Correct after re-indexing", "Recommended actions",
            "Questions answered", "Recommended", "Not needed", "Admits it doesn't know"} <= labels
    text = " ".join(m.value for m in at.markdown)
    assert "Checked, not recommended" in text and "Top recommendation" in text
    assert "In this run:" in text                       # takeaways built from the run are shown
    assert "(F11)" not in text and "D27:" not in text   # no internal document codes on screen


def test_export_demo_copies_only_what_the_dashboard_reads(tmp_path):
    from scripts.export_demo import export
    src = DEMO / list_runs(DEMO)[0]
    copied = export(src, tmp_path / "demo")
    out = tmp_path / "demo" / src.name
    assert (out / "impact.json").exists() and (out / "recommendations.json").exists()
    assert "longform/outdated_verdict.json" in copied
    assert list_runs(tmp_path / "demo") == [src.name]


def test_model_names_are_readable():
    from driftmon.dashboard.runs import model_name
    assert model_name("claude-sonnet-5-5") == "Claude Sonnet 5.5"
    assert model_name("claude-opus-5-5") == "Claude Opus 5.5"
    assert model_name("some-other-model") == "some-other-model"
    assert model_name(None) == "unknown"


def test_run_models_come_from_the_run_and_fall_back_to_the_config(tmp_path):
    import json
    import shutil
    from driftmon.dashboard.runs import run_models
    config = {"llm": {"generator_model": "claude-sonnet-5-5", "judge_model": "claude-opus-5-5"},
              "embeddings": {"model": "BAAI/bge-small-en-v1.5"}}
    src = DEMO / list_runs(DEMO)[0]
    run_path = tmp_path / src.name
    shutil.copytree(src, run_path)
    (run_path / "run_info.json").unlink(missing_ok=True)          # an older run: no provenance file
    old = run_models(load_run(tmp_path, src.name), config)
    assert old == {"writer": "Claude Sonnet 5.5", "judge": "Claude Opus 5.5",
                   "embedding": "BAAI/bge-small-en-v1.5", "source": "config"}
    info = {"models": {"writer": "claude-haiku-5-5", "judge": "claude-fable-5-1"}, "embedding_model": "e5-small"}
    (run_path / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    new = run_models(load_run(tmp_path, src.name), config)          # the run's own record wins
    assert new == {"writer": "Claude Haiku 5.5", "judge": "Claude Fable 5.1", "embedding": "e5-small",
                   "source": "run"}
    (run_path / "run_info.json").write_text(json.dumps({"models": None, "embedding_model": "e5"}), "utf-8")
    assert run_models(load_run(tmp_path, src.name), config)["writer"].startswith("none")


def test_run_labels_say_when_and_which_agents(tmp_path):
    from driftmon.dashboard.runs import run_label
    full, partial = tmp_path / "20261109-144329", tmp_path / "20261105-083957"
    for d, files in ((full, ["monitoring.json", "drift.json", "retrieval.json", "impact.json",
                             "recommendations.json"]), (partial, ["monitoring.json", "drift.json"])):
        d.mkdir()
        for f in files:
            (d / f).write_text("{}", encoding="utf-8")
    assert run_label(tmp_path, full.name) == "9 Nov 2026, 14:43 · all agents"
    assert run_label(tmp_path, partial.name) == "5 Nov 2026, 08:39 · agents A–B"
    assert run_label(tmp_path, "my-run") == "my-run · no reports"
