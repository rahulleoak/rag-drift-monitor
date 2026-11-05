"""Tests for the charts and the dashboard, on a run built from a synthetic archive."""
import os
from pathlib import Path

import pandas as pd
import pytest

from driftmon.dashboard_data import list_runs, load_run, severity_grid
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


def test_make_all_charts(run_dir, tmp_path):
    cfg = _config()
    names = make_all(cfg, run_dir, [run_dir / "charts", tmp_path / "copy"])
    assert {"drift_per_year.png", "severity_by_section.png"} <= set(names)
    for n in names:
        assert (run_dir / "charts" / n).stat().st_size > 5_000
        assert (tmp_path / "copy" / n).exists()


def test_dashboard_renders_every_tab(run_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DRIFTMON_RUNS_DIR", str(run_dir.parent))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Data monitoring", "Drift", "Retrieval",
                                          "Impact", "Recommendations"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Flagged (vs baseline)"] == "2"
    assert metrics["Next step"] == "full evaluation"
    segment = next(s for s in at.selectbox if s.label == "Segment")
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
