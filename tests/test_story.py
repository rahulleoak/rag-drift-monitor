"""The dashboard's takeaways are built from the run's numbers, and their wording follows the result."""
import copy
import dataclasses
from pathlib import Path

import pytest

from driftmon.dashboard import story
from driftmon.dashboard.runs import default_run, list_runs, load_run

DEMO = Path(__file__).resolve().parents[1] / "demo" / "runs"


@pytest.fixture(scope="module")
def run():
    return load_run(DEMO, default_run(DEMO, list_runs(DEMO)))


def _with_impact(run, **changes):
    """A copy of the run whose impact report has some fields replaced."""
    return dataclasses.replace(run, impact=run.impact.model_copy(update=changes))


def test_overview_states_the_main_findings_from_the_numbers(run):
    text = " ".join(story.overview_findings(run))
    h = story.answer_headline(run)
    assert f"{h['questions']} questions" in text
    assert "0.3%" in text and "99%" in text
    assert "19 of 19 sections" in text and "flagged only 7" in text
    assert "biggest change is 2020" in text and "hides gradual drift" in text


def test_control_sentence_follows_the_result(run):
    assert "100% on both" in story.answers_takeaway(run)
    rows = copy.deepcopy(run.impact.by_period)
    base = next(r for r in rows if str(r["period"]) == run.baseline)
    base["refreshed_correct"] = 0.9                     # suppose re-indexing hurt old questions
    text = story.answers_takeaway(_with_impact(run, by_period=rows))
    assert "on both" not in text and "100% stale vs 90% refreshed" in text


def test_harm_sentence_when_drift_detection_caught_everything(run):
    segs = copy.deepcopy(run.impact.segments)
    for s in segs:
        if s["class"] == "harm without drift":
            s["class"] = "drift that hurts"
    changed = _with_impact(run, segments=segs)
    assert "flagged all of them" in " ".join(story.overview_findings(changed))
    assert "missed" not in story.harm_takeaway(changed)


def test_no_gradual_drift_claim_unless_the_numbers_show_it(run):
    whole = run.whole.copy()
    last = run.monitoring.periods[-1]
    mask = (whole["period"].astype(str) == last) & (whole["compared_with"] != run.baseline)
    whole.loc[mask, "embedding_band"] = "medium"      # last year no longer looks ordinary year on year
    text = " ".join(story.overview_findings(dataclasses.replace(run, whole=whole)))
    assert "hides gradual drift" not in text


def test_every_takeaway_is_empty_when_the_run_lacks_the_data(run):
    bare = dataclasses.replace(run, impact=None, retrieval=None, longform={})
    assert story.answers_takeaway(bare) == story.harm_takeaway(bare) == ""
    assert story.retrieval_takeaway(bare) == story.ranking_takeaway(bare) == story.wikipedia_takeaway(bare) == ""
    assert story.correlation_takeaway(bare) == ""
    assert all("questions about" not in s for s in story.overview_findings(bare))


def test_route_sentence_names_the_decision(run):
    assert "ran the full answer check" in story.route_sentence(run)
    assert "regression check" in story.route_sentence(dataclasses.replace(run, next_step="regression_check"))


def test_wikipedia_takeaway_includes_the_hidden_point(run):
    text = story.wikipedia_takeaway(run)
    assert "50%" in text and "2%" in text and "92%" in text and "95%" in text


def test_flagged_share_per_year(run):
    from driftmon.dashboard.runs import flagged_share_by_year
    f = flagged_share_by_year(run)
    assert list(f["period"]) == list(run.monitoring.periods)
    assert ((f["flagged"] <= f["groups"]) & (f["share"].between(0, 1))).all()
    text = story.flag_share_takeaway(run)
    top = f.sort_values("share").iloc[-1]
    assert f"{int(top['flagged'])} of {int(top['groups'])}" in text and "the most of any year" in text


def test_section_mix_stacks_to_one_and_marks_new_sections(run):
    from driftmon.dashboard.runs import OTHER, section_mix
    m = section_mix(run)
    sums = m.groupby("period", observed=True)["share"].sum()
    assert (sums - 1).abs().max() < 1e-9
    assert str(m["period"].iloc[0]) == run.baseline            # baseline first
    assert not m[m["period"] == run.baseline]["new"].any()     # nothing is new in the baseline itself
    y = m[m["period"] == run.monitoring.periods[1]]
    assert (y["y1"].diff().dropna() > 0).all() and abs(y["y1"].max() - 1) < 1e-9   # stacked without gaps
    assert y[y["new"]]["y0"].min() >= y[~y["new"]]["y1"].max() - 1e-9                # new sections on top
    assert OTHER in set(m["section"])
