"""Tests for the outdated-answer test (D27 harm step, D28)."""
import json
import re

import pandas as pd

from driftmon.evaluation import outdated as od
from driftmon.config import load_config


def _snaps():
    rows = []
    texts = {"2016": "The bridge, built in 1932, is 120 metres long and crosses the river. "
                     "The old market town on the river has 8,000 people.",
             "2016-17": "The bridge, built in 1932, is 125 metres long and crosses the river. "
                        "The old market town by the river has 8,000 people."}
    for period, text in texts.items():
        rows.append({"page_id": 1, "title": "Town", "event": False, "period": period,
                     "sections": json.dumps([["Lead", text]])})
    return pd.DataFrame(rows)


def test_edit_items_puts_changed_and_reworded_sentences_in_their_groups():
    items = od.edit_items(_snaps(), ["2016", "2016-17"], 0.6)
    g = items.set_index("group")
    assert set(items["group"]) == {od.FLAGGED, od.COMPARISON}
    assert "125 metres" in g.at[od.FLAGGED, "after"] and "by the river" in g.at[od.COMPARISON, "after"]
    s = od.sample_items(items, 5, seed=0)
    assert s["item_id"].is_unique and set(s["item_id"].str[0]) == {"F", "C"}


def test_question_writer_never_sees_the_old_sentence_or_group():
    cfg = load_config()
    items = od.edit_items(_snaps(), ["2016", "2016-17"], 0.6)
    for r in od.question_requests(items, cfg):
        assert "120 metres" not in r.prompt and not re.search("flagged|comparison|before", r.prompt, re.I)


def test_change_judge_is_not_told_which_version_is_which():
    cfg = load_config()
    reqs = od.change_requests([f"i{i}" for i in range(20)], ["Q?"] * 20, ["OLD"] * 20, ["NEW"] * 20, cfg)
    firsts = [r.prompt.split("Answer 1: ")[1].split("\n")[0] for r in reqs]
    assert set(firsts) == {"OLD", "NEW"}                  # order shuffled
    assert not any(re.search("old|new|version", r.prompt.replace("OLD", "").replace("NEW", ""), re.I)
                   for r in reqs)


def test_verdict_needs_a_large_and_significant_difference():
    cfg = load_config()

    def items(f_changed, c_changed, n=60):
        return pd.DataFrame({"group": [od.FLAGGED] * n + [od.COMPARISON] * n,
                             "verdict": ["different"] * f_changed + ["same"] * (n - f_changed)
                             + ["different"] * c_changed + ["same"] * (n - c_changed)})
    assert od.verdict(items(30, 5), cfg)["useful"] is True
    assert od.verdict(items(12, 5), cfg)["useful"] is False      # difference below 20 points
    assert od.verdict(items(3, 1, n=4), cfg)["useful"] is False  # big difference, too few items


def test_contains_sentence_ignores_whitespace():
    assert od.contains_sentence("The town has  8,000 people.", ["Intro.\nThe town has 8,000\npeople. More."])
    assert not od.contains_sentence("The town has 9,000 people.", ["The town has 8,000 people."])
