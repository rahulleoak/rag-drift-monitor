"""Tests for the targeted re-index check (D32)."""
import json

import pandas as pd

from driftmon.config import load_config
from driftmon.evaluation import reindex as rx


def _snaps():
    texts = {
        1: ("The bridge, built in 1932, is 120 metres long and crosses the river.",
            "The bridge, built in 1932, is 125 metres long and crosses the river."),
        2: ("The old market town on the river is led by mayor Smith and his council.",
            "The old market town on the river is led by mayor Jones and his council."),
    }
    rows = []
    for pid, (old, new) in texts.items():
        for period, text in (("2016-17", old), ("2022", new)):
            rows.append({"page_id": pid, "title": f"Page {pid}", "event": False, "period": period,
                         "sections": json.dumps([["Lead", text]])})
    return pd.DataFrame(rows)


def test_listed_pages_ignore_changes_inside_the_baseline():
    f = pd.DataFrame({"page_id": [1, 2, 3], "period": ["2016-17", "2018", "2022"]})
    assert rx.listed_pages(f, "2016-17") == {2, 3}


def test_edit_strata_put_unlisted_name_changes_apart():
    items = rx.edit_strata(_snaps(), "2016-17", "2022", {1}, 0.6)
    by_page = items.set_index("page_id")["stratum"].to_dict()
    assert by_page == {1: "listed_number", 2: "unlisted"}
    s = rx.sample_strata(items, {"listed_number": 5, "unlisted": 5}, seed=0)
    assert s["item_id"].is_unique and set(s["item_id"].str[0]) == {"N", "U"}


def _chunks():
    rows = []
    for pid in (1, 2, 3):
        for period in ("2016-17", "2022"):
            if pid == 3 and period == "2016-17":
                continue  # an article new since the baseline
            rows.append({"chunk_id": f"{pid}-{period}", "page_id": pid, "period": period, "title": "t",
                         "section": "s", "text": "x"})
    return pd.DataFrame(rows)


def test_index_chunks_and_work():
    t = rx.index_chunks(_chunks(), "2016-17", "2022", {1})
    assert set(t["stale"]["chunk_id"]) == {"1-2016-17", "2-2016-17"}
    assert set(t["targeted"]["chunk_id"]) == {"1-2022", "2-2016-17"}
    assert set(t["full"]["page_id"]) == {1, 2, 3}
    w = rx.work(t)
    assert w["passages_targeted"] == 1 and w["passages_full"] == 3


def _judged(stale, targeted, strata, listed):
    return pd.DataFrame({"stratum": strata, "listed": listed, "verdict_stale": stale, "verdict_targeted": targeted,
                         "outdated_stale": rx.outdated(pd.Series(stale)),
                         "outdated_targeted": rx.outdated(pd.Series(targeted))})


def test_recovered_is_weighted_by_stratum_size():
    # listed: stale outdated 2 of 2, targeted fixes both; unlisted: stale outdated 1 of 2, targeted fixes none
    df = _judged(["different", "one_declined", "different", "same"], ["same", "same", "different", "same"],
                 ["listed_number", "listed_number", "unlisted", "unlisted"], [True, True, False, False])
    assert rx.recovered(df, {"listed_number": 100, "unlisted": 100}) == 1.0 / 1.5
    assert rx.recovered(df, {"listed_number": 300, "unlisted": 100}) == 3.0 / 3.5


def test_verdict_by_the_fixed_thresholds():
    cfg = load_config()
    good = _judged(["different"] * 10 + ["same"] * 10, ["same"] * 20,
                   ["listed_number"] * 10 + ["unlisted"] * 10, [True] * 10 + [False] * 10)
    v = rx.verdict(good, {"listed_number": 10, "unlisted": 10}, cfg)
    assert v["enough"] and v["recovered"] == 1.0 and v["listed_agreement"] == 1.0
    bad = _judged(["different"] * 20, ["same"] * 10 + ["different"] * 10,
                  ["listed_number"] * 10 + ["unlisted"] * 10, [True] * 10 + [False] * 10)
    v = rx.verdict(bad, {"listed_number": 10, "unlisted": 10}, cfg)
    assert not v["enough"] and abs(v["recovered"] - 0.5) < 1e-9
    assert rx.outdated(pd.Series(["both_declined", "same"])).sum() == 0
