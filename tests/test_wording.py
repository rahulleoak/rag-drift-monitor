"""Tests for the newer-wording test (D31)."""
from collections import Counter

import numpy as np
import pandas as pd

from driftmon.config import load_config
from driftmon.evaluation import wording as wd
from driftmon.llm.prompts import MeaningCheck


def _questions():
    rows = []
    for i, (period, status) in enumerate([("2018", "flagged")] * 4 + [("2019", "control")] * 3
                                         + [("2022", "flagged")] * 2 + [("2016-17", "baseline")] * 2):
        rows.append({"question_id": f"q{i}", "doc_id": f"d{i}", "period": period, "category": "POLITICS",
                     "date": f"{period[:4]}-05-01", "question": f"What did the senator say about the plan {i}?",
                     "set": "answerable", "drift_status": status})
    rows.append({**rows[0], "question_id": "qu", "set": "unanswerable"})
    return pd.DataFrame(rows)


def test_sample_keeps_question_periods_answerable_and_caps_each_group():
    cfg = load_config()
    cfg["evaluation"]["newer_wording"]["per_group"] = 3
    s = wd.sample_questions(_questions(), cfg)
    assert set(s["period"]) <= set(cfg["evaluation"]["newer_wording"]["question_periods"])
    assert "qu" not in set(s["question_id"]) and s["group"].value_counts().max() <= 3
    assert set(s["group"]) == {"flagged", "control", "baseline"}


def test_rewording_writer_sees_question_and_date_not_the_article_or_group():
    cfg = load_config()
    items = wd.sample_questions(_questions().assign(text="SECRET ARTICLE TEXT"), cfg)
    for r in wd.rewording_requests(items, cfg):
        assert "SECRET" not in r.prompt and "flagged" not in r.prompt and "Article published" in r.prompt
        assert "late 2022" in r.system


def test_newer_terms_need_significance_and_a_rising_rate():
    old = Counter({"coronavirus": 200, "virus": 100, "said": 1000, "plan": 300})
    new = Counter({"covid": 150, "coronavirus": 5, "virus": 100, "said": 1000, "plan": 300})
    terms = wd.newer_terms({"2020": old, "2022": new}, "2020", "2022", 1.96, 1.2)
    assert "covid" in terms and "coronavirus" not in terms and "said" not in terms


def test_code_checks_reject_unchanged_or_unsupported_rewrites():
    items = pd.DataFrame([
        {"question": "How many coronavirus cases did Ohio report?", "period": "2020", "has_newer": True,
         "same_era": "What number of coronavirus cases did Ohio report?",
         "newer": "What number of COVID cases did Ohio report?"},
        {"question": "Who won the vote?", "period": "2020", "has_newer": False, "same_era": "Who won the ballot?",
         "newer": ""},
        {"question": "Who won the vote?", "period": "2020", "has_newer": True, "same_era": "Who won the ballot?",
         "newer": "Who triumphed in the ballot?"},
        {"question": "Who won the vote?", "period": "2020", "has_newer": True, "same_era": "Who won the vote?",
         "newer": "Who won the covid vote?"},
    ])
    out = wd.code_checks(items, {"2020": {"covid"}})
    assert out.loc[0, "rejected_by"] == "" and out.loc[0, "newer_words"] == "covid"
    assert out.loc[1, "rejected_by"] == "no newer wording"
    assert out.loc[2, "rejected_by"] == "no newer word by the data"
    assert "same_era equals original" in out.loc[3, "rejected_by"]


def test_meaning_judge_order_is_shuffled_and_mapped_back():
    cfg = load_config()
    items = pd.DataFrame([{"question": f"Q{i}", "same_era": f"SAME{i}", "newer": f"NEW{i}"} for i in range(20)])
    reqs, flips = wd.meaning_requests(items, cfg)
    assert 0 < flips.sum() < len(flips)
    for r, flip in zip(reqs, flips):
        assert (r.prompt.index("NEW") < r.prompt.index("SAME")) == bool(flip)
        assert "newer" not in r.prompt.lower() and "same_era" not in r.prompt
    # judge says only the rewording shown first keeps the meaning
    checks = [MeaningCheck(reason="x", rewording_1_same=True, rewording_2_same=False) for _ in reqs]
    back = wd.unshuffle(checks, flips)
    assert (back["newer_same"] == flips).all() and (back["same_era_same"] == ~flips).all()


def test_ranks_and_found_at_k():
    items = pd.DataFrame({"doc_id": ["a", "b"]})
    ranked = {"original": [["a", "x"], ["b", "x"]], "same_era": [["x", "a"], ["x", "y"]],
              "newer": [["x", "y", "z", "w", "v", "a"], ["b"]]}
    out = wd.add_ranks(items, ranked, hit_k=5)
    assert out["rank_original"].tolist() == [1, 1] and out["found_newer"].tolist() == [False, True]
    assert out["rr_same_era"].tolist() == [0.5, 0.0]


def _kept(same, newer, groups):
    return pd.DataFrame({"found_same_era": same, "found_newer": newer, "found_original": same,
                         "rr_original": 1.0, "rr_same_era": 1.0, "rr_newer": 1.0, "group": groups})


def test_verdict_flags_a_large_significant_drop_and_a_flagged_gap():
    cfg = load_config()
    n = 30
    kept = _kept([True] * 2 * n, [False] * n + [True] * n, ["flagged"] * n + ["control"] * n)
    v = wd.verdict(kept, cfg)
    assert v["lost"] == n and v["gained"] == 0 and v["wording_hurts_search"]
    assert v["flagged_drop"] == 1.0 and v["control_drop"] == 0.0 and v["drift_level_matters"]


def test_verdict_no_effect_when_versions_agree():
    cfg = load_config()
    kept = _kept([True, False] * 20, [True, False] * 20, ["flagged", "control"] * 20)
    v = wd.verdict(kept, cfg)
    assert v["drop"] == 0 and not v["wording_hurts_search"] and v["drift_level_matters"] is False
    assert v["mcnemar_p"] == 1.0
    s = wd.summarise(kept, cfg)
    assert set(s["group"]) == {"flagged", "control", "all"} and np.allclose(s["drop"], 0)


def test_mcnemar_is_one_sided():
    assert wd.mcnemar_p(10, 0) < 0.01 and wd.mcnemar_p(0, 10) > 0.9
