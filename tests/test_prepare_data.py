"""End-to-end test of scripts/prepare_data.py on a small synthetic raw file."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from driftmon.config import load_config
from driftmon.embeddings import Embedder
from scripts.prepare_data import check_raw_file, run


def fake_model(batch):
    """Random 32-d vector seeded by the text: identical texts match, others barely."""
    out = []
    for t in batch:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=32))
    return np.array(out)


TEN_WORDS = "one two three four five six seven eight nine ten"
RAW = [
    # baseline 2016-17
    ("2016-02-01", "POLITICS", "Senate passes budget", TEN_WORDS),
    ("2016-03-01", "HEALTHY LIVING", "Sleep matters more than you think", TEN_WORDS),
    ("2016-04-01", "POLITICS", "Senate passes budget", TEN_WORDS),           # copy of row 1
    ("2016-05-01", "POLITICS", "Headline only story here", ""),               # empty description
    ("2017-01-01", "COMEDY", "Ha", "Short."),                                  # too short
    ("2017-02-01", "TRAVEL", "Sunday Roundup", "word " * 90),                  # too long
    ("2017-03-01", "WORLD NEWS", "Elections abroad draw crowds", TEN_WORDS),
    # outside every study period
    ("2015-06-01", "POLITICS", "Old news from 2015 here", TEN_WORDS),
    # later period 2020
    ("2020-03-01", "POLITICS", "Coronavirus spreads across states", TEN_WORDS),
    ("2020-04-01", "WELLNESS", "Masks become common in cities", TEN_WORDS),
    ("2020-05-01", "U.S. NEWS", "Schools move classes online", TEN_WORDS),
]


def _setup(tmp_path):
    raw = tmp_path / "raw.json"
    with raw.open("w", encoding="utf-8") as f:
        for i, (d, cat, head, desc) in enumerate(RAW):
            f.write(json.dumps({"link": f"x{i}", "headline": head, "category": cat,
                                "short_description": desc, "authors": "", "date": d}) + "\n")
    cfg = copy.deepcopy(load_config())
    cfg["periods"]["baseline"] = {"name": "2016-17", "start": "2016-01-01", "end": "2017-12-31"}
    cfg["periods"]["later"] = [{"name": "2020", "start": "2020-01-01", "end": "2020-12-31"}]
    cfg["periods"]["sample_per_period"] = 2
    embedder = Embedder("fake", tmp_path / "cache", encode_fn=fake_model)
    return raw, cfg, embedder


def test_run_applies_rules_in_order_and_writes_outputs(tmp_path):
    raw, cfg, embedder = _setup(tmp_path)
    table = run(cfg, raw, tmp_path / "processed", tmp_path / "report", embedder)

    base = table.loc["2016-17"]
    assert base["raw"] == 7
    assert (base["empty_description"], base["too_short"], base["too_long"],
            base["near_duplicate"], base["kept"]) == (1, 1, 1, 1, 3)
    assert table.loc["2020", "kept"] == 3
    assert table.loc["total", "raw"] == 10  # the 2015 record is outside every period
    assert table.loc["2016-17", "in_drift_sample"] == 2 == table.loc["2020", "in_drift_sample"]

    docs = pd.read_parquet(tmp_path / "processed" / "documents.parquet")
    assert len(docs) == 6
    kept_budget = docs[docs.title == "Senate passes budget"]
    assert list(kept_budget.date.astype(str)) == ["2016-02-01"]  # earliest copy kept
    sleep = docs[docs.title.str.startswith("Sleep")].iloc[0]
    assert (sleep.category, sleep.category_raw) == ("WELLNESS", "HEALTHY LIVING")
    assert docs.groupby("period").in_drift_sample.sum().to_dict() == {"2016-17": 2, "2020": 2}

    removed = pd.read_csv(tmp_path / "report" / "removed.csv")
    assert sorted(removed.rule) == ["empty_description", "near_duplicate", "too_long", "too_short"]
    assert (tmp_path / "report" / "retention.csv").exists()


def test_run_is_repeatable_and_reuses_cached_embeddings(tmp_path):
    raw, cfg, embedder = _setup(tmp_path)
    run(cfg, raw, tmp_path / "p1", tmp_path / "r1", embedder)
    again = Embedder("fake", tmp_path / "cache", encode_fn=fake_model)
    run(cfg, raw, tmp_path / "p2", tmp_path / "r2", again)
    assert again.model_calls == 0
    a = pd.read_parquet(tmp_path / "p1" / "documents.parquet")
    b = pd.read_parquet(tmp_path / "p2" / "documents.parquet")
    assert a.equals(b)


def test_check_raw_file(tmp_path):
    f = tmp_path / "f.json"
    f.write_bytes(b"abc")
    good = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert check_raw_file(f, good) == []
    assert len(check_raw_file(f, {"bytes": 4, "sha256": "0" * 64})) == 2
    assert check_raw_file(f, {}) == []
