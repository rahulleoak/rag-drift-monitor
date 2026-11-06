"""Tests for the query stream and query drift (driftmon.queries)."""
import numpy as np
import pandas as pd

from driftmon import queries

RNG = np.random.default_rng(0)
OLD = ["recipe", "garden", "budget", "senate", "travel", "school", "music", "movie"]
NEW = ["virus", "mask", "lockdown", "vaccine", "quarantine", "testing"]


def _stream(shift: float = 1.0, n: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    rows, vecs = [], []
    centre = RNG.normal(size=16)
    moved = centre + shift * 3 * RNG.normal(size=16)
    for period, dates, words, c in [("2016-17", ["2016-03-01", "2017-03-01"], OLD, centre),
                                    ("2020", ["2020-04-01"], NEW if shift else OLD, moved)]:
        for i in range(n):
            rows.append({"period": period, "date": dates[i % len(dates)], "set": "answerable",
                         "sample_pool": "random", "question": "what about the " + " ".join(RNG.choice(words, 5))})
            vecs.append(c + RNG.normal(size=16))
    v = np.array(vecs)
    return pd.DataFrame(rows), v / np.linalg.norm(v, axis=1, keepdims=True)


def test_shifted_queries_are_significant_and_read_against_an_ordinary_year():
    stream, vecs = _stream()
    t = queries.query_drift_table(stream, vecs, "2016-17", ["2020"], permutations=100)
    assert list(t["period"]) == ["2017 (inside baseline)", "2020"]
    ref, p2020 = t.iloc[0], t.iloc[1]
    assert not ref["significant"] and np.isnan(ref["mmd_q"])         # the reference is not a test
    assert p2020["significant"] and p2020["mmd_q"] <= 0.05 and p2020["jsd_q"] <= 0.05
    assert p2020["new_term_rate"] > 0.3
    assert set(p2020["emerging_terms"].split(", ")) <= set(NEW)
    assert p2020["examples"] and all(p2020["emerging_terms"].split(", ")[0] in q for q in p2020["examples"])


def test_unchanged_queries_are_not_significant():
    stream, vecs = _stream(shift=0.0)
    t = queries.query_drift_table(stream, vecs, "2016-17", ["2020"], permutations=100)
    assert not t.iloc[1]["significant"]


def test_stream_uses_only_pools_drawn_alike_in_every_period():
    stream, _ = _stream(n=10)
    stream.loc[:4, "sample_pool"] = "flagged"   # chosen by document drift: not part of the stream
    cfg = {"evaluation": {"query_stream_pools": ["random", "oversample"]}}

    class FakeEmbedder:
        def embed_queries(self, texts):
            return np.ones((len(texts), 4))
    got = queries.query_stream(cfg, stream, FakeEmbedder())
    assert got is not None and len(got[0]) == len(stream) - 5 and (got[0]["sample_pool"] == "random").all()
    assert queries.query_stream(cfg, None) is None


def test_stream_summary():
    stream, _ = _stream(n=10)
    stream.loc[stream.index[-1], "set"] = "unanswerable"
    s = queries.stream_summary(stream, "2016-17", ["2020"]).set_index("period")
    assert s.loc["2016-17", "queries"] == 10 and s.loc["2020", "unanswerable_share"] == 0.1
    assert s.loc["2020", "new_term_rate"] > 0.3 and "new_term_rate" in s.columns
