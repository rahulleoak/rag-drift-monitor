"""Tests for Agent E's rules (D7, D29): each action fires on its evidence pattern."""
import pandas as pd
import pytest

from driftmon import policy as pol
from driftmon.config import load_config
from driftmon.evaluation.answers import DRIFT_HARMLESS, DRIFT_HURTS, HARM_WITHOUT_DRIFT, JUST_OUTSIDE, UNCLEAR
from driftmon.schemas import EvidenceItem, ImpactReport, Recommendation

BASE = "2016-17"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _seg(sid, cls, loss=0.9, refreshed=1.0, retrieval=1.0, cause=pol.MISSING_CONTENT, n=10):
    period, cat = sid.split("|")
    return {"segment_id": f"{sid}|vs {BASE}", "period": period, "category": cat, "questions": n,
            "stale_correct": refreshed - loss, "refreshed_correct": refreshed, "loss_correct": loss,
            "loss_low": loss - 0.05, "loss_high": loss + 0.05, "refreshed_retrieval": retrieval,
            "class": cls, "cause": cause}


def _period(period, stale, refreshed, n=50, invented=0.0):
    return {"period": period, "questions": n, "stale_correct": stale, "refreshed_correct": refreshed,
            "stale_hallucinated": invented, "refreshed_hallucinated": invented, "stale_declined_ok": 1.0}


def _impact(segments, by_period=None, correlations=None, evidence=None, borderline=None):
    return ImpactReport(questions=100, threshold=0.1, segments=segments,
                        by_period=by_period or [_period(BASE, 1.0, 1.0), _period("2020", 0.0, 0.98)],
                        correlations=correlations or [], evidence=evidence or [], borderline=borderline or [])


def _x(cfg, impact, **kw):
    return pol.Inputs(cfg=cfg, impact=impact, **kw)


def test_reindex_on_a_schedule_when_drift_misses_harm(cfg):
    imp = _impact([_seg("2020|WELLNESS", DRIFT_HURTS), _seg("2020|POLITICS", HARM_WITHOUT_DRIFT)],
                  correlations=[{"measure": "mmd2", "spearman": 0.2, "p_value": 0.4}])
    r = pol.reindex(_x(cfg, imp))
    assert r.status == "recommended" and "on a schedule" in r.finding
    assert set(r.scope) == {"2020 WELLNESS", "2020 POLITICS"}
    assert any("not flagged by drift detection" in e.claim for e in r.evidence)


def test_reindex_on_drift_alerts_when_drift_finds_every_loss(cfg):
    imp = _impact([_seg("2020|WELLNESS", DRIFT_HURTS)],
                  correlations=[{"measure": "mmd2", "spearman": 0.8, "p_value": 0.01}])
    r = pol.reindex(_x(cfg, imp))
    assert r.status == "recommended" and "whenever drift detection flags" in r.finding


def test_no_reindex_without_material_loss(cfg):
    r = pol.reindex(_x(cfg, _impact([_seg("2020|WELLNESS", "healthy", loss=0.0, cause="")])))
    assert r.status == "not needed"


def test_rules_report_not_checked_without_their_inputs(cfg):
    x = pol.Inputs(cfg=cfg)
    for rule in pol.RULES:
        assert rule(x).status == "not checked", rule.__name__


def test_poor_refreshed_index_with_poor_retrieval_triggers_model_evaluation(cfg):
    bad = _seg("2020|WELLNESS", DRIFT_HURTS, loss=0.2, refreshed=0.3, retrieval=0.2, cause=pol.RETRIEVAL_OR_MODEL)
    x = _x(cfg, _impact([bad]))
    me = pol.model_evaluation(x)
    assert me.status == "recommended" and me.scope == ["2020 WELLNESS"]
    assert pol.retraining(x, me).status == "not checked"      # only after the evaluation has run


def test_model_evaluation_not_needed_when_reindexing_fixes_everything(cfg):
    x = _x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)]))
    me = pol.model_evaluation(x)
    assert me.status == "not needed" and pol.retraining(x, me).status == "not needed"


def test_found_but_wrong_answers_point_at_the_prompt(cfg):
    step = _seg("2020|WELLNESS", DRIFT_HURTS, refreshed=0.4, retrieval=0.9, cause=pol.ANSWER_STEP)
    assert pol.prompts(_x(cfg, _impact([step]))).status == "recommended"
    invented = _impact([_seg("2020|WELLNESS", DRIFT_HURTS)],
                       by_period=[_period(BASE, 1.0, 1.0, invented=0.2), _period("2020", 0.0, 0.9, invented=0.2)])
    assert pol.prompts(_x(cfg, invented)).status == "recommended"
    assert pol.prompts(_x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)]))).status == "not needed"


def test_answers_just_outside_the_top_5_update_retrieval_settings(cfg):
    ev = [{"segment_id": f"2020|WELLNESS|vs {BASE}", "conclusion": JUST_OUTSIDE, "stale_correct_top5": 0.1,
           "stale_correct_top10": 0.6}]
    r = pol.retrieval_config(_x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)], evidence=ev)))
    assert r.status == "recommended" and "10%" in r.evidence[0].claim and "60%" in r.evidence[0].claim


def test_escalation_reasons(cfg):
    # harm without drift
    r = pol.escalate(_x(cfg, _impact([_seg("2020|POLITICS", HARM_WITHOUT_DRIFT)])))
    assert r.status == "recommended" and "2020 POLITICS" in r.scope and "Lesson" in r.finding
    # an unclear loop result
    ev = [{"segment_id": f"2020|WELLNESS|vs {BASE}", "conclusion": UNCLEAR, "loss_top5": 0.2, "loss_top10": 0.15,
           "loss_top10_low": -0.05, "loss_top10_high": 0.35}]
    r = pol.escalate(_x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)], evidence=ev)))
    assert r.status == "recommended" and "unclear" in r.finding
    # a judge that disagrees with the hand labels
    labels = pd.DataFrame([{"who": "a labeller", "part": "relevance", "answers_agreement": 0.5, "agreement": 0.4}])
    r = pol.escalate(_x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)]), labels=labels))
    assert r.status == "recommended" and "the relevance judge" in r.scope
    # none of them
    ok = pd.DataFrame([{"who": "a labeller", "part": "relevance", "answers_agreement": 0.9, "agreement": 0.8}])
    assert pol.escalate(_x(cfg, _impact([_seg("2020|WELLNESS", DRIFT_HURTS)]), labels=ok)).status == "not needed"


def test_drift_without_harm_means_keep_monitoring(cfg):
    from types import SimpleNamespace
    drift = SimpleNamespace(whole_period=[], segments=[], drifted_segments=[])
    imp = _impact([_seg("2020|WELLNESS", DRIFT_HARMLESS, loss=0.0, cause="")])
    r = pol.monitor(_x(cfg, imp, drift=drift))
    assert r.status == "recommended" and "2020 WELLNESS" in r.scope


def _longform(useful=True, beyond=0.9, truncated=0.5, chunks=0.87):
    changes = pd.DataFrame({"title": ["Germany", "Germany", "Delhi"], "period": ["2018", "2020", "2021"],
                            "page_id": [1, 1, 2]})
    verdict = {"useful": useful, "flagged_changed": 0.5, "comparison_changed": 0.02, "p_value": 1e-9}
    summ = pd.DataFrame([{"period": "2022", "median_tokens": 8000, "share_of_text_beyond_512": beyond}])
    ev = pd.DataFrame({"method": ["truncated", "averaged", "chunks_256"],
                       "spearman_all": [truncated, 0.78, chunks]})
    return {"baseline": BASE, "fact_changes": changes, "outdated_verdict": verdict,
            "longform_summary": summ, "chunking_eval": ev}


def test_changed_facts_that_change_answers_mean_targeted_reindex(cfg):
    r = pol.reindex_documents(_x(cfg, None, longform=_longform()))
    assert r.status == "recommended" and r.scope == ["Germany", "Delhi"]
    assert pol.reindex_documents(_x(cfg, None, longform=_longform(useful=False))).status == "not needed"


def test_chunking_only_when_documents_are_long_and_chunks_help(cfg):
    assert pol.chunking(_x(cfg, None, longform=_longform())).status == "recommended"
    assert pol.chunking(_x(cfg, None, longform=_longform(beyond=0.1))).status == "not needed"
    assert pol.chunking(_x(cfg, None, longform=_longform(chunks=0.52))).status == "not needed"


def test_a_recommendation_without_evidence_is_rejected():
    bare = Recommendation(action="a", key=pol.REINDEX, status="recommended", rule="r", finding="f")
    assert pol.check_evidence(bare).status == "rejected"
    no_numbers = bare.model_copy(update={"evidence": [EvidenceItem(source="Agent D", claim="it got worse")]})
    assert pol.check_evidence(no_numbers).status == "rejected"
    ok = bare.model_copy(update={"evidence": [EvidenceItem(source="Agent D", claim="loss 90%")]})
    assert pol.check_evidence(ok).status == "recommended"


def test_decide_considers_every_action_once_and_ranks_recommended_first(cfg):
    imp = _impact([_seg("2020|WELLNESS", DRIFT_HURTS), _seg("2020|POLITICS", HARM_WITHOUT_DRIFT)])
    recs = pol.decide(_x(cfg, imp, longform=_longform()))
    assert sorted(r.key for r in recs) == sorted(pol.ACTIONS)
    statuses = [r.status for r in recs]
    n = statuses.count("recommended")
    assert statuses[:n] == ["recommended"] * n and [r.priority for r in recs[:n]] == list(range(1, n + 1))
    kinds = [pol.KIND[r.key] for r in recs[:n]]
    assert kinds == sorted(kinds)                 # measured answer loss first, maintenance last
    assert recs[0].key == pol.REINDEX


def test_applied_reindex_is_cited_and_a_shortfall_asks_for_a_scheduled_full_reindex(cfg):
    applied = {"enough": False, "recovered": 0.77, "min_recovered": 0.8, "outdated_stale": 0.129,
               "outdated_targeted": 0.03, "targeted_share": 0.64,
               "by_stratum": {"listed_number": {"items": 40, "outdated_stale": 0.675, "outdated_targeted": 0.0}}}
    r = pol.reindex_documents(_x(cfg, None, longform={**_longform(), "reindex_verdict": applied}))
    assert r.status == "recommended" and "scheduled full re-index" in r.finding
    claim = next(e.claim for e in r.evidence if e.source == "Wikipedia: targeted re-index applied")
    assert "13%" in claim and "3%" in claim and "64%" in claim
    ok = pol.reindex_documents(_x(cfg, None, longform={**_longform(), "reindex_verdict": {**applied, "enough": True}}))
    assert "alone was enough" in ok.finding
    plain = pol.reindex_documents(_x(cfg, None, longform=_longform()))
    assert all(e.source != "Wikipedia: targeted re-index applied" for e in plain.evidence)
