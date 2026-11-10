"""Tests for Agent E's node: explanations are number-checked, the report is complete (D29)."""
from driftmon.agents import recommendation as rec
from driftmon.config import load_config
from driftmon.llm.client import LLMResult
from driftmon.llm.prompts import Explanation
from driftmon.schemas import EvidenceItem, Recommendation


class FakeLLM:
    """Returns the given drafts in order; records the prompts it saw."""

    def __init__(self, drafts):
        self.drafts, self.prompts = list(drafts), []

    def spent(self):
        return 0.0

    def complete_many(self, requests, batch=None):
        out = []
        for r in requests:
            self.prompts.append(r.prompt)
            out.append(LLMResult(key=r.key, parsed=Explanation(text=self.drafts.pop(0)), text="", stop_reason="",
                                 model=r.model, input_tokens=0, output_tokens=0, cost_usd=0.0))
        return out


def _rec():
    return Recommendation(action="Update or re-index the knowledge base", key="reindex", status="recommended",
                          rule="r", finding="Re-index on a schedule.", scope=["2020 POLITICS"], priority=1,
                          evidence=[EvidenceItem(source="Agent D", claim="0-1% correct on the stale index vs "
                                                                         "96-100% refreshed (1,700 questions).")])


def test_numbers_are_normalised():
    assert rec.numbers("1,700 questions, 0.87 and 96-100%.") == {"1700", "0.87", "96", "100"}
    assert rec.unsupported_numbers("About 1700 of 2020 POLITICS questions, 97% correct.",
                                   ["1,700 questions", "2020 POLITICS", "96-100%"]) == ["97"]


def test_explanation_with_invented_number_is_retried_then_accepted():
    cfg = load_config()
    llm = FakeLLM(["Re-index: 97% would be correct.", "Re-index: 96-100% correct once refreshed, vs 0-1%."])
    out, stats = rec.explain([_rec()], llm, cfg)
    assert out[0].explanation.startswith("Re-index: 96-100%")
    assert out[0].explanation_check == "LLM, numbers checked (attempt 2)"
    assert stats["rejected_drafts"] == 1 and stats["requests"] == 2
    assert "97" in llm.prompts[1] and "not in the evidence" in llm.prompts[1]


def test_explanation_falls_back_to_the_rule_text_after_two_bad_drafts():
    cfg = load_config()
    out, stats = rec.explain([_rec()], FakeLLM(["97%.", "Still 98%."]), cfg)
    assert out[0].explanation == "Re-index on a schedule."
    assert "98" in out[0].explanation_check and stats["fallbacks"] == 1


def test_no_llm_keeps_the_rule_text_and_only_recommended_actions_are_explained():
    cfg = load_config()
    other = _rec().model_copy(update={"status": "not needed", "key": "prompts", "priority": 0})
    out, stats = rec.explain([_rec(), other], None, cfg)
    assert [o.explanation for o in out] == ["Re-index on a schedule."] * 2
    assert stats["requests"] == 0 and out[0].explanation_check == "rule text (no LLM)"
    llm = FakeLLM(["Re-index on a schedule: 96-100% vs 0-1%."])
    out, stats = rec.explain([_rec(), other], llm, cfg)
    assert stats["requests"] == 1 and out[1].explanation_check.startswith("rule text")


def test_node_without_reports_still_lists_every_action():
    cfg = load_config()
    report = rec.recommendation({"config": cfg})["recommendations"]
    assert len(report.recommendations) == 9
    assert report.summary == {"not checked": 9}
    md = rec.to_markdown(report)
    assert "Checked, not recommended" in md and "Escalate cases for human review" in md
