"""Tests for driftmon.llm with a fake API client: no network, no key, no spend."""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from driftmon.llm import BudgetExceeded, LLMClient, LLMRequest, json_schema

CONFIG = {"budget_usd": 1.0, "use_batch_api": True,
          "prices": {"m-small": {"input": 2.0, "output": 10.0}, "m-big": {"input": 4.0, "output": 20.0}},
          "jobs": {"questions": {"expected_output": 500}}}


class Answer(BaseModel):
    question: str
    tags: list[str]


class Outer(BaseModel):
    items: list[Answer]
    note: str


def _msg(text, stop="end_turn", model="m-small", tin=1000, tout=200):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, model=model,
                           usage=SimpleNamespace(input_tokens=tin, output_tokens=tout,
                                                 cache_creation_input_tokens=0, cache_read_input_tokens=0))


class FakeAPI:
    """Stands in for anthropic.Anthropic: records calls, answers from a responder."""

    def __init__(self, responder):
        self.responder = responder
        self.live_calls, self.batches = [], []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._live))
        self.messages = SimpleNamespace(batches=SimpleNamespace(
            create=self._create, retrieve=self._retrieve, results=self._results))
        self._polls = 0

    def _live(self, **kw):
        self.live_calls.append(kw)
        return self.responder(kw, live=True)

    def _create(self, requests):
        self.batches.append(requests)
        return SimpleNamespace(id=f"batch{len(self.batches)}")

    def _retrieve(self, batch_id):
        self._polls += 1
        return SimpleNamespace(processing_status="ended" if self._polls > 1 else "in_progress")

    def _results(self, batch_id):
        out = []
        for r in self.batches[int(batch_id[5:]) - 1]:
            kind, msg = self.responder(r["params"], live=False)
            out.append(SimpleNamespace(custom_id=r["custom_id"],
                                       result=SimpleNamespace(type=kind, message=msg)))
        return out


def good(kw, live):
    msg = _msg(json.dumps({"question": kw["messages"][0]["content"], "tags": ["a"]}))
    return msg if live else ("succeeded", msg)


def _client(tmp_path, responder=good, budget=1.0):
    return LLMClient(config={**CONFIG, "budget_usd": budget}, cache_dir=tmp_path / "cache",
                     ledger=tmp_path / "ledger.csv", client=FakeAPI(responder), sleep=lambda s: None)


def _req(prompt="What happened?", version="v1"):
    return LLMRequest(job="questions", model="m-small", system="You write questions.", prompt=prompt,
                      schema=Answer, prompt_version=version, effort="low", max_tokens=2000)


def test_json_schema_is_strict_at_every_level():
    s = json_schema(Outer)
    assert s["additionalProperties"] is False and set(s["required"]) == {"items", "note"}
    inner = s["$defs"]["Answer"]
    assert inner["additionalProperties"] is False and set(inner["required"]) == {"question", "tags"}


def test_live_call_is_validated_costed_logged_and_cached(tmp_path):
    llm = _client(tmp_path)
    r = llm.complete(_req())
    assert r.ok and r.parsed.question == "What happened?" and not r.cached
    assert r.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)
    call = llm.client.live_calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "temperature" not in call
    again = llm.complete(_req())
    assert again.cached and again.parsed == r.parsed and len(llm.client.live_calls) == 1
    fresh = _client(tmp_path)  # a new client reads the ledger and cache from disk
    assert fresh.spent() == pytest.approx(r.cost_usd)
    assert fresh.complete(_req()).cached
    assert len(fresh.client.live_calls) == 0


def test_prompt_version_change_is_a_new_request(tmp_path):
    llm = _client(tmp_path)
    llm.complete(_req(version="v1"))
    llm.complete(_req(version="v2"))
    assert len(llm.client.live_calls) == 2


def test_budget_is_enforced_before_spending(tmp_path):
    llm = _client(tmp_path, budget=0.001)  # estimate ~ (24+... tokens) + 500 output tokens ~ $0.005
    with pytest.raises(BudgetExceeded):
        llm.complete(_req())
    assert llm.client.live_calls == []
    with pytest.raises(BudgetExceeded):
        llm.complete_many([_req(f"q{i}") for i in range(3)])
    assert llm.client.batches == []


def test_refusal_and_broken_output(tmp_path):
    def responder(kw, live):
        p = kw["messages"][0]["content"]
        if p == "refuse me":
            return _msg("", stop="refusal")
        return _msg("not json at all")
    llm = _client(tmp_path, responder)
    refused = llm.complete(_req("refuse me"))
    assert not refused.ok and refused.error == "refused"
    assert llm.complete(_req("refuse me")).cached  # a live refusal (after fallback) is kept
    broken = llm.complete(_req("broken"))
    assert not broken.ok and broken.error.startswith("invalid output")
    llm.complete(_req("broken"))                   # broken replies are retried, not cached
    assert len(llm.client.live_calls) == 3
    assert llm.spent() == pytest.approx(3 * (1000 * 2 + 200 * 10) / 1e6)


def test_batch_runs_only_uncached_requests_at_half_price_in_order(tmp_path):
    llm = _client(tmp_path)
    llm.complete(_req("q0"))  # cached beforehand
    reqs = [_req(f"q{i}") for i in range(4)] + [_req("q1")]  # includes a duplicate
    out = llm.complete_many(reqs, batch=True)
    assert [r.parsed.question for r in out] == ["q0", "q1", "q2", "q3", "q1"]
    assert out[0].cached
    assert len(llm.client.batches) == 1 and len(llm.client.batches[0]) == 3  # q1..q3 once each
    assert out[1].cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6 * 0.5)


def test_batch_refusal_retried_live_and_errors_not_cached(tmp_path):
    def responder(kw, live):
        p = kw["messages"][0]["content"]
        if live:
            return _msg(json.dumps({"question": p, "tags": []}))
        if p == "sensitive":
            return ("succeeded", _msg("", stop="refusal"))
        if p == "flaky":
            return ("errored", None)
        return ("succeeded", _msg(json.dumps({"question": p, "tags": []})))
    llm = _client(tmp_path, responder)
    out = llm.complete_many([_req("fine"), _req("sensitive"), _req("flaky")], batch=True)
    assert out[0].ok and out[1].ok          # the refusal was retried live (with fallback)
    assert [c["messages"][0]["content"] for c in llm.client.live_calls] == ["sensitive"]
    assert not out[2].ok and out[2].error == "batch errored"
    assert llm.cached(_req("flaky")) is None  # errors are retried next time


def test_api_key_ignores_an_empty_environment_variable(tmp_path, monkeypatch):
    from driftmon.llm import api_key
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert api_key(env) == "from-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert api_key(env) == "from-env"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert api_key(tmp_path / "missing.env") is None
