"""Agent E: Recommendation (D7, D29).

Reads: Agent D's ``ImpactReport`` (sections, causes, the evidence loop),
Agent C's ``RetrievalReport``, Agent B's ``DriftReport`` and Agent A's
``MonitoringReport``, plus evidence produced outside the graph: the
long-form results (changed facts, the outdated-answer test, chunking) and
the hand-label checks (``driftmon.longform.evidence``).
Does: ``driftmon.policy`` runs one rule per action in the brief and decides
(in code) what to recommend, with the evidence. Sonnet then writes a short
explanation for each recommended action from that evidence only; code
rejects an explanation with a number that is not in the evidence, retries
once naming the numbers, and otherwise keeps the rule's own text.
Writes: ``RecommendationReport``. The pipeline saves it as
``recommendations.json`` and a readable ``recommendations.md``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from driftmon import policy
from driftmon.llm.client import LLMRequest
from driftmon.llm.prompts import EXPLAIN_RETRY, EXPLAIN_SYSTEM, EXPLAIN_TEMPLATE, EXPLAIN_VERSION, Explanation
from driftmon.schemas import PipelineState, Recommendation, RecommendationReport

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    """The numbers in a text, normalised ("1,700" -> "1700"; "0.50" stays "0.50")."""
    out = set()
    for m in NUMBER.findall(text):
        m = m.rstrip(".,")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", m):
            m = m.replace(",", "")
        out.add(m)
    return out


def unsupported_numbers(text: str, sources: Iterable[str]) -> list[str]:
    """Numbers in ``text`` that appear in none of ``sources``."""
    allowed = set().union(*(numbers(s) for s in sources))
    return sorted(numbers(text) - allowed, key=lambda s: (len(s), s))


def _sources(rec: Recommendation) -> list[str]:
    return [rec.action, rec.finding, *rec.scope, *(e.claim for e in rec.evidence)]


def explain_request(rec: Recommendation, cfg: Mapping[str, Any], bad: Sequence[str] = ()) -> LLMRequest:
    """One Sonnet request: explain this recommendation from its evidence."""
    job = cfg["llm"]["jobs"]["explanations"]
    scope = ", ".join(rec.scope[:12]) + (f" and {len(rec.scope) - 12} more" if len(rec.scope) > 12 else "")
    prompt = EXPLAIN_TEMPLATE.format(action=rec.action, finding=rec.finding, scope=scope or "all",
                                     evidence="\n".join(f"- [{e.source}] {e.claim}" for e in rec.evidence))
    if bad:
        prompt += EXPLAIN_RETRY.format(numbers=", ".join(bad))
    return LLMRequest(job="explanations", model=cfg["llm"]["generator_model"], system=EXPLAIN_SYSTEM,
                      prompt=prompt, schema=Explanation, prompt_version=EXPLAIN_VERSION, effort=job["effort"],
                      max_tokens=int(job["max_tokens"]))


def explain(recs: list[Recommendation], llm: Any, cfg: Mapping[str, Any]) -> tuple[list[Recommendation], dict]:
    """Add an explanation to every recommended action; the rest keep the
    rule's finding. Returns the recommendations and a summary of the step."""
    todo = [i for i, r in enumerate(recs) if r.status == "recommended"]
    out = [r.model_copy(update={"explanation": r.finding, "explanation_check": "rule text (not recommended)"})
           for r in recs]
    stats = {"requests": 0, "rejected_drafts": 0, "fallbacks": 0, "cost_usd": 0.0}
    if llm is None or not cfg["recommendation"].get("explain", True) or not todo:
        for i in todo:
            out[i] = out[i].model_copy(update={"explanation_check": "rule text (no LLM)"})
        return out, stats
    before = llm.spent()
    bad: dict[int, list[str]] = {i: [] for i in todo}
    pending = list(todo)
    for attempt in (1, 2):
        results = llm.complete_many([explain_request(recs[i], cfg, bad[i]) for i in pending], batch=False)
        stats["requests"] += len(pending)
        retry = []
        for i, res in zip(pending, results):
            if not res.ok:
                out[i] = out[i].model_copy(update={"explanation_check": f"rule text (LLM failed: {res.error})"})
                stats["fallbacks"] += 1
                continue
            text = res.parsed.text.strip()
            wrong = unsupported_numbers(text, _sources(recs[i]))
            if not wrong:
                out[i] = out[i].model_copy(update={"explanation": text, "explanation_check":
                                                   f"LLM, numbers checked (attempt {attempt})"})
            else:
                stats["rejected_drafts"] += 1
                bad[i] = wrong
                retry.append(i)
        pending = retry
        if not pending:
            break
    for i in pending:
        out[i] = out[i].model_copy(update={"explanation_check": f"rule text (LLM used numbers not in the evidence: "
                                                                f"{', '.join(bad[i])})"})
        stats["fallbacks"] += 1
    stats["cost_usd"] = round(llm.spent() - before, 4)
    return out, stats


def recommendation(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: decide (policy), explain (LLM, checked), report."""
    cfg = state["config"]
    longform = dict(state.get("longform") or {})
    mon = state.get("monitoring")
    if mon is not None and mon.period_summary:
        words = sorted(r["median_words"] for r in mon.period_summary)
        longform.setdefault("news_median_words", int(round(words[len(words) // 2])))
    x = policy.Inputs(cfg=cfg, monitoring=mon, drift=state.get("drift"), retrieval=state.get("retrieval"),
                      impact=state.get("impact"), longform=longform, labels=state.get("label_checks"))
    recs = policy.decide(x)
    recs, stats = explain(recs, state.get("llm"), cfg)
    summary: dict[str, int] = {}
    for r in recs:
        summary[r.status] = summary.get(r.status, 0) + 1
    notes = []
    if longform.get("fact_changes") is None:
        notes.append("No long-form evidence found: the changed-facts and chunking rules were not checked.")
    if state.get("label_checks") is None:
        notes.append("No hand-label checks found: the judge-agreement part of the escalation rule was not checked.")
    return {"recommendations": RecommendationReport(recommendations=recs, summary=summary, explanations=stats,
                                                    notes=notes)}


def to_markdown(report: RecommendationReport, title: str = "Recommendations") -> str:
    """A readable report: recommended actions with evidence, then the rest."""
    lines = [f"# {title}", "",
             "Decided by rules in code (`driftmon/policy.py`, D29); explanations written by an LLM from the "
             "evidence only, and rejected if they use a number that is not in the evidence.", "",
             "Summary: " + ", ".join(f"{n} {s}" for s, n in report.summary.items()), ""]
    for r in report.recommended():
        lines += [f"## {r.priority}. {r.action}", "", r.explanation, ""]
        if r.scope:
            shown = ", ".join(r.scope[:10]) + (f", and {len(r.scope) - 10} more" if len(r.scope) > 10 else "")
            lines += [f"**Applies to ({len(r.scope)}):** {shown}", ""]
        lines += ["**Evidence:**", ""] + [f"- *{e.source}:* {e.claim}" for e in r.evidence]
        lines += ["", f"*Rule:* {r.rule}. *Explanation:* {r.explanation_check}.", ""]
    rest = [r for r in report.recommendations if r.status != "recommended"]
    if rest:
        lines += ["## Checked, not recommended", "", "| Action | Status | Why |", "|---|---|---|"]
        for r in rest:
            why = r.finding + (" " + " ".join(e.claim for e in r.evidence) if r.evidence else "")
            lines.append(f"| {r.action} | {r.status} | {why.replace('|', '/')} |")
        lines.append("")
    for n in report.notes:
        lines.append(f"Note: {n}")
    return "\n".join(lines).rstrip() + "\n"
