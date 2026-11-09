"""Decision rules for Agent E: evidence to corrective actions (D7, D29).

One plain function per action in the brief. Each reads the agents' reports
(and the long-form evidence and hand-label checks), and returns a
``Recommendation`` with a status:

    recommended   the rule fired; the evidence says why
    not needed    the rule was checked; the evidence says what ruled it out
    not checked   an input the rule needs is missing

The rules, in short:
    re-index the knowledge base   material answer loss that the refreshed index recovers
    re-index specific documents   changed facts flagged, and the outdated-answer test says they matter
    improve chunking              documents far longer than the embedding model reads, and
                                  chunks follow real edits clearly better
    update retrieval settings     the loop found answers just outside the top k
    improve prompts               retrieval finds answers but answers are wrong, or claims are invented
    model evaluation              poor answers on the refreshed index, with poor retrieval
    retraining                    a model evaluation that showed the problem persists
    escalate to a human           harm without drift, an unclear loop result, or a weak judge
    continue monitoring           drift without material loss

No LLM is involved here. ``check_evidence`` rejects a recommendation that
carries no evidence.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from driftmon.evaluation.answers import DRIFT_HARMLESS, DRIFT_HURTS, HARM_WITHOUT_DRIFT, JUST_OUTSIDE, UNCLEAR
from driftmon.schemas import (DriftReport, EvidenceItem, ImpactReport, MonitoringReport, Recommendation,
                              RetrievalReport)

REINDEX, REINDEX_DOCS, RETRIEVAL_CONFIG = "reindex", "reindex_documents", "retrieval_config"
PROMPTS, MODEL_EVAL, RETRAIN = "prompts", "model_evaluation", "retraining"
ESCALATE, CHUNKING, MONITOR = "escalate", "chunking", "monitor"

# the brief's wording (section 2, Agent E)
ACTIONS = {
    REINDEX: "Update or re-index the knowledge base",
    REINDEX_DOCS: "Re-index specific documents (changed facts)",
    RETRIEVAL_CONFIG: "Update retrieval configurations",
    PROMPTS: "Improve prompts or retrieval strategies",
    MODEL_EVAL: "Trigger model evaluation",
    RETRAIN: "Recommend retraining",
    ESCALATE: "Escalate cases for human review",
    CHUNKING: "Improve document chunking or embedding strategies",
    MONITOR: "Continue monitoring without intervention",
}
# order of recommended actions: measured answer loss first, then escalation, then maintenance
KIND = {REINDEX: 1, REINDEX_DOCS: 1, RETRIEVAL_CONFIG: 1, PROMPTS: 1, MODEL_EVAL: 1, RETRAIN: 1,
        ESCALATE: 2, CHUNKING: 3, MONITOR: 3}
MATERIAL = (DRIFT_HURTS, HARM_WITHOUT_DRIFT)
MISSING_CONTENT = "missing content (re-index)"
ANSWER_STEP = "answer step (articles found, answers wrong)"
RETRIEVAL_OR_MODEL = "retrieval or embedding model (poor on the refreshed index too)"


@dataclass
class Inputs:
    """Everything the rules read. Any report may be missing (None)."""

    cfg: Mapping[str, Any]
    monitoring: MonitoringReport | None = None
    drift: DriftReport | None = None
    retrieval: RetrievalReport | None = None
    impact: ImpactReport | None = None
    longform: Mapping[str, Any] = field(default_factory=dict)   # see ``driftmon.longform.evidence``
    labels: pd.DataFrame | None = None                          # hand labels vs the LLM, one row per (who, part)

    @property
    def rc(self) -> Mapping[str, Any]:
        return self.cfg["recommendation"]

    def segments(self) -> pd.DataFrame:
        """Agent D's classified segments (enough questions, not the baseline)."""
        if self.impact is None or not self.impact.segments:
            return pd.DataFrame()
        s = pd.DataFrame(self.impact.segments)
        return s[s["class"].isin([DRIFT_HURTS, DRIFT_HARMLESS, HARM_WITHOUT_DRIFT, "healthy"])]

    def baseline_name(self) -> str:
        return self.cfg["periods"]["baseline"]["name"]


# ---------- formatting: every number a rule cites goes through these ----------

def pct(x: float) -> str:
    v = 100 * float(x)
    return f"{v:.1f}%" if 0 < v < 1 else f"{v:.0f}%"


def pct_range(values: Sequence[float]) -> str:
    lo, hi = round(100 * min(values)), round(100 * max(values))
    return f"{lo}%" if lo == hi else f"{lo}-{hi}%"


def points(values: Sequence[float]) -> str:
    """A range of differences between two shares, in percentage points."""
    lo, hi = round(100 * min(values)), round(100 * max(values))
    return f"{lo} percentage points" if lo == hi else f"{lo}-{hi} percentage points"


def num(x: float, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def _rec(key: str, status: str, rule: str, finding: str, evidence: list[EvidenceItem] | None = None,
         scope: list[str] | None = None) -> Recommendation:
    return Recommendation(action=ACTIONS[key], key=key, status=status, rule=rule, finding=finding,
                          evidence=evidence or [], scope=scope or [])


def _seg_name(segment_id: str) -> str:
    """'2020|WELLNESS|vs 2016-17' -> '2020 WELLNESS'."""
    period, cat = segment_id.split("|")[:2]
    return f"{period} {cat}"


def _later(rows: Sequence[Mapping[str, Any]], key: str, baseline: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if r[key] != baseline]


# ---------- the rules ----------

def reindex(x: Inputs) -> Recommendation:
    """Material loss that the refreshed index recovers: the content was missing."""
    rule = ("a section loses at least the material share of correct answers on the stale index, "
            "and the refreshed index answers it well")
    seg = x.segments()
    if not len(seg):
        return _rec(REINDEX, "not checked", rule, "No classified sections from Agent D.")
    hit = seg[seg["class"].isin(MATERIAL) & (seg["cause"] == MISSING_CONTENT)].copy()
    if not len(hit):
        return _rec(REINDEX, "not needed", rule,
                    f"No section loses a material share of answers that re-indexing recovers "
                    f"({len(seg)} sections checked).",
                    [EvidenceItem(source="Agent D: sections", claim=f"{len(seg)} sections with enough questions; "
                                  f"none has a material loss recovered by the refreshed index.")])
    hit["lost"] = hit["loss_correct"] * hit["questions"]
    hit = hit.sort_values("lost", ascending=False)
    base = x.baseline_name()
    later = _later(x.impact.by_period, "period", base)
    ev = [EvidenceItem(source="Agent D: answers by period",
                       claim=f"Questions about {later[0]['period']} to {later[-1]['period']}: "
                             f"{pct_range([r['stale_correct'] for r in later])} answered correctly from the stale "
                             f"index vs {pct_range([r['refreshed_correct'] for r in later])} from the refreshed "
                             f"index ({sum(r['questions'] for r in later)} questions)."),
          EvidenceItem(source="Agent D: sections",
                       claim=f"{len(hit)} of {len(seg)} sections with enough questions lose a material share "
                             f"of answers (correct answers drop by {points(hit['loss_correct'])}), and the "
                             f"refreshed index answers {pct_range(hit['refreshed_correct'])} of their questions "
                             f"correctly.")]
    old = [r for r in x.impact.by_period if r["period"] == base]
    if old:
        ev.append(EvidenceItem(source="Agent D: historical control",
                               claim=f"Questions about the baseline ({base}): {pct(old[0]['stale_correct'])} "
                                     f"correct on the stale index and {pct(old[0]['refreshed_correct'])} on the "
                                     f"refreshed one, so re-indexing does not hurt older questions."))
    silent = _silence(x, base)
    if silent:
        ev.append(silent)
    trigger, why = _trigger(x, hit)
    ev.extend(why)
    if x.impact.examples:
        e = x.impact.examples[0]
        ev.append(EvidenceItem(source="Agent D: example",
                               claim=f"\"{e['question']}\" Stale index: \"{_short(e['stale_answer'])}\" "
                                     f"Refreshed index: \"{_short(e['refreshed_answer'])}\""))
    when = ("on a schedule, with every new batch of articles, not only when drift is flagged"
            if trigger == "schedule" else "whenever drift detection flags a section")
    finding = (f"Re-index the archive {when}. Without it, questions about new events lose almost every "
               f"answer: in {len(hit)} sections correct answers drop by {points(hit['loss_correct'])}.")
    return _rec(REINDEX, "recommended", rule, finding, ev, [_seg_name(s) for s in hit["segment_id"]])


def _trigger(x: Inputs, hit: pd.DataFrame) -> tuple[str, list[EvidenceItem]]:
    """Schedule or drift alert: does drift detection find the harm?"""
    missed = hit[hit["class"] == HARM_WITHOUT_DRIFT]
    alpha = float(x.cfg["drift"]["alpha"])
    corr = [c for c in (x.impact.correlations or []) if c.get("p_value") is not None]
    any_sig = any(c["p_value"] <= alpha for c in corr)
    ev = []
    if len(missed):
        ev.append(EvidenceItem(source="Agent D: which drift matters",
                               claim=f"{len(missed)} of the {len(hit)} sections that lose answers were not "
                                     f"flagged by drift detection (for example "
                                     f"{', '.join(_seg_name(s) for s in missed['segment_id'][:3])})."))
    if corr:
        rho = [c["spearman"] for c in corr]
        claim = (f"Some drift measures track the size of the loss across sections (Spearman up to "
                 f"{num(max(rho))}, p <= {alpha})." if any_sig else
                 f"No drift measure predicts the size of the loss across sections (Spearman {num(min(rho))} "
                 f"to {num(max(rho))}, all p > {alpha}).")
        ev.append(EvidenceItem(source="Agent D: drift scores vs loss", claim=claim))
    schedule = len(missed) > 0 or (bool(corr) and not any_sig)
    return ("schedule" if schedule else "drift alert"), ev


def _silence(x: Inputs, base: str) -> EvidenceItem | None:
    """Stale-index answers: declined, or invented?"""
    later = _later(x.impact.by_period, "period", base)
    n = sum(r["questions"] for r in later)
    if not n or "stale_declined_ok" not in later[0]:
        return None
    declined = sum(r["stale_declined_ok"] * r["questions"] for r in later) / n
    invented = round(sum(r["stale_hallucinated"] * r["questions"] for r in later))
    return EvidenceItem(source="Agent D: how the stale index fails",
                        claim=f"It fails silently: when nothing relevant is found it declines "
                              f"({pct(declined)} of those cases); {invented} of {n} stale answers added a claim "
                              f"not in the articles.")


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def reindex_documents(x: Inputs) -> Recommendation:
    """Changed facts flagged and shown to change answers: re-index those documents."""
    rule = "the changed-facts check flags documents, and the outdated-answer test shows flagged edits change answers"
    lf = x.longform
    changes, verdict = lf.get("fact_changes"), lf.get("outdated_verdict")
    if changes is None or verdict is None:
        return _rec(REINDEX_DOCS, "not checked", rule, "No long-form evidence (changed facts and the outdated-answer "
                                                       "test) found.")
    since = changes[changes["period"] != lf.get("baseline", x.baseline_name())]
    summ = lf.get("outdated_summary")
    ev = [EvidenceItem(source="Wikipedia: outdated-answer test",
                       claim=f"Questions on flagged edits got a different answer from the old and the new version "
                             f"{pct(verdict['flagged_changed'])} of the time, vs {pct(verdict['comparison_changed'])} "
                             f"for edits that kept their numbers (Fisher p = {verdict['p_value']:.1e}).")]
    if not verdict.get("useful") or not len(since):
        why = ("the outdated-answer test does not show that flagged edits change answers"
               if not verdict.get("useful") else "no changed facts since the index was built")
        return _rec(REINDEX_DOCS, "not needed", rule, f"Not needed: {why}.", ev)
    titles = since.groupby("title").size().sort_values(ascending=False)
    n_articles = lf.get("articles")
    ev.insert(0, EvidenceItem(source="Wikipedia: changed-facts check",
                              claim=f"{len(since)} changed numbers or dates in {len(titles)}"
                                    f"{f' of {n_articles}' if n_articles else ''} articles since the index was "
                                    f"built ({since['period'].min()} to {since['period'].max()})."))
    low = lf.get("low_band_share")
    if low is not None:
        ev.append(EvidenceItem(source="Wikipedia: changed facts vs drift scores",
                               claim=f"{pct(low)} of articles with a changed fact had a \"low\" drift score, "
                                     f"so drift alerts alone would miss them."))
    if summ is not None and len(summ):
        f = summ.set_index("group").loc["flagged"]
        ev.append(EvidenceItem(source="Wikipedia: outdated-answer test",
                               claim=f"The old index still retrieves the edited passage {pct(f['edit_retrieved_old'])} "
                                     f"of the time, so it serves the outdated fact with confidence."))
    prec = _label_rate(x.labels, "facts", "agreement")
    if prec:
        ev.append(EvidenceItem(source="Hand labels",
                               claim=f"Blind hand labels put the check's precision at "
                                     f"{' and '.join(pct(v) for _, v in prec)} ({', '.join(w for w, _ in prec)})."))
    ex = lf.get("example")
    if ex:
        ev.append(EvidenceItem(source="Wikipedia: example",
                               claim=f"{ex['title']}: \"{_short(ex['before'], 140)}\" became "
                                     f"\"{_short(ex['after'], 140)}\"."))
    finding = (f"Re-index the {len(titles)} articles whose numbers or dates changed, and check the flagged "
               f"sentences: on an edited archive the stale index gives outdated answers with confidence, and drift "
               f"scores do not see it.")
    return _rec(REINDEX_DOCS, "recommended", rule, finding, ev, list(titles.index))


def _label_rate(labels: pd.DataFrame | None, part: str, column: str) -> list[tuple[str, float]]:
    """(who, rate) for one part of the hand-label checks."""
    if labels is None or not len(labels) or column not in labels:
        return []
    rows = labels[(labels["part"] == part) & labels[column].notna()]
    return [(r["who"], float(r[column])) for r in rows.to_dict("records")]


def chunking(x: Inputs) -> Recommendation:
    """Long documents: chunk them, and measure drift per chunk."""
    rule = ("a typical document is far longer than the embedding model reads, and chunks follow real edits "
            "clearly better than the first 512 tokens")
    summ, ev_tab = x.longform.get("longform_summary"), x.longform.get("chunking_eval")
    if summ is None or ev_tab is None:
        return _rec(CHUNKING, "not checked", rule, "No long-form chunking evidence found.")
    last = summ.iloc[-1]
    m = ev_tab.set_index("method")["spearman_all"]
    chunk_rows = [k for k in m.index if k.startswith("chunks_") and not k.endswith("_max")]
    best = max(chunk_rows, key=lambda k: m[k])
    gain = float(m[best] - m["truncated"])
    beyond = float(last["share_of_text_beyond_512"])
    sizes = ", ".join(f"chunks of {k.split('_')[1]} tokens {num(m[k])}" for k in sorted(chunk_rows))
    ev = [EvidenceItem(source="Wikipedia: long documents",
                       claim=f"{pct(beyond)} of a typical Wikipedia article's text lies past the 512 tokens the "
                             f"embedding model reads (median {int(last['median_tokens'])} tokens, {last['period']})."),
          EvidenceItem(source="Wikipedia: chunking experiment",
                       claim=f"How well each way of reading an article follows its real edits (Spearman): first 512 "
                             f"tokens {num(m['truncated'])}, one averaged vector {num(m['averaged'])}, {sizes}.")]
    news = _news_length(x)
    if news:
        ev.append(news)
    if beyond < float(x.rc["min_text_beyond_model"]) or gain < float(x.rc["min_chunking_gain"]):
        return _rec(CHUNKING, "not needed", rule, "Not needed: documents fit the model, or chunks do not follow "
                                                  "edits clearly better.", ev)
    size = best.split("_")[1]
    finding = (f"For long documents, keep section-aware chunks (about {x.cfg['longform']['chunking']['size_tokens']} "
               f"tokens) and measure drift per chunk and per document against its own earlier version; the first "
               f"512 tokens or one averaged vector hide most edits. Chunks of {size} tokens tracked edits best and "
               f"are worth testing for retrieval. News articles are short, so this does not apply to the archive.")
    return _rec(CHUNKING, "recommended", rule, finding, ev, ["long-form documents (Wikipedia set)"])


def _news_length(x: Inputs) -> EvidenceItem | None:
    lf = x.longform.get("news_median_words")
    if lf is None:
        return None
    return EvidenceItem(source="Agent A: data monitoring",
                        claim=f"News articles are short (median {lf} words per article), well within the model's limit.")


def retrieval_config(x: Inputs) -> Recommendation:
    """The loop back to C found answers just outside the top k."""
    rule = "the evidence loop finds answers just outside the top 5 (ranks 6-10)"
    if x.impact is None:
        return _rec(RETRIEVAL_CONFIG, "not checked", rule, "No Agent D report.")
    near = [e for e in x.impact.evidence if e.get("conclusion") == JUST_OUTSIDE]
    if near:
        ev = [EvidenceItem(source="Agent D: evidence loop",
                           claim=f"{_seg_name(e['segment_id'])}: correct answers from the stale index rise from "
                                 f"{pct(e['stale_correct_top5'])} with the top 5 to {pct(e['stale_correct_top10'])} "
                                 f"with the top 10.") for e in near]
        return _rec(RETRIEVAL_CONFIG, "recommended", rule,
                    f"Read more articles per question (top 10 instead of top 5) in {len(near)} section(s): the answers "
                    f"were just outside the top 5.", ev, [_seg_name(e["segment_id"]) for e in near])
    ev = []
    if x.retrieval is not None and x.retrieval.judged_by_period:
        later = _later(x.retrieval.judged_by_period, "period", x.baseline_name())
        ev.append(EvidenceItem(source="Agent C: judged retrieval",
                               claim=f"On the refreshed index the top 5 holds an answering article for "
                                     f"{pct_range([r['refreshed_answer'] for r in later])} of new questions."))
    ev.append(EvidenceItem(source="Agent D: evidence loop",
                           claim=f"{len(x.impact.borderline)} sections were borderline, so the loop back to Agent C "
                                 f"{'ran' if x.impact.evidence else 'did not run'}."))
    return _rec(RETRIEVAL_CONFIG, "not needed", rule,
                "Not needed: the refreshed index already puts an answer in the top 5, and no section had answers "
                "just outside it.", ev)


def prompts(x: Inputs) -> Recommendation:
    """Retrieval finds answers, but answers are wrong or ungrounded."""
    rule = ("retrieval finds answering articles but the answers are wrong, or answers add claims the articles do "
            "not make")
    seg = x.segments()
    if x.impact is None or not x.impact.by_period:
        return _rec(PROMPTS, "not checked", rule, "No Agent D report.")
    rows = x.impact.by_period
    n = sum(r["questions"] for r in rows)
    invented = sum((r["stale_hallucinated"] + r["refreshed_hallucinated"]) * r["questions"] for r in rows)
    rate = invented / (2 * n) if n else 0.0
    step = seg[seg["cause"] == ANSWER_STEP] if len(seg) else seg
    later = _later(rows, "period", x.baseline_name())
    ev = [EvidenceItem(source="Agent D: answers by period",
                       claim=f"On the refreshed index {pct_range([r['refreshed_correct'] for r in later])} of new "
                             f"questions are answered correctly."),
          EvidenceItem(source="Agent D: grounding",
                       claim=f"{round(invented)} of {2 * n} answers added a claim not in the retrieved articles "
                             f"({pct(rate)}).")]
    if len(step) or rate > float(x.rc["max_hallucinated"]):
        scope = [_seg_name(s) for s in step["segment_id"]] if len(step) else ["all sections"]
        return _rec(PROMPTS, "recommended", rule,
                    "Revise the answer prompt: the right articles are found but the answers do not use them well.",
                    ev, scope)
    return _rec(PROMPTS, "not needed", rule,
                "Not needed: where the right articles are found, the answers are correct and stay within them.", ev)


def model_evaluation(x: Inputs) -> Recommendation:
    """Poor on the refreshed index, with poor retrieval: suspect the embedding model."""
    rule = ("a section stays poor on the refreshed index, and its top 5 rarely holds an answer although the source "
            "article is in the index")
    seg = x.segments()
    if not len(seg):
        return _rec(MODEL_EVAL, "not checked", rule, "No classified sections from Agent D.")
    bad = seg[seg["cause"] == RETRIEVAL_OR_MODEL]
    if len(bad):
        ev = [EvidenceItem(source="Agent D: sections",
                           claim=f"{_seg_name(r['segment_id'])}: {pct(r['refreshed_correct'])} correct and an answer "
                                 f"in the top 5 for {pct(r['refreshed_retrieval'])} of questions on the refreshed index.")
              for r in bad.to_dict("records")]
        return _rec(MODEL_EVAL, "recommended", rule,
                    f"Evaluate a second embedding model on {len(bad)} section(s) where re-indexing did not help.",
                    ev, [_seg_name(s) for s in bad["segment_id"]])
    ev = [EvidenceItem(source="Agent D: sections",
                       claim=f"The refreshed index answers {pct_range(seg['refreshed_correct'])} of questions "
                             f"correctly in every one of the {len(seg)} sections with enough questions "
                             f"(threshold {pct(x.cfg['impact']['recovered_correct'])}).")]
    return _rec(MODEL_EVAL, "not needed", rule,
                "Not needed: re-indexing fixes every section, so the embedding model handles the new content.", ev)


def retraining(x: Inputs, model_eval: Recommendation) -> Recommendation:
    """Only after a model evaluation shows the problem persists."""
    rule = "a model evaluation shows quality stays poor with a second embedding model and a fresh index"
    if model_eval.status != "recommended":
        return _rec(RETRAIN, "not needed" if model_eval.status == "not needed" else "not checked", rule,
                    "Not needed: model evaluation was not triggered, and retraining only follows a failed evaluation.",
                    [EvidenceItem(source="Agent E: model evaluation rule",
                                  claim=f"Model evaluation: {model_eval.status}.")])
    return _rec(RETRAIN, "not checked", rule,
                "Decide after the model evaluation: retrain only if the second model is also poor.",
                [EvidenceItem(source="Agent E: model evaluation rule", claim="Model evaluation: recommended.")])


def escalate(x: Inputs) -> Recommendation:
    """Harm the drift scores did not see, an unclear loop result, or a weak judge."""
    rule = "a section loses answers without a drift flag, the evidence loop is unclear, or the judge disagrees with " \
           "the hand labels"
    seg = x.segments()
    if not len(seg):
        return _rec(ESCALATE, "not checked", rule, "No classified sections from Agent D.")
    missed = seg[seg["class"] == HARM_WITHOUT_DRIFT].sort_values("questions", ascending=False)
    unclear = [e for e in x.impact.evidence if e.get("conclusion") == UNCLEAR]
    weak = [(w, p, v) for p, col in (("relevance", "answers_agreement"), ("answers", "agreement"))
            for w, v in _label_rate(x.labels, p, col) if v < float(x.rc["min_judge_agreement"])]
    ev, scope, reasons = [], [], []
    if len(missed):
        names = [_seg_name(s) for s in missed["segment_id"]]
        scope += names
        reasons.append(f"{len(missed)} sections lost answers without a drift flag")
        ev.append(EvidenceItem(source="Agent D: harm without drift",
                               claim=f"In {len(missed)} sections correct answers drop by "
                                     f"{points(missed['loss_correct'])}, but drift detection did not flag them: "
                                     f"{', '.join(names[:5])}{' and others' if len(names) > 5 else ''}."))
        ev.extend(_why_missed(x, missed["segment_id"]))
    for e in unclear:
        scope.append(_seg_name(e["segment_id"]))
        reasons.append("the evidence loop was unclear")
        ev.append(EvidenceItem(source="Agent D: evidence loop",
                               claim=f"{_seg_name(e['segment_id'])}: loss {pct(e['loss_top5'])} with the top 5 and "
                                     f"{pct(e['loss_top10'])} with the top 10, interval "
                                     f"{pct(e['loss_top10_low'])} to {pct(e['loss_top10_high'])}."))
    for who, part, v in weak:
        scope.append(f"the {part} judge")
        reasons.append(f"the {part} judge agrees with {who} only {pct(v)} of the time")
        ev.append(EvidenceItem(source="Hand labels",
                               claim=f"The {part} judge agrees with {who} on {pct(v)} of blind items "
                                     f"(minimum {pct(x.rc['min_judge_agreement'])})."))
    if not reasons:
        checks = [f"{w}: {pct(v)}" for p, c in (("relevance", "answers_agreement"), ("answers", "agreement"))
                  for w, v in _label_rate(x.labels, p, c)]
        ev = [EvidenceItem(source="Agent D: sections", claim=f"Every section that lost answers was flagged by drift "
                                                             f"detection ({len(seg)} sections checked).")]
        if checks:
            ev.append(EvidenceItem(source="Hand labels",
                                   claim=f"The judges agree with blind hand labels: {'; '.join(checks)}."))
        return _rec(ESCALATE, "not needed", rule, "Not needed: drift detection found every loss, and the judges "
                                                  "agree with the hand labels.", ev)
    finding = ("Send these to a person: " + "; ".join(dict.fromkeys(reasons)) + ".")
    if len(missed):
        scored = {s.segment_id for s in (x.drift.segments if x.drift else [])}
        lessons = []
        if any(s in scored for s in missed["segment_id"]):
            lessons.append("new events in a familiar section do not shift the section's overall profile, so "
                           "drift alerts miss them")
        if any(s not in scored for s in missed["segment_id"]):
            lessons.append("a section that did not exist in the baseline cannot be scored for drift until it has "
                           "a baseline of its own")
        finding += (" Lesson for the reviewer: " + "; and ".join(lessons) + ". On a news archive, rely on "
                    "scheduled re-indexing, and use drift scores to explain what changed.")
    return _rec(ESCALATE, "recommended", rule, finding, ev, scope)


def _why_missed(x: Inputs, segment_ids: Sequence[str]) -> list[EvidenceItem]:
    """For sections with harm but no flag: scored as ordinary drift, or not scored at all?"""
    scored = {s.segment_id: s for s in (x.drift.segments if x.drift else [])}
    reasons = {(s["period"], s["category"]): s["reason"] for s in (x.monitoring.segments if x.monitoring else [])}
    measured = [scored[s] for s in segment_ids if s in scored]
    unscored = [s for s in segment_ids if s not in scored]
    ev = []
    if measured:
        t = [s.times_ordinary_embedding for s in measured]
        sev = sorted({s.severity for s in measured})
        ev.append(EvidenceItem(source="Agent B: drift scores",
                               claim=f"{len(measured)} of them were scored: embedding drift {num(min(t), 1)}-"
                                     f"{num(max(t), 1)}x ordinary change, severity {' or '.join(sev)}."))
    if unscored:
        why = sorted({reasons.get(tuple(s.split("|")[:2]), "") or "not scored" for s in unscored})
        ev.append(EvidenceItem(source="Agent A: data monitoring",
                               claim=f"{len(unscored)} of them could not be scored for drift ({'; '.join(why)}): "
                                     f"{', '.join(_seg_name(s) for s in unscored)}. Agent A reports these as new "
                                     f"sections."))
    return ev


def monitor(x: Inputs) -> Recommendation:
    """Drift without material loss: keep watching, no action."""
    rule = "drift is detected but answers do not suffer"
    seg = x.segments()
    if x.impact is None or x.drift is None:
        return _rec(MONITOR, "not checked", rule, "Missing Agent B or Agent D report.")
    base = x.baseline_name()
    harmless = seg[seg["class"] == DRIFT_HARMLESS] if len(seg) else seg
    old = [r for r in x.impact.by_period if r["period"] == base]
    stable_old = bool(old) and old[0]["refreshed_correct"] >= old[0]["stale_correct"] - \
        float(x.cfg["impact"]["min_answer_quality_drop"])
    ev, scope = [], []
    if len(harmless):
        scope += [_seg_name(s) for s in harmless["segment_id"]]
        ev.append(EvidenceItem(source="Agent D: drift that does not hurt",
                               claim=f"{len(harmless)} flagged sections show no material loss in correct answers."))
    if stable_old:
        scope.append(f"questions about the baseline ({base})")
        ev.append(EvidenceItem(source="Agent D: historical control",
                               claim=f"Questions about {base}: {pct(old[0]['stale_correct'])} correct before and "
                                     f"{pct(old[0]['refreshed_correct'])} after re-indexing."))
        shift = _newer_share(x)
        if shift:
            ev.append(shift)
    yoy = _year_on_year(x)
    if yoy:
        ev.append(yoy)
    if not scope:
        return _rec(MONITOR, "not needed", rule, "Not needed as the only action: every flagged section lost answers.",
                    ev)
    finding = ("Keep monitoring, without intervention, where drift does not hurt answers"
               + (": questions about older events are unaffected even as newer articles join their results" if
                  stable_old else "") + ". Compare each new batch with the baseline, not only with the previous "
               "year, or gradual drift goes unnoticed.")
    return _rec(MONITOR, "recommended", rule, finding, ev, scope)


def _newer_share(x: Inputs) -> EvidenceItem | None:
    if x.retrieval is None or not x.retrieval.ranking_change:
        return None
    last = x.retrieval.ranking_change[-1]
    return EvidenceItem(source="Agent C: ranking change",
                        claim=f"By {last['index'].replace('refreshed_', '')}, newer articles make up "
                              f"{pct(last['newer_share'])} of the top 10 for those questions.")


def _year_on_year(x: Inputs) -> EvidenceItem | None:
    """The last period vs the previous one and vs the baseline."""
    rows = x.drift.whole_period
    if not rows:
        return None
    last = rows[-1].period
    mine = {r.compared_with: r for r in rows if r.period == last}
    base = x.baseline_name()
    prev = [c for c in mine if c != base]
    if base not in mine or not prev:
        return None
    b, p = mine[base], mine[prev[0]]
    return EvidenceItem(source="Agent B: drift, whole periods",
                        claim=f"{last} vs {prev[0]} is {num(p.times_ordinary_embedding, 1)}x ordinary change in "
                              f"embedding drift ({p.severity}), but {last} vs {base} is "
                              f"{num(b.times_ordinary_embedding, 1)}x ({b.severity}).")


RULES: list[Callable[[Inputs], Recommendation]] = [reindex, reindex_documents, retrieval_config, prompts,
                                                    model_evaluation, escalate, chunking, monitor]


def check_evidence(rec: Recommendation) -> Recommendation:
    """A recommendation must cite evidence with numbers, or it is rejected."""
    if rec.status != "recommended":
        return rec
    ok = rec.evidence and all(e.source.strip() and any(ch.isdigit() for ch in e.claim) for e in rec.evidence)
    if ok:
        return rec
    return rec.model_copy(update={"status": "rejected",
                                  "finding": rec.finding + " (Rejected: no evidence with numbers.)"})


def decide(x: Inputs) -> list[Recommendation]:
    """Run every rule; recommended first (by kind of evidence, then rule order)."""
    recs = [rule(x) for rule in RULES]
    me = next(r for r in recs if r.key == MODEL_EVAL)
    recs.insert(recs.index(me) + 1, retraining(x, me))
    recs = [check_evidence(r) for r in recs]
    order = {r.key: i for i, r in enumerate(recs)}
    rank = {"recommended": 0, "rejected": 1, "not needed": 2, "not checked": 3}
    recs.sort(key=lambda r: (rank[r.status], KIND[r.key] if r.status == "recommended" else 0, order[r.key]))
    n = 0
    out = []
    for r in recs:
        if r.status == "recommended":
            n += 1
            r = r.model_copy(update={"priority": n})
        out.append(r)
    return out
