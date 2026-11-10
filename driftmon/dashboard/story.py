"""Plain-language takeaways for the dashboard, built from a run's numbers.

The dashboard's fixed text only says how to read a chart. What a run *shows*
comes from here: each sentence is filled in from the saved reports, and its
wording depends on the result, so pointing the dashboard at another run can
never leave a sentence that contradicts the chart next to it. Code decides
what is said; nothing here calls an LLM.

Every function returns an empty string or list when the run lacks the data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from driftmon.dashboard.runs import Run, harm_vs_drift
from driftmon.longform.evidence import low_band_share
from driftmon.policy import pct, pct_range


def _later(rows: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("period")) != baseline]


def _weighted(rows: list[dict[str, Any]], col: str) -> float:
    n = sum(r["questions"] for r in rows)
    return sum(r[col] * r["questions"] for r in rows) / n if n else float("nan")


def _top(run: Run, key: str) -> str:
    """'top 5' from the run's settings, or 'top results' when the run does not record them."""
    k = run.settings.get(key)
    return f"top {k}" if k else "top results"


def _years(periods: list[str]) -> str:
    return f"{periods[0]}–{periods[-1]}" if len(periods) > 1 else periods[0]


def answer_headline(run: Run) -> dict[str, Any]:
    """New-event questions and how many each index answers correctly (Agent D)."""
    if run.impact is None or not run.impact.by_period:
        return {}
    later = _later(run.impact.by_period, run.baseline)
    if not later:
        return {}
    n = sum(r["questions"] for r in later)
    out = {"questions": n, "stale": _weighted(later, "stale_correct"),
           "refreshed": _weighted(later, "refreshed_correct"),
           "years": _years([str(r["period"]) for r in later])}
    if "stale_declined_ok" in later[0]:
        out["declined"] = _weighted(later, "stale_declined_ok")
        out["invented"] = round(sum(r["stale_hallucinated"] * r["questions"] for r in later))
    base = [r for r in run.impact.by_period if str(r["period"]) == run.baseline]
    if base:
        out["control_stale"], out["control_refreshed"] = base[0]["stale_correct"], base[0]["refreshed_correct"]
    return out


def harm_counts(run: Run) -> dict[str, Any]:
    """Sections with enough questions: how many lost answers, and how many of those drift missed."""
    seg = harm_vs_drift(run)
    if not len(seg):
        return {}
    harmed = seg[seg["class"].isin(["drift that hurts", "harm without drift"])]
    missed = harmed[harmed["class"] == "harm without drift"]
    return {"judged": len(seg), "harmed": len(harmed), "missed": len(missed),
            "missed_low": int(missed["times_ordinary_embedding"].notna().sum()),
            "missed_unscored": int(missed["times_ordinary_embedding"].isna().sum()),
            "loss_range": pct_range(harmed["loss_correct"]) if len(harmed) else ""}


def _baseline_rows(run: Run) -> pd.DataFrame:
    w = run.whole
    return w[w["compared_with"] == run.baseline] if len(w) else w


def drift_peak(run: Run) -> dict[str, Any]:
    """The year that moved furthest from the baseline, and the gradual-drift check on the last year."""
    vs_base = _baseline_rows(run)
    if not len(vs_base):
        return {}
    top = vs_base.sort_values("times_ordinary_embedding", ascending=False).iloc[0]
    out = {"year": str(top["period"]), "times": float(top["times_ordinary_embedding"]),
           "band": top["embedding_band"]}
    last = str(run.monitoring.periods[-1])
    prev = run.whole[(run.whole["period"].astype(str) == last) & (run.whole["compared_with"] != run.baseline)]
    lb = vs_base[vs_base["period"].astype(str) == last]
    if len(prev) and len(lb):
        out.update(last=last, prev=str(prev.iloc[0]["compared_with"]),
                   last_vs_prev=float(prev.iloc[0]["times_ordinary_embedding"]),
                   last_vs_prev_band=prev.iloc[0]["embedding_band"],
                   last_vs_base=float(lb.iloc[0]["times_ordinary_embedding"]),
                   last_vs_base_band=lb.iloc[0]["embedding_band"])
    return out


def overview_findings(run: Run) -> list[str]:
    """The run's main findings, one sentence each, for the top of the Overview."""
    out = []
    h = answer_headline(run)
    if h:
        s = (f"**Without re-indexing, the assistant answers {pct(h['stale'])} of {h['questions']} questions "
             f"about {h['years']} correctly; after re-indexing, {pct(h['refreshed'])}.**")
        if "declined" in h and h["declined"] >= 0.9:
            s += (f" It fails silently: it says the archive does not cover the question "
                  f"({pct(h['declined'])} of cases) rather than inventing an answer.")
        out.append(s)
    c = harm_counts(run)
    if c and c["harmed"]:
        s = f"**{c['harmed']} of {c['judged']} sections with enough questions lost answers"
        if c["missed"]:
            s += (f", and drift detection flagged only {c['harmed'] - c['missed']} of them.** Drift scores show "
                  "where content changed, not where answers break, so re-index on a schedule rather than "
                  "waiting for a drift alert.")
        else:
            s += ", and drift detection flagged all of them.**"
        out.append(s)
    p = drift_peak(run)
    if p:
        s = f"**The biggest change is {p['year']}:** {p['times']:.1f}× ordinary change from the baseline ({p['band']})."
        if "last" in p and p["last_vs_prev_band"] == "low" and p["last_vs_base_band"] != "low":
            s += (f" {p['last']} looks ordinary next to {p['prev']} ({p['last_vs_prev']:.1f}×) but has moved "
                  f"{p['last_vs_base']:.1f}× from the baseline: comparing only with last year hides gradual drift.")
        out.append(s)
    return out


def route_sentence(run: Run) -> str:
    """What the routing decisions were, in words: after Agent B, and the evidence loop after Agent D."""
    n = len(run.drift.drifted_segments)
    if run.next_step == "full_evaluation":
        s = (f"Drift detection flagged {n} year × section groups, so the system ran the full answer check "
             "on every question.")
    elif run.next_step == "regression_check":
        s = "No group was flagged, so the system ran a small regression check instead of the full answer check."
    else:
        s = f"Routing after drift detection: {run.next_step.replace('_', ' ')}."
    loop = run.impact.evidence if run.impact is not None else []
    if loop:
        s += (f" {len(loop)} borderline section{'s' if len(loop) != 1 else ''} went back to Agent C for more "
              "evidence (the evidence loop, see Impact).")
    return s


def _segment_name(segment_id: str) -> str:
    """'2020|WORLD NEWS|vs 2016-17' -> '2020 WORLD NEWS'."""
    period, cat = segment_id.split("|")[:2]
    return f"{period} {cat}"


def fault_banner(run: Run) -> str:
    """For a run with a simulated failure (D34): what was broken, so it is never mistaken for a normal run."""
    f = run.info.get("fault_injection")
    if not f:
        return ""
    return (f"Simulated failure: {float(f['drop_share']):.0%} of the {f['period']} {f['category']} articles were "
            "left out of the refreshed index, as if an ingestion batch had failed. This run demonstrates the "
            "evidence loop; its numbers are not the project's results.")


def evidence_loop_takeaway(run: Run) -> str:
    """What the evidence loop found, per borderline section, and what Agent E did with it."""
    rows = run.impact.evidence if run.impact is not None else []
    if not rows:
        return ""
    parts = []
    for e in rows:
        parts.append(f"{_segment_name(e['segment_id'])} ({e['questions']} questions): {pct(e['loss_top5'])} fewer "
                     f"correct answers on the stale index when answering from the {_top(run, 'answer_k')}, "
                     f"{pct(e['loss_top10'])} from the {_top(run, 'top_k')} (interval {pct(e['loss_top10_low'])} to "
                     f"{pct(e['loss_top10_high'])}). Conclusion: {e['conclusion']}.")
    return " ".join(parts)


def drift_chart_takeaway(run: Run) -> str:
    p = drift_peak(run)
    if not p:
        return ""
    s = f"{p['year']} moved furthest from the baseline ({p['times']:.1f}×, {p['band']})."
    if "last" in p:
        s += (f" {p['last']}: {p['last_vs_prev']:.1f}× against {p['prev']} (blue), {p['last_vs_base']:.1f}× "
              "against the baseline (orange).")
    return s


def monitoring_points(run: Run) -> list[str]:
    """What arrived each year (Agent A), as short sentences."""
    rows = pd.DataFrame(run.monitoring.period_summary)
    if not len(rows):
        return []
    rows["period"] = rows["period"].astype(str)
    later = rows[rows["period"] != run.baseline]
    base = rows[rows["period"] == run.baseline]
    out = []
    if len(base) and len(later):
        out.append(f"**Volume:** {int(base['documents'].iloc[0]):,} articles in the baseline, then "
                   f"{int(later['documents'].min()):,} to {int(later['documents'].max()):,} a year. A smaller "
                   "batch is a property of the source, not drift.")
    if "new_section_share" in later and later["new_section_share"].max() > 0:
        top = later.sort_values("new_section_share").iloc[-1]
        out.append(f"**New sections:** up to {pct(top['new_section_share'])} of a year's articles "
                   f"({top['period']}) are in sections that did not exist in the baseline, so drift cannot be "
                   "scored for them.")
    if "mix_cramers_v" in later and later["mix_cramers_v"].notna().any():
        out.append(f"**Section mix:** how much the share of each section changed, from 0 (not at all) to 1 "
                   f"(completely): {later['mix_cramers_v'].min():.2f} to {later['mix_cramers_v'].max():.2f}.")
    if "new_term_rate" in later and later["new_term_rate"].notna().any():
        top = later.sort_values("new_term_rate").iloc[-1]
        out.append(f"**New words:** words never used in the baseline peak at {pct(top['new_term_rate'])} of all "
                   f"words in {top['period']}.")
    seg = pd.DataFrame(run.monitoring.segments)
    if len(seg):
        reasons = seg.loc[~seg["eligible"], "reason"].value_counts()
        parts = [f"{n} {r}" for r, n in reasons.items()]
        out.append(f"**Enough to analyse:** {int(seg['eligible'].sum())} of {len(seg)} year × section groups"
                   + (f" ({'; '.join(parts)})." if parts else "."))
    return out


def retrieval_takeaway(run: Run) -> str:
    rep = run.retrieval
    if rep is None or not rep.judged_by_period:
        return ""
    later = _later(rep.judged_by_period, run.baseline)
    base = [r for r in rep.judged_by_period if str(r["period"]) == run.baseline]
    if not later:
        return ""
    s = (f"For questions about {_years([str(r['period']) for r in later])}, an answer is in the {_top(run, 'judge_k')} "
         f"for "
         f"{pct_range([r['stale_answer'] for r in later])} on the stale index and "
         f"{pct_range([r['refreshed_answer'] for r in later])} on the refreshed one.")
    if base:
        b = base[0]
        same = abs(b["stale_answer"] - b["refreshed_answer"]) < 0.005
        s += (f" {run.baseline} questions: {pct(b['stale_answer'])} on both, so adding new articles does not hurt "
              "old questions." if same else
              f" {run.baseline} questions: {pct(b['stale_answer'])} stale vs {pct(b['refreshed_answer'])} "
              "refreshed.")
    return s


def ranking_takeaway(run: Run) -> str:
    rc = run.retrieval.ranking_change if run.retrieval is not None else []
    if not rc:
        return ""
    first, last = rc[0], rc[-1]
    if "overlap" not in first:
        return ""
    s = (f"As new articles are added, the old questions' {_top(run, 'top_k')} change gradually: "
         f"overlap with the stale "
         f"index falls from {pct(first['overlap'])} to {pct(last['overlap'])}")
    if "newer_share" in first:
        s += f", and newer articles grow from {pct(first['newer_share'])} to {pct(last['newer_share'])} of the results"
    s += "."
    if all("source_displaced" in r and r["source_displaced"] == 0 for r in rc):
        s += f" The article each question came from is never pushed out of the {_top(run, 'top_k')}."
    return s


def answers_takeaway(run: Run) -> str:
    h = answer_headline(run)
    if not h:
        return ""
    s = f"{h['years']}: {pct(h['stale'])} correct on the stale index vs {pct(h['refreshed'])} on the refreshed one."
    if "control_stale" in h:
        same = abs(h["control_stale"] - h["control_refreshed"]) < 0.005
        s += (f" {run.baseline} (the control): {pct(h['control_stale'])} on both indexes."
              if same else f" {run.baseline} (the control): {pct(h['control_stale'])} stale vs "
              f"{pct(h['control_refreshed'])} refreshed.")
    if "invented" in h:
        s += (f" On the stale index the assistant fails silently: it declines rather than inventing an answer "
              f"({h['invented']} of {h['questions']} answers added a claim not in the articles).")
    return s


def harm_takeaway(run: Run) -> str:
    c = harm_counts(run)
    if not c:
        return ""
    if not c["harmed"]:
        return f"None of the {c['judged']} sections with enough questions lost a material share of answers."
    s = (f"{c['harmed']} of {c['judged']} sections lost {c['loss_range']} of their correct answers, whatever "
         "their drift score.")
    if c["missed"]:
        s += f" Drift detection missed {c['missed']}: "
        bits = []
        if c["missed_low"]:
            bits.append(f"{c['missed_low']} scored low")
        if c["missed_unscored"]:
            bits.append(f"{c['missed_unscored']} are in sections new since the baseline, which it cannot score "
                        "(listed below the chart)")
        s += " and ".join(bits) + "."
    return s


def correlation_takeaway(run: Run) -> str:
    rows = run.impact.correlations if run.impact is not None else []
    if not rows:
        return ""
    r = pd.DataFrame(rows)
    sig = r[r["p_value"] < 0.05]
    s = (f"Across {int(r['segments'].max())} scored sections, drift measures and the size of the loss correlate "
         f"from {r['spearman'].min():.2f} to {r['spearman'].max():.2f} (1 = always together, 0 = unrelated).")
    s += (" None is significant: no drift measure predicts how much a section loses." if sig.empty else
          f" Significant: {', '.join(sig['measure'])}.")
    return s


def wikipedia_takeaway(run: Run) -> str:
    lf = run.longform
    summary, verdict = lf.get("outdated_summary"), lf.get("outdated_verdict")
    if summary is None or not len(summary):
        return ""
    g = summary.set_index("group")
    if "flagged" not in g.index or "comparison" not in g.index:
        return ""
    f, c = g.loc["flagged"], g.loc["comparison"]
    s = (f"When an edit replaced a number or date, the answer changed {pct(f['answer_changed'])} of the time; "
         f"for edits that kept their numbers, {pct(c['answer_changed'])}")
    if verdict:
        s += f" (Fisher p = {verdict['p_value']:.1e})"
    s += "."
    if "edit_retrieved_old" in f:
        s += (f" The old index still retrieved the edited passage {pct(f['edit_retrieved_old'])} of the time, "
              "so it serves the outdated fact with confidence.")
    fs = lf.get("fact_summary")
    if fs is not None and len(fs):
        s += (f" Drift detection would not have warned: {pct(low_band_share(fs))} of the articles with a changed "
              "fact had a \"low\" drift score.")
    return s


def flag_share_takeaway(run: Run) -> str:
    """Which year had the largest share of its analysable sections flagged."""
    from driftmon.dashboard.runs import flagged_share_by_year
    f = flagged_share_by_year(run)
    if not len(f) or not f["flagged"].any():
        return ""
    top = f.sort_values(["share", "groups"], ascending=False).iloc[0]
    rest = f[f["period"] != top["period"]]
    s = (f"{top['period']}: {int(top['flagged'])} of {int(top['groups'])} analysable sections flagged "
         f"({pct(top['share'])}), the most of any year")
    if len(rest):
        s += f"; other years {pct_range(rest['share'])}"
    return s + "."


def severity_takeaway(run: Run) -> str:
    """The year with the most flagged sections, and which ones moved most."""
    seg = run.segments
    if not len(seg):
        return ""
    f = seg[(seg["compared_with"] == run.baseline) & seg["flagged"]]
    if not len(f):
        return "No section was flagged against the baseline."
    year = f["period"].astype(str).value_counts().idxmax()
    top = f[f["period"].astype(str) == year].sort_values("times_ordinary_embedding", ascending=False)
    names = ", ".join(c.title() for c in top["category"].head(4))
    return (f"{year} has the most flagged sections ({len(top)}); the largest shifts are in {names}. "
            f"{len(f)} section-years are flagged in all.")


def section_trend_takeaway(run: Run) -> str:
    """The biggest section moves from the baseline, and the new sections' peak."""
    from driftmon.dashboard.runs import ALL_OTHER, section_shares
    d = section_shares(run)
    if not len(d):
        return ""
    d = d.assign(period=d["period"].astype(str))
    last = str(run.monitoring.periods[-1])
    wide = d.pivot_table(index="section", columns="period", values="share", observed=True)
    out = []
    old = wide[(wide.index != ALL_OTHER) & wide[run.baseline].notna()] if run.baseline in wide else wide.iloc[0:0]
    if len(old) and last in old:
        move = (old[last] - old[run.baseline]).dropna()
        if len(move):
            s = move.abs().idxmax()
            out.append(f"{s.title()} went from {pct(old.loc[s, run.baseline])} of articles in the baseline to "
                       f"{pct(old.loc[s, last])} in {last}.")
    new = d[d["new"]]
    if len(new):
        peak = new.sort_values("share").iloc[-1]
        out.append(f"{peak['section'].title()}, a section new since the baseline, peaked at {pct(peak['share'])} "
                   f"of articles in {peak['period']}.")
    return " ".join(out)


def topics_takeaway(run: Run) -> str:
    t = run.topics
    if not len(t) or "docs_period" not in t:
        return ""
    top = t.sort_values("docs_period", ascending=False).iloc[0]
    words = ", ".join(str(top["label"]).split(", ")[:3])
    return (f"The largest is {top['period']}: {words}, with {int(top['docs_period'])} articles against "
            f"{int(top['docs_baseline'])} in the baseline sample. Most topics existed before and grew.")


def meaning_takeaway(run: Run) -> str:
    s = run.term_shifts
    if not len(s) or "times_ordinary" not in s:
        return ""
    s = s.dropna(subset=["times_ordinary"]).sort_values("times_ordinary", ascending=False)
    if not len(s):
        return ""
    top = s.iloc[0]
    before = ", ".join(str(top["context_before"]).split(", ")[:3])
    after = ", ".join(str(top["context_after"]).split(", ")[:3])
    return (f"\"{top['term']}\" in {top['period']} moved most, {top['times_ordinary']:.1f}× ordinary change: "
            f"its neighbouring words went from {before} to {after}.")


def query_drift_takeaway(run: Run) -> str:
    q = run.drift.query_drift
    if isinstance(q, str) or not q:
        return ""
    d = pd.DataFrame(q)
    d = d[~d["period"].astype(str).str.contains("inside baseline")]
    if not len(d) or "times_ordinary_embedding" not in d:
        return ""
    top = d.sort_values("times_ordinary_embedding").iloc[-1]
    terms = str(top.get("emerging_terms") or "").split(", ")[:3]
    s = f"The questions shift most in {top['period']} ({top['times_ordinary_embedding']:.1f}× the reference"
    return s + (f"), with new words such as {', '.join(terms)}." if terms and terms[0] else ").")
