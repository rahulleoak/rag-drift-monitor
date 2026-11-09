"""Streamlit dashboard. Reads saved outputs from ``runs/<run_id>/``; it does not
run the pipeline itself, so the demo cannot fail on a live LLM call (Q16).

Pages (tabs): Overview, Data monitoring (Agent A), Drift (Agent B), Retrieval
(Agent C), Impact (Agent D) and Recommendations (Agent E).

Run (from the repository root):
    streamlit run app/streamlit_app.py
It reads ``runs/`` if it holds any pipeline runs, and otherwise the run
committed under ``demo/runs/`` (no API key or rerun needed). Set
DRIFTMON_RUNS_DIR to read runs from another folder (used by the tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.dashboard.charts import DARK as T  # noqa: E402  (Rivvun dark palette, validated)
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.dashboard.runs import (answers_stale_vs_refreshed, harm_vs_drift, list_runs, load_run,  # noqa: E402
                                     run_models, severity_grid, stale_vs_refreshed)

st.set_page_config(page_title="RAG drift monitor", layout="wide")
config = load_config()
runs_dir = Path(os.environ.get("DRIFTMON_RUNS_DIR") or resolve_path(config["paths"]["runs"]))
runs = list_runs(runs_dir)
demo = False
if not runs and not os.environ.get("DRIFTMON_RUNS_DIR"):   # fresh clone: the committed demo run
    runs_dir = resolve_path(config["paths"]["demo_runs"])
    runs, demo = list_runs(runs_dir), True

st.title("RAG drift monitor")
st.caption("Newsroom archive assistant · HuffPost 2016–2022 · which drift matters?")
if not runs:
    st.warning(f"No pipeline runs in {runs_dir}. Run `python scripts/run_pipeline.py` first.")
    st.stop()

run_id = st.sidebar.selectbox("Pipeline run", runs, help="Saved runs, newest first")
run = load_run(runs_dir, run_id)
models = run_models(run, config)
st.sidebar.caption(f"Baseline: {run.baseline}\n\nPeriods: {', '.join(run.monitoring.periods)}\n\n"
                   f"Writes: {models['writer']}\n\nJudges: {models['judge']}\n\nEmbeddings: {models['embedding']}"
                   + ("" if models["source"] == "run" else
                      "\n\n(Models from the current config: this run predates run_info.json.)"))
if demo:
    st.sidebar.info("Showing the bundled demo run (`demo/runs/`). Run `python scripts/run_pipeline.py` "
                    "to see your own.")

ORDINAL = alt.Scale(domain=["low", "medium", "high"], range=list(T.bands.values()))


def themed(chart: alt.Chart) -> alt.Chart:
    """Rivvun dark styling: navy surface, off-white ink, hairline slate grid."""
    return (chart.configure(background=T.surface, font="Inter, Segoe UI, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=T.ink_2, titleColor=T.ink_2, gridColor=T.grid,
                            domainColor=T.axis, tickColor=T.axis)
            .configure_legend(labelColor=T.ink_2, titleColor=T.ink_2)
            .configure_title(color=T.ink, anchor="start", fontWeight=600))


def drift_line(whole: pd.DataFrame, col: str, measure: str, title: str) -> alt.Chart:
    """× ordinary change per year, vs the baseline and vs the previous year."""
    ref = run.drift.reference["whole_period_null"]
    typical = ref["typical"][measure]
    th = pd.DataFrame([{"band": b, "y": v / typical} for b, v in ref["thresholds"][measure].items()])
    d = whole.assign(vs=lambda x: x["compared_with"].map(
        lambda c: "vs baseline" if c == run.baseline else "vs previous year"))
    color = alt.Color("vs:N", scale=alt.Scale(domain=["vs baseline", "vs previous year"],
                                              range=list(T.series[:2])), title=None)
    base = alt.Chart(d).encode(x=alt.X("period:O", title=None), y=alt.Y(f"{col}:Q", title="× ordinary change"),
                               color=color)
    lines = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=70, stroke=T.surface,
                                                            strokeWidth=2).encode(
        tooltip=["period", "vs", alt.Tooltip(f"{col}:Q", format=".2f", title="× ordinary"),
                 "embedding_band", "lexical_band", "kind"])
    rules = alt.Chart(th).mark_rule(color=T.muted, strokeWidth=1).encode(y="y:Q", tooltip=["band"])
    labels = alt.Chart(th).mark_text(align="left", dx=4, color=T.ink_2, fontSize=11).encode(
        y="y:Q", x=alt.value(0), text="band")
    return (rules + labels + lines).properties(title=title, height=280)


def severity_heatmap(grid: pd.DataFrame) -> alt.Chart:
    order = (grid[grid["category"] != "ALL SECTIONS"].groupby("category")["n_period"].sum()
             .sort_values(ascending=False).index.tolist())
    y = alt.Y("category:N", sort=["ALL SECTIONS"] + order, title=None)
    base = alt.Chart(grid).encode(x=alt.X("period:O", title=None), y=y)
    rect = base.mark_rect(stroke=T.surface, strokeWidth=2).encode(
        color=alt.Color("severity:N", scale=ORDINAL, title="severity"),
        tooltip=["category", "period", "severity", "kind",
                 alt.Tooltip("times_ordinary_embedding:Q", format=".1f", title="embedding ×"),
                 alt.Tooltip("times_ordinary_lexical:Q", format=".1f", title="words ×"),
                 alt.Tooltip("n_period:Q", title="articles")])
    text = base.transform_filter("datum.flagged").mark_text(fontSize=11).encode(
        text="kind", color=alt.Color("severity:N", scale=alt.Scale(
            domain=["low", "medium", "high"], range=[T.on(c) for c in T.bands.values()]), legend=None))
    return (rect + text).properties(height=24 * grid["category"].nunique() + 20)


tabs = st.tabs(["Overview", "Data monitoring", "Drift", "Retrieval", "Impact", "Recommendations"])

with tabs[0]:
    flagged = run.flagged()
    c = st.columns(4)
    c[0].metric("Periods monitored", len(run.monitoring.periods))
    c[1].metric("Flagged (vs baseline)", len(flagged))
    c[2].metric("Control segments", len(run.drift.control_segments))
    c[3].metric("Next step", run.next_step.replace("_", " "))
    if run.impact is not None and run.impact.by_period:
        new = [r for r in run.impact.by_period if r["period"] != run.baseline]
        n = sum(r["questions"] for r in new)
        stale = sum(r["stale_correct"] * r["questions"] for r in new) / n
        fresh = sum(r["refreshed_correct"] * r["questions"] for r in new) / n
        c = st.columns(4)
        c[0].metric("New-event questions", n)
        c[1].metric("Correct, stale index", f"{stale:.1%}" if 0 < stale < 0.01 else f"{stale:.0%}")
        c[2].metric("Correct, refreshed index", f"{fresh:.0%}")
        if run.recommendations is not None:
            c[3].metric("Recommended actions", len(run.recommendations.recommended()))
    if run.recommendations is not None and run.recommendations.recommended():
        top = run.recommendations.recommended()[0]
        st.markdown(f"**Top recommendation: {top.action}.** {top.finding}")
    left, right = st.columns(2)
    left.altair_chart(themed(drift_line(run.whole, "times_ordinary_embedding", "mmd2", "Embedding drift")),
                      width="stretch")
    right.altair_chart(themed(drift_line(run.whole, "times_ordinary_lexical", "jsd_excess", "Word drift")),
                       width="stretch")
    st.subheader("Most drifted segments")
    st.dataframe(flagged[["category", "period", "severity", "kind", "times_ordinary_embedding",
                          "times_ordinary_lexical", "emerging_terms"]].round(2),
                 hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("Agent A · what came in, compared with the baseline")
    summary = pd.DataFrame(run.monitoring.period_summary)
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.markdown("**Data quality (cleaning)**")
    a.dataframe(pd.DataFrame(run.monitoring.data_quality), hide_index=True, width="stretch")
    b.markdown("**Renamed sections (merged for analysis)**")
    b.dataframe(pd.DataFrame(run.monitoring.renamed_sections), hide_index=True, width="stretch")
    seg = pd.DataFrame(run.monitoring.segments)
    st.markdown(f"**Segments:** {int(seg['eligible'].sum())} of {len(seg)} period × section groups "
                "have enough articles to analyse")
    st.dataframe(seg, hide_index=True, width="stretch")
    st.markdown("**Query stream** (simulated questions from the random and oversample pools)")
    if isinstance(run.monitoring.query_stream, str):
        st.info("Query stream: " + run.monitoring.query_stream)
    else:
        st.dataframe(pd.DataFrame(run.monitoring.query_stream).round(3), hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Agent B · drift by section and year")
    st.altair_chart(themed(severity_heatmap(severity_grid(run))), width="stretch")
    st.markdown("**Drill down**")
    ids = run.whole["segment_id"].tolist() + run.segments["segment_id"].tolist()
    pick = st.selectbox("Segment", ids, index=ids.index(run.flagged()["segment_id"].iloc[0])
                        if len(run.flagged()) else 0)
    row = pd.concat([run.whole, run.segments]).set_index("segment_id").loc[pick]
    m = st.columns(4)
    m[0].metric("Embedding × ordinary", f"{row['times_ordinary_embedding']:.1f}", row["embedding_band"])
    m[1].metric("Words × ordinary", f"{row['times_ordinary_lexical']:.1f}", row["lexical_band"])
    m[2].metric("Kind of drift", row["kind"])
    m[3].metric("Novel articles", f"{row['novel_share']:.0%}")
    st.markdown(f"**Emerging words:** {', '.join(row['emerging_terms'])}  \n"
                f"**Declining words:** {', '.join(row['declining_terms'])}")
    if row["examples"]:
        st.markdown("**Example headlines:**\n" + "\n".join(f"- {t}" for t in row["examples"]))
    st.caption(f"Compared with {row['compared_with']} · {row['n_period']} vs {row['n_reference']} articles · "
               f"ordinary change from {row['null_window']} pairs at n = {row['null_n']}")
    for title, key, table in [("Emerging topics", "emerging_topics", run.topics),
                              ("Words that changed meaning", "words_that_changed_meaning", run.term_shifts),
                              ("How the drift measures relate", "measure_correlations", run.correlations)]:
        st.markdown(f"**{title}**")
        if key in run.charts or f"dark/{key}" in run.charts:
            st.image(str(run.charts.get(f"dark/{key}") or run.charts[key]))
        with st.expander("Table"):
            st.dataframe(table.round(3) if len(table) else table, width="stretch")
    st.markdown("**Query drift** (each period's questions vs the baseline questions)")
    if isinstance(run.drift.query_drift, str):
        st.info("Query drift: " + run.drift.query_drift)
    else:
        qd = pd.DataFrame(run.drift.query_drift)
        st.dataframe(qd.drop(columns=["examples"], errors="ignore").round(3), hide_index=True,
                     width="stretch")
        st.caption("The questions are written from each period's articles, so query drift partly "
                   "mirrors document drift by construction; a real query log can drift on its own.")

def dumbbell(long: pd.DataFrame, title: str) -> alt.Chart:
    """Per year: the stale and the refreshed index, joined by a hairline."""
    order = ["Stale index (2016-17 only)", "Refreshed index"]
    colour = alt.Color("index:N", scale=alt.Scale(domain=order, range=[T.series[1], T.series[0]]),
                       legend=alt.Legend(title=None, orient="top"))
    y = alt.Y("period:N", title=None, sort=sorted(long["period"].unique()), axis=alt.Axis(labelOverlap=False))
    x = alt.X("value:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("period:N", title="Year"), alt.Tooltip("index:N", title="Index"),
           alt.Tooltip("value:Q", title=title, format=".0%"), alt.Tooltip("questions:Q", title="Questions"),
           alt.Tooltip("delta:Q", title="Refreshed − stale", format="+.0%"),
           alt.Tooltip("delta_low:Q", title="95% CI low", format="+.0%"),
           alt.Tooltip("delta_high:Q", title="95% CI high", format="+.0%")]
    rule = alt.Chart(long).mark_rule(color=T.axis, strokeWidth=2).encode(
        y=y, x=alt.X("min(value):Q"), x2="max(value):Q")
    dots = alt.Chart(long).mark_circle(size=140, opacity=1, stroke=T.surface, strokeWidth=2).encode(
        y=y, x=x, color=colour, tooltip=tip)
    return (rule + dots).properties(height={"step": 36})


with tabs[3]:
    st.subheader("Agent C · retrieval, stale vs refreshed")
    rep = run.retrieval
    if rep is None:
        st.info("Not run yet. This page shows stale vs refreshed retrieval once Agent C has run.")
    else:
        m = st.columns(4)
        m[0].metric("Questions", rep.questions)
        m[1].metric("Mode", rep.mode.replace("_", " "))
        m[2].metric("Judged questions", rep.judge.get("judged_questions", 0))
        m[3].metric("LLM cost, this run", f"${rep.judge.get('cost_usd', 0):.2f}",
                    help="Cached replies are free, so a rerun costs little; every call is logged in runs/llm_costs.csv.")
        long = stale_vs_refreshed(run, "answer")
        if len(long):
            st.markdown("**(a) Does the top 5 contain an answer?** Each year's questions on the stale "
                        "index and on the refreshed index. 2016–17 questions are the historical control.")
            st.altair_chart(themed(dumbbell(long, "Share of questions answered by the top 5")), width="stretch")
            with st.expander("Table: by year and by drift status, with 95% intervals"):
                st.dataframe(pd.DataFrame(rep.judged_by_period).round(3), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(rep.judged_by_status).round(3), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(rep.judged_by_type).round(3), hide_index=True, width="stretch")
        else:
            st.info("The judged comparison was not run (no LLM judge in this run).")
        a, b = st.columns(2)
        a.markdown("**(b) Known item on the same refreshed index** (weighted down: F11)")
        a.dataframe(pd.DataFrame(rep.known_item).round(3), hide_index=True, width="stretch")
        b.markdown("**(c) Old questions, stale vs refreshed top 10**")
        b.dataframe(pd.DataFrame(rep.ranking_change).round(3), hide_index=True, width="stretch")
        if rep.worst_queries:
            with st.expander("Questions that lost most on the stale index"):
                for w in rep.worst_queries:
                    st.markdown(f"**{w['question']}** ({w['period']}, {w['category']}, {w['drift_status']}) · "
                                f"nDCG stale {w['stale_ndcg']:.2f} → refreshed {w['refreshed_ndcg']:.2f}  \n"
                                f"Stale top 3: {' · '.join(w['stale_top3'])}  \n"
                                f"Refreshed top 3: {' · '.join(w['refreshed_top3'])}")
        for note in rep.notes:
            st.caption(note)

def harm_scatter(seg: pd.DataFrame) -> alt.Chart:
    """Each classified section: its drift score against its loss in correct answers."""
    classes = ["drift that hurts", "harm without drift", "drift that does not hurt", "healthy"]
    colour = alt.Color("class:N", scale=alt.Scale(domain=classes, range=[*T.series, T.muted]),
                       legend=alt.Legend(title=None, orient="top"))
    x = alt.X("times_ordinary_embedding:Q", title="Embedding drift (× an ordinary year)")
    y = alt.Y("loss_correct:Q", title="Drop in correct answers", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("section:N", title="Section"), alt.Tooltip("class:N", title="Class"),
           alt.Tooltip("loss_correct:Q", title="Drop", format=".0%"),
           alt.Tooltip("loss_low:Q", title="95% CI low", format=".0%"),
           alt.Tooltip("loss_high:Q", title="95% CI high", format=".0%"),
           alt.Tooltip("times_ordinary_embedding:Q", title="Embedding ×", format=".1f"),
           alt.Tooltip("severity:N", title="Drift severity"), alt.Tooltip("questions:Q", title="Questions")]
    base = alt.Chart(seg).encode(x=x)
    err = base.mark_rule(color=T.axis, strokeWidth=2).encode(y=alt.Y("loss_low:Q"), y2="loss_high:Q")
    dots = base.mark_circle(size=130, opacity=1, stroke=T.surface, strokeWidth=2).encode(
        y=y, color=colour, tooltip=tip)
    return (err + dots).properties(height=300)


def outdated_chart(summary: pd.DataFrame) -> alt.Chart:
    """Wikipedia: how often the answer changed between the old and the new version."""
    names = {"flagged": "Flagged changed facts", "comparison": "Edits that kept their numbers"}
    d = summary.assign(group=summary["group"].map(names))
    y = alt.Y("group:N", title=None, sort=list(names.values()), axis=alt.Axis(labelLimit=260))
    x = alt.X("answer_changed:Q", title="Questions whose answer changed", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("group:N", title="Edits"), alt.Tooltip("items:Q", title="Questions"),
           alt.Tooltip("answer_changed:Q", title="Answer changed", format=".0%"),
           alt.Tooltip("changed_low:Q", title="95% CI low", format=".0%"),
           alt.Tooltip("changed_high:Q", title="95% CI high", format=".0%"),
           alt.Tooltip("edit_retrieved_old:Q", title="Old index found the passage", format=".0%")]
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=y, x=x, tooltip=tip,
        color=alt.Color("group:N", scale=alt.Scale(domain=list(names.values()), range=[T.series[0], T.muted]),
                        legend=None))
    err = alt.Chart(d).mark_rule(color=T.ink_2, strokeWidth=1.5).encode(y=y, x="changed_low:Q",
                                                                        x2="changed_high:Q")
    label = alt.Chart(d).mark_text(align="left", dx=8, color=T.ink, fontSize=12).encode(
        y=y, x="changed_high:Q", text=alt.Text("answer_changed:Q", format=".0%"))
    return (bars + err + label).properties(height={"step": 48})


with tabs[4]:
    st.subheader("Agent D · answers, stale vs refreshed, and which drift matters")
    imp = run.impact
    if imp is None or not imp.questions:
        st.info("Not run yet. This page shows answer quality once Agent D has run with an LLM.")
    else:
        m = st.columns(4)
        m[0].metric("Questions answered (both indexes)", imp.questions)
        m[1].metric("Material loss threshold", f"{imp.threshold:.0%}")
        un = {r["index"]: r for r in imp.unanswerable}
        if "stale" in un:
            m[2].metric("Unanswerable: declined (stale)", f"{un['stale']['declined']:.0%}")
        m[3].metric("LLM cost, this run", f"${imp.judge.get('cost_usd', 0):.2f}",
                    help="Cached replies are free, so a rerun costs little; every call is logged in runs/llm_costs.csv.")
        st.markdown(f"**Does the assistant answer correctly?** {models['writer']} answers from each index's "
                    f"top 5; {models['judge']} grades each answer alone, not told the index. {run.baseline} "
                    "questions are the historical control.")
        st.altair_chart(themed(dumbbell(answers_stale_vs_refreshed(run), "Share of questions answered correctly")),
                        width="stretch")
        st.caption("2016–17: both indexes answer 100%, so the two dots overlap. On the stale index the "
                   "assistant fails silently: it says the archive does not have the answer rather than "
                   "inventing one.")
        seg = harm_vs_drift(run)
        if len(seg):
            st.markdown("**Which drift matters?** Every section with enough questions, by its drift score "
                        "and its drop in correct answers. More drift does not mean more harm.")
            scored = seg.dropna(subset=["times_ordinary_embedding"])
            st.altair_chart(themed(harm_scatter(scored)), width="stretch")
            unscored = seg[seg["times_ordinary_embedding"].isna()]
            counts = ", ".join(f"{n} {c}" for c, n in seg["class"].value_counts().items())
            st.caption(f"{len(seg)} sections: {counts}. "
                       + (f"Not plotted, because drift detection could not score them (not in the baseline): "
                          f"{', '.join(unscored['section'])}." if len(unscored) else ""))
            with st.expander("Table: every section, with its class and likely cause"):
                st.dataframe(pd.DataFrame(imp.segments).round(3), hide_index=True, width="stretch")
        if imp.correlations:
            with st.expander("Drift measures vs the loss (Spearman, across sections)"):
                st.dataframe(pd.DataFrame(imp.correlations).round(3), hide_index=True, width="stretch")
        if imp.evidence:
            st.markdown("**Evidence loop** (borderline sections, answered again from the top 10)")
            st.dataframe(pd.DataFrame(imp.evidence).round(3), hide_index=True, width="stretch")
        else:
            st.caption("No section was borderline, so the loop back to Agent C did not run.")
        if imp.examples:
            with st.expander("Examples: questions only the refreshed index answers"):
                for e in imp.examples:
                    st.markdown(f"**{e['question']}** ({e['period']}, {e['category']})  \n"
                                f"*Stale:* {e['stale_answer']}  \n*Refreshed:* {e['refreshed_answer']}")
        lf = run.longform
        if "outdated_summary" in lf:
            st.markdown("**Edited documents (Wikipedia): does a changed fact change the answer?** Questions "
                        "written from the new version of an edited sentence, answered from the old and the "
                        "new version of the archive.")
            st.altair_chart(themed(outdated_chart(lf["outdated_summary"])), width="stretch")
            v = lf.get("outdated_verdict", {})
            if v:
                st.caption(f"Difference {v['difference']:.0%} (needed {v['min_difference']:.0%}), Fisher "
                           f"p = {v['p_value']:.1e}: the changed-facts check is "
                           f"{'useful' if v['useful'] else 'not shown to be useful'}.")
            items = lf.get("outdated_items")
            if items is not None and len(items):
                with st.expander("Examples: same question, old and new version"):
                    hit = items[(items["group"] == "flagged") & (items["verdict"] == "different")].head(6)
                    for r in hit.to_dict("records"):
                        st.markdown(f"**{r['title']}** · {r['question']}  \n*Before:* {r['before']}  \n"
                                    f"*After:* {r['after']}  \n*Old answer:* {r['answer_old']}  \n"
                                    f"*New answer:* {r['answer_new']}")
        for note in imp.notes:
            st.caption(note)

with tabs[5]:
    st.subheader("Agent E · recommended actions, with their evidence")
    rec = run.recommendations
    if rec is None:
        st.info("Not run yet. This page shows Agent E's recommendations once the full pipeline has run.")
    else:
        m = st.columns(4)
        for i, status in enumerate(["recommended", "not needed", "not checked"]):
            m[i].metric(status.capitalize(), rec.summary.get(status, 0))
        m[3].metric("LLM cost, this run", f"${rec.explanations.get('cost_usd', 0):.2f}")
        st.caption("Rules in code decide (one per action in the brief); an LLM explains each recommended "
                   "action from its evidence, and any explanation with a number that is not in the evidence "
                   "is rejected.")
        for r in rec.recommended():
            st.markdown(f"#### {r.priority}. {r.action}")
            st.write(r.explanation)
            if r.scope:
                shown = ", ".join(r.scope[:12]) + (f", and {len(r.scope) - 12} more" if len(r.scope) > 12 else "")
                st.caption(f"Applies to ({len(r.scope)}): {shown}")
            with st.expander("Evidence"):
                for e in r.evidence:
                    st.markdown(f"- **{e.source}:** {e.claim}")
                st.caption(f"Rule: {r.rule}. Explanation: {r.explanation_check}.")
        rest = [r for r in rec.recommendations if r.status != "recommended"]
        if rest:
            st.markdown("#### Checked, not recommended")
            for r in rest:
                st.markdown(f"- **{r.action}** ({r.status}): {r.finding} "
                            + " ".join(e.claim for e in r.evidence))
        for note in rec.notes:
            st.caption(note)
