"""Streamlit dashboard. Reads saved outputs from ``runs/<run_id>/``; it does not
run the pipeline itself, so the demo cannot fail on a live LLM call (Q16).

Pages (tabs): Overview, Data monitoring (Agent A), Drift (Agent B), and
placeholders for Retrieval (C), Impact (D) and Recommendations (E), which fill
in as those agents are built.

Run (from the repository root):
    streamlit run app/streamlit_app.py
Set DRIFTMON_RUNS_DIR to read runs from another folder (used by the tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.charts import DARK as T  # noqa: E402  (Rivvun dark palette, validated)
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.dashboard_data import list_runs, load_run, severity_grid  # noqa: E402

st.set_page_config(page_title="RAG drift monitor", layout="wide")
config = load_config()
runs_dir = Path(os.environ.get("DRIFTMON_RUNS_DIR") or resolve_path(config["paths"]["runs"]))
runs = list_runs(runs_dir)

st.title("RAG drift monitor")
st.caption("Newsroom archive assistant · HuffPost 2016–2022 · which drift matters?")
if not runs:
    st.warning(f"No pipeline runs in {runs_dir}. Run `python scripts/run_pipeline.py` first.")
    st.stop()

run_id = st.sidebar.selectbox("Pipeline run", runs, help="Saved runs, newest first")
run = load_run(runs_dir, run_id)
st.sidebar.caption(f"Baseline: {run.baseline}\n\nPeriods: {', '.join(run.monitoring.periods)}")

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
    st.info("Query stream: " + run.monitoring.query_stream)

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
        if key in run.charts:
            st.image(str(run.charts.get(f"dark/{key}", run.charts[key])))
        with st.expander("Table"):
            st.dataframe(table.round(3) if len(table) else table, width="stretch")
    st.info("Explanations written by the LLM will appear here once the LLM parts are built.")

for tab, agent, what in [(tabs[3], "C · Retrieval evaluation", "stale vs refreshed retrieval per segment"),
                         (tabs[4], "D · RAG impact assessment", "answer quality and which drift matters"),
                         (tabs[5], "E · Recommendations", "actions with their evidence")]:
    with tab:
        st.subheader(f"Agent {agent}")
        st.info(f"Not run yet. This page will show {what}. Next step chosen after Agent B: "
                f"**{run.next_step.replace('_', ' ')}**.")
