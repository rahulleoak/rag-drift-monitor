"""Streamlit dashboard. Reads saved outputs from ``runs/<run_id>/``; it does not
run the pipeline itself, so the demo cannot fail on a live LLM call (Q16).

Pages (tabs): Overview, Data monitoring (Agent A), Drift (Agent B), Retrieval
(Agent C), Impact (Agent D) and Recommendations (Agent E).

Text on the page follows one rule: fixed text only says what a chart is and
how to read it; what a run *shows* is built from its numbers by
``driftmon.dashboard.story``, so it cannot contradict the charts.

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

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.dashboard import story  # noqa: E402
from driftmon.dashboard.charts import DARK as T  # noqa: E402  (Rivvun dark palette, validated)
from driftmon.dashboard.runs import (answers_stale_vs_refreshed, harm_vs_drift, list_runs, load_run,  # noqa: E402
                                     run_label, run_models, severity_grid, stale_label, stale_vs_refreshed,
                                     with_settings)
from driftmon.policy import pct  # noqa: E402

st.set_page_config(page_title="RAG drift monitor", layout="wide")
config = load_config()
runs_dir = Path(os.environ.get("DRIFTMON_RUNS_DIR") or resolve_path(config["paths"]["runs"]))
runs = list_runs(runs_dir)
demo = False
if not runs and not os.environ.get("DRIFTMON_RUNS_DIR"):   # fresh clone: the committed demo run
    runs_dir = resolve_path(config["paths"]["demo_runs"])
    runs, demo = list_runs(runs_dir), True

st.title("RAG drift monitor", anchor=False)
if not runs:
    st.warning(f"No pipeline runs in {runs_dir}. Run `python scripts/run_pipeline.py` first.")
    st.stop()

run_id = st.sidebar.selectbox("Pipeline run", runs, format_func=lambda r: run_label(runs_dir, r),
                              help="Saved runs, newest first, with the agents each one holds. Older runs "
                              "can be kept in runs/archive/, which is not listed.")
run = with_settings(load_run(runs_dir, run_id), config)   # settings: from the run, else the config
models = run_models(run, config)
first = run.baseline.split("-")[0] if run.baseline[:4].isdigit() else run.baseline
st.caption(f"{run.settings['description']}, {first}–{run.monitoring.periods[-1]}: which drift matters?")

# ---- sidebar: what produced this run -----------------------------------------------------------
about = [f"**Baseline:** {run.baseline}", f"**Later years:** {', '.join(run.monitoring.periods)}",
         f"**Writes answers:** {models['writer']}", f"**Judges:** {models['judge']}",
         f"**Embeddings:** {models['embedding']}"]
if run.info.get("created"):
    about.append(f"**Run on:** {run.info['created'][:16].replace('T', ' ')}")
if run.info.get("llm_cost_usd") is not None:
    about.append(f"**LLM cost of this run:** ${run.info['llm_cost_usd']:.2f} (cached replies are free)")
st.sidebar.markdown("\n\n".join(about))
if models["source"] != "run":
    st.sidebar.caption("Model names from the current config: this run predates run_info.json.")
if demo:
    st.sidebar.info("Showing the bundled demo run (`demo/runs/`). Run `python scripts/run_pipeline.py` "
                    "to see your own.")

ORDINAL = alt.Scale(domain=["low", "medium", "high"], range=list(T.bands.values()))
PERCENT = st.column_config.NumberColumn(format="percent")


def top(key: str) -> str:
    """'top 5' for the run's setting ``key`` (retrieval.judge_k, retrieval.top_k ...), else 'top results'."""
    k = run.settings.get(key)
    return f"top {k}" if k else "top results"


def at_least(key: str, unit: str) -> str:
    v = run.settings.get(key)
    return f"at least {v:g} {unit}" if v else f"enough {unit}"


def tilted(n: int) -> int:
    """Axis label angle: flat for a few periods, tilted when many would collide."""
    return 0 if n <= 8 else -45


N_PERIODS = len(run.monitoring.periods) + 1


# ---- small layout helpers ----------------------------------------------------------------------
def heading(text: str) -> None:
    st.markdown(f"#### {text}", anchors=False)


def intro(text: str) -> None:
    """What this tab is for: one or two sentences in the body font."""
    st.markdown(text)


def read(text: str) -> None:
    """How to read the chart above (fixed: true for any run)."""
    st.caption(f"How to read it: {text}")


def shows(text: str) -> None:
    """What this run shows (built from the run's numbers by driftmon.dashboard.story)."""
    if text:
        st.markdown(f"**In this run:** {text}")


def table(df: pd.DataFrame, columns: dict[str, str], percent: tuple[str, ...] = (),
          decimals: dict[str, int] | tuple[str, ...] = ()) -> None:
    """Show only ``columns`` (old name -> readable name), in that order. Shares in ``percent`` show as
    percentages with one decimal; columns in ``decimals`` with that many decimals (a tuple means one);
    whole numbers get thousands separators."""
    keep = [c for c in columns if c in df.columns]
    view = df[keep].copy()
    for c in percent:
        if c in keep:
            view[c] = pd.to_numeric(view[c], errors="coerce") * 100
    view = view.rename(columns=columns)
    digits = decimals if isinstance(decimals, dict) else {c: 1 for c in decimals}
    cfg = {columns[c]: st.column_config.NumberColumn(format="%.1f%%") for c in percent if c in keep}
    cfg.update({columns[c]: st.column_config.NumberColumn(format=f"%.{d}f")
                for c, d in digits.items() if c in keep})
    for c in keep:
        if columns[c] not in cfg and pd.api.types.is_integer_dtype(df[c]):
            cfg[columns[c]] = st.column_config.NumberColumn(format="localized")
    st.dataframe(view, hide_index=True, width="stretch", column_config=cfg)


def segment_label(segment_id: str) -> str:
    """'2020|FOOD & DRINK|vs 2016-17' -> '2020 · FOOD & DRINK (vs 2016-17)'."""
    period, category, vs = (segment_id.split("|") + ["", ""])[:3]
    return f"{period} · {category or 'ALL SECTIONS'} ({vs})" if vs else segment_id


# ---- charts ------------------------------------------------------------------------------------
def themed(chart: alt.Chart) -> alt.Chart:
    """Rivvun dark styling: navy surface, off-white ink, hairline slate grid."""
    return (chart.configure(background=T.surface, font="Inter, Segoe UI, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=T.ink_2, titleColor=T.ink_2, gridColor=T.grid,
                            domainColor=T.axis, tickColor=T.axis, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=T.ink_2, titleColor=T.ink_2, labelFontSize=12)
            .configure_title(color=T.ink, anchor="start", fontWeight=600, fontSize=14))


def drift_line(whole: pd.DataFrame, col: str, measure: str, title: str) -> alt.Chart:
    """× ordinary change per year, vs the baseline and vs the previous year."""
    ref = run.drift.reference["whole_period_null"]
    typical = ref["typical"][measure]
    th = pd.DataFrame([{"band": b, "y": v / typical} for b, v in ref["thresholds"][measure].items()])
    d = whole.assign(vs=lambda x: x["compared_with"].map(
        lambda c: "vs the baseline" if c == run.baseline else "vs the previous year"))
    color = alt.Color("vs:N", scale=alt.Scale(domain=["vs the baseline", "vs the previous year"],
                                              range=list(T.series[:2])),
                      legend=alt.Legend(title=None, orient="top"))
    base = alt.Chart(d).encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=tilted(N_PERIODS))),
                               y=alt.Y(f"{col}:Q", title="× ordinary change"), color=color)
    band_of = "embedding_band" if "embedding" in col else "lexical_band"
    lines = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=70, stroke=T.surface,
                                                            strokeWidth=2).encode(
        tooltip=[alt.Tooltip("period:N", title="Year"), alt.Tooltip("vs:N", title="Compared"),
                 alt.Tooltip(f"{col}:Q", format=".2f", title="× ordinary change"),
                 alt.Tooltip(f"{band_of}:N", title="Severity")])
    rules = alt.Chart(th).mark_rule(color=T.muted, strokeWidth=1, strokeDash=[4, 3]).encode(y="y:Q")
    labels = alt.layer(*[alt.Chart(th[th["band"] == b]).mark_text(
        align="left", dy=dy, color=T.ink_2, fontSize=11).encode(y="y:Q", x=alt.value(4), text="band")
        for b, dy in (("high", -7), ("medium", 9))])
    return (rules + labels + lines).properties(title=title, height=260)


def severity_heatmap(grid: pd.DataFrame) -> alt.Chart:
    """Section × year, coloured by severity; flagged cells say what kind of drift."""
    order = (grid[grid["category"] != "ALL SECTIONS"].groupby("category")["n_period"].sum()
             .sort_values(ascending=False).index.tolist())
    y = alt.Y("category:N", sort=["ALL SECTIONS"] + order, title=None, axis=alt.Axis(labelOverlap=False))
    base = alt.Chart(grid).encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=tilted(N_PERIODS), orient="top")),
                                  y=y)
    rect = base.mark_rect(stroke=T.surface, strokeWidth=2).encode(
        color=alt.Color("severity:N", scale=ORDINAL, title="Drift severity",
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("category:N", title="Section"), alt.Tooltip("period:N", title="Year"),
                 alt.Tooltip("severity:N", title="Severity"), alt.Tooltip("kind:N", title="Kind of drift"),
                 alt.Tooltip("times_ordinary_embedding:Q", format=".1f", title="Meaning × ordinary"),
                 alt.Tooltip("times_ordinary_lexical:Q", format=".1f", title="Words × ordinary"),
                 alt.Tooltip("n_period:Q", title="Articles")])
    text = base.transform_filter("datum.flagged").mark_text(fontSize=11).encode(
        text="kind:N", color=alt.Color("severity:N", scale=alt.Scale(
            domain=["low", "medium", "high"], range=[T.on(c) for c in T.bands.values()]), legend=None))
    # independent colour scales: the labels need their own (contrasting) colours, and sharing the
    # scale with the cells hid both the labels and the legend
    return (rect + text).resolve_scale(color="independent").properties(
        height=26 * grid["category"].nunique() + 20)


def dumbbell(long: pd.DataFrame, title: str) -> alt.Chart:
    """Per year: the stale and the refreshed index, joined by a hairline."""
    order = [stale_label(run.baseline), "Refreshed index"]
    colour = alt.Color("index:N", scale=alt.Scale(domain=order, range=[T.series[1], T.series[0]]),
                       legend=alt.Legend(title=None, orient="top"))
    y = alt.Y("period:N", title=None, sort=sorted(long["period"].unique()), axis=alt.Axis(labelOverlap=False))
    x = alt.X("value:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("period:N", title="Year"), alt.Tooltip("index:N", title="Index"),
           alt.Tooltip("value:Q", title=title, format=".0%"), alt.Tooltip("questions:Q", title="Questions"),
           alt.Tooltip("delta:Q", title="Refreshed − stale", format="+.0%"),
           alt.Tooltip("delta_low:Q", title="95% interval, low", format="+.0%"),
           alt.Tooltip("delta_high:Q", title="95% interval, high", format="+.0%")]
    rule = alt.Chart(long).mark_rule(color=T.axis, strokeWidth=2).encode(
        y=y, x=alt.X("min(value):Q"), x2="max(value):Q")
    dots = alt.Chart(long).mark_circle(size=140, opacity=1, stroke=T.surface, strokeWidth=2).encode(
        y=y, x=x, color=colour, tooltip=tip)
    return (rule + dots).properties(height={"step": 36})


CLASS_COLOURS = {"drift that hurts": T.series[0], "harm without drift": T.series[1],
                 "drift that does not hurt": T.series[2], "healthy": T.muted}


def harm_scatter(seg: pd.DataFrame) -> alt.Chart:
    """Each classified section: its drift score against its loss in correct answers."""
    present = [c for c in CLASS_COLOURS if c in set(seg["class"])]   # only classes this run has
    colour = alt.Color("class:N", scale=alt.Scale(domain=present, range=[CLASS_COLOURS[c] for c in present]),
                       legend=alt.Legend(title=None, orient="top"))
    x = alt.X("times_ordinary_embedding:Q", title="Drift score: meaning, × ordinary change")
    y = alt.Y("loss_correct:Q", title="Drop in correct answers", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("section:N", title="Section"), alt.Tooltip("class:N", title="Class"),
           alt.Tooltip("loss_correct:Q", title="Drop", format=".0%"),
           alt.Tooltip("loss_low:Q", title="95% interval, low", format=".0%"),
           alt.Tooltip("loss_high:Q", title="95% interval, high", format=".0%"),
           alt.Tooltip("times_ordinary_embedding:Q", title="Meaning × ordinary", format=".1f"),
           alt.Tooltip("severity:N", title="Drift severity"), alt.Tooltip("questions:Q", title="Questions")]
    base = alt.Chart(seg).encode(x=x)
    err = base.mark_rule(color=T.axis, strokeWidth=2).encode(y=alt.Y("loss_low:Q"), y2="loss_high:Q")
    dots = base.mark_circle(size=130, opacity=1, stroke=T.surface, strokeWidth=2).encode(
        y=y, color=colour, tooltip=tip)
    return (err + dots).properties(height=300)


def outdated_chart(summary: pd.DataFrame) -> alt.Chart:
    """Wikipedia: how often the answer changed between the old and the new version."""
    names = {"flagged": "A number or date changed", "comparison": "Reworded, numbers kept"}
    d = summary.assign(group=summary["group"].map(names))
    y = alt.Y("group:N", title=None, sort=list(names.values()), axis=alt.Axis(labelLimit=260))
    x = alt.X("answer_changed:Q", title="Questions whose answer changed", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("group:N", title="Edits"), alt.Tooltip("items:Q", title="Questions"),
           alt.Tooltip("answer_changed:Q", title="Answer changed", format=".0%"),
           alt.Tooltip("changed_low:Q", title="95% interval, low", format=".0%"),
           alt.Tooltip("changed_high:Q", title="95% interval, high", format=".0%"),
           alt.Tooltip("edit_retrieved_old:Q", title="Old index found the edited passage", format=".0%")]
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=y, x=x, tooltip=tip,
        color=alt.Color("group:N", scale=alt.Scale(domain=list(names.values()), range=[T.series[0], T.muted]),
                        legend=None))
    err = alt.Chart(d).mark_rule(color=T.ink_2, strokeWidth=1.5).encode(y=y, x="changed_low:Q",
                                                                        x2="changed_high:Q")
    label = alt.Chart(d).mark_text(align="left", dx=8, color=T.ink, fontSize=12).encode(
        y=y, x="changed_high:Q", text=alt.Text("answer_changed:Q", format=".0%"))
    return (bars + err + label).properties(height={"step": 48})


def year_bars(df: pd.DataFrame, col: str, title: str) -> alt.Chart:
    """One measure per later year, as labelled bars (a single series: no legend)."""
    x = alt.X("period:N", title=None, axis=alt.Axis(labelAngle=tilted(len(df))))
    y = alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format="%", tickCount=4))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=T.series[0], width={"band": 0.6}).encode(
        x=x, y=y, tooltip=[alt.Tooltip("period:N", title="Year"), alt.Tooltip(f"{col}:Q", title=title,
                                                                                 format=".1%")])
    text = alt.Chart(df).mark_text(dy=-8, color=T.ink, fontSize=12).encode(
        x=x, y=y, text=alt.Text(f"{col}:Q", format=".0%"))
    return (bars + text).properties(title=title, height=200)


tabs = st.tabs(["Overview", "Data monitoring", "Drift", "Retrieval", "Impact", "Recommendations"])

# ---- Overview ----------------------------------------------------------------------------------
with tabs[0]:
    intro(f"The assistant's search index was built from {run.baseline} articles; each later year arrives as a "
          "new batch. Five agents check what arrived, how much it drifted, whether search and answers got "
          "worse, and what to do. The main findings of this run:")
    findings = story.overview_findings(run)
    if findings:
        with st.container(border=True):
            for f in findings:
                st.markdown(f"- {f}")
            recs = run.recommendations.recommended() if run.recommendations is not None else []
            if recs:
                st.markdown(f"- **Top recommendation: {recs[0].action}.** {recs[0].finding}")
    h = story.answer_headline(run)
    c = story.harm_counts(run)
    if h:
        m = st.columns(4)
        m[0].metric("Questions about new events", h["questions"],
                    help=f"Test questions about articles from {h['years']}.")
        m[1].metric("Correct without re-indexing", pct(h["stale"]),
                    help=f"Answered from the stale index, built from {run.baseline} only.")
        m[2].metric("Correct after re-indexing", pct(h["refreshed"]),
                    help="Answered from the refreshed index, rebuilt with each year's articles.")
        if c:
            m[3].metric("Sections that lost answers", f"{c['harmed']} of {c['judged']}",
                        help=f"Year × section groups with {at_least('min_segment_questions', 'questions')} "
                             "whose correct answers dropped by a material amount.")
    m = st.columns(4)
    m[0].metric("Years monitored", len(run.monitoring.periods), help=", ".join(run.monitoring.periods))
    m[1].metric("Groups flagged for drift", len(run.flagged()),
                help="Year × section groups (and whole years) whose drift is significant and at least medium "
                     "severity, compared with the baseline.")
    m[2].metric("Control groups", len(run.drift.control_segments),
                help="Groups with low drift, kept for comparison: if they lose answers too, drift is not the "
                     "cause.")
    if run.recommendations is not None:
        m[3].metric("Recommended actions", len(run.recommendations.recommended()),
                    help="See the Recommendations tab, each with its evidence.")
    st.caption(story.route_sentence(run))

    heading("How much did the archive change each year?")
    left, right = st.columns(2)
    left.altair_chart(themed(drift_line(run.whole, "times_ordinary_embedding", "mmd2",
                                        "Meaning (embedding drift)")), width="stretch")
    right.altair_chart(themed(drift_line(run.whole, "times_ordinary_lexical", "jsd_excess",
                                         "Vocabulary (word drift)")), width="stretch")
    window = run.drift.reference.get("whole_period_null", {}).get("window", "")
    read(f"1.0 = as much change as is normal inside {run.baseline}"
         + (f" (the same {window} a year apart)" if window in ("month", "quarter") else "") + "; 2.5 = two "
         "and a half times that. Orange compares each year with the baseline, blue with the year before. Above the dashed "
         "medium and high lines, change counts as unusual.")
    shows(story.drift_chart_takeaway(run))

    heading("Most drifted groups")
    flagged = run.flagged().head(10)
    table(flagged, {"period": "Year", "category": "Section", "severity": "Severity", "kind": "Kind of drift",
                    "times_ordinary_embedding": "Meaning × ordinary", "times_ordinary_lexical": "Words × ordinary",
                    "emerging_terms": "Words that became common"},
          decimals=("times_ordinary_embedding", "times_ordinary_lexical"))
    read("the ten groups that changed most. Kind of drift: *meaning* (what articles are about), *wording* "
         "(the words used) or *both*. More detail on the Drift tab.")

# ---- Data monitoring (Agent A) -----------------------------------------------------------------
with tabs[1]:
    st.subheader("Agent A · what arrived each year", anchor=False)
    intro("Before any drift analysis, this agent checks each new batch of articles and questions: how many, "
          "how long, which sections, what new words, and whether there is enough to analyse.")
    points = story.monitoring_points(run)
    if points:
        with st.container(border=True):
            for p in points:
                st.markdown(f"- {p}")
    summary = pd.DataFrame(run.monitoring.period_summary)
    summary["period"] = summary["period"].astype(str)
    later = summary[summary["period"] != run.baseline]
    if len(later):
        a, b = st.columns(2)
        a.altair_chart(themed(year_bars(later, "new_section_share", "Articles in sections new since the baseline")),
                       width="stretch")
        b.altair_chart(themed(year_bars(later, "new_term_rate", "Words never used in the baseline")),
                       width="stretch")
    heading("Each year compared with the baseline")
    table(summary, {"period": "Year", "documents": "Articles", "median_words": "Words per article",
                    "new_sections": "New sections", "new_section_share": "In new sections",
                    "mix_cramers_v": "Mix change (0–1)", "new_term_rate": "New words"},
          percent=("new_section_share", "new_term_rate"), decimals={"mix_cramers_v": 2, "median_words": 0})
    read("mix change is Cramér's V: 0 = every section kept its share, 1 = a completely different mix.")
    a, b = st.columns(2)
    with a:
        heading("Data quality (cleaning)")
        dq = pd.DataFrame(run.monitoring.data_quality)
        total = dq[dq["period"] == "total"] if "period" in dq else dq
        if len(total) and "kept" in total:
            t = total.iloc[0]
            st.caption(f"{int(t['kept']):,} of {int(t['raw']):,} articles kept ({t['kept_pct']:.1f}%). Removed: "
                       "empty text, too short or too long, near-duplicates.")
        table(dq, {"period": "Year", "raw": "Raw", "empty_description": "Empty", "too_short": "Too short",
                   "too_long": "Too long", "near_duplicate": "Near-duplicate", "kept": "Kept",
                   "in_drift_sample": "Drift sample"})
    with b:
        heading("Renamed sections")
        st.caption("Sections the source renamed over time are merged, so a rename does not look like drift.")
        table(pd.DataFrame(run.monitoring.renamed_sections),
              {"category_raw": "Name in the source", "category": "Merged into", "period": "Year",
               "documents": "Articles"})
    seg = pd.DataFrame(run.monitoring.segments)
    if len(seg):
        heading("Which year × section groups can be analysed")
        st.caption(f"{int(seg['eligible'].sum())} of {len(seg)} groups have enough articles in that year and in "
                   "the baseline. The rest are reported but not scored.")
        with st.expander("Every group, with the reason it was left out"):
            table(seg, {"period": "Year", "category": "Section", "n_period": "Articles this year",
                        "n_baseline": "Articles in the baseline", "eligible": "Analysed", "reason": "Why not"})
    heading("The question stream")
    st.caption("No real query log exists, so an LLM wrote test questions from sampled articles: most from random "
               "articles, some from articles likely to state a fact that changes. A few are built to be "
               "unanswerable, to test whether the assistant admits it does not know.")
    if isinstance(run.monitoring.query_stream, str):
        st.info("Question stream: " + run.monitoring.query_stream)
    else:
        table(pd.DataFrame(run.monitoring.query_stream),
              {"period": "Year", "queries": "Questions", "unanswerable_share": "Built to be unanswerable",
               "median_words": "Median words", "new_term_rate": "New words"},
              percent=("unanswerable_share", "new_term_rate"))

# ---- Drift (Agent B) ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Agent B · what changed, where, and how much", anchor=False)
    intro("Drift is measured two ways, for every year and every section with enough articles: **meaning** "
          "(each article becomes a vector of what it is about; the year's vectors are compared with the "
          "baseline's) and **vocabulary** (how often each word is used). Both are reported as a multiple of "
          f"ordinary change inside {run.baseline}. These scores show where content changed; the Impact tab "
          "shows whether that is where answers break.")
    heading("Drift severity by section and year")
    st.altair_chart(themed(severity_heatmap(severity_grid(run))), width="stretch")
    read("each cell is one section in one year, compared with the baseline. Coloured cells carry a label when "
         "flagged: the kind of drift (meaning, wording or both). Empty cells had too few articles to analyse. "
         "Hover for the numbers.")
    heading("Drill down into one group")
    ids = run.whole["segment_id"].tolist() + run.segments["segment_id"].tolist()
    first = ids.index(run.flagged()["segment_id"].iloc[0]) if len(run.flagged()) else 0
    pick = st.selectbox("Group", ids, index=first, format_func=segment_label,
                        help="Opens on the most drifted group.")
    row = pd.concat([run.whole, run.segments]).set_index("segment_id").loc[pick]
    m = st.columns(4)
    m[0].metric("Meaning × ordinary", f"{row['times_ordinary_embedding']:.1f}", row["embedding_band"],
                delta_color="off", help="How far what the articles are about moved, as a multiple of ordinary.")
    m[1].metric("Words × ordinary", f"{row['times_ordinary_lexical']:.1f}", row["lexical_band"],
                delta_color="off", help="How far the vocabulary moved, as a multiple of ordinary.")
    m[2].metric("Kind of drift", row["kind"], help="Meaning, wording, both, or none.")
    m[3].metric("Unlike anything in the baseline", f"{row['novel_share']:.0%}",
                help="Articles far from every baseline article. Low means familiar content recombined around "
                     "new events, not brand-new content.")
    st.markdown(f"**Words that became common:** {', '.join(row['emerging_terms'])}  \n"
                f"**Words that faded:** {', '.join(row['declining_terms'])}")
    if row["examples"]:
        st.markdown("**Example headlines:**\n" + "\n".join(f"- {t}" for t in row["examples"]))
    st.caption(f"Compared with {row['compared_with']} · {row['n_period']} vs {row['n_reference']} articles · "
               f"ordinary change measured on {row['null_window']} pairs inside the baseline")
    charts = [("Topics that grew", "emerging_topics", run.topics,
               "clusters of articles about one subject whose share of a year is "
               + (f"at least {run.settings['topic_min_ratio']:g}× " if run.settings.get("topic_min_ratio") else
                  "several times ") + "their share of the baseline. Orange = articles that year, blue = in the baseline. Most topics existed before and "
               "grew; they did not appear from nothing."),
              ("Words that changed meaning", "words_that_changed_meaning", run.term_shifts,
               "the same word used in a different context. The bar is how far the articles using it moved; the "
               "text shows its neighbouring words before → after. A word count would miss this, because the "
               "word itself did not change.")]
    for title, key, data, how in charts:
        heading(title)
        if key in run.charts or f"dark/{key}" in run.charts:
            st.image(str(run.charts.get(f"dark/{key}") or run.charts[key]))
            read(how)
        else:
            st.caption(f"Chart not drawn for this run: `python scripts/make_charts.py --run {run.run_id}` draws it. "
                       "The numbers are below.")
        with st.expander("The numbers"):
            st.dataframe(data.round(3) if len(data) else data, width="stretch", hide_index=True)
    with st.expander("Method check: do the drift measures agree with each other?"):
        if "measure_correlations" in run.charts or "dark/measure_correlations" in run.charts:
            st.image(str(run.charts.get("dark/measure_correlations") or run.charts["measure_correlations"]))
        read("rank correlation between drift measures across all comparisons: 1 = always agree, 0 = unrelated. "
             "The two meaning measures (MMD² and centroid shift) nearly always agree, so one would do; the "
             "vocabulary measure agrees only partly, so it adds information. That is why two techniques are "
             "used.")
    heading("Do the questions drift too?")
    if isinstance(run.drift.query_drift, str):
        st.info("Query drift: " + run.drift.query_drift)
    else:
        qd = pd.DataFrame(run.drift.query_drift)
        table(qd.assign(period=qd["period"].str.replace(" (inside baseline)", " (reference)", regex=False)),
              {"period": "Year", "n_period": "Questions", "times_ordinary_embedding": "Meaning ×",
               "times_ordinary_lexical": "Words ×", "new_term_rate": "New words", "significant": "Significant",
               "emerging_terms": "Words that became common"},
              percent=("new_term_rate",), decimals=("times_ordinary_embedding", "times_ordinary_lexical"))
        per_year = qd["n_period"].median() if "n_period" in qd else None
        read("the same analysis on the test questions instead of the articles, each year against the "
             "baseline questions. "
             + (f"With about {per_year:.0f} questions a year, " if per_year else "")
             + "× is measured against the change inside the baseline (the reference row), a rough scale. The questions were "
             "written from each year's articles, so they partly mirror the article drift by construction; a "
             "real query log could drift on its own.")

# ---- Retrieval (Agent C) -----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Agent C · does search still find the answer?", anchor=False)
    intro("An assistant like this works in two steps: **search** finds a few articles, then the LLM **writes an "
          f"answer** from them. This tab tests search alone: for each question, is an article that answers it "
          f"in the {top('judge_k')}? {models['judge']} reads each article found and decides, without being "
          "told which index found it.")
    rep = run.retrieval
    if rep is None:
        st.info("Not run yet. This page shows stale vs refreshed retrieval once Agent C has run.")
    else:
        long = stale_vs_refreshed(run, "answer")
        if len(long):
            later = [r for r in rep.judged_by_period if str(r["period"]) != run.baseline]
            base = [r for r in rep.judged_by_period if str(r["period"]) == run.baseline]
            m = st.columns(4)
            m[0].metric("Questions tested", rep.questions)
            m[1].metric("Found without re-indexing",
                        pct(sum(r["stale_answer"] * r["questions"] for r in later) /
                            sum(r["questions"] for r in later)),
                        help=f"Share of questions about later years with an answer in the {top('judge_k')}, "
                             "stale index.")
            m[2].metric("Found after re-indexing",
                        pct(sum(r["refreshed_answer"] * r["questions"] for r in later) /
                            sum(r["questions"] for r in later)))
            if base:
                m[3].metric("Old questions still found", pct(base[0]["refreshed_answer"]),
                            help=f"{run.baseline} questions on the refreshed index: adding new articles should "
                                 "not hurt them (the historical control).")
            heading(f"Is an answer in the {top('judge_k')}?")
            st.altair_chart(themed(dumbbell(long, f"Share of questions with an answer in the {top('judge_k')}")),
                            width="stretch")
            read("each row is a year of questions. Blue = the stale index (never updated), orange = the "
                 "refreshed one; the line between them is what re-indexing gains. "
                 f"{run.baseline} questions are the control.")
            shows(story.retrieval_takeaway(run))
            if rep.worst_queries:
                heading("What that looks like")
                for w in rep.worst_queries[:2]:
                    with st.container(border=True):
                        st.markdown(f"**{w['question']}**  \n"
                                    f"{w['period']} · {w['category']}  \n"
                                    f"*Stale index found:* {' · '.join(w['stale_top3'])}  \n"
                                    f"*Refreshed index found:* {' · '.join(w['refreshed_top3'])}")
                if len(rep.worst_queries) > 2:
                    with st.expander(f"{len(rep.worst_queries) - 2} more questions that lost most"):
                        for w in rep.worst_queries[2:]:
                            st.markdown(f"**{w['question']}** ({w['period']}, {w['category']})  \n"
                                        f"*Stale:* {' · '.join(w['stale_top3'])}  \n"
                                        f"*Refreshed:* {' · '.join(w['refreshed_top3'])}")
            with st.expander("The numbers, with 95% intervals, by year and by drift status"):
                cols = {"period": "Year", "drift_status": "Drift status", "questions": "Questions",
                        "stale_answer": "Answer found, stale", "refreshed_answer": "Answer found, refreshed",
                        "stale_ndcg": "Ranking quality, stale", "refreshed_ndcg": "Ranking quality, refreshed"}
                pc = ("stale_answer", "refreshed_answer")
                table(pd.DataFrame(rep.judged_by_period), cols, percent=pc)
                table(pd.DataFrame(rep.judged_by_status), cols, percent=pc)
                st.caption("Ranking quality is nDCG: 1 = the answering articles are ranked first.")
        else:
            st.info("The judged comparison was not run (no LLM judge in this run).")
        if rep.ranking_change:
            heading("Do old questions suffer when new articles are added?")
            shows(story.ranking_takeaway(run))
            rc = pd.DataFrame(rep.ranking_change)
            table(rc.assign(index=rc["index"].str.replace("refreshed_", "", regex=False)),
                  {"index": "Refreshed up to", "questions": "Questions", "overlap": f"Same {top('top_k')} as stale",
                   "newer_share": f"Newer articles in {top('top_k')}", "source_displaced": "Source article pushed out"},
                  percent=("overlap", "newer_share", "source_displaced"))
        if rep.known_item:
            with st.expander("Sanity check: is the article each question came from found?"):
                st.caption("Each question was written from one article, so it shares that article's words and is "
                           "easy to find. A high score here is expected; it checks the search works, not that it "
                           "copes with drift.")
                ki = pd.DataFrame(rep.known_item)
                table(ki.assign(index=ki["index"].str.replace("refreshed_", "", regex=False)),
                      {"index": "Refreshed up to", "asked_by": "Questions", "questions": "Count",
                       "reciprocal_rank": "Source article's rank score (1 = first)",
                       "recall": f"Source article in {top('top_k')}"}, percent=("recall",))

# ---- Impact (Agent D) --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Agent D · do the answers suffer, and which drift matters?", anchor=False)
    imp = run.impact
    if imp is None or not imp.questions:
        st.info("Skipped: this run did not use the LLM (`--no-judge`), so no answers were written or graded."
                if run.info.get("mode") in ("no_judge", "estimate") else
                "Not run yet. This page shows answer quality once Agent D has run with an LLM.")
    else:
        intro(f"Now the whole assistant runs: {models['writer']} answers every question from each index's "
              f"{top('answer_k')}, "
              f"and {models['judge']} grades each answer on its own (correct, grounded in the articles, or "
              "declined), without being told which index it came from. This is where *which drift matters* is "
              "answered.")
        h = story.answer_headline(run)
        un = {r["index"]: r for r in imp.unanswerable}
        n_un = un.get("refreshed", {}).get("questions", 0)
        m = st.columns(4)
        m[0].metric("Questions answered", imp.questions,
                    help=f"Answered from both indexes: {imp.questions - n_un} normal questions plus {n_un} built "
                         "to be unanswerable.")
        if h:
            m[1].metric("Correct without re-indexing", pct(h["stale"]), help=f"Questions about {h['years']}.")
            m[2].metric("Correct after re-indexing", pct(h["refreshed"]))
        if "refreshed" in un:
            r = un["refreshed"]
            m[3].metric("Admits it doesn't know", pct(r["declined"]),
                        help=f"Of the {r['questions']} unanswerable questions, the share the refreshed index "
                             f"declined; {pct(r['hallucinated'])} got an invented answer.")
        heading("Does the assistant answer correctly?")
        st.altair_chart(themed(dumbbell(answers_stale_vs_refreshed(run), "Share of questions answered correctly")),
                        width="stretch")
        read(f"like the Retrieval chart, but this is the final answer, not just search. {run.baseline} questions "
             "are the control.")
        shows(story.answers_takeaway(run))

        seg = harm_vs_drift(run)
        if len(seg):
            heading("Which drift matters?")
            scored = seg.dropna(subset=["times_ordinary_embedding"])
            st.altair_chart(themed(harm_scatter(scored)), width="stretch")
            read("each dot is a year × section with enough questions. Across: its drift score; up: how much its "
                 "correct answers dropped (lines are 95% intervals). If drift predicted harm, dots would rise from "
                 "left to right. Orange = flagged by drift detection and harmed; blue = harmed but never flagged.")
            shows(story.harm_takeaway(run))
            unscored = seg[seg["times_ordinary_embedding"].isna()]
            if len(unscored):
                st.caption("Not plotted, because drift detection could not score them (no baseline to compare "
                           f"with): {', '.join(unscored['section'])}.")
            loss = (f"at least {imp.threshold:.0%} and the 95% interval excludes zero" if imp.threshold else "")
            with st.expander("Every section, with its class"):
                st.caption(f"A section counts as harmed when its correct answers drop by {loss}; this was fixed "
                           "before looking at the results. Sections without "
                           f"{at_least('min_segment_questions', 'questions')} are not classified."
                           if loss else "")
                table(pd.DataFrame(imp.segments),
                      {"period": "Year", "category": "Section", "questions": "Questions",
                       "stale_correct": "Correct, stale", "refreshed_correct": "Correct, refreshed",
                       "loss_correct": "Drop", "flagged": "Flagged for drift", "class": "Class",
                       "cause": "Likely cause"}, percent=("stale_correct", "refreshed_correct", "loss_correct"))
        if imp.correlations:
            shows(story.correlation_takeaway(run))
            with st.expander("Each drift measure against the size of the loss"):
                table(pd.DataFrame(imp.correlations), {"measure": "Drift measure", "segments": "Sections",
                                                       "spearman": "Rank correlation", "p_value": "p-value"})
        if imp.evidence:
            heading(f"Evidence loop: borderline sections, answered again from the {top('top_k')}")
            st.dataframe(pd.DataFrame(imp.evidence).round(3), hide_index=True, width="stretch")
        else:
            st.caption("Evidence loop: when a section's result is borderline, this agent sends it back to Agent C "
                       f"to search the {top('top_k')} instead of the {top('answer_k')} and answer again. No "
                       "section was borderline in "
                       "this run, so every result above is clear-cut.")
        if imp.examples:
            heading("What that looks like")
            e = imp.examples[0]
            with st.container(border=True):
                st.markdown(f"**{e['question']}**  \n{e['period']} · {e['category']}  \n"
                            f"*Stale index:* {e['stale_answer']}  \n*Refreshed index:* {e['refreshed_answer']}")
            if len(imp.examples) > 1:
                with st.expander(f"{len(imp.examples) - 1} more questions only the refreshed index answers"):
                    for e in imp.examples[1:]:
                        st.markdown(f"**{e['question']}** ({e['period']}, {e['category']})  \n"
                                    f"*Stale:* {e['stale_answer']}  \n*Refreshed:* {e['refreshed_answer']}")
        lf = run.longform
        if "outdated_summary" in lf:
            heading("Edited documents: does a changed fact change the answer?")
            intro("A separate test on long Wikipedia articles, captured at the end of each year. For edits that "
                  "replaced a number or date, and for edits that only reworded the text, a question was written "
                  "from the new version and answered from the old and the new archive.")
            st.altair_chart(themed(outdated_chart(lf["outdated_summary"])), width="stretch")
            read("the share of questions whose answer differs between the old and the new archive; lines are 95% "
                 "intervals.")
            shows(story.wikipedia_takeaway(run))
            items = lf.get("outdated_items")
            if items is not None and len(items):
                hit = items[(items["group"] == "flagged") & (items["verdict"] == "different")].head(6)
                with st.expander("Examples: the same question, old and new version"):
                    for r in hit.to_dict("records"):
                        st.markdown(f"**{r['title']}** · {r['question']}  \n*Before:* {r['before']}  \n"
                                    f"*After:* {r['after']}  \n*Old answer:* {r['answer_old']}  \n"
                                    f"*New answer:* {r['answer_new']}")
        for note in imp.notes:
            st.caption(note)

# ---- Recommendations (Agent E) -----------------------------------------------------------------
with tabs[5]:
    st.subheader("Agent E · what to do, with the evidence", anchor=False)
    rec = run.recommendations
    if rec is None:
        st.info("Not run yet. This page shows Agent E's recommendations once the full pipeline has run.")
    else:
        intro(f"{len(rec.recommendations)} possible actions, taken from the project brief. For each, a rule written in code checks the "
              f"other agents' results and decides whether it is needed. {models['writer']} then explains each "
              "recommended action from its evidence, and code rejects any explanation that uses a number not in "
              "the evidence, so the text cannot invent or change a result.")
        m = st.columns(4)
        m[0].metric("Recommended", rec.summary.get("recommended", 0))
        m[1].metric("Not needed", rec.summary.get("not needed", 0),
                    help="Checked and ruled out, with the number that ruled it out (below).")
        m[2].metric("Could not be checked", rec.summary.get("not checked", 0),
                    help="Actions whose rule lacked an input (for example a run without the LLM). 0 means every "
                         "action was tested.")
        m[3].metric("Explanations checked", f"{rec.explanations.get('requests', 0)}",
                    help=f"{rec.explanations.get('rejected_drafts', 0)} drafts rejected for unsupported numbers; "
                         f"{rec.explanations.get('fallbacks', 0)} fell back to the rule's own text.")
        st.caption("In order of the kind of evidence: measured loss of answers first, then cases for a person, "
                   "then maintenance.")
        for r in rec.recommended():
            heading(f"{r.priority}. {r.action}")
            st.write(r.explanation)
            if r.scope:
                shown = ", ".join(r.scope[:12]) + (f", and {len(r.scope) - 12} more" if len(r.scope) > 12 else "")
                st.caption(f"Applies to ({len(r.scope)}): {shown}")
            with st.expander("Evidence"):
                for e in r.evidence:
                    st.markdown(f"- **{e.source}:** {e.claim}")
                check = r.explanation_check
                note = (f"written by {models['writer']}; every number checked against the evidence"
                        + (" (passed first time)" if "attempt 1" in check else
                           " (passed on a second attempt)" if "attempt 2" in check else f" ({check})")
                        if check.startswith("LLM") else check)
                st.caption(f"Decided by the rule: {r.rule}. Explanation {note}.")
        rest = [r for r in rec.recommendations if r.status != "recommended"]
        if rest:
            heading("Checked, not recommended")
            for r in rest:
                reason = r.finding.split(": ", 1)[1] if r.finding.lower().startswith(("not needed:", "not checked:")) \
                    else r.finding
                deciding = r.evidence[0].claim if r.evidence else ""
                st.markdown(f"- **{r.action}** ({r.status}). {reason[:1].upper() + reason[1:]} {deciding}")
        for note in rec.notes:
            st.caption(note)
