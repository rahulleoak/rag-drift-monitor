"""Streamlit dashboard. Reads saved outputs from ``runs/<run_id>/``; it does not
run the pipeline itself, so the demo cannot fail on a live LLM call (Q16).

Pages (left bar): Overview, Data Monitoring (Agent A), Drift (Agent B),
Retrieval (Agent C), Impact (Agent D) and Recommendations (Agent E).

Text follows one rule: fixed text only says what a chart is and how to read it
(in the ⓘ tooltip on each heading); what a run *shows* is built from its numbers
by ``driftmon.dashboard.story`` and stated under the heading, so it cannot
contradict the charts. Charts come from ``driftmon.dashboard.live``.

Run (from the repository root):
    streamlit run app/streamlit_app.py
It reads ``runs/`` if it holds any pipeline runs, and otherwise the run
committed under ``demo/runs/`` (no API key or rerun needed). Set
DRIFTMON_RUNS_DIR to read runs from another folder (used by the tests).
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.dashboard import live, story  # noqa: E402
from driftmon.dashboard.charts import DARK as T  # noqa: E402  (Rivvun dark palette, validated)
from driftmon.dashboard.runs import (ALL_OTHER, answers_stale_vs_refreshed, default_run,  # noqa: E402
                                     flagged_share_by_year, harm_vs_drift, list_runs, load_run, run_label,
                                     run_models, section_shares, severity_grid, stale_label, stale_vs_refreshed,
                                     term_shift, with_settings)
from driftmon.policy import pct  # noqa: E402

st.set_page_config(page_title="RAG Drift Monitor", layout="wide", initial_sidebar_state="expanded")
config = load_config()
runs_dir = Path(os.environ.get("DRIFTMON_RUNS_DIR") or resolve_path(config["paths"]["runs"]))
runs = list_runs(runs_dir)
demo = False
if not runs and not os.environ.get("DRIFTMON_RUNS_DIR"):   # fresh clone: the committed demo run
    runs_dir = resolve_path(config["paths"]["demo_runs"])
    runs, demo = list_runs(runs_dir), True

# ---- styling: softer body text, bright bold numbers, readable captions, chips -------------------
st.markdown(f"""<style>
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  color: #C7CBD2; line-height: 1.6; }}
.num {{ color: {T.ink}; font-weight: 650; }}
.stmt p, p.stmt {{ color: #E4E2DC; font-size: 1.02rem; margin-top: -0.35rem; }}
.dot-stale {{ color: {T.series[1]}; }} .dot-new {{ color: {T.series[0]}; }}
[data-testid="stCaptionContainer"] p {{ color: #AEB3BB !important; font-size: 0.9rem; line-height: 1.55; }}
.chip {{ display: inline-block; padding: 2px 10px; margin: 2px 6px 4px 0; border-radius: 999px;
  background: #1A2232; color: #D3D6DB; font-size: 0.8rem; white-space: nowrap; }}
.chip b {{ color: {T.ink}; font-weight: 600; }}
.topbar {{ margin: -0.4rem 0 0.6rem 0; }}
.agent {{ color: #AEB3BB; font-size: 0.95rem; margin-top: -0.6rem; }}
.subtitle {{ color: #AEB3BB; font-size: 0.98rem; margin-top: -0.9rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: 1.02rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 0.3rem 0; }}
</style>""", unsafe_allow_html=True)

PAGES = ["Overview", "Data Monitoring", "Drift", "Retrieval", "Impact", "Recommendations"]
page = st.sidebar.radio("Pages", PAGES, label_visibility="collapsed")

head, picker = st.columns([3.2, 1], vertical_alignment="bottom")
head.title("RAG Drift Monitor", anchor=False)
head.markdown('<p class="subtitle">Tracks how a document archive drifts over time, and whether that drift breaks '
              'the answers of a RAG (retrieval-augmented generation) assistant.</p>', unsafe_allow_html=True)
if not runs:
    st.warning(f"No pipeline runs in {runs_dir}. Run `python scripts/run_pipeline.py` first.")
    st.stop()
run_id = picker.selectbox("Pipeline run", runs, index=runs.index(default_run(runs_dir, runs)),
                          format_func=lambda r: run_label(runs_dir, r),
                          help="Saved runs, newest first, with the agents each one holds. Older runs can be kept "
                               "in runs/archive/, which is not listed.")
run = with_settings(load_run(runs_dir, run_id), config)   # settings: from the run, else the config
models = run_models(run, config)

# ---- top bar: what this run is, at a glance ----------------------------------------------------
periods = list(run.monitoring.periods)
facts = [("Archive", run.settings.get("description", "")), ("Baseline", run.baseline),
         ("Later years", f"{periods[0]}–{periods[-1]}" if len(periods) > 1 else periods[0])]
if run.longform:
    facts.append(("Long documents", "Wikipedia snapshots"))
facts += [("Writes", models["writer"]), ("Judges", models["judge"]),
          ("Embeddings", models["embedding"].split("/")[-1])]
if run.info.get("created"):
    facts.append(("Run", run.info["created"][:16].replace("T", " ")))
if run.info.get("llm_cost_usd") is not None:
    facts.append(("LLM cost", f"${run.info['llm_cost_usd']:.2f}"))
st.markdown('<div class="topbar">' + "".join(f'<span class="chip">{html.escape(k)}: <b>{html.escape(str(v))}</b>'
                                             '</span>' for k, v in facts) + "</div>", unsafe_allow_html=True)
if models["source"] != "run":
    st.caption("Model names from the current config: this run predates run_info.json.")
if story.fault_banner(run):
    st.warning(story.fault_banner(run), icon="⚠️")
if demo:
    st.sidebar.info("Showing the bundled demo run (`demo/runs/`). Run `python scripts/run_pipeline.py` to see "
                    "your own.")


# ---- text helpers -------------------------------------------------------------------------------
NUMBER = re.compile(r"(?<![\w#&\-–])(?:[−+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-−]?\d+)?"
                    r"(?:\s?[-–]\s?\d+(?:,\d{3})*(?:\.\d+)?)?(?:%|×|x\b)?)")
YEARS = re.compile(r"^(?:19|20)\d\d(?:[-–](?:19|20)?\d\d)?$")


def emphasise(text: str) -> str:
    """Bold-white numbers (not bare years), and a coloured dot before 'stale index' / 'refreshed index'."""
    def num(m: re.Match) -> str:
        s = m.group(0)
        return s if YEARS.match(s) or not any(ch.isdigit() for ch in s) else f'<span class="num">{s}</span>'
    out = NUMBER.sub(num, text)
    out = re.sub(r"\b(stale index)", r'<span class="dot-stale">●</span> \1', out, count=1, flags=re.I)
    out = re.sub(r"\b(refreshed (?:index|one))", r'<span class="dot-new">●</span> \1', out, count=1, flags=re.I)
    return out


def say(text: str) -> None:
    """Body text with numbers emphasised."""
    if text:
        st.markdown(emphasise(text), unsafe_allow_html=True)


def statement(text: str) -> None:
    """What this run shows, stated under a heading (built from the run by driftmon.dashboard.story)."""
    if text:
        st.markdown(f'<div class="stmt">\n\n{emphasise(text)}\n\n</div>', unsafe_allow_html=True)


def section(title: str, how: str = "", shows: str = "") -> None:
    """A section heading, its 'how to read' in the ⓘ tooltip, and this run's finding underneath."""
    st.subheader(title, anchor=False, help=how or None)
    statement(shows)


def page_header(name: str, agent: str) -> None:
    st.header(name, anchor=False)
    st.markdown(f'<p class="agent">{agent}</p>', unsafe_allow_html=True)


def chips(items: list[str], limit: int = 14) -> str:
    shown = "".join(f'<span class="chip">{html.escape(str(i))}</span>' for i in items[:limit])
    more = f'<span class="chip">+{len(items) - limit} more</span>' if len(items) > limit else ""
    return shown + more


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def table(df: pd.DataFrame, columns: dict[str, str], percent: tuple[str, ...] = (),
          decimals: dict[str, int] | tuple[str, ...] = (), help: dict[str, str] | None = None) -> None:
    """Show only ``columns`` (old name -> readable name), in that order. Shares in ``percent`` show as
    percentages with one decimal; columns in ``decimals`` with that many decimals (a tuple means one);
    whole numbers get thousands separators; ``help`` adds a hover note to a column header."""
    keep = [c for c in columns if c in df.columns]
    view = df[keep].copy()
    for c in percent:
        if c in keep:
            view[c] = pd.to_numeric(view[c], errors="coerce") * 100
    view = view.rename(columns=columns)
    digits = decimals if isinstance(decimals, dict) else {c: 1 for c in decimals}
    help = help or {}
    cfg = {columns[c]: st.column_config.NumberColumn(format="%.1f%%", help=help.get(c)) for c in percent if c in keep}
    cfg.update({columns[c]: st.column_config.NumberColumn(format=f"%.{d}f", help=help.get(c))
                for c, d in digits.items() if c in keep})
    for c in keep:
        if columns[c] not in cfg:
            cfg[columns[c]] = (st.column_config.NumberColumn(format="localized", help=help.get(c))
                               if pd.api.types.is_integer_dtype(df[c]) else st.column_config.Column(help=help.get(c)))
    st.dataframe(view, hide_index=True, width="stretch", column_config=cfg)


def top(key: str) -> str:
    """'top 5' for the run's setting ``key`` (retrieval.judge_k, retrieval.top_k ...), else 'top results'."""
    k = run.settings.get(key)
    return f"top {k}" if k else "top results"


def at_least(key: str, unit: str) -> str:
    v = run.settings.get(key)
    return f"at least {v:g} {unit}" if v else f"enough {unit}"


def segment_label(segment_id: str) -> str:
    """'2020|FOOD & DRINK|vs 2016-17' -> '2020 · FOOD & DRINK (vs 2016-17)'."""
    period, category, vs = (segment_id.split("|") + ["", ""])[:3]
    return f"{period} · {category or 'ALL SECTIONS'} ({vs})" if vs else segment_id


def chart(c, stretch: bool = True) -> None:
    st.altair_chart(live.themed(c), width="stretch" if stretch else "content")


ORDINARY = (f"1.0 = as much change as is normal inside {run.baseline}"
            + (f" (the same {w} a year apart)" if (w := run.drift.reference.get("whole_period_null", {})
                                                   .get("window", "")) in ("month", "quarter") else "")
            + "; 2.0 = twice that.")


# ---- pages --------------------------------------------------------------------------------------
def overview() -> None:
    page_header("Overview", f"The search index was built from {run.baseline} articles; each later year arrives as a "
                            "new batch. Five agents check what arrived, how much it drifted, whether search and "
                            "answers got worse, and what to do.")
    findings = story.overview_findings(run)
    recs = run.recommendations.recommended() if run.recommendations is not None else []
    if findings:
        with st.container(border=True):
            st.markdown("**Main findings**")
            say(bullets(findings + ([f"**Top recommendation: {recs[0].action}.** {recs[0].finding}"] if recs else [])))
    h, c = story.answer_headline(run), story.harm_counts(run)
    if h:
        m = st.columns(4)
        m[0].metric("Questions about new events", h["questions"], help=f"Test questions about articles from {h['years']}.")
        m[1].metric("Correct without re-indexing", pct(h["stale"]),
                    help=f"Answered from the stale index, built from {run.baseline} only.")
        m[2].metric("Correct after re-indexing", pct(h["refreshed"]),
                    help="Answered from the refreshed index, rebuilt with each year's articles.")
        if c:
            m[3].metric("Sections that lost answers", f"{c['harmed']} of {c['judged']}",
                        help=f"Year × section groups with {at_least('min_segment_questions', 'questions')} "
                             "whose correct answers dropped by a material amount.")
    m = st.columns(4)
    m[0].metric("Years monitored", len(periods), help=", ".join(periods))
    m[1].metric("Groups flagged for drift", len(run.flagged()),
                help="Year × section groups (and whole years) whose drift is significant and at least medium "
                     "severity, compared with the baseline.")
    m[2].metric("Control groups", len(run.drift.control_segments),
                help="Groups with low drift, kept for comparison: if they lose answers too, drift is not the cause.")
    if run.recommendations is not None:
        m[3].metric("Recommended actions", len(recs), help="See Recommendations, each with its evidence.")
    st.caption(story.route_sentence(run))

    section("Drift per year",
            f"{ORDINARY} Orange compares each year with the baseline, blue with the year before. Above the dashed "
            "medium and high lines, change counts as unusual. Meaning = what the articles are about (embeddings); "
            "vocabulary = the words used.", story.drift_chart_takeaway(run))
    left, right = st.columns(2)
    ref = run.drift.reference["whole_period_null"]
    with left:
        chart(live.drift_line(run.whole, "times_ordinary_embedding", "mmd2", "Meaning", ref, run.baseline))
    with right:
        chart(live.drift_line(run.whole, "times_ordinary_lexical", "jsd_excess", "Vocabulary", ref, run.baseline))

    section("Most drifted groups", "The ten year × section groups that changed most against the baseline. Kind of "
            "drift: meaning, wording or both. More on the Drift page.")
    table(run.flagged().head(10),
          {"period": "Year", "category": "Section", "severity": "Severity", "kind": "Kind of drift",
           "times_ordinary_embedding": "Meaning ×", "times_ordinary_lexical": "Words ×",
           "emerging_terms": "Words that became common"},
          decimals=("times_ordinary_embedding", "times_ordinary_lexical"),
          help={"times_ordinary_embedding": "How far what the articles are about moved, × ordinary change.",
                "times_ordinary_lexical": "How far the vocabulary moved, × ordinary change.",
                "severity": "Low, medium or high, against the archive's own ordinary change."})


def data_monitoring() -> None:
    page_header("Data Monitoring", "Agent A checks each new batch of articles and questions before any drift "
                                   "analysis: how many, how long, which sections, what new words.")
    points = story.monitoring_points(run)
    if points:
        with st.container(border=True):
            st.markdown("**What arrived**")
            say(bullets(points))
    summary = pd.DataFrame(run.monitoring.period_summary)
    summary["period"] = summary["period"].astype(str)
    later = summary[summary["period"] != run.baseline]
    if len(later):
        section("New sections and new words",
                "Left: the share of each year's articles in sections that did not exist in the baseline (drift "
                "cannot be scored for them). Right: the share of words never used in the baseline.")
        a, b = st.columns(2)
        with a:
            chart(live.year_bars(later, "new_section_share", "Articles in new sections"))
        with b:
            chart(live.year_bars(later, "new_term_rate", "Words never used in the baseline"))
    shares = section_shares(run)
    if len(shares):
        section("Section shares over time",
                "Each line is one section's share of the year's articles. Dashed lines are sections new since the "
                "baseline. Sections that never reach 5% of a year are summed in the grey line. Hover a point for "
                "the numbers.", story.section_trend_takeaway(run))
        chart(live.section_trends(shares, ALL_OTHER))
    section("Each year against the baseline", "Mix change is Cramér's V: 0 = every section kept its share, 1 = a "
            "completely different mix.")
    table(summary, {"period": "Year", "documents": "Articles", "median_words": "Words per article",
                    "new_sections": "New sections", "new_section_share": "In new sections",
                    "mix_cramers_v": "Mix change (0–1)", "new_term_rate": "New words"},
          percent=("new_section_share", "new_term_rate"), decimals={"mix_cramers_v": 2, "median_words": 0},
          help={"mix_cramers_v": "Cramér's V: 0 = same section mix as the baseline, 1 = completely different.",
                "new_term_rate": "Share of words never used in the baseline."})
    a, b = st.columns(2)
    with a:
        dq = pd.DataFrame(run.monitoring.data_quality)
        total = dq[dq["period"] == "total"] if "period" in dq else dq
        kept = (f"{int(total.iloc[0]['kept']):,} of {int(total.iloc[0]['raw']):,} articles kept "
                f"({total.iloc[0]['kept_pct']:.1f}%).") if len(total) and "kept" in total else ""
        section("Data quality", "Articles removed in cleaning: empty text, too short or too long, near-duplicates.",
                kept)
        table(dq, {"period": "Year", "raw": "Raw", "empty_description": "Empty", "too_short": "Too short",
                   "too_long": "Too long", "near_duplicate": "Near-duplicate", "kept": "Kept",
                   "in_drift_sample": "Drift sample"},
              help={"in_drift_sample": "Equal-sized random sample per year used by the drift statistics."})
    with b:
        section("Renamed sections", "Sections the source renamed over time are merged, so a rename does not look "
                "like drift.")
        table(pd.DataFrame(run.monitoring.renamed_sections),
              {"category_raw": "Name in the source", "category": "Merged into", "period": "Year",
               "documents": "Articles"})
    seg = pd.DataFrame(run.monitoring.segments)
    if len(seg):
        section("Groups that can be analysed", "A year × section group is analysed when it has enough articles that "
                "year and in the baseline.",
                f"{int(seg['eligible'].sum())} of {len(seg)} year × section groups have enough articles; the rest "
                "are reported but not scored.")
        with st.expander("Every group, with the reason it was left out"):
            table(seg, {"period": "Year", "category": "Section", "n_period": "Articles this year",
                        "n_baseline": "Articles in the baseline", "eligible": "Analysed", "reason": "Why not"})
    section("Question stream", "No real query log exists, so an LLM wrote test questions from sampled articles: "
            "most from random articles, some from articles likely to state a fact that changes. A few are built to "
            "be unanswerable, to test whether the assistant admits it does not know.")
    if isinstance(run.monitoring.query_stream, str):
        st.info("Question stream: " + run.monitoring.query_stream)
    else:
        table(pd.DataFrame(run.monitoring.query_stream),
              {"period": "Year", "queries": "Questions", "unanswerable_share": "Built to be unanswerable",
               "median_words": "Median words", "new_term_rate": "New words"},
              percent=("unanswerable_share", "new_term_rate"))


def drift() -> None:
    page_header("Drift", "Agent B measures drift two ways, for every year and every section with enough articles: "
                         "meaning (what articles are about) and vocabulary (the words used), each against the "
                         "archive's own ordinary change. These scores show where content changed; Impact shows "
                         "whether that is where answers break.")
    section("Drift by section and year",
            "Each cell is one section in one year, compared with the baseline, coloured by severity. Flagged cells "
            "carry the kind of drift (meaning, wording or both). Empty cells had too few articles. Hover for the "
            "numbers.", story.severity_takeaway(run))
    chart(live.severity_heatmap(severity_grid(run)))
    fs = flagged_share_by_year(run)
    if len(fs):
        section("Share of sections flagged",
                "Of the sections with enough articles to analyse that year, the share flagged: significant drift "
                "of at least medium severity against the baseline.", story.flag_share_takeaway(run))
        chart(live.flagged_bars(fs))

    section("One group in detail", f"{ORDINARY} Pick any year × section group; it opens on the most drifted one.")
    ids = run.whole["segment_id"].tolist() + run.segments["segment_id"].tolist()
    first = ids.index(run.flagged()["segment_id"].iloc[0]) if len(run.flagged()) else 0
    pick = st.selectbox("Group", ids, index=first, format_func=segment_label, help="Opens on the most drifted group.")
    row = pd.concat([run.whole, run.segments]).set_index("segment_id").loc[pick]
    m = st.columns(4)
    m[0].metric("Meaning × ordinary", f"{row['times_ordinary_embedding']:.1f}", row["embedding_band"],
                delta_color="off", help="How far what the articles are about moved, as a multiple of ordinary.")
    m[1].metric("Words × ordinary", f"{row['times_ordinary_lexical']:.1f}", row["lexical_band"],
                delta_color="off", help="How far the vocabulary moved, as a multiple of ordinary.")
    m[2].metric("Kind of drift", row["kind"], help="Meaning, wording, both, or none.")
    m[3].metric("Unlike anything in the baseline", f"{row['novel_share']:.0%}",
                help="Articles far from every baseline article. Low means familiar content recombined around new "
                     "events, not brand-new content.")
    words = term_shift(row)
    if len(words):
        st.markdown("**Words before and after**", help=f"How often each word is used, per 1,000 words: blue in "
                    f"{row['compared_with']} (left), orange in {row['period']} (right). ↑ became common, ↓ faded; "
                    "the five strongest each way, and each moved at least 20%.")
        chart(live.butterfly(words, str(row["compared_with"]), str(row["period"])), stretch=False)
    if row["examples"]:
        st.markdown("**Example headlines**")
        say(bullets(row["examples"]))
    st.caption(f"Compared with {row['compared_with']}: {row['n_period']} articles against {row['n_reference']}. "
               f"Ordinary change measured on {row['null_window']} pairs inside the baseline.")

    if len(run.topics):
        section("Topics that grew",
                "Clusters of articles about one subject whose share of a year is "
                + (f"at least {run.settings['topic_min_ratio']:g}× " if run.settings.get("topic_min_ratio") else
                   "several times ") + "their share of the baseline. Orange = articles that year, blue = in the "
                "baseline sample. Hover for the full topic words and the sections involved.",
                story.topics_takeaway(run))
        chart(live.topics_bars(run.topics))
    if len(run.term_shifts) and run.term_shifts["times_ordinary"].notna().any():
        section("Words that changed meaning",
                "The same word used in a different context. The bar is how far the articles using it moved (× "
                "ordinary change; the dashed line is ordinary); the text shows its neighbouring words before → "
                "after. A word count would miss this, because the word itself did not change.",
                story.meaning_takeaway(run))
        chart(live.meaning_bars(run.term_shifts), stretch=False)
    corr = run.correlations
    if len(corr) and corr.notna().to_numpy().any():
        with st.expander("Method check: do the drift measures agree with each other?"):
            st.caption("Rank correlation between drift measures across all comparisons: 1 = always agree, 0 = "
                       "unrelated. The two meaning measures (MMD² and centroid shift) nearly always agree, so one "
                       "would do; the vocabulary measure agrees only partly, so it adds information. That is why "
                       "two techniques are used.")
            chart(live.correlation_grid(corr), stretch=False)
    section("Question drift",
            "The same analysis on the test questions instead of the articles, each year against the baseline "
            "questions, scaled by the change inside the baseline (the reference row). The questions were written "
            "from each year's articles, so they partly mirror the article drift; a real query log could drift on "
            "its own.", story.query_drift_takeaway(run))
    if isinstance(run.drift.query_drift, str):
        st.info("Query drift: " + run.drift.query_drift)
    else:
        qd = pd.DataFrame(run.drift.query_drift)
        table(qd.assign(period=qd["period"].str.replace(" (inside baseline)", " (ref.)", regex=False)),
              {"period": "Year", "n_period": "Questions", "times_ordinary_embedding": "Meaning ×",
               "times_ordinary_lexical": "Words ×", "new_term_rate": "New words", "significant": "Significant",
               "emerging_terms": "Words that became common"},
              percent=("new_term_rate",), decimals=("times_ordinary_embedding", "times_ordinary_lexical"))


def example_pair(question: str, where: str, left_title: str, left: list[str] | str, right_title: str,
                 right: list[str] | str) -> None:
    """One question with what the stale and the refreshed index gave, side by side."""
    with st.container(border=True):
        st.markdown(f"**{question}**")
        st.caption(where)
        a, b = st.columns(2)
        for col, title, body, cls in ((a, left_title, left, "dot-stale"), (b, right_title, right, "dot-new")):
            with col:
                st.markdown(f'<span class="{cls}">●</span> **{title}**', unsafe_allow_html=True)
                st.markdown(bullets(body) if isinstance(body, list) else body)


def retrieval() -> None:
    page_header("Retrieval", "Agent C tests search alone. An assistant like this works in two steps: search finds a "
                             f"few articles, then the LLM writes an answer from them. Here, for each question, is an "
                             f"article that answers it in the {top('judge_k')}? {models['judge']} reads each article "
                             "found and decides, without being told which index found it.")
    rep = run.retrieval
    if rep is None:
        st.info("Not run yet. This page shows stale vs refreshed retrieval once Agent C has run.")
        return
    long = stale_vs_refreshed(run, "answer")
    if len(long):
        later = [r for r in rep.judged_by_period if str(r["period"]) != run.baseline]
        base = [r for r in rep.judged_by_period if str(r["period"]) == run.baseline]
        n = sum(r["questions"] for r in later)
        m = st.columns(4)
        m[0].metric("Questions tested", rep.questions)
        m[1].metric("Found without re-indexing", pct(sum(r["stale_answer"] * r["questions"] for r in later) / n),
                    help=f"Questions about later years with an answer in the {top('judge_k')}, stale index.")
        m[2].metric("Found after re-indexing", pct(sum(r["refreshed_answer"] * r["questions"] for r in later) / n))
        if base:
            m[3].metric("Old questions still found", pct(base[0]["refreshed_answer"]),
                        help=f"{run.baseline} questions on the refreshed index: adding new articles should not hurt "
                             "them (the historical control).")
        section(f"Answer found in the {top('judge_k')}",
                f"Each pair of bars is a year of questions: blue = the stale index (never updated), orange = the "
                f"refreshed one. {run.baseline} questions are the control: both should score the same.",
                story.retrieval_takeaway(run))
        chart(live.grouped_bars(long, f"Questions with an answer in the {top('judge_k')}", stale_label(run.baseline),
                                run.baseline))
        if rep.worst_queries:
            section("Examples", "Questions where the stale index lost most: what each index found (top 3 titles).")
            for w in rep.worst_queries[:2]:
                example_pair(w["question"], f"{w['period']} · {w['category']}", "Stale index found", w["stale_top3"],
                             "Refreshed index found", w["refreshed_top3"])
            if len(rep.worst_queries) > 2:
                with st.expander(f"{len(rep.worst_queries) - 2} more questions"):
                    for w in rep.worst_queries[2:]:
                        example_pair(w["question"], f"{w['period']} · {w['category']}", "Stale index found",
                                     w["stale_top3"], "Refreshed index found", w["refreshed_top3"])
        with st.expander("The numbers, with 95% intervals, by year and by drift status"):
            jk = run.settings.get("judge_k") or 5
            cols = {"period": "Year", "drift_status": "Drift status", "questions": "Questions",
                    "stale_answer": "Answer found, stale", "refreshed_answer": "Answer found, refreshed",
                    "stale_precision": f"Precision@{jk}, stale", "refreshed_precision": f"Precision@{jk}, refreshed",
                    "stale_ndcg": f"nDCG@{jk}, stale", "refreshed_ndcg": f"nDCG@{jk}, refreshed"}
            prec_tip = (f"Share of the top {jk} judged relevant. Most questions have only one or two articles "
                        f"that answer them, so around 40% is typical even for a good search.")
            tips = {"stale_precision": prec_tip, "refreshed_precision": prec_tip,
                    "stale_ndcg": "Ranking quality: 1 = the answering articles are ranked first.",
                    "refreshed_ndcg": "Ranking quality: 1 = the answering articles are ranked first."}
            pc = ("stale_answer", "refreshed_answer", "stale_precision", "refreshed_precision")
            table(pd.DataFrame(rep.judged_by_period), cols, percent=pc, help=tips)
            table(pd.DataFrame(rep.judged_by_status), cols, percent=pc, help=tips)
    else:
        st.info("The judged comparison was not run (no LLM judge in this run).")
    if rep.ranking_change:
        section("Old questions after re-indexing",
                f"The {run.baseline} questions asked on each refreshed index: how much of their {top('top_k')} "
                "stays the same as on the stale index, and how much is newer articles.",
                story.ranking_takeaway(run))
        rc = pd.DataFrame(rep.ranking_change)
        table(rc.assign(index=rc["index"].str.replace("refreshed_", "", regex=False)),
              {"index": "Refreshed up to", "questions": "Questions", "overlap": f"Same {top('top_k')} as stale",
               "newer_share": f"Newer articles in {top('top_k')}", "source_displaced": "Source article pushed out"},
              percent=("overlap", "newer_share", "source_displaced"))
    if rep.known_item:
        with st.expander("Sanity check: is the article each question came from found?"):
            st.caption("Each question was written from one article, so it shares that article's words and is easy "
                       "to find. A high score is expected; it checks the search works, not that it copes with drift.")
            ki = pd.DataFrame(rep.known_item)
            table(ki.assign(index=ki["index"].str.replace("refreshed_", "", regex=False)),
                  {"index": "Refreshed up to", "asked_by": "Questions", "questions": "Count",
                   "reciprocal_rank": "Source article's rank (MRR)",
                   "recall": f"Source article in {top('top_k')} (Recall@{run.settings.get('top_k') or 10})"},
                  percent=("recall",), help={"reciprocal_rank": "Mean reciprocal rank: 1 = always first, 0.5 = "
                                                                "second on average."})


def impact() -> None:
    page_header("Impact", "Agent D runs the whole assistant: "
                f"{models['writer']} answers every question from each index's {top('answer_k')}, and "
                f"{models['judge']} grades each answer on its own (correct, grounded in the articles, or declined), "
                "without being told which index it came from. This is where the drift that matters shows.")
    imp = run.impact
    if imp is None or not imp.questions:
        st.info("Skipped: this run did not use the LLM (`--no-judge`), so no answers were written or graded."
                if run.info.get("mode") in ("no_judge", "estimate") else
                "Not run yet. This page shows answer quality once Agent D has run with an LLM.")
        return
    h = story.answer_headline(run)
    un = {r["index"]: r for r in imp.unanswerable}
    n_un = un.get("refreshed", {}).get("questions", 0)
    m = st.columns(4)
    m[0].metric("Questions answered", imp.questions,
                help=f"Answered from both indexes: {imp.questions - n_un} normal questions plus {n_un} built to be "
                     "unanswerable.")
    if h:
        m[1].metric("Correct without re-indexing", pct(h["stale"]), help=f"Questions about {h['years']}.")
        m[2].metric("Correct after re-indexing", pct(h["refreshed"]))
    if "refreshed" in un:
        r = un["refreshed"]
        m[3].metric("Admits it doesn't know", pct(r["declined"]),
                    help=f"Of the {r['questions']} unanswerable questions, the share the refreshed index declined; "
                         f"{pct(r['hallucinated'])} got an invented answer.")
    section("Correct answers", f"Like Retrieval, but for the final answer, not just search: blue = stale index, "
            f"orange = refreshed. {run.baseline} questions are the control.", story.answers_takeaway(run))
    chart(live.grouped_bars(answers_stale_vs_refreshed(run), "Questions answered correctly", stale_label(run.baseline),
                            run.baseline))

    seg = harm_vs_drift(run)
    if len(seg):
        section("Drift score against lost answers",
                "Each dot is a year × section with enough questions. Across: its drift score; up: how much its "
                "correct answers dropped (lines are 95% intervals; the axis starts near the lowest value). If drift "
                "predicted harm, dots would rise from left to right. Orange = flagged by drift detection; blue = "
                "never flagged.", story.harm_takeaway(run))
        chart(live.harm_scatter(seg.dropna(subset=["times_ordinary_embedding"])))
        unscored = seg[seg["times_ordinary_embedding"].isna()]
        if len(unscored):
            st.markdown("**Not plotted** (no baseline to score them against): "
                        + chips(list(unscored["section"])), unsafe_allow_html=True)
        loss = f"at least {imp.threshold:.0%} and the 95% interval excludes zero" if imp.threshold else ""
        with st.expander("Every section, with its class"):
            if loss:
                st.caption(f"A section counts as harmed when its correct answers drop by {loss}; this was fixed "
                           f"before looking at the results. Sections without "
                           f"{at_least('min_segment_questions', 'questions')} are not classified.")
            table(pd.DataFrame(imp.segments),
                  {"period": "Year", "category": "Section", "questions": "Questions",
                   "stale_correct": "Correct, stale", "refreshed_correct": "Correct, refreshed",
                   "loss_correct": "Drop", "flagged": "Flagged for drift", "class": "Class", "cause": "Likely cause"},
                  percent=("stale_correct", "refreshed_correct", "loss_correct"))
    if imp.correlations:
        statement(story.correlation_takeaway(run))
        with st.expander("Each drift measure against the size of the loss"):
            table(pd.DataFrame(imp.correlations), {"measure": "Drift measure", "segments": "Sections",
                                                   "spearman": "Rank correlation", "p_value": "p-value"},
                  help={"spearman": "1 = more drift always means more loss; 0 = unrelated."})
    if imp.evidence:
        section(f"Evidence loop: borderline sections, answered again from the {top('top_k')}",
                f"A section is borderline when its drop passes the threshold but its interval still reaches zero. "
                f"The graph sends it back to Agent C, which grades ranks 6-{run.settings.get('top_k', 10)}; this "
                f"agent then answers from the {top('top_k')} and concludes in code: harm confirmed, answers just "
                "outside the top results (a retrieval setting), or unclear (a person should look). Agent E acts on "
                "the conclusion.", story.evidence_loop_takeaway(run))
        loop = pd.DataFrame(imp.evidence)
        loop["segment_id"] = loop["segment_id"].map(story.segment_name)
        table(loop,
              {"segment_id": "Section", "questions": "Questions", "loss_top5": f"Drop, {top('answer_k')}",
               "loss_top10": f"Drop, {top('top_k')}", "loss_top10_low": "Interval low",
               "loss_top10_high": "Interval high", "conclusion": "Conclusion"},
              percent=("loss_top5", "loss_top10", "loss_top10_low", "loss_top10_high"))
    else:
        st.caption(f"Evidence loop: when a section's result is borderline, this agent sends it back to Agent C to "
                   f"search the {top('top_k')} instead of the {top('answer_k')} and answer again. No section was "
                   "borderline in this run, so every result above is clear-cut.")
    if imp.examples:
        section("Examples", "Questions only the refreshed index answers: the two answers side by side.")
        e = imp.examples[0]
        example_pair(e["question"], f"{e['period']} · {e['category']}", "Stale index", e["stale_answer"],
                     "Refreshed index", e["refreshed_answer"])
        if len(imp.examples) > 1:
            with st.expander(f"{len(imp.examples) - 1} more questions only the refreshed index answers"):
                for e in imp.examples[1:]:
                    example_pair(e["question"], f"{e['period']} · {e['category']}", "Stale index", e["stale_answer"],
                                 "Refreshed index", e["refreshed_answer"])
    lf = run.longform
    if "outdated_summary" in lf:
        section("Edited documents",
                "A separate test on long Wikipedia articles captured at the end of each year. For edits that replaced "
                "a number or date, and for edits that only reworded the text, a question was written from the new "
                "version and answered from the old and the new archive. Bars: the share whose answer differs; lines "
                "are 95% intervals.", story.wikipedia_takeaway(run))
        chart(live.outdated_chart(lf["outdated_summary"]))
        items = lf.get("outdated_items")
        if items is not None and len(items):
            hit = items[(items["group"] == "flagged") & (items["verdict"] == "different")].head(6)
            with st.expander("Examples: the same question, old and new version"):
                for r in hit.to_dict("records"):
                    with st.container(border=True):
                        st.markdown(f"**{r['title']}**: {r['question']}")
                        a, b = st.columns(2)
                        a.markdown(f"**Before:** {r['before']}\n\n**Old answer:** {r['answer_old']}")
                        b.markdown(f"**After:** {r['after']}\n\n**New answer:** {r['answer_new']}")
    for note in imp.notes:
        st.caption(note)


def recommendations() -> None:
    rec = run.recommendations
    n = len(rec.recommendations) if rec is not None else 0
    page_header("Recommendations", f"Agent E checks {n or 'the'} possible actions from the project brief. For each, a "
                                   "rule written in code decides whether it is needed from the other agents' results. "
                                   f"{models['writer']} then explains each recommended action from its evidence, and "
                                   "code rejects any explanation that uses a number not in the evidence.")
    if rec is None:
        st.info("Not run yet. This page shows Agent E's recommendations once the full pipeline has run.")
        return
    m = st.columns(4)
    m[0].metric("Recommended", rec.summary.get("recommended", 0))
    m[1].metric("Not needed", rec.summary.get("not needed", 0),
                help="Checked and ruled out, with the number that ruled it out (below).")
    m[2].metric("Could not be checked", rec.summary.get("not checked", 0),
                help="Actions whose rule lacked an input (for example a run without the LLM). 0 means every action "
                     "was tested.")
    m[3].metric("Explanations checked", f"{rec.explanations.get('requests', 0)}",
                help=f"{rec.explanations.get('rejected_drafts', 0)} drafts rejected for unsupported numbers; "
                     f"{rec.explanations.get('fallbacks', 0)} fell back to the rule's own text.")
    st.caption("In order of the kind of evidence: measured loss of answers first, then cases for a person, then "
               "maintenance.")
    for r in rec.recommended():
        st.subheader(f"{r.priority}. {r.action}", anchor=False)
        say(r.explanation)
        if r.scope:
            st.markdown(f"**Applies to {len(r.scope)}:** " + chips(list(r.scope)), unsafe_allow_html=True)
        with st.expander("Evidence"):
            groups: dict[str, list[tuple[str, str]]] = {}
            for e in r.evidence:
                who, _, what = e.source.partition(": ")
                groups.setdefault(who if what else "Evidence", []).append((what or who, e.claim))
            for who, items in groups.items():
                st.markdown(f"**From {who}**" if who.startswith("Agent") else f"**{who}**")
                say("\n".join(f"- *{what[:1].upper() + what[1:]}*. {claim}" for what, claim in items))
            check = r.explanation_check
            note = (f"written by {models['writer']}; every number checked against the evidence"
                    + (" (passed first time)" if "attempt 1" in check else
                       " (passed on a second attempt)" if "attempt 2" in check else f" ({check})")
                    if check.startswith("LLM") else check)
            st.caption(f"Decided by the rule: {r.rule}. Explanation {note}.")
    rest = [r for r in rec.recommendations if r.status != "recommended"]
    if rest:
        st.subheader("Checked, not recommended", anchor=False)
        for r in rest:
            reason = r.finding.split(": ", 1)[1] if r.finding.lower().startswith(("not needed:", "not checked:")) \
                else r.finding
            deciding = r.evidence[0].claim if r.evidence else ""
            st.markdown(f'**{r.action}** <span class="chip">{r.status}</span>', unsafe_allow_html=True)
            say(f"{reason[:1].upper() + reason[1:]} {deciding}")
    for note in rec.notes:
        st.caption(note)


{"Overview": overview, "Data Monitoring": data_monitoring, "Drift": drift, "Retrieval": retrieval,
 "Impact": impact, "Recommendations": recommendations}[page]()
