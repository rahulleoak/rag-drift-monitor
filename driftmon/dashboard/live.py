"""Interactive (Altair) charts for the dashboard, built from a saved run's tables.

Each function takes plain DataFrames and returns an Altair chart; ``themed`` adds
the dark Rivvun styling. No Streamlit here, so the charts are testable on their
own. Colours follow ``driftmon.dashboard.charts.DARK`` (validated): blue = the
baseline / stale index (old), orange = the later year / refreshed index (new),
used the same way in every chart. The section-trend chart needs more series and
uses the validated categorical order in ``SECTION_COLOURS`` (checked against the
dark surface with the dataviz validator, 2026-11-09: adjacent CVD ΔE >= 8.4,
normal-vision ΔE >= 19.3, contrast >= 3:1), with every line labelled by name.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import altair as alt
import pandas as pd

from driftmon.dashboard.charts import DARK as T

ORDINAL = alt.Scale(domain=["low", "medium", "high"], range=list(T.bands.values()))
SECTION_COLOURS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
CLASS_COLOURS = {"drift that hurts": T.series[0], "harm without drift": T.series[1],
                 "drift that does not hurt": T.series[2], "healthy": T.muted}
MEASURE_NAMES = {"mmd2": "MMD²", "centroid_shift": "Centroid shift", "jsd_excess": "Word drift (JSD)",
                 "novelty_mean": "Novelty (mean)", "novel_share": "Novel share", "new_term_rate": "New-term rate"}


def themed(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Rivvun dark styling: navy surface, off-white ink, hairline slate grid."""
    return (chart.configure(background=T.surface, font="Inter, Segoe UI, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=T.ink_2, titleColor=T.ink_2, gridColor=T.grid, domainColor=T.axis,
                            tickColor=T.axis, labelFontSize=12, titleFontSize=12, titleFontWeight=400)
            .configure_legend(labelColor=T.ink_2, titleColor=T.ink_2, labelFontSize=12, titleFontSize=12)
            .configure_title(color=T.ink, anchor="start", fontWeight=600, fontSize=14))


def tilted(n: int) -> int:
    """Axis label angle: flat for a few periods, tilted when many would collide."""
    return 0 if n <= 8 else -45


def drift_line(whole: pd.DataFrame, col: str, measure: str, title: str, reference: Mapping,
               baseline: str) -> alt.Chart:
    """× ordinary change per year, vs the baseline and vs the previous year, with severity lines."""
    typical = reference["typical"][measure]
    th = pd.DataFrame([{"band": b, "y": v / typical} for b, v in reference["thresholds"][measure].items()])
    d = whole.assign(vs=lambda x: x["compared_with"].map(
        lambda c: "vs the baseline" if c == baseline else "vs the previous year"))
    color = alt.Color("vs:N", scale=alt.Scale(domain=["vs the baseline", "vs the previous year"],
                                              range=[T.series[0], T.series[1]]),
                      legend=alt.Legend(title=None, orient="top"))
    n = d["period"].nunique()
    base = alt.Chart(d).encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=tilted(n))),
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
    n = grid["period"].nunique()
    y = alt.Y("category:N", sort=["ALL SECTIONS"] + order, title=None, axis=alt.Axis(labelOverlap=False))
    base = alt.Chart(grid).encode(x=alt.X("period:O", title=None,
                                          axis=alt.Axis(labelAngle=tilted(n), orient="top")), y=y)
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
    # independent colour scales: the labels need contrasting colours; a shared scale hid labels and legend
    return (rect + text).resolve_scale(color="independent").properties(
        height=26 * grid["category"].nunique() + 20)


def grouped_bars(long: pd.DataFrame, title: str, stale_name: str, baseline: str) -> alt.Chart:
    """Per year, two bars: the stale index (blue) and the refreshed index (orange), each labelled.
    The baseline year is the historical control and is named so on the axis."""
    order = [stale_name, "Refreshed index"]
    d = long.assign(year=[f"{p} (control)" if p == baseline else p for p in long["period"]])
    years = [f"{p} (control)" if p == baseline else p for p in sorted(long["period"].unique())]
    x = alt.X("year:N", title=None, sort=years, axis=alt.Axis(labelAngle=tilted(len(years))))
    y = alt.Y("value:Q", title=title, scale=alt.Scale(domain=[0, 1.08]), axis=alt.Axis(format="%", tickCount=5,
                                                                                       values=[0, .25, .5, .75, 1]))
    colour = alt.Color("index:N", scale=alt.Scale(domain=order, range=[T.series[1], T.series[0]]),
                       legend=alt.Legend(title=None, orient="top"))
    tip = [alt.Tooltip("period:N", title="Year"), alt.Tooltip("index:N", title="Index"),
           alt.Tooltip("value:Q", title=title, format=".1%"), alt.Tooltip("questions:Q", title="Questions"),
           alt.Tooltip("delta:Q", title="Refreshed − stale", format="+.0%"),
           alt.Tooltip("delta_low:Q", title="95% interval, low", format="+.0%"),
           alt.Tooltip("delta_high:Q", title="95% interval, high", format="+.0%")]
    off = alt.XOffset("index:N", sort=order)
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3, width={"band": 0.42}).encode(
        x=x, xOffset=off, y=y, color=colour, tooltip=tip)
    labels = alt.Chart(d).transform_calculate(
        lab="datum.value > 0 && datum.value < 0.01 ? format(datum.value, '.1%') : format(datum.value, '.0%')"
    ).mark_text(dy=-8, fontSize=11, color=T.ink).encode(x=x, xOffset=off, y=y, text="lab:N")
    return (bars + labels).properties(height=280)


def harm_scatter(seg: pd.DataFrame) -> alt.Chart:
    """Each classified section: its drift score against its loss in correct answers, zoomed to the data."""
    present = [c for c in CLASS_COLOURS if c in set(seg["class"])]   # only classes this run has
    colour = alt.Color("class:N", scale=alt.Scale(domain=present, range=[CLASS_COLOURS[c] for c in present]),
                       legend=alt.Legend(title=None, orient="top"))
    low = max(0.0, (float(seg["loss_low"].min()) if "loss_low" in seg else float(seg["loss_correct"].min())) - 0.05)
    low = min(low, 0.8)
    x = alt.X("times_ordinary_embedding:Q", title="Drift score (meaning, × ordinary change)",
              scale=alt.Scale(zero=True))
    y = alt.Y("loss_correct:Q", title="Drop in correct answers", scale=alt.Scale(domain=[low, 1.01], clamp=True),
              axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("section:N", title="Section"), alt.Tooltip("class:N", title="Class"),
           alt.Tooltip("loss_correct:Q", title="Drop", format=".0%"),
           alt.Tooltip("loss_low:Q", title="95% interval, low", format=".0%"),
           alt.Tooltip("loss_high:Q", title="95% interval, high", format=".0%"),
           alt.Tooltip("times_ordinary_embedding:Q", title="Meaning × ordinary", format=".1f"),
           alt.Tooltip("severity:N", title="Drift severity"), alt.Tooltip("questions:Q", title="Questions")]
    base = alt.Chart(seg).encode(x=x)
    err = base.mark_rule(color=T.axis, strokeWidth=2).encode(y=alt.Y("loss_low:Q", scale=alt.Scale(
        domain=[low, 1.01], clamp=True)), y2="loss_high:Q")
    dots = base.mark_circle(size=150, opacity=1, stroke=T.surface, strokeWidth=2).encode(
        y=y, color=colour, tooltip=tip)
    return (err + dots).properties(height=320)


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
    err = alt.Chart(d).mark_rule(color=T.ink_2, strokeWidth=1.5).encode(y=y, x="changed_low:Q", x2="changed_high:Q")
    label = alt.Chart(d).mark_text(align="left", dx=8, color=T.ink, fontSize=12).encode(
        y=y, x="changed_high:Q", text=alt.Text("answer_changed:Q", format=".0%"))
    return (bars + err + label).properties(height={"step": 48})


def year_bars(df: pd.DataFrame, col: str, title: str) -> alt.Chart:
    """One measure per later year, as labelled bars (a single series: no legend)."""
    x = alt.X("period:N", title=None, axis=alt.Axis(labelAngle=tilted(len(df))))
    y = alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format="%", tickCount=4))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=T.series[0], width={"band": 0.6}).encode(
        x=x, y=y, tooltip=[alt.Tooltip("period:N", title="Year"),
                           alt.Tooltip(f"{col}:Q", title=title, format=".1%")])
    text = alt.Chart(df).mark_text(dy=-8, color=T.ink, fontSize=12).encode(
        x=x, y=y, text=alt.Text(f"{col}:Q", format=".0%"))
    return (bars + text).properties(title=title, height=200)


def flagged_bars(f: pd.DataFrame) -> alt.Chart:
    """Per year: share of analysable sections flagged, labelled 'n of m' (one series: no legend)."""
    f = f.assign(label=[f"{a} of {b}" for a, b in zip(f["flagged"], f["groups"])])
    x = alt.X("period:N", title=None, axis=alt.Axis(labelAngle=tilted(len(f))))
    y = alt.Y("share:Q", title="Sections flagged", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", tickCount=5))
    tip = [alt.Tooltip("period:N", title="Year"), alt.Tooltip("flagged:Q", title="Flagged"),
           alt.Tooltip("groups:Q", title="Analysable sections"), alt.Tooltip("share:Q", title="Share", format=".0%")]
    bars = alt.Chart(f).mark_bar(cornerRadiusEnd=4, color=T.series[0], width={"band": 0.55}).encode(
        x=x, y=y, tooltip=tip)
    text = alt.Chart(f).mark_text(dy=-8, color=T.ink, fontSize=12).encode(x=x, y=y, text="label:N")
    return (bars + text).properties(height=240)


def _dodge(values: Sequence[float], gap: float) -> list[float]:
    """Spread end-of-line label positions so no two are closer than ``gap`` (keeps their order)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def section_trends(d: pd.DataFrame, other: str) -> alt.Chart:
    """One line per large section: its share of each year's articles. Sections new since the baseline
    are dashed; every line is labelled at its end, so colour is not the only cue."""
    periods = ([str(p) for p in d["period"].cat.categories] if hasattr(d["period"], "cat")
               else sorted(d["period"].astype(str).unique()))   # baseline first, then later years
    d = d.assign(period=d["period"].astype(str))
    names = d.sort_values("rank")["section"].drop_duplicates().tolist()
    named = [s for s in names if s != other]
    colours = {s: SECTION_COLOURS[i % len(SECTION_COLOURS)] for i, s in enumerate(named)}
    colours[other] = T.muted
    d = d.assign(display=d["section"].map(lambda s: s.title() if s != other else s),
                 status=["new since the baseline" if n else "in the baseline" for n in d["new"]])
    disp = [s.title() if s != other else s for s in names]
    colour = alt.Color("display:N", sort=disp, scale=alt.Scale(domain=disp, range=[colours[s] for s in names]),
                       legend=alt.Legend(title=None, orient="top", columns=4))
    dash = alt.StrokeDash("new:N", scale=alt.Scale(domain=[False, True], range=[[1, 0], [6, 4]]), legend=None)
    x = alt.X("period:N", title=None, sort=periods, scale=alt.Scale(paddingOuter=0.05),
              axis=alt.Axis(labelAngle=tilted(len(periods))))
    top = float(d["share"].max())
    y = alt.Y("share:Q", title="Share of the year's articles", scale=alt.Scale(domain=[0, top * 1.08]),
              axis=alt.Axis(format="%", tickCount=5))
    tip = [alt.Tooltip("display:N", title="Section"), alt.Tooltip("period:N", title="Year"),
           alt.Tooltip("share:Q", title="Share", format=".1%"), alt.Tooltip("articles:Q", title="Articles", format=","),
           alt.Tooltip("status:N", title="")]
    lines = alt.Chart(d).mark_line(strokeWidth=2.2).encode(x=x, y=y, color=colour, strokeDash=dash, detail="display:N")
    points = alt.Chart(d).mark_point(filled=True, size=55, stroke=T.surface, strokeWidth=1.5).encode(
        x=x, y=y, color=colour, tooltip=tip)
    last = d[d["period"] == periods[-1]].copy()
    last["label_y"] = _dodge(last["share"].tolist(), gap=top * 0.07)
    last["label"] = [f"{s}{' (new)' if n else ''}" for s, n in zip(last["display"], last["new"])]
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=11, color=T.ink_2).encode(
        x=alt.X("period:N", sort=periods), y=alt.Y("label_y:Q"), text="label:N")
    return (lines + points + labels).properties(height=400, padding={"left": 5, "top": 5, "bottom": 5,
                                                                    "right": 150})


def butterfly(d: pd.DataFrame, baseline: str, period: str) -> alt.HConcatChart:
    """Words before vs after: use per 1,000 words in the baseline (left, blue) and in the year (right, orange)."""
    order = d["label"].tolist()
    top = float(max(d["per_1000_reference"].max(), d["per_1000_period"].max()) * 1.15) or 1.0
    y = alt.Y("label:N", sort=order, title=None, axis=None)
    tip = [alt.Tooltip("term:N", title="Word"), alt.Tooltip("direction:N", title="Became"),
           alt.Tooltip("per_1000_reference:Q", title=f"Per 1,000 words, {baseline}", format=".2f"),
           alt.Tooltip("per_1000_period:Q", title=f"Per 1,000 words, {period}", format=".2f")]
    left = alt.Chart(d).mark_bar(color=T.series[1], cornerRadiusEnd=3, height=14).encode(
        x=alt.X("per_1000_reference:Q", title=f"{baseline} (per 1,000 words)",
                scale=alt.Scale(domain=[0, top], reverse=True)), y=y, tooltip=tip).properties(
        width=260, height={"step": 24})
    words = alt.Chart(d).mark_text(color=T.ink, fontSize=12).encode(y=y, text="label:N").properties(
        width=130, height={"step": 24})
    right = alt.Chart(d).mark_bar(color=T.series[0], cornerRadiusEnd=3, height=14).encode(
        x=alt.X("per_1000_period:Q", title=f"{period} (per 1,000 words)", scale=alt.Scale(domain=[0, top])),
        y=y, tooltip=tip).properties(width=260, height={"step": 24})
    return alt.hconcat(left, words, right, spacing=4)


def topics_bars(topics: pd.DataFrame, top: int = 10) -> alt.Chart:
    """Largest topic-years that grew: articles in the year (orange) vs in the baseline sample (blue)."""
    d = topics.sort_values("docs_period", ascending=False).head(top).copy()
    d["name"] = [f"{p} · " + ", ".join(str(lab).split(", ")[:3]) for p, lab in zip(d["period"], d["label"])]
    order = d["name"].tolist()
    long = pd.concat([d.assign(side="in that year", articles=d["docs_period"]),
                      d.assign(side="in the baseline", articles=d["docs_baseline"])])
    colour = alt.Color("side:N", scale=alt.Scale(domain=["in that year", "in the baseline"],
                                                 range=[T.series[0], T.series[1]]),
                       legend=alt.Legend(title=None, orient="top"))
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=320))
    tip = [alt.Tooltip("period:N", title="Year"), alt.Tooltip("label:N", title="Topic words"),
           alt.Tooltip("docs_period:Q", title="Articles that year"),
           alt.Tooltip("docs_baseline:Q", title="Articles in the baseline sample"),
           alt.Tooltip("ratio:Q", title="Growth (×)", format=".1f"), alt.Tooltip("top_sections:N", title="Sections")]
    off = alt.YOffset("side:N", sort=["in that year", "in the baseline"])
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3, height={"band": 0.8}).encode(
        y=y, yOffset=off, x=alt.X("articles:Q", title="Articles in the drift sample"), color=colour, tooltip=tip)
    labels = alt.Chart(long[long["side"] == "in that year"]).mark_text(align="left", dx=5, fontSize=11,
                                                                        color=T.ink_2).encode(
        y=y, yOffset=off, x="articles:Q", text="articles:Q")
    return (bars + labels).properties(height={"step": 26})


def meaning_bars(shifts: pd.DataFrame, top: int = 10) -> alt.Chart:
    """Words whose surrounding articles moved most (× ordinary), with their context before → after."""
    s = shifts.dropna(subset=["times_ordinary"]).sort_values("times_ordinary", ascending=False)
    s = s.drop_duplicates("term").head(top).copy()
    s["name"] = [f'"{w}" {p}' for w, p in zip(s["term"], s["period"])]
    s["context"] = [", ".join(str(b).split(", ")[:3]) + "  →  " + ", ".join(str(a).split(", ")[:3])
                    for b, a in zip(s["context_before"], s["context_after"])]
    order = s["name"].tolist()
    y = alt.Y("name:N", sort=order, title=None)
    xmax = float(s["times_ordinary"].max()) * 1.05
    tip = [alt.Tooltip("term:N", title="Word"), alt.Tooltip("period:N", title="Year"),
           alt.Tooltip("times_ordinary:Q", title="× ordinary change", format=".1f"),
           alt.Tooltip("context_before:N", title="Neighbouring words before"),
           alt.Tooltip("context_after:N", title="Neighbouring words after")]
    bars = alt.Chart(s).mark_bar(color=T.series[0], cornerRadiusEnd=3, height=16).encode(
        y=y, x=alt.X("times_ordinary:Q", title="× ordinary change", scale=alt.Scale(domain=[0, xmax])),
        tooltip=tip)
    ordinary = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=T.muted, strokeDash=[4, 3]).encode(x="x:Q")
    left = (bars + ordinary).properties(width=300, height={"step": 30})
    context = alt.Chart(s).mark_text(align="left", fontSize=11.5, color=T.ink_2).encode(
        y=alt.Y("name:N", sort=order, axis=None), text="context:N", tooltip=tip).properties(
        width=440, height={"step": 30})
    return alt.hconcat(left, context, spacing=12)


def correlation_grid(corr: pd.DataFrame, names: Mapping[str, str] = MEASURE_NAMES) -> alt.Chart:
    """Lower triangle of rank correlations between drift measures; blue = opposite, orange = agree."""
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i and pd.notna(corr.loc[a, b]):
                rows.append({"row": names.get(a, a), "col": names.get(b, b), "rho": float(corr.loc[a, b])})
    d = pd.DataFrame(rows)
    order = [names.get(c, c) for c in cols]
    rect = alt.Chart(d).mark_rect(stroke=T.surface, strokeWidth=2).encode(
        x=alt.X("col:N", sort=order[:-1], title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("row:N", sort=order[1:], title=None),
        color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[T.diverging[0], T.diverging[2],
                                                                           T.diverging[-1]]),
                        legend=alt.Legend(title="Agreement", orient="right")),
        tooltip=[alt.Tooltip("row:N", title=""), alt.Tooltip("col:N", title=""),
                 alt.Tooltip("rho:Q", title="Rank correlation", format=".2f")])
    text = alt.Chart(d).mark_text(fontSize=12, color=T.ink).encode(
        x=alt.X("col:N", sort=order[:-1]), y=alt.Y("row:N", sort=order[1:]), text=alt.Text("rho:Q", format=".2f"))
    return (rect + text).properties(width=440, height=300)
