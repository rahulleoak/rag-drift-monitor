"""Static charts (PNG) of the drift results, for findings, README and slides.

Each function takes saved results (DataFrames read from ``runs/<run_id>/``)
and returns a matplotlib Figure; ``scripts/make_charts.py`` saves them.

Style follows one validated palette (checked with the dataviz validator,
2026-11-05): categorical slots blue / orange / aqua (all-pairs CVD-safe for
three series), an ordinal blue ramp for low / medium / high, and a blue-red
diverging map with a gray midpoint for correlations. Thin marks, hairline
grids, text in ink colors (never the series color), a legend for two or
more series plus selective direct labels.
"""
from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # files only; no window
import logging  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
BANDS = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}  # ordinal ramp 250/450/650
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", NEUTRAL,
                                                                 "#ec8a89", "#b42f2f"])

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # glyph fallback notices

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK_2, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def drift_per_year(whole: pd.DataFrame, thresholds: Mapping[str, Mapping[str, float]],
                   typical: Mapping[str, float]) -> plt.Figure:
    """Two panels (embedding, words): × ordinary change per year, vs the baseline
    and vs the previous year, with the medium and high thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    whole = whole.copy()
    whole["vs"] = np.where(whole["compared_with"] == whole["compared_with"].iloc[0],
                           "vs baseline (2016–17)", "vs previous year")
    panels = [("times_ordinary_embedding", "mmd2", "Embedding drift (MMD²)"),
              ("times_ordinary_lexical", "jsd_excess", "Word drift (excess JSD)")]
    for ax, (col, measure, title) in zip(axes, panels):
        med = thresholds[measure]["medium"] / typical[measure]
        high = thresholds[measure]["high"] / typical[measure]
        top = max(whole[col].max(), high) * 1.15
        ax.axhspan(0, med, color=NEUTRAL, zorder=0, lw=0)
        for y, name in ((med, "medium"), (high, "high")):
            ax.axhline(y, color=MUTED, lw=1, zorder=1)
            ax.text(1.0, y, f" {name}", transform=ax.get_yaxis_transform(), color=INK_2,
                    fontsize=8, va="center", ha="left")
        ax.text(0.98, 0.03, "ordinary range (below medium)", transform=ax.transAxes,
                color=MUTED, fontsize=8, va="bottom", ha="right")
        for k, (label, grp) in enumerate(whole.groupby("vs", sort=False)):
            grp = grp.sort_values("period")
            ax.plot(grp["period"], grp[col], color=SERIES[k], marker="o", ms=6,
                    mec=SURFACE, mew=2, label=label, zorder=3)
        peak = whole.loc[whole[col].idxmax()]
        ax.annotate(f"{peak[col]:.1f}×", (peak["period"], peak[col]), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=INK, fontsize=9, fontweight="semibold")
        ax.set_ylim(0, top)
        ax.set_title(title, pad=24)
        _subtitle(ax, "× median ordinary change (same month, a year apart, inside the baseline)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("× ordinary change")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def band_heatmap(segments: pd.DataFrame, whole: pd.DataFrame, periods: Sequence[str],
                 baseline: str) -> plt.Figure:
    """Section × year grid coloured by severity; gray = too small to analyse."""
    seg = segments[segments["compared_with"] == baseline]
    rows = sorted(seg["category"].unique(), key=lambda c: (-seg[seg["category"] == c]["n_period"].sum(), c))
    rows = ["ALL SECTIONS"] + rows
    allp = whole[whole["compared_with"] == baseline].assign(category="ALL SECTIONS")
    grid = pd.concat([allp, seg]).set_index(["category", "period"])
    order = {"low": 0, "medium": 1, "high": 2}
    mat = np.full((len(rows), len(periods)), np.nan)
    for i, c in enumerate(rows):
        for j, p in enumerate(periods):
            if (c, p) in grid.index:
                mat[i, j] = order[grid.loc[(c, p), "severity"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.6))
    cmap = ListedColormap([BANDS["low"], BANDS["medium"], BANDS["high"]]).with_extremes(bad=NEUTRAL)
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(periods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(periods)), periods)
    ax.set_yticks(range(len(rows)), rows)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    for i, c in enumerate(rows):  # label only the flagged cells, with their kind of drift
        for j, p in enumerate(periods):
            if (c, p) in grid.index and grid.loc[(c, p), "flagged"]:
                kind = grid.loc[(c, p), "kind"]
                ax.text(j, i, kind, ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if mat[i, j] >= 1 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BANDS[b]) for b in BANDS] + \
              [plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    ax.legend(handles, ["low", "medium", "high", "too few articles"], ncol=4, loc="lower left",
              bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1.2)
    ax.set_title("Drift severity by section and year (vs 2016–17)", pad=48)
    ax.text(0, 1.045, "Flagged cells name the kind of drift (meaning, wording, or both)",
            transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.tight_layout()
    return fig


def term_rates(rates: pd.DataFrame, terms: Sequence[str], first_year: int = 2016) -> plt.Figure:
    """Share of each year's articles mentioning each term (up to three terms)."""
    years = [c for c in rates.columns if str(c).isdigit() and int(c) >= first_year]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, t in enumerate(terms[:3]):
        vals = rates.loc[t, years].astype(float)
        ax.plot([int(y) for y in years], vals, color=SERIES[k], marker="o", ms=5, mec=SURFACE,
                mew=2, label=f'"{t}"')
        peak_year = vals.idxmax()
        ax.annotate(f'"{t}" {vals.max():.1f}%', (int(peak_year), vals.max()), xytext=(6, 4),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_ylabel("% of the year's articles")
    ax.set_xticks([int(y) for y in years])
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("The words of the pandemic", pad=24)
    _subtitle(ax, "Share of each year's articles that mention the word (headline + description)")
    fig.tight_layout()
    return fig


def term_shift_bars(shifts: pd.DataFrame, top: int = 10) -> plt.Figure:
    """Words whose surrounding articles moved most, with before → after context."""
    s = shifts.dropna(subset=["times_ordinary"]).sort_values("times_ordinary", ascending=False)
    s = s.drop_duplicates("term").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(s) + 1.4))
    y = np.arange(len(s))
    ax.barh(y, s["times_ordinary"], height=0.5, color=SERIES[0])
    for i, r in enumerate(s.itertuples()):
        before = ", ".join(r.context_before.split(", ")[:3])
        after = ", ".join(r.context_after.split(", ")[:3])
        ax.text(r.times_ordinary + 0.08, i, f"{r.times_ordinary:.1f}×   {before}  →  {after}",
                va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(y, [f'"{t}"  {p}' for t, p in zip(s["term"], s["period"])])
    ax.set_xlim(0, s["times_ordinary"].max() * 2.15)
    ax.axvline(1, color=MUTED, lw=1)
    ax.text(1.05, -0.75, "ordinary", color=MUTED, fontsize=8, va="bottom")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("× ordinary shift of the articles using the word")
    ax.set_title("Words that changed meaning", pad=26)
    _subtitle(ax, "Strongest year per word; context words before (2016–17) → after")
    fig.tight_layout()
    return fig


def emerging_topics_bars(topics: pd.DataFrame, top: int = 10) -> plt.Figure:
    """Largest emerging topic-years: articles in the year vs in the baseline sample."""
    t = topics.sort_values("docs_period", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y + 0.14, t["docs_period"], height=0.28, color=SERIES[0], label="in that year")
    ax.barh(y - 0.14, t["docs_baseline"], height=0.28, color=SERIES[1], label="in the baseline")
    labels = [f"{p}  " + textwrap.shorten(l, 42, placeholder="…")
              for p, l in zip(t["period"], t["label"])]
    ax.set_yticks(y, labels)
    for i, r in enumerate(t.itertuples()):
        ax.text(r.docs_period + 1.5, i + 0.14, f"{r.docs_period}", va="center", color=INK_2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("articles in the drift sample (1,398 per period)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Emerging topics: found by growth, not novelty", pad=26)
    _subtitle(ax, "Topic clusters whose share of a year is at least 3× their share of the baseline")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame, names: Mapping[str, str], n_comparisons: int) -> plt.Figure:
    """Lower-triangle Spearman correlations between drift measures (P6)."""
    labels = [names.get(c, c) for c in corr.columns]
    full = corr.to_numpy(dtype=float)
    mat = full[1:, :-1]                      # lower triangle without the empty row/column
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(np.ma.masked_where(mask, mat), cmap=DIVERGING, vmin=-1, vmax=1)
    n = len(mat)
    for i in range(n):
        for j in range(i + 1):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(n), labels[:-1], rotation=35, ha="right")
    ax.set_yticks(range(n), labels[1:])
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.set_title(f"How the drift measures relate (Spearman, {n_comparisons} comparisons)", pad=12)
    fig.tight_layout()
    return fig
