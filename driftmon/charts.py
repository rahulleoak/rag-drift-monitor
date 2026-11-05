"""Static charts (PNG) of the drift results, for findings, README and slides.

Each function takes saved results (DataFrames read from ``runs/<run_id>/``)
and a ``Theme``, and returns a matplotlib Figure; ``scripts/make_charts.py``
saves them.

Colours follow the Rivvun AI brand (rivvun.ai design tokens, read
2026-11-05): warm cream or deep navy surfaces, navy / off-white ink, and the
burnt-orange accent #C14100 as the first series and the severity ramp. The
brand has one accent, so two supporting series hues (a blue and a teal /
green) were chosen to sit with it. Every palette was checked with the
dataviz validator against the theme's own surface (2026-11-05):
    light  series #C14100 #2F6CB0 #1E9A8A  all-pairs CVD dE >= 13.7, contrast >= 3:1
           ramp   #E39A73 #C14100 #7A2900  light end 2.09:1
    dark   series #D9581E #6383E0 #1F9E70  all-pairs CVD dE >= 9.8, contrast >= 3:1
           ramp   #8A3000 #C9500F #F29A6B  dark end 2.15:1
Thin marks, hairline grids, text in ink colours (never the series colour),
a legend for two or more series plus selective direct labels.
"""
from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # files only; no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # glyph fallback notices

FONTS = ["Inter", "Segoe UI", "DejaVu Sans"]  # Rivvun's stack: Inter, then system fonts


@dataclass(frozen=True)
class Theme:
    """One chart theme: surfaces, ink and the validated palettes."""

    name: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    neutral: str                      # empty cells, the "ordinary" band
    series: tuple[str, str, str]      # categorical slots 1-3
    bands: dict[str, str] = field(default_factory=dict)  # ordinal low / medium / high
    diverging: tuple[str, ...] = ()   # negative pole ... neutral ... positive pole

    @property
    def rc(self) -> dict:
        return {
            "font.family": FONTS, "font.size": 10, "axes.titlesize": 12,
            "axes.titleweight": "semibold", "axes.titlelocation": "left",
            "axes.titlecolor": self.ink, "axes.labelcolor": self.ink_2, "text.color": self.ink,
            "axes.edgecolor": self.axis, "axes.linewidth": 1.0, "axes.facecolor": self.surface,
            "figure.facecolor": self.surface, "savefig.facecolor": self.surface,
            "xtick.color": self.muted, "ytick.color": self.muted,
            "xtick.labelcolor": self.ink_2, "ytick.labelcolor": self.ink_2,
            "axes.grid": True, "axes.axisbelow": True, "grid.color": self.grid,
            "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
            "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": self.ink_2,
            "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        }

    def on(self, fill: str) -> str:
        """Text colour for a label set inside a fill: white on dark fills, navy on light ones."""
        r, g, b = to_rgb(fill)
        light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45
        return "#0B1A2B" if light else "#FFFFFF"


LIGHT = Theme(
    name="light", surface="#F6F4EF", ink="#0B1A2B", ink_2="#4D5563", muted="#7C8290",
    grid="#E6E2DA", axis="#C7C2B7", neutral="#ECE7DC",
    series=("#C14100", "#2F6CB0", "#1E9A8A"),
    bands={"low": "#E39A73", "medium": "#C14100", "high": "#7A2900"},
    diverging=("#1B3F6E", "#7FA3CF", "#ECE7DC", "#E39A73", "#7A2900"))
DARK = Theme(
    name="dark", surface="#0D1725", ink="#F3F1EB", ink_2="#ACACA8", muted="#8A9099",
    grid="#242B37", axis="#39404A", neutral="#1C2330",
    series=("#D9581E", "#6383E0", "#1F9E70"),
    bands={"low": "#8A3000", "medium": "#C9500F", "high": "#F29A6B"},
    diverging=("#2E5FA8", "#3B4D6B", "#1C2330", "#7A3A1C", "#E0642A"))
THEMES = {"light": LIGHT, "dark": DARK}


def _subtitle(ax, text: str, t: Theme) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=t.ink_2, fontsize=9, va="bottom")


def drift_per_year(whole: pd.DataFrame, thresholds: Mapping[str, Mapping[str, float]],
                   typical: Mapping[str, float], t: Theme = LIGHT) -> plt.Figure:
    """Two panels (embedding, words): × ordinary change per year, vs the baseline
    and vs the previous year, with the medium and high thresholds."""
    with plt.rc_context(t.rc):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        whole = whole.copy()
        whole["vs"] = np.where(whole["compared_with"] == whole["compared_with"].iloc[0],
                               "vs baseline (2016–17)", "vs previous year")
        panels = [("times_ordinary_embedding", "mmd2", "Embedding drift (MMD²)"),
                  ("times_ordinary_lexical", "jsd_excess", "Word drift (excess JSD)")]
        for ax, (col, measure, title) in zip(axes, panels):
            med = thresholds[measure]["medium"] / typical[measure]
            high = thresholds[measure]["high"] / typical[measure]
            ax.axhspan(0, med, color=t.neutral, zorder=0, lw=0)
            for y, name in ((med, "medium"), (high, "high")):
                ax.axhline(y, color=t.muted, lw=1, zorder=1)
                ax.text(1.0, y, f" {name}", transform=ax.get_yaxis_transform(), color=t.ink_2,
                        fontsize=8, va="center", ha="left")
            ax.text(0.98, 0.03, "ordinary range (below medium)", transform=ax.transAxes,
                    color=t.muted, fontsize=8, va="bottom", ha="right")
            for k, (label, grp) in enumerate(whole.groupby("vs", sort=False)):
                grp = grp.sort_values("period")
                ax.plot(grp["period"], grp[col], color=t.series[k], marker="o", ms=6,
                        mec=t.surface, mew=2, label=label, zorder=3)
            peak = whole.loc[whole[col].idxmax()]
            ax.annotate(f"{peak[col]:.1f}×", (peak["period"], peak[col]), textcoords="offset points",
                        xytext=(0, 8), ha="center", color=t.ink, fontsize=9, fontweight="semibold")
            ax.set_ylim(0, max(whole[col].max(), high) * 1.15)
            ax.set_title(title, pad=24)
            _subtitle(ax, "× median ordinary change (same month, a year apart, inside the baseline)", t)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("× ordinary change")
        axes[0].legend(loc="upper left", fontsize=9)
        fig.tight_layout()
    return fig


def band_heatmap(segments: pd.DataFrame, whole: pd.DataFrame, periods: Sequence[str],
                 baseline: str, t: Theme = LIGHT) -> plt.Figure:
    """Section × year grid coloured by severity; neutral = too few articles to analyse."""
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
    fills = [t.bands["low"], t.bands["medium"], t.bands["high"]]
    with plt.rc_context(t.rc):
        fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.6))
        cmap = ListedColormap(fills).with_extremes(bad=t.neutral)
        ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        ax.set_xticks(np.arange(-0.5, len(periods)), minor=True)  # 2px surface gap between cells
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color=t.surface, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        ax.set_xticks(range(len(periods)), periods)
        ax.set_yticks(range(len(rows)), rows)
        ax.get_yticklabels()[0].set_fontweight("semibold")
        for i, c in enumerate(rows):  # label only flagged cells, with their kind of drift
            for j, p in enumerate(periods):
                if (c, p) in grid.index and grid.loc[(c, p), "flagged"]:
                    ax.text(j, i, grid.loc[(c, p), "kind"], ha="center", va="center", fontsize=7.5,
                            color=t.on(fills[int(mat[i, j])]))
        for s in ax.spines.values():
            s.set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=f) for f in fills + [t.neutral]]
        ax.legend(handles, ["low", "medium", "high", "too few articles"], ncol=4, loc="lower left",
                  bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1.2)
        ax.set_title("Drift severity by section and year (vs 2016–17)", pad=48)
        ax.text(0, 1.045, "Flagged cells name the kind of drift (meaning, wording, or both)",
                transform=ax.transAxes, color=t.ink_2, fontsize=9, va="bottom")
        fig.tight_layout()
    return fig


def term_rates(rates: pd.DataFrame, terms: Sequence[str], first_year: int = 2016,
               t: Theme = LIGHT) -> plt.Figure:
    """Share of each year's articles mentioning each term (up to three terms)."""
    years = [c for c in rates.columns if str(c).isdigit() and int(c) >= first_year]
    with plt.rc_context(t.rc):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for k, term in enumerate(terms[:3]):
            vals = rates.loc[term, years].astype(float)
            ax.plot([int(y) for y in years], vals, color=t.series[k], marker="o", ms=5,
                    mec=t.surface, mew=2, label=f'"{term}"')
            ax.annotate(f'"{term}" {vals.max():.1f}%', (int(vals.idxmax()), vals.max()), xytext=(6, 4),
                        textcoords="offset points", color=t.ink, fontsize=9)
        ax.set_ylabel("% of the year's articles")
        ax.set_xticks([int(y) for y in years])
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title("The words of the pandemic", pad=24)
        _subtitle(ax, "Share of each year's articles that mention the word (headline + description)", t)
        fig.tight_layout()
    return fig


def term_shift_bars(shifts: pd.DataFrame, top: int = 10, t: Theme = LIGHT) -> plt.Figure:
    """Words whose surrounding articles moved most, with before → after context."""
    s = shifts.dropna(subset=["times_ordinary"]).sort_values("times_ordinary", ascending=False)
    s = s.drop_duplicates("term").head(top).iloc[::-1]
    with plt.rc_context(t.rc):
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(s) + 1.4))
        y = np.arange(len(s))
        ax.barh(y, s["times_ordinary"], height=0.5, color=t.series[0])
        for i, r in enumerate(s.itertuples()):
            before = ", ".join(r.context_before.split(", ")[:3])
            after = ", ".join(r.context_after.split(", ")[:3])
            ax.text(r.times_ordinary + 0.08, i, f"{r.times_ordinary:.1f}×   {before}  →  {after}",
                    va="center", color=t.ink_2, fontsize=8.5)
        ax.set_yticks(y, [f'"{w}"  {p}' for w, p in zip(s["term"], s["period"])])
        ax.set_xlim(0, s["times_ordinary"].max() * 2.15)
        ax.axvline(1, color=t.muted, lw=1)
        ax.text(1.05, -0.75, "ordinary", color=t.muted, fontsize=8, va="bottom")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("× ordinary shift of the articles using the word")
        ax.set_title("Words that changed meaning", pad=26)
        _subtitle(ax, "Strongest year per word; context words before (2016–17) → after", t)
        fig.tight_layout()
    return fig


def emerging_topics_bars(topics: pd.DataFrame, top: int = 10, t: Theme = LIGHT) -> plt.Figure:
    """Largest emerging topic-years: articles in the year vs in the baseline sample."""
    d = topics.sort_values("docs_period", ascending=False).head(top).iloc[::-1]
    with plt.rc_context(t.rc):
        fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(d) + 1.4))
        y = np.arange(len(d))
        ax.barh(y + 0.14, d["docs_period"], height=0.28, color=t.series[0], label="in that year")
        ax.barh(y - 0.14, d["docs_baseline"], height=0.28, color=t.series[1], label="in the baseline")
        labels = [f"{p}  " + textwrap.shorten(lab, 42, placeholder="…")
                  for p, lab in zip(d["period"], d["label"])]
        ax.set_yticks(y, labels)
        for i, r in enumerate(d.itertuples()):
            ax.text(r.docs_period + 1.5, i + 0.14, f"{r.docs_period}", va="center", color=t.ink_2, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("articles in the drift sample (1,398 per period)")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Emerging topics: found by growth, not novelty", pad=26)
        _subtitle(ax, "Topic clusters whose share of a year is at least 3× their share of the baseline", t)
        fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame, names: Mapping[str, str], n_comparisons: int,
                        t: Theme = LIGHT) -> plt.Figure:
    """Lower-triangle Spearman correlations between drift measures (P6)."""
    labels = [names.get(c, c) for c in corr.columns]
    mat = corr.to_numpy(dtype=float)[1:, :-1]   # lower triangle without the empty row/column
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list(f"div_{t.name}", list(t.diverging))
    with plt.rc_context(t.rc):
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ax.imshow(np.ma.masked_where(mask, mat), cmap=cmap, vmin=-1, vmax=1)
        n = len(mat)
        for i in range(n):
            for j in range(i + 1):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color=t.on(to_hex(cmap((v + 1) / 2))))
        ax.set_xticks(range(n), labels[:-1], rotation=35, ha="right")
        ax.set_yticks(range(n), labels[1:])
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=t.surface, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=t.muted, labelcolor=t.ink_2, length=0)
        ax.set_title(f"How the drift measures relate (Spearman, {n_comparisons} comparisons)", pad=12)
        fig.tight_layout()
    return fig
