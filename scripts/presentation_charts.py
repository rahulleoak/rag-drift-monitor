"""Draw the two slide charts that have no dashboard screenshot (dark theme).

    17_chunking.png      how well each way of reading a long article follows real edits (F16)
    18_applied_fix.png   outdated answers, 2017 index vs the targeted re-index (F24)

Numbers come from the saved results (runs/longform/), not typed in.

Usage (from the repository root):
    python scripts/presentation_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from driftmon.dashboard.charts import DARK as T, FONTS  # noqa: E402

OUT = ROOT / "docs" / "presentation"
LONGFORM = ROOT / "runs" / "longform"
OLD, NEW = T.series[1], T.series[0]  # blue = before / stale, orange = after / refreshed


def _frame(ax, fig):
    fig.patch.set_facecolor(T.surface)
    ax.set_facecolor(T.surface)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T.axis)
    ax.tick_params(colors=T.ink_2, labelsize=11, length=0)
    ax.xaxis.grid(True, color=T.grid, linewidth=0.8)
    ax.set_axisbelow(True)


def chunking() -> None:
    ev = pd.read_csv(LONGFORM / "chunking_eval.csv").set_index("method")
    rows = [("First 512 tokens\n(whole-article vector)", "truncated"),
            ("One averaged vector\nof all chunks", "averaged"),
            ("256-token chunks\n(our default)", "chunks_256"),
            ("128-token chunks", "chunks_128")]
    labels = [r[0] for r in rows]
    vals = [float(ev.loc[r[1], "spearman_all"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
    _frame(ax, fig)
    y = range(len(rows))[::-1]
    colors = [OLD, T.muted, NEW, T.muted]
    ax.barh(list(y), vals, height=0.58, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", color=T.ink, fontsize=13, fontweight="bold")
    ax.set_yticks(list(y), labels, color=T.ink, fontsize=11.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("How well the change score follows real edits (rank correlation, 1 = perfectly)",
                  color=T.ink_2, fontsize=11)
    ax.set_title("Reading a long article: how well each method sees real edits", loc="left",
                 color=T.ink, fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "17_chunking.png", facecolor=T.surface)
    plt.close(fig)


def applied_fix() -> None:
    v = json.loads((LONGFORM / "reindex_verdict.json").read_text(encoding="utf-8"))
    s = v["by_stratum"]
    rows = [("Changed facts,\nflagged articles", s["listed_number"]["outdated_stale"],
             s["listed_number"]["outdated_targeted"]),
            ("Other edits,\nflagged articles", s["listed_wording"]["outdated_stale"],
             s["listed_wording"]["outdated_targeted"]),
            ("Edits in the other\n75 articles", s["unlisted"]["outdated_stale"], s["unlisted"]["outdated_targeted"]),
            ("All edits\n(weighted)", v["outdated_stale"], v["outdated_targeted"])]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    _frame(ax, fig)
    h = 0.36
    for i, (label, old, new) in enumerate(rows[::-1]):
        ax.barh(i + h / 2 + 0.02, old * 100, height=h, color=OLD)
        ax.barh(i - h / 2 - 0.02, new * 100, height=h, color=NEW)
        for yy, val in ((i + h / 2 + 0.02, old), (i - h / 2 - 0.02, new)):
            ax.text(val * 100 + 1, yy, f"{val * 100:.1f}%" if label.startswith("All") else f"{val * 100:.0f}%",
                    va="center", color=T.ink, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=T.ink, fontsize=11.5)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Answers that are outdated (differ from a full re-index, or cannot answer)",
                  color=T.ink_2, fontsize=11)
    ax.set_title("Outdated answers before and after re-indexing only the 90 flagged articles", loc="left",
                 color=T.ink, fontsize=15, fontweight="bold", pad=34)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OLD), plt.Rectangle((0, 0), 1, 1, color=NEW)]
    leg = ax.legend(handles, ["Index built in 2017 (stale)", "After the targeted re-index"], loc="lower left",
                    bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=11, handlelength=1.2)
    for t in leg.get_texts():
        t.set_color(T.ink)
    fig.tight_layout()
    fig.savefig(OUT / "18_applied_fix.png", facecolor=T.surface)
    plt.close(fig)


def main() -> int:
    plt.rcParams["font.family"] = FONTS
    chunking()
    applied_fix()
    print("Wrote 17_chunking.png and 18_applied_fix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
