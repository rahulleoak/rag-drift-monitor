"""Draw the agent-flow diagram for the presentation (dark theme) into docs/presentation/.

Usage (from the repository root):
    python scripts/presentation_diagram.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftmon.dashboard.charts import DARK as T  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "presentation" / "00_architecture.png"


def main() -> int:
    fig, ax = plt.subplots(figsize=(13.8, 5.4), dpi=160)
    fig.patch.set_facecolor(T.surface)
    ax.set_facecolor(T.surface)
    ax.axis("off")
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.4)

    def box(x, y, w, h, kicker, title, sub, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=1.6, edgecolor=edge, facecolor="#16202F"))
        if kicker:
            ax.text(x + w / 2, y + h * 0.78, kicker, ha="center", va="center", color=T.muted, fontsize=9)
        ax.text(x + w / 2, y + h * (0.52 if kicker else 0.62), title, ha="center", va="center", color=T.ink,
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * (0.25 if kicker else 0.3), sub, ha="center", va="center", color=T.ink_2,
                fontsize=9.5)

    def arrow(x0, y0, x1, y1, dashed=False, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
                                     color=T.ink_2, linestyle="--" if dashed else "-",
                                     connectionstyle=f"arc3,rad={rad}"))

    y, h, w, gap = 2.0, 1.4, 2.35, 0.4
    xs = [0.25 + i * (w + gap) for i in range(5)]
    agents = [("Agent A", "Data Monitoring", "what came in"), ("Agent B", "Semantic Drift", "what changed, how much"),
              ("Agent C", "Retrieval", "stale vs refreshed search"),
              ("Agent D", "RAG Impact", "stale vs refreshed answers"),
              ("Agent E", "Recommendation", "actions + evidence")]
    for (k, t, s), x in zip(agents, xs):
        box(x, y, w, h, k, t, s, T.series[0] if k == "Agent E" else T.axis)
    for i in range(4):
        arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    arrow(xs[3] + 0.5, y, xs[2] + w - 0.5, y, dashed=True, rad=-0.5)
    ax.text((xs[2] + xs[3] + w) / 2, y - 0.62, "borderline section: more evidence (once)", ha="center",
            color=T.muted, fontsize=9)
    box(0.25, 4.15, 4.9, 0.9, None, "Archive, replayed year by year", "HuffPost 2016–2022 · simulated questions",
        T.series[1])
    arrow(1.4, 4.15, 1.4, y + h)
    box(xs[4] + w - 4.9, 4.15, 4.9, 0.9, None, "Long documents", "Wikipedia snapshots · changed facts · chunking",
        T.series[1])
    arrow(xs[4] + w / 2, 4.15, xs[4] + w / 2, y + h)
    box(xs[4], 0.15, w, 0.95, None, "Saved run", "reports · dashboard", T.axis)
    arrow(xs[4] + w / 2, y, xs[4] + w / 2, 1.1)
    ax.text(0.25, 0.75, "Each agent passes a typed report to the next (LangGraph); routing is decided in code.",
            color=T.ink_2, fontsize=10, va="center")
    ax.text(0.25, 0.35, "The LLM writes questions, answers and explanations, and judges; rules in code decide.",
            color=T.ink_2, fontsize=10, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=T.surface, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
