"""Three charts, saved as PNG files. Plain matplotlib, no display needed."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"A": "#1f77b4", "B": "#d95f02", "C": "#2ca02c", "Unknown": "#7f7f7f"}


def scatter_hours_final(data, corr, out: Path) -> None:
    sub = data[["hours", "final", "section"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sec, g in sub.groupby("section"):
        ax.scatter(g["hours"], g["final"], s=18, alpha=0.75, color=COLORS.get(sec, "#7f7f7f"), label=f"Section {sec}" if sec != "Unknown" else "Section unknown")
    slope, intercept = np.polyfit(sub["hours"], sub["final"], 1)
    xs = np.linspace(sub["hours"].min(), sub["hours"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="black", lw=1.5, label=f"straight-line fit (+{slope:.2f} per hour)")
    ax.set_xlabel("Study hours per week")
    ax.set_ylabel("Final score")
    ax.set_title(f"Hours vs final score (n={corr['n']}; Pearson {corr['pearson']:.2f}, Spearman {corr['spearman']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def section_means(table, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(table))
    err = [table["mean_final"] - table["ci_low"], table["ci_high"] - table["mean_final"]]
    ax.errorbar(x, table["mean_final"], yerr=err, fmt="o", capsize=6, color="black")
    for i, row in table.reset_index(drop=True).iterrows():
        ax.annotate(f"n={int(row['n'])}", (i, row["ci_low"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xticks(x, [f"Section {s}" for s in table["section"]])
    ax.set_ylabel("Average final score (95% interval)")
    ax.set_title("Average final score by section")
    ax.margins(y=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def cost_curves(curves: dict[float, list[float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for rate, costs in curves.items():
        ax.plot(range(len(costs)), costs, label=f"rate {rate}")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title("Gradient descent: cost against step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
