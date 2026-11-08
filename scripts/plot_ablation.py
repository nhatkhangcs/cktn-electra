#!/usr/bin/env python3
"""Recreate the ablation bar chart (Figure 4) from the paper's reported values.

Example:
    python scripts/plot_ablation.py --out assets/ablation.png
"""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cktn_electra.utils import REPO_ROOT  # noqa: E402

# Values reported in Figure 4 of the paper.
VALUES = {
    "CKTN-ELECTRA": {"Accuracy": 0.9214, "Macro-F1": 0.7103, "MRR@10": 0.7718, "Recall@10": 0.9039},
    "No script-aware": {"Accuracy": 0.0304, "Macro-F1": 0.0012, "MRR@10": 0.7678, "Recall@10": 0.9021},
    "No linear lambda": {"Accuracy": 0.0304, "Macro-F1": 0.0012, "MRR@10": 0.7712, "Recall@10": 0.9047},
}
PANELS = [("Category Classification", ["Accuracy", "Macro-F1"]), ("Information Retrieval", ["MRR@10", "Recall@10"])]
COLORS = ["#1f6f78", "#c65d32"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "ablation.png")
    args = parser.parse_args()

    methods = list(VALUES)
    x = np.arange(len(methods))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (title, metrics) in zip(axes, PANELS):
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + (i - 0.5) * width, [VALUES[m][metric] for m in methods], width,
                          label=metric, color=COLORS[i], edgecolor="black", linewidth=0.6)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax.set_xticks(x, [fill(m, 14) for m in methods])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
