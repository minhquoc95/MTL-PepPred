"""Fig S1 (R2 #6): pooled training-sequence length histogram, p90/p99 marked.

Reads  ../results/A1_lengths_hist.csv, ../results/A1_lengths_summary.csv
Writes ../results/Figure_S1_length.png / .tif  (600 DPI)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
OUT = RESULTS / "Figure_S1_length"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

hist = pd.read_csv(RESULTS / "A1_lengths_hist.csv")
summary = pd.read_csv(RESULTS / "A1_lengths_summary.csv").iloc[0]
p90, p99 = summary["p90"], summary["p99"]

fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
ax.bar(hist["length"], hist["count"], width=1.0, color="#1F6F6A", edgecolor="none", zorder=3)
ax.axvline(p90, color="#D9844A", linestyle="--", linewidth=1.0, zorder=4)
ax.axvline(p99, color="#8B2E2E", linestyle="--", linewidth=1.0, zorder=4)
ax.axvline(128, color="black", linestyle=":", linewidth=1.0, zorder=4)
ymax = hist["count"].max()
ax.set_ylim(0, ymax * 1.28)
# p90 and MAX_LENGTH sit only 7 residues apart -- stack at different heights and push
# each label's text away from its own line (p90 label to the left, MAX_LENGTH to the
# right) so they cannot merge even though the two vertical lines are close together.
ax.text(p90 - 3, ymax * 1.22, f"p90={p90:.0f}", color="#D9844A", ha="right", va="top",
        fontsize=7, fontweight="bold")
ax.text(128 + 3, ymax * 1.08, "max=128", color="black", ha="left", va="top", fontsize=7)
ax.text(p99 + 3, ymax * 1.22, f"p99={p99:.0f}", color="#8B2E2E", ha="left", va="top",
        fontsize=7, fontweight="bold")

ax.set_xlabel("Sequence length (residues)", fontsize=9)
ax.set_ylabel("Count (pooled training sequences, n=59,997)", fontsize=8.5)
ax.set_xlim(0, 260)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved {OUT.name}.png / .tif -- p90={p90:.0f} p99={p99:.0f} (MAX_LENGTH=128 truncates between p95 and p99)")
