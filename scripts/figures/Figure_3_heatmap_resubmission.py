"""
Figure 3 (resubmission): per-task MTL-PepPred performance heatmap, mean ± SD of 5 runs.

Reads  ../results/C0_seed{42,1,2,3,4}_pertask_metrics.csv
Writes ../results/Figure_3_updated.png  and  .tif  (600 DPI)

Cell = mean of 5 seeds; small text under it = sample SD across seeds. Computed from the
per-seed values rather than read from the benchmark workbook, whose 2-dp rounding would be
rounded a second time to the 1 dp shown here (e.g. 0.4461 -> 0.45 -> 0.5 instead of 0.4).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
OUT = RESULTS / "Figure_3_updated"
SEEDS = [42, 1, 2, 3, 4]
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8})

NAME_MAP = {
    "ACE_inhibitory": "ACE Inh.", "DPPIV_inhibitory": "DPPIV Inh.", "Bitter": "Bitter",
    "Umami": "Umami", "Antimicrobial": "Antimicrobial", "Antimalarial_alt": "Antimal. (alt)",
    "Antimalarial": "Antimal. (main)", "Quorum_sensing": "Quorum",
    "Anticancer_alt": "Anticancer (alt)", "Anticancer": "Anticancer (main)",
    "AntiMRSA": "Anti-MRSA", "TTCA": "TTCA", "BBP": "BBP", "Anti_parasitic": "Antiparasitic",
    "NeuroPred": "Neuro.", "Antibacterial": "Antibact.", "Antifungal": "Antifung.",
    "Antiviral": "Antiviral", "Toxicity": "Toxicity", "Antioxidant": "Antioxidant",
    "Signal_peptide": "Signal Pep.*",
}

per_seed = pd.concat([pd.read_csv(RESULTS / f"C0_seed{s}_pertask_metrics.csv") for s in SEEDS])
metrics = ["ACC", "AUC", "PR-AUC", "MCC"]
cols = ["ACC", "AUC", "PR_AUC", "MCC"]
grouped = per_seed.groupby("task")[cols]
assert (grouped.size() == len(SEEDS)).all(), "every task needs all five seeds"
mean = 100 * grouped.mean().values
sd = 100 * grouped.std(ddof=1).values
names = [NAME_MAP[t] for t in grouped.mean().index]
order = np.argsort(mean.mean(axis=1))[::-1]
mean, sd, names = mean[order], sd[order], [names[i] for i in order]

cmap = LinearSegmentedColormap.from_list("white_green_navy", ["#FFFFFF", "#70AD47", "#1F3864"])
fig, ax = plt.subplots(figsize=((40 + 22 * len(metrics)) * MM, 170 * MM))
im = ax.imshow(mean, cmap=cmap, aspect="auto", vmin=30, vmax=100)

ax.set_frame_on(False)
ax.tick_params(which="both", length=0)
ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=8.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)

for i in range(len(names)):
    for j in range(len(metrics)):
        v, s = mean[i, j], sd[i, j]
        col = "white" if v > 75 else "black"
        ax.text(j, i - 0.13, f"{v:.1f}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=col)
        if not np.isnan(s):
            ax.text(j, i + 0.27, f"±{s:.1f}", ha="center", va="center", fontsize=5.5, color=col)

cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
cbar.set_label("Score (%)", fontsize=8.5)
cbar.ax.tick_params(labelsize=7.5)
cbar.outline.set_linewidth(0.5)
ax.set_xlabel("Evaluation metrics (mean ± SD, 5 runs)", fontsize=8.5)
ax.set_ylabel("Bioactivities", fontsize=8.5)

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tif"), dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved {OUT.name}.png / .tif (600 DPI), {len(names)} tasks x {len(metrics)} metrics")
