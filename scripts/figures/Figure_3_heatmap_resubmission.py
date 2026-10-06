"""
Figure 3 (resubmission): per-task MTL-PepPred performance heatmap, mean ± SD of 5 runs.

Reads  ../Benchmark Summary - resubmission.xlsx   (run update_benchmark_resubmission.py first)
Writes ../results/Figure_3_updated.png  and  .tif  (600 DPI)

Cell = mean of 5 seeds; small text under it = SD across seeds.
PR-AUC column is added automatically once MTL-PepPred PR-AUC is filled in (after C0).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
XLSX = HERE.parents[1] / "results" / "Benchmark Summary - resubmission.xlsx"
OUT = HERE.parents[1] / "results" / "Figure_3_updated"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8})

NAME_MAP = {
    "ACE inhibitory activity": "ACE Inh.",
    "DPP IV inhibitory activity": "DPPIV Inh.",
    "Bitter": "Bitter",
    "Umami": "Umami",
    "Antimicrobial activity": "Antimicrobial",
    "Antimalarial activity (alternative dataset)": "Antimal. (alt)",
    "Antimalarial activity (main dataset)": "Antimal. (main)",
    "Quorum sensing activity": "Quorum",
    "Anticancer activity (alternative dataset)": "Anticancer (alt)",
    "Anticancer activity (main dataset)": "Anticancer (main)",
    "Anti-MRSA strains activity": "Anti-MRSA",
    "Tumor T cell antigens": "TTCA",
    "Blood-Brain Barrier": "BBP",
    "Antiparasitic activity": "Antiparasitic",
    "Neuropeptide": "Neuro.",
    "Antibacterial activity": "Antibact.",
    "Antifungal activity": "Antifung.",
    "Antiviral activity": "Antiviral",
    "Toxicity": "Toxicity",
    "Antioxidant activity": "Antioxidant",
    "Signal_peptide": "Signal Pep.*",
}

df = pd.read_excel(XLSX, sheet_name="Sheet1")
df["Bioactivity"] = df["Bioactivity"].ffill()
df["Model"] = df["Model"].astype(str).str.strip()
mtl = df[df["Model"] == "MTL-PepPred"].copy()

ALL = ["ACC", "AUC", "PR-AUC", "MCC"]
for m in ALL:
    for c in (m, f"{m}_SD"):
        mtl[c] = pd.to_numeric(mtl[c], errors="coerce")
metrics = [m for m in ALL if mtl[m].notna().all()]
if "PR-AUC" not in metrics:
    print("NOTE: MTL-PepPred PR-AUC not available yet (C0 pending) -> PR-AUC column omitted.")

mean = mtl[metrics].values
sd = mtl[[f"{m}_SD" for m in metrics]].values
names = [NAME_MAP.get(t, t) for t in mtl["Bioactivity"]]
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
