"""Figure 5 (R2 #4 panel a supports #5; R1 #6 / R2 #5 panel b): merged two-panel figure,
decided 2026-09-21 (README §4B) to replace the separate S2 (TUM variance) and S3 (transfer
deltas) supplementary figures with one main-text figure sharing a single task ordering.

(a) Learned TUM per-task variance sigma^2, mean +/- SD of 5 seeds (content of old S2).
(b) Single-task transfer effect, delta AUC and delta MCC in percentage points, MTL minus
    single-task (content of old S3).

Both panels use the SAME row order -- sorted by sigma^2 descending -- so the reader can see
whether high-uncertainty tasks are also the ones that transfer worst. Y-tick labels (task
display names) appear once, on panel (a) only. AntiMRSA and Signal_peptide (R2 #5's named
ceiling-effect tasks) are starred with one shared footnote.

Reads  ../results/B4_task_variances.csv, ../results/B2_transfer_deltas.csv
Writes ../results/Figure_5_transfer_variance.png / .tif  (600 DPI)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
OUT = RESULTS / "Figure_5_transfer_variance"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# same short display names as Figure_3_heatmap_resubmission.py / S2 / S3
DISPLAY = {
    "ACE_inhibitory": "ACE Inh.", "DPPIV_inhibitory": "DPP-IV Inh.", "Bitter": "Bitter",
    "Umami": "Umami", "Antimicrobial": "Antimicrobial", "Antimalarial_alt": "Antimal. (alt)",
    "Antimalarial": "Antimal. (main)", "Quorum_sensing": "Quorum", "Anticancer_alt": "Anticancer (alt)",
    "Anticancer": "Anticancer (main)", "AntiMRSA": "Anti-MRSA", "TTCA": "TTCA", "BBP": "BBP",
    "Anti_parasitic": "Antiparasitic", "NeuroPred": "Neuro.", "Antibacterial": "Antibact.",
    "Antifungal": "Antifung.", "Antiviral": "Antiviral", "Toxicity": "Toxicity",
    "Antioxidant": "Antioxidant", "Signal_peptide": "Signal Pep.*",
}
CEILING = {"AntiMRSA", "Signal_peptide"}

var = pd.read_csv(RESULTS / "B4_task_variances.csv").set_index("task")
tdf = pd.read_csv(RESULTS / "B2_transfer_deltas.csv")
auc = tdf[tdf["metric"] == "auc"].set_index("task")["delta_MTL_minus_single"]
mcc = tdf[tdf["metric"] == "mcc"].set_index("task")["delta_MTL_minus_single"]

# single shared order for BOTH panels: sigma^2 ascending -> barh's default puts the largest
# value at the TOP of the chart, matching the old S2 figure's convention exactly
tasks = var.sort_values("sigma2_mean", ascending=True).index.tolist()
sig_mean = var.loc[tasks, "sigma2_mean"].values
sig_sd = var.loc[tasks, "sigma2_sd"].values
auc_pp = (auc.loc[tasks] * 100).values
mcc_pp = (mcc.loc[tasks] * 100).values


def label_for(t):
    name = DISPLAY.get(t, t).rstrip("*").rstrip()
    return f"{name} *" if t in CEILING else name


labels = [label_for(t) for t in tasks]
y = np.arange(len(tasks))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(190 * MM, 172 * MM))

# ---- panel (a): TUM variance ----
axA.barh(y, sig_mean, xerr=sig_sd, color="#1F6F6A", edgecolor="black", linewidth=0.4,
         error_kw={"elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6}, zorder=3)
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=8.5)
axA.set_xlabel(r"Learned TUM variance $\sigma^2$ (mean $\pm$ SD, 5 seeds)", fontsize=9.5)
axA.tick_params(axis="x", labelsize=8.5)
axA.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
axA.spines[["top", "right"]].set_visible(False)
axA.set_ylim(-0.8, len(tasks) - 0.2)
axA.text(-0.02, 1.02, "a", transform=axA.transAxes, fontsize=12, fontweight="bold", va="bottom")

# ---- panel (b): transfer deltas ----
h = 0.36
axB.barh(y + h / 2, auc_pp, h * 0.92, color="#1F6F6A", edgecolor="black", linewidth=0.3,
         label=r"$\Delta$AUC (MTL $-$ single-task)", zorder=3)
axB.barh(y - h / 2, mcc_pp, h * 0.92, color="#D9844A", edgecolor="black", linewidth=0.3,
         label=r"$\Delta$MCC (MTL $-$ single-task)", zorder=3)
axB.axvline(0, color="black", linewidth=0.8, zorder=4)
axB.set_yticks(y)
axB.set_yticklabels(labels, fontsize=8.5)  # task names repeated on (b) at the supervisor's request
axB.set_xlabel(r"Transfer effect, percentage points (MTL $-$ single-task)", fontsize=9.5)
axB.tick_params(axis="x", labelsize=8.5)
axB.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
axB.spines[["top", "right"]].set_visible(False)
axB.set_ylim(-0.8, len(tasks) - 0.2)
axB.text(-0.02, 1.02, "b", transform=axB.transAxes, fontsize=12, fontweight="bold", va="bottom")
axB.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08),
           bbox_transform=axB.transAxes, ncol=2)
axB.text(0.5, -0.155, "* ceiling-effect tasks (single-task AUC already above 0.99)",
         transform=axB.transAxes, fontsize=8, style="italic", ha="center", va="top")

fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved {OUT.name}.png / .tif -- {len(tasks)} tasks, shared order by sigma^2 descending "
      f"(top={tasks[-1]}, bottom={tasks[0]})")
