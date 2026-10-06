"""Figure 4 (R1 #1): ROC and PR curves, seed-42 reference model, all-task micro-average plus
three representative tasks spanning the performance range (Signal_peptide near-ceiling,
Toxicity mid-range, Antimalarial-main most imbalanced / lowest PR-AUC, Anti_parasitic lowest
overall AUC).

Renamed from the supplementary "Figure S4" to a main-text figure (decided 2026-09-21, §4B) --
content is unchanged from the version accepted in the fourth QA pass, only the file name and
output name changed. The original Figure_S4_roc_pr_resubmission.py / Figure_S4_roc_pr.png/.tif
are left in place, not deleted.

Reads  ../results/C0_seed42_test_probs.csv   (task, sequence, label, prob)
Writes ../results/Figure_4_roc_pr.png / .tif  (600 DPI)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
OUT = RESULTS / "Figure_4_roc_pr"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

df = pd.read_csv(RESULTS / "C0_seed42_test_probs.csv")

REPRESENTATIVE = ["Signal_peptide", "Toxicity", "Antimalarial", "Anti_parasitic"]
COLORS = {"micro-average (all 21 tasks)": "black", "Signal_peptide": "#1F6F6A",
          "Toxicity": "#D9844A", "Antimalarial": "#79C2B6", "Anti_parasitic": "#8B2E2E"}
# same short display names as Figure_3_heatmap_resubmission.py / S2 / S3
DISPLAY = {"Signal_peptide": "Signal Pep.*", "Toxicity": "Toxicity",
           "Antimalarial": "Antimal. (main)", "Anti_parasitic": "Antiparasitic"}
LABELS = {"micro-average (all 21 tasks)": "micro-average (all 21 tasks)", **DISPLAY}

series = {"micro-average (all 21 tasks)": (df["label"].values, df["prob"].values)}
for t in REPRESENTATIVE:
    sub = df[df["task"] == t]
    series[t] = (sub["label"].values, sub["prob"].values)

fig, axes = plt.subplots(1, 2, figsize=(160 * MM, 95 * MM))
axR, axP = axes

# same 5 curves/colours appear in both panels -- one shared legend combining both numbers
# per series, placed well below both x-axis labels, instead of a separate legend per panel
handles, combo_labels = [], []
for name, (y, p) in series.items():
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    lw = 1.6 if name.startswith("micro") else 1.0
    ls = "-" if name.startswith("micro") else "--"
    (line,) = axR.plot(fpr, tpr, color=COLORS[name], linewidth=lw, linestyle=ls)

    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    axP.plot(rec, prec, color=COLORS[name], linewidth=lw, linestyle=ls)

    handles.append(line)
    combo_labels.append(f"{LABELS.get(name, name)} (AUC {auc:.3f} / PR-AUC {ap:.3f})")

axR.plot([0, 1], [0, 1], color="grey", linewidth=0.6, linestyle=":", zorder=1)
axR.set_xlabel("False positive rate", fontsize=8.5)
axR.set_ylabel("True positive rate", fontsize=8.5)
axR.set_title("ROC", fontsize=9)
axR.spines[["top", "right"]].set_visible(False)

axP.set_xlabel("Recall", fontsize=8.5)
axP.set_ylabel("Precision", fontsize=8.5)
axP.set_title("Precision-Recall", fontsize=9)
axP.spines[["top", "right"]].set_visible(False)
axP.set_ylim(0, 1.05)

fig.tight_layout(rect=(0, 0.20, 1, 1))
# placed via fig-level coordinates, well clear of both axes' x-axis labels above it
fig.legend(handles, combo_labels, loc="lower center", fontsize=6.5, frameon=False,
           ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved {OUT.name}.png / .tif")
for name, (y, p) in series.items():
    print(f"  {name:30s} AUC={roc_auc_score(y, p):.4f}  PR-AUC={average_precision_score(y, p):.4f}  n={len(y)}")
