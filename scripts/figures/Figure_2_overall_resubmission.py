"""
Figure 2 (resubmission): overall performance, MTL-PepPred vs UniDL4BioPep vs PDeepPP.

Reads  ../Benchmark Summary - resubmission.xlsx   (run update_benchmark_resubmission.py first)
Writes ../results/Figure_2_updated.png  and  .tif  (600 DPI, 140 mm = 1.5 column width)

Bar  = mean over tasks (MTL-PepPred: each task value is already the mean of 5 seeds).
Error bar = SEM across tasks, same definition for all three models.
PR-AUC is drawn only if MTL-PepPred PR-AUC values are present (after C0).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
XLSX = HERE.parents[1] / "results" / "Benchmark Summary - resubmission.xlsx"
OUT = HERE.parents[1] / "results" / "Figure_2_updated"

# Original colours of the submitted Figure 2, restored at the user's request (2026-09-21).
COLORS = {"MTL-PepPred": "#89B0AE", "UniDL4BioPep": "#BEE3DB", "PDeepPP": "#FFD6BA"}
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

df = pd.read_excel(XLSX, sheet_name="Sheet1")
df["Bioactivity"] = df["Bioactivity"].ffill()
df["Model"] = df["Model"].astype(str).str.strip()
ALL_METRICS = ["ACC", "AUC", "PR-AUC", "MCC"]
for m in ALL_METRICS:
    df[m] = pd.to_numeric(df[m], errors="coerce")
    if df[m].max() <= 1.5:          # accept fractions or percentages
        df[m] *= 100

models = ["MTL-PepPred", "UniDL4BioPep", "PDeepPP"]
sub = {m: df[df["Model"] == m] for m in models}

metrics = [m for m in ALL_METRICS if sub["MTL-PepPred"][m].notna().sum() == 21]
if "PR-AUC" not in metrics:
    print("NOTE: MTL-PepPred PR-AUC not available yet (C0 pending) -> PR-AUC group omitted.")


def mean_sem(s):
    v = s.dropna().values
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


stats = {m: [mean_sem(sub[m][k]) for k in metrics] for m in models}
for m in models:
    print(f"{m:13s} " + "  ".join(f"{k}={s[0]:.2f}±{s[1]:.2f} (n={s[2]})"
                                     for k, s in zip(metrics, stats[m])))

# Layout follows the originally submitted Figure 2: grouped bars without outlines,
# SEM error bars, bold value labels above each bar, framed legend upper left,
# full axis box, dashed horizontal grid, y-axis from 65 to 102.
fig, ax = plt.subplots(figsize=(140 * MM, 95 * MM))
x = np.arange(len(metrics))
w = 0.25
for i, m in enumerate(models):
    means = [s[0] for s in stats[m]]
    sems = [s[1] for s in stats[m]]
    n = stats[m][0][2]
    label = f"MTL-PepPred (Ours, n={n})" if m == "MTL-PepPred" else f"{m} (n={n})"
    bars = ax.bar(x + (i - 1) * w, means, w, color=COLORS[m], yerr=sems, capsize=3,
                  error_kw={"elinewidth": 0.8, "capthick": 0.8}, label=label, zorder=3)
    for b, mu, se in zip(bars, means, sems):
        ax.text(b.get_x() + b.get_width() / 2, mu + se + 0.6, f"{mu:.1f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8.5)
ax.set_ylabel("Score (%)", fontsize=9.5)
ax.set_xlabel("Metric", fontsize=9.5)
ax.set_ylim(65, 107)
ax.set_yticks(range(65, 101, 5))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3, zorder=0)
ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.95, edgecolor="#cccccc")

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tif"), dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved {OUT.name}.png / .tif (600 DPI)")
