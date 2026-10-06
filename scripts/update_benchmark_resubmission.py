"""
Build the resubmission benchmark workbook from the 5-seed B1 runs.

Input
  ../Benchmark Summary - new.xlsx          original Table 4 (single run, seed 42)
  ../results/test_results_seed_{42,1,2,3,4}.json

Output
  ../Benchmark Summary - resubmission.xlsx
    Sheet1          same layout as the original; MTL-PepPred rows replaced by the
                    5-seed mean, SD in columns L-R. Baseline rows untouched.
    MTL_per_seed    per-task, per-seed raw values
    Notes           what changed and what is still pending

MTL-PepPred metrics per seed:
  ACC, AUC, MCC, Sn (= recall)  : read directly from test_results_seed_*.json
  Sp, BACC                       : derived exactly from ACC, precision, recall
                                   (no rounding, confusion matrix reconstructed)
  PR-AUC                         : from C0_seed{42,1,2,3,4}_pertask_metrics.csv (B6,
                                   2026-09-21) -- AMP-off inference on the seed 1-4 weights,
                                   which were never deleted (only checkpoint.pt was), so this
                                   is the exact trained models, not a retrain. sklearn
                                   average_precision_score, matching the manuscript's own
                                   Table 4 method (see the C0 changelog entries).
"""
import json
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill

HERE = Path(__file__).resolve().parent
RESUB = HERE.parent
SRC_XLSX = RESUB / "results" / "Benchmark Summary - new.xlsx"
OUT_XLSX = RESUB / "results" / "Benchmark Summary - resubmission.xlsx"
RESULTS = RESUB / "results"
SEEDS = [42, 1, 2, 3, 4]

# task key in the training code  ->  Bioactivity label in Table 4
TASK_MAP = {
    "ACE_inhibitory": "ACE inhibitory activity",
    "DPPIV_inhibitory": "DPP IV inhibitory activity",
    "Bitter": "Bitter",
    "Umami": "Umami",
    "Antimicrobial": "Antimicrobial activity",
    "Antimalarial_alt": "Antimalarial activity (alternative dataset)",
    "Antimalarial": "Antimalarial activity (main dataset)",
    "Quorum_sensing": "Quorum sensing activity",
    "Anticancer_alt": "Anticancer activity (alternative dataset)",
    "Anticancer": "Anticancer activity (main dataset)",
    "AntiMRSA": "Anti-MRSA strains activity",
    "TTCA": "Tumor T cell antigens",
    "BBP": "Blood-Brain Barrier",
    "Anti_parasitic": "Antiparasitic activity",
    "NeuroPred": "Neuropeptide",
    "Antibacterial": "Antibacterial activity",
    "Antifungal": "Antifungal activity",
    "Antiviral": "Antiviral activity",
    "Toxicity": "Toxicity",
    "Antioxidant": "Antioxidant activity",
    "Signal_peptide": "Signal_peptide",
}
METRICS = ["ACC", "AUC", "PR-AUC", "BACC", "Sn", "Sp", "MCC"]


def derive_sp(acc, prec, rec):
    """Specificity from accuracy, precision and recall.

    With prevalence pi = P/(P+N):  FP/N_total = pi*rec*(1-prec)/prec
    acc = 1 - pi + pi*rec*(2*prec-1)/prec  ->  solve for pi, then Sp.
    Returns (sp, pi).
    """
    if prec == 0 or rec == 0:
        return np.nan, np.nan
    k = rec * (2 * prec - 1) / prec
    pi = (1 - acc) / (1 - k)
    fp_rate_total = pi * rec * (1 - prec) / prec
    sp = 1 - fp_rate_total / (1 - pi)
    return sp, pi


def load_seed_metrics():
    """ACC/AUC/MCC/Sn from test_results_seed_*.json (train_mtl.py's own one-shot AMP-on
    evaluation); PR-AUC from C0_seed*_pertask_metrics.csv (B6, AMP-off, sklearn
    average_precision_score -- the seed-1..4 weights were never deleted, only checkpoint.pt
    was, so this is inference on the exact same models, not a retrain)."""
    import csv as _csv

    pr_auc_by_seed_task = {}
    for s in SEEDS:
        with open(RESULTS / f"C0_seed{s}_pertask_metrics.csv", newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                pr_auc_by_seed_task[(row["task"], s)] = float(row["PR_AUC"])

    rows = {}  # (task, seed) -> dict
    for s in SEEDS:
        d = json.loads((RESULTS / f"test_results_seed_{s}.json").read_text())
        for task, m in d["test_metrics"].items():
            sp, pi = derive_sp(m["accuracy"], m["precision"], m["recall"])
            rows[(task, s)] = {
                "ACC": m["accuracy"], "AUC": m["auc"], "MCC": m["mcc"],
                "Sn": m["recall"], "Sp": sp, "BACC": (m["recall"] + sp) / 2,
                "PR-AUC": pr_auc_by_seed_task.get((task, s), np.nan), "prevalence": pi,
            }
    return rows


def main():
    per_seed = load_seed_metrics()
    tasks = sorted({t for t, _ in per_seed})
    assert set(tasks) == set(TASK_MAP), f"task mismatch: {set(tasks) ^ set(TASK_MAP)}"

    summary = {}
    for t in tasks:
        summary[t] = {}
        for met in METRICS:
            v = np.array([per_seed[(t, s)][met] for s in SEEDS], dtype=float)
            if np.all(np.isnan(v)):
                summary[t][met] = (None, None)
            else:
                summary[t][met] = (100 * np.nanmean(v), 100 * np.nanstd(v, ddof=1))

    wb = openpyxl.load_workbook(SRC_XLSX)
    ws = wb["Sheet1"]
    col = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1) if ws.cell(1, c).value}

    # SD columns to the right of Reference
    sd_start = col["Reference"] + 1
    sd_cols = {}
    for i, met in enumerate(METRICS):
        c = sd_start + i
        ws.cell(1, c, f"{met}_SD").font = Font(bold=True)
        sd_cols[met] = c
    note_col = sd_start + len(METRICS)
    ws.cell(1, note_col, "Note").font = Font(bold=True)

    hl = PatternFill("solid", fgColor="E2EFDA")
    label_to_task = {v: k for k, v in TASK_MAP.items()}
    current_bio, replaced = None, 0
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, col["Bioactivity"]).value:
            current_bio = ws.cell(r, col["Bioactivity"]).value
        model = (ws.cell(r, col["Model"]).value or "").strip()
        if model != "MTL-PepPred":
            continue
        task = label_to_task[current_bio]
        for met in METRICS:
            mean, sd = summary[task][met]
            c = ws.cell(r, col[met])
            if mean is None:
                c.value = None
            else:
                c.value = round(mean, 2)
                ws.cell(r, sd_cols[met], round(sd, 2))
            c.fill = hl
        ws.cell(r, col["Reference"], "This work (mean of 5 seeds)")
        ws.cell(r, note_col, "mean of 5 runs")
        replaced += 1
    assert replaced == 21, replaced

    # per-seed sheet
    ps = wb.create_sheet("MTL_per_seed")
    hdr = ["task", "Bioactivity", "seed"] + METRICS + ["derived prevalence (pos/total)"]
    ps.append(hdr)
    for c in range(1, len(hdr) + 1):
        ps.cell(1, c).font = Font(bold=True)
    for t in tasks:
        for s in SEEDS:
            m = per_seed[(t, s)]
            ps.append([t, TASK_MAP[t], s] +
                      [None if np.isnan(m[k]) else round(100 * m[k], 2) for k in METRICS] +
                      [round(m["prevalence"], 4)])

    nt = wb.create_sheet("Notes")
    for line in [
        "Built by Coding/update_benchmark_resubmission.py",
        "MTL-PepPred rows (green) = mean of 5 independent runs, seeds 42, 1, 2, 3, 4 (B1). SD (sample, ddof=1) in *_SD columns.",
        "All baseline rows are unchanged from 'Benchmark Summary - new.xlsx'.",
        "ACC, AUC, MCC, Sn read from results/test_results_seed_*.json. Sp and BACC derived exactly from accuracy, precision and recall.",
        "PR-AUC (mean of 5 runs, with SD) read from results/C0_seed{42,1,2,3,4}_pertask_metrics.csv (B6, 2026-09-21): AMP-off inference on the seed 1-4 weights, which were never deleted, using sklearn average_precision_score to match the manuscript's own method.",
        "Check: UniDL4BioPep Antioxidant row in the original sheet reads ACC=AUC=PR-AUC=80.4 and MCC=87.2, which looks like a data-entry error. Verify against Du et al. 2023 before using it in the figures.",
    ]:
        nt.append([line])

    wb.save(OUT_XLSX)

    # console summary
    print(f"Saved: {OUT_XLSX.name}")
    print(f"{'task':18s} {'ACC':>12s} {'AUC':>12s} {'MCC':>12s} {'Sp':>12s} {'prev':>6s}")
    for t in tasks:
        f = lambda k: f"{summary[t][k][0]:6.2f}±{summary[t][k][1]:4.2f}"
        print(f"{t:18s} {f('ACC'):>12s} {f('AUC'):>12s} {f('MCC'):>12s} {f('Sp'):>12s} "
              f"{per_seed[(t, 42)]['prevalence']:6.3f}")
    for k in ["ACC", "AUC", "MCC", "BACC"]:
        print(f"mean over 21 tasks {k}: {np.mean([summary[t][k][0] for t in tasks]):.2f}")


if __name__ == "__main__":
    main()
