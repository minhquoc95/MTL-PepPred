"""
Build the resubmission benchmark workbook from the 5-seed B1 runs.

Input
  results/Benchmark Summary - new.xlsx          original Table 4 (single run, seed 42)
  results/C0_seed{42,1,2,3,4}_pertask_metrics.csv

Output
  results/Benchmark Summary - resubmission.xlsx
    Sheet1          same layout as the original; MTL-PepPred rows replaced by the
                    5-seed mean, SD in columns L-R. Baseline rows untouched.
    MTL_per_seed    per-task, per-seed raw values
    Notes           what changed

MTL-PepPred metrics per seed: ACC, AUC, PR-AUC, BACC, Sn, Sp and MCC are all read from
C0_seed*_pertask_metrics.csv (exp_seed_pertask_metrics.py: AMP-off inference, PR-AUC =
average precision), so every column of Table 4 comes from the same five evaluations.
Means and sample SDs (ddof=1) are stored at full precision and displayed at 2 dp.
"""
import csv
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


def load_seed_metrics():
    """(task, seed) -> every Table 4 metric, from C0_seed*_pertask_metrics.csv."""
    rows = {}
    for s in SEEDS:
        with open(RESULTS / f"C0_seed{s}_pertask_metrics.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["task"], s)] = {
                    "ACC": float(r["ACC"]), "AUC": float(r["AUC"]), "PR-AUC": float(r["PR_AUC"]),
                    "BACC": float(r["BACC"]), "Sn": float(r["Sn"]), "Sp": float(r["Sp"]),
                    "MCC": float(r["MCC"]), "prevalence": int(r["n_pos"]) / int(r["n"]),
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
                # stored at full precision, displayed at 2 dp: anything reading this sheet
                # and rounding again (Figure 3 prints 1 dp) must not round a rounded value
                c.value = round(mean, 6)
                c.number_format = "0.00"
                sdc = ws.cell(r, sd_cols[met])
                sdc.value, sdc.number_format = round(sd, 6), "0.00"
            c.fill = hl
        ws.cell(r, col["Reference"], "This work (mean of 5 seeds)")
        ws.cell(r, note_col, "mean of 5 runs")
        replaced += 1
    assert replaced == 21, replaced

    # The UniDL4BioPep antioxidant row in the source sheet is shifted by one column from AUC
    # onwards (it read AUC = PR-AUC = 80.4, MCC = 87.2). Restore it, checking the bad values
    # first so the fix can never land on the wrong row.
    current_bio, fixed = None, 0
    for r in range(2, ws.max_row + 1):
        current_bio = ws.cell(r, col["Bioactivity"]).value or current_bio
        if current_bio == "Antioxidant activity" and (ws.cell(r, col["Model"]).value or "").strip() == "UniDL4BioPep":
            bad = {m: ws.cell(r, col[m]).value for m in ("AUC", "PR-AUC", "MCC")}
            assert bad == {"AUC": 80.4, "PR-AUC": 80.4, "MCC": 87.2}, bad
            for m, v in {"AUC": 87.2, "PR-AUC": None, "BACC": 80.45, "Sn": 81, "Sp": 79.9, "MCC": 60.8}.items():
                ws.cell(r, col[m]).value = v  # ws.cell(r, c, None) would leave the old value
            fixed += 1
    assert fixed == 1, fixed

    # per-seed sheet
    ps = wb.create_sheet("MTL_per_seed")
    hdr = ["task", "Bioactivity", "seed"] + METRICS + ["prevalence (pos/total)"]
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
        "Built by scripts/update_benchmark_resubmission.py",
        "MTL-PepPred rows (green) = mean of 5 independent runs, seeds 42, 1, 2, 3, 4. SD (sample, ddof=1) in *_SD columns. Stored at full precision, displayed at 2 dp; Table 4 shows the 2-dp rounding.",
        "All baseline rows are unchanged from 'Benchmark Summary - new.xlsx', except the UniDL4BioPep antioxidant row, whose values from AUC onwards were shifted by one column in the source and are restored here.",
        "All MTL-PepPred metrics read from results/C0_seed{42,1,2,3,4}_pertask_metrics.csv (AMP-off inference; PR-AUC = average precision).",
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
