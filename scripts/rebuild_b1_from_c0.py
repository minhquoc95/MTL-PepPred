"""Five-run means for Table 4 / Figures 2-3: build B1_multiseed_overall.csv and
B1_multiseed_pertask.csv from the five per-seed C0_seed{42,1,2,3,4}_pertask_metrics.csv files
(written by exp_seed_pertask_metrics.py: AMP off, PR-AUC = average precision).

Overall = mean of the 21 per-task values within each seed, then mean and SD (ddof=1) of
those five per-seed values. Per-task = mean and SD across the five seeds.

Usage (from the repository root):
  python scripts/rebuild_b1_from_c0.py --results_dir results
"""
import argparse
import csv
import statistics as st
from pathlib import Path

SEEDS = [42, 1, 2, 3, 4]
METRICS = ["ACC", "AUC", "PR_AUC", "MCC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    results = Path(ap.parse_args().results_dir)

    per_task, seed_vals = {}, {s: {m: [] for m in METRICS} for s in SEEDS}
    for seed in SEEDS:
        with open(results / f"C0_seed{seed}_pertask_metrics.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                per_task.setdefault(row["task"], {m: [] for m in METRICS})
                for m in METRICS:
                    per_task[row["task"]][m].append(float(row[m]))
                    seed_vals[seed][m].append(float(row[m]))
    assert all(len(v["ACC"]) == len(SEEDS) for v in per_task.values()), "a task is missing a seed"

    with open(results / "B1_multiseed_pertask.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_seeds", "accuracy_mean", "accuracy_sd", "auc_mean", "auc_sd",
                    "pr_auc_mean", "pr_auc_sd", "mcc_mean", "mcc_sd"])
        for t in sorted(per_task):
            row = [t, len(SEEDS)]
            for m in METRICS:
                row += [round(st.mean(per_task[t][m]), 4), round(st.stdev(per_task[t][m]), 4)]
            w.writerow(row)

    names = {"ACC": "test_avg_acc", "AUC": "test_avg_auc", "PR_AUC": "test_avg_pr_auc", "MCC": "test_avg_mcc"}
    with open(results / "B1_multiseed_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "sd"])
        for m in METRICS:
            vals = [st.mean(seed_vals[s][m]) for s in SEEDS]
            w.writerow([names[m], round(st.mean(vals), 4), round(st.stdev(vals), 4)])
            print(f"{m:7s} {st.mean(vals):.4f} +/- {st.stdev(vals):.4f}")


if __name__ == "__main__":
    main()
