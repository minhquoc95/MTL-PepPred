"""Aggregate training outputs into the tables the manuscript needs.

Reads the `test_results.json` files the training script writes
(keys: test_avg_acc/auc/mcc, test_metrics[task]={accuracy,auc,pr_auc?,mcc,...}).

B1 - multi-seed mean +/- SD:
  python aggregate.py b1 --glob "checkpoints/seed_*/test_results.json" \
      --variances_glob "checkpoints/seed_*/task_variances.json" --out_dir <results_dir>

B2 - single-task transfer deltas (MTL minus single-task):
  python aggregate.py b2 --mtl checkpoints/seed_42/test_results.json \
      --single_glob "checkpoints/single_*/test_results.json" --out_dir <results_dir>
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = ["accuracy", "auc", "pr_auc", "mcc"]


def load(path):
    with open(path) as f:
        return json.load(f)


def _get(task_metrics, key):
    # tolerate naming variants
    aliases = {"accuracy": ["accuracy", "acc"], "auc": ["auc"],
               "pr_auc": ["pr_auc", "prauc", "auprc"], "mcc": ["mcc"]}
    for k in aliases[key]:
        if k in task_metrics:
            return float(task_metrics[k])
    return float("nan")


def b1(args):
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    overall = {m: [] for m in ["test_avg_acc", "test_avg_auc", "test_avg_mcc"]}
    per_task = {}
    for f in files:
        d = load(f)
        for m in overall:
            if m in d:
                overall[m].append(float(d[m]))
        for task, tm in d.get("test_metrics", {}).items():
            per_task.setdefault(task, {m: [] for m in METRICS})
            for m in METRICS:
                v = _get(tm, m)
                if not np.isnan(v):
                    per_task[task][m].append(v)

    os.makedirs(args.out_dir, exist_ok=True)
    ov = {k: (round(np.mean(v), 4), round(np.std(v, ddof=1) if len(v) > 1 else 0.0, 4))
          for k, v in overall.items() if v}
    pd.DataFrame([{"metric": k, "mean": m, "sd": s} for k, (m, s) in ov.items()]).to_csv(
        os.path.join(args.out_dir, "B1_multiseed_overall.csv"), index=False)

    rows = []
    for task, mm in per_task.items():
        row = {"task": task, "n_seeds": max(len(v) for v in mm.values())}
        for m in METRICS:
            v = mm[m]
            row[f"{m}_mean"] = round(np.mean(v), 4) if v else float("nan")
            row[f"{m}_sd"] = round(np.std(v, ddof=1), 4) if len(v) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "B1_multiseed_pertask.csv"), index=False)

    if args.variances_glob:
        vfiles = sorted(glob.glob(args.variances_glob))
        acc = {}
        for vf in vfiles:
            for t, s in load(vf).items():
                acc.setdefault(t, []).append(float(s))
        if acc:
            vr = [{"task": t, "sigma2_mean": round(np.mean(s), 4),
                   "sigma2_sd": round(np.std(s, ddof=1) if len(s) > 1 else 0.0, 4)}
                  for t, s in acc.items()]
            vr.sort(key=lambda r: -r["sigma2_mean"])
            pd.DataFrame(vr).to_csv(os.path.join(args.out_dir, "B4_task_variances.csv"), index=False)
            print("wrote B4_task_variances.csv (sorted, highest sigma^2 first)")

    print("B1 overall:", ov)


def b2(args):
    mtl = load(args.mtl).get("test_metrics", {})
    rows = []
    for f in sorted(glob.glob(args.single_glob)):
        d = load(f).get("test_metrics", {})
        # single-task run has exactly one task in test_metrics
        for task, tm in d.items():
            if task not in mtl:
                continue
            for m in ["auc", "mcc", "accuracy"]:
                mtl_v = _get(mtl[task], m)
                sin_v = _get(tm, m)
                rows.append({"task": task, "metric": m,
                             "MTL": round(mtl_v, 4), "single_task": round(sin_v, 4),
                             "delta_MTL_minus_single": round(mtl_v - sin_v, 4)})
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "B2_transfer_deltas.csv"), index=False)
    pos = df[(df.metric == "auc") & (df.delta_MTL_minus_single > 0)]
    neg = df[(df.metric == "auc") & (df.delta_MTL_minus_single < 0)]
    print(f"B2: AUC positive transfer on {len(pos)} tasks, negative transfer on {len(neg)} tasks.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("b1"); p1.add_argument("--glob", required=True)
    p1.add_argument("--variances_glob", default=""); p1.add_argument("--out_dir", default=".")
    p2 = sub.add_parser("b2"); p2.add_argument("--mtl", required=True)
    p2.add_argument("--single_glob", required=True); p2.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    (b1 if args.cmd == "b1" else b2)(args)


if __name__ == "__main__":
    main()
