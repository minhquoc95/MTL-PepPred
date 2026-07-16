"""
Ablation Study Report Generator
Reads results from all ablation variant checkpoints and prints a comparison table.

Usage:
    python ablation_report.py
    python ablation_report.py --results_dir checkpoints_run2
    python ablation_report.py --csv_out per_task.csv --json_out per_task.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Ablation variants to collect (checkpoint subfolder name)
VARIANTS = [
    "full_model",
    "no_cnn",
    "no_transformer",
    "no_tum",
    "esm_ratio_1p0",
    "esm_ratio_0p5",
    "unfreeze_esm",
    "transformer_2L",
]


def load_variant(variant_name, results_dir):
    """
    Load metrics from a trained ablation variant's results.json.
    All variants (including the baseline full_model) read the same
    best-validation-epoch metrics produced by train_mtl.py's
    evaluate_all_tasks(), so the comparison table is apples-to-apples.
    """
    results_path = Path(results_dir) / variant_name / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        r = json.load(f)
    fm = r.get("final_metrics", {})
    if not fm:
        return None
    tasks = list(fm.keys())
    return {
        "variant": variant_name,
        "avg_acc":   sum(fm[t]["accuracy"] for t in tasks) / len(tasks) * 100,
        "avg_auc":   sum(fm[t]["auc"]      for t in tasks) / len(tasks) * 100,
        "avg_mcc":   sum(fm[t]["mcc"]      for t in tasks) / len(tasks) * 100,
        "avg_f1":    sum(fm[t]["f1"]       for t in tasks) / len(tasks) * 100,
        "num_tasks": len(tasks),
        "source": str(results_path),
    }


def load_ablation_config(variant_name, results_dir):
    """Load ablation config saved alongside checkpoint."""
    cfg_path = Path(results_dir) / variant_name / "best_model" / "ablation_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def delta(val, baseline):
    """Format a delta vs baseline with sign and colour hint."""
    d = val - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def print_report(rows, baseline_row):
    """Print formatted comparison table."""
    header = f"{'Variant':<28} {'ACC':>7} {'AUC':>7} {'MCC':>7} {'F1':>7}  {'dACC':>7} {'dAUC':>7} {'dMCC':>7}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  ABLATION STUDY - MTL Peptide Classifier (21 Tasks)")
    print("=" * len(header))
    print(header)
    print(sep)

    for row in rows:
        d_acc = delta(row["avg_acc"],   baseline_row["avg_acc"])
        d_auc = delta(row["avg_auc"],   baseline_row["avg_auc"])
        d_mcc = delta(row["avg_mcc"],   baseline_row["avg_mcc"])
        is_baseline = "baseline" in row["variant"]
        marker = " *" if is_baseline else "  "
        print(
            f"{marker}{row['variant']:<26} "
            f"{row['avg_acc']:>6.2f}% "
            f"{row['avg_auc']:>6.2f}% "
            f"{row['avg_mcc']:>6.2f}% "
            f"{row['avg_f1']:>6.2f}%  "
            f"{d_acc:>7} "
            f"{d_auc:>7} "
            f"{d_mcc:>7}"
        )

    print(sep)
    print("  * = baseline (full model)")
    print()
    print("Interpretation:")
    print("  Negative dACC/dAUC/dMCC  --> removing that component hurts performance")
    print("  Positive delta            --> removing it had no cost (or helped)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Report Generator")
    parser.add_argument("--results_dir", type=str, default="checkpoints",
                        help="Directory containing per-variant checkpoint subfolders (default: checkpoints)")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Optional path to write the per-task ACC%% breakdown as CSV")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Optional path to write the per-task ACC%% breakdown as JSON")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    rows = []

    # Baseline (full_model) is required — every delta in the report is computed
    # against it, so a missing/corrupt file must hard-fail, not silently
    # default to a zero baseline that would make every delta meaningless.
    baseline_path = results_dir / "full_model" / "results.json"
    baseline = load_variant("full_model", results_dir)
    if not baseline:
        print(f"[ERROR] Required baseline file not found or empty: {baseline_path}")
        print("        Run train_mtl.py for the full_model variant first.")
        sys.exit(1)
    baseline["variant"] = "full_model (baseline)"
    rows.append(baseline)

    # Load each ablation variant
    for v in VARIANTS:
        if v == "full_model":
            continue  # already loaded as baseline
        row = load_variant(v, results_dir)
        if row:
            rows.append(row)
        else:
            print(f"[SKIP] {v}: results.json not found (not yet trained)")

    print_report(rows, baseline)

    # Per-task breakdown for completed variants
    print("=" * 60)
    print("  PER-TASK BREAKDOWN (ACC%)")
    print("=" * 60)

    # Collect per-task data
    task_data = {}

    # Per-task ACC for every variant, all read from the same results.json format
    for v in VARIANTS:
        results_path = results_dir / v / "results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            r = json.load(f)
        for task, m in r.get("final_metrics", {}).items():
            task_data.setdefault(task, {})[v] = m["accuracy"] * 100

    if task_data:
        completed_variants = [
            v for v in VARIANTS
            if (results_dir / v / "results.json").exists()
        ]
        col_w = 10
        header2 = f"{'Task':<22}" + "".join(f"{v[:col_w]:>{col_w}}" for v in completed_variants)
        print(header2)
        print("-" * len(header2))
        for task in sorted(task_data.keys()):
            row_str = f"{task:<22}"
            for v in completed_variants:
                val = task_data[task].get(v)
                row_str += f"{val:>{col_w}.1f}" if val is not None else f"{'N/A':>{col_w}}"
            print(row_str)
        print()

        if args.csv_out:
            with open(args.csv_out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Task"] + completed_variants)
                for task in sorted(task_data.keys()):
                    writer.writerow(
                        [task] + [task_data[task].get(v, "") for v in completed_variants]
                    )
            print(f"[OK] Per-task CSV written to {args.csv_out}")

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(task_data, f, indent=2)
            print(f"[OK] Per-task JSON written to {args.json_out}")
    elif args.csv_out or args.json_out:
        print("[WARN] No per-task data available — skipping CSV/JSON export.")


if __name__ == "__main__":
    main()
