"""B6: per-task PR-AUC (average precision) + per-sequence probabilities for any trained
seed's EXISTING weights (no retraining -- heads.pt/shared_backbone.pt for seeds 1-4 were
never deleted, only checkpoint.pt was, so this reuses the exact bit-identical B1 models).

Generalised version of exp_C0_pertask_metrics.py: takes --seed_tag so it can be run once per
seed, uses sklearn average_precision_score for PR-AUC throughout (matching the manuscript's
own method and the C0/seed-42 correction), not the trapezoid pr_auc() in metrics.py.

Usage (from the repository root, once per seed 42, 1, 2, 3, 4):
  PYTHONPATH=. python scripts/exp_seed_pertask_metrics.py \
      --checkpoint_dir checkpoints/seed_1/best_model --datasets_dir datasets \
      --out_dir results --seed_tag 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import metrics as M
from model_loader import load_model, predict_probs
from mtl_peptide_classifier import get_all_peptide_tasks


def read_xy(path):
    df = pd.read_csv(path)
    sc = "sequence" if "sequence" in df.columns else ("Sequence" if "Sequence" in df.columns else df.columns[0])
    lc = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[1])
    df = df.dropna(subset=[sc, lc])
    seqs = df[sc].astype(str).tolist()
    labels = df[lc].astype(int).tolist()
    keep = [(s, y) for s, y in zip(seqs, labels) if s and s.lower() != "nan"]
    return [s for s, _ in keep], [y for _, y in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--datasets_dir", default="datasets")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed_tag", required=True, help="e.g. 1, 2, 3, 4 -- used in output filenames")
    args = ap.parse_args()

    task_configs = get_all_peptide_tasks(args.datasets_dir)
    model, tokenizer, device, task_names = load_model(args.checkpoint_dir)
    print(f"Loaded seed_{args.seed_tag} model with {len(task_names)} heads; "
          f"{len(task_configs)} tasks detected in datasets_dir")
    # only evaluate tasks the model actually has a head for (e.g. seeds 1-4 predate the
    # B5 "Hemolytic" task registered later in datasets_dir -- skip anything not in model.heads)
    task_configs = {t: cfg for t, cfg in task_configs.items() if t in task_names}
    skipped = set(get_all_peptide_tasks(args.datasets_dir)) - set(task_configs)
    if skipped:
        print(f"Skipping tasks not present in this checkpoint's heads: {sorted(skipped)}")

    rows = []
    prob_rows = []
    for task_name, cfg in task_configs.items():
        prefix = cfg["csv_prefix"]
        test_path = Path(args.datasets_dir) / f"{prefix}_test.csv"
        if not test_path.exists():
            print(f"SKIP {task_name}: no test csv at {test_path}")
            continue
        seqs, labels = read_xy(test_path)
        probs = predict_probs(model, tokenizer, seqs, tasks=[task_name],
                              batch_size=args.batch_size, device=device)[task_name]
        probs = np.asarray(probs)
        y = np.asarray(labels)
        m = M.all_metrics(y, probs, threshold=0.5)  # ACC/AUC/MCC/Sn/Sp/BACC (not PR-AUC)
        pr_auc = average_precision_score(y, probs)  # correct method, matches C0/manuscript
        rows.append({
            "task": task_name, "n": m["n"], "n_pos": m["n_pos"],
            "ACC": round(m["acc"], 4), "AUC": round(m["auc"], 4), "PR_AUC": round(float(pr_auc), 4),
            "MCC": round(m["mcc"], 4), "Sn": round(m["sn"], 4), "Sp": round(m["sp"], 4),
            "BACC": round(m["bacc"], 4),
        })
        for s, yy, p in zip(seqs, labels, probs):
            prob_rows.append({"task": task_name, "sequence": s, "label": yy, "prob": round(float(p), 4)})
        print(f"{task_name:20s} n={m['n']:4d} ACC={m['acc']:.4f} AUC={m['auc']:.4f} "
              f"PR_AUC={pr_auc:.4f} MCC={m['mcc']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"C0_seed{args.seed_tag}_pertask_metrics.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(out_dir / f"C0_seed{args.seed_tag}_test_probs.csv", index=False)
    print(f"\nWrote {len(rows)} tasks / {len(prob_rows)} sequence rows to {out_dir} "
          f"(seed_tag={args.seed_tag})")


if __name__ == "__main__":
    main()
