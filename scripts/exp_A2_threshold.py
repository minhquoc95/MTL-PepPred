"""A2 (Reviewer 2 #7) - decision-threshold calibration.

Selects the threshold that maximises MCC on a held-out validation split of the TRAINING
data, then reports the change on the untouched TEST set versus the default 0.5.

Method note (state this in the manuscript): to keep the test set untouched, calibration
uses an independent 20% validation split of each task's training CSV (numpy seed 42). The
test CSV is used only once, for final reporting.

Run once per task, e.g. Bitter and Anti_parasitic:
  python exp_A2_threshold.py --task Bitter \
      --train_csv <repo>/datasets/3__Bitter_train.csv \
      --test_csv  <repo>/datasets/3__Bitter_test.csv \
      --checkpoint_dir <model_dir> --out_dir <results_dir>

`--task` must be the exact head name in task_config.json
(e.g. Bitter, Anti_parasitic, Signal_peptide).
"""
import argparse
import os

import numpy as np
import pandas as pd

import metrics as M
from model_loader import load_model, predict_probs


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
    ap.add_argument("--task", required=True, help="exact head name in task_config.json")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tokenizer, device, task_names = load_model(args.checkpoint_dir)
    if args.task not in task_names:
        raise SystemExit(f"Task '{args.task}' not in model heads: {task_names}")

    # ---- validation split of the training data (test stays untouched) ----
    tr_seqs, tr_y = read_xy(args.train_csv)
    idx = np.arange(len(tr_seqs))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx = idx[:n_val]
    val_seqs = [tr_seqs[i] for i in val_idx]
    val_y = [tr_y[i] for i in val_idx]

    val_p = predict_probs(model, tokenizer, val_seqs, tasks=[args.task],
                          batch_size=args.batch_size, device=device)[args.task]
    val_p = np.asarray(val_p)
    val_y = np.asarray(val_y)

    # ---- sweep threshold, maximise MCC on validation ----
    best_t, best_mcc = 0.5, -2.0
    for t in np.linspace(0.01, 0.99, 99):
        m = M.mcc(val_y, (val_p >= t).astype(int))
        if m > best_mcc:
            best_mcc, best_t = m, float(t)

    # ---- apply to the untouched test set ----
    te_seqs, te_y = read_xy(args.test_csv)
    te_p = np.asarray(predict_probs(model, tokenizer, te_seqs, tasks=[args.task],
                                    batch_size=args.batch_size, device=device)[args.task])
    te_y = np.asarray(te_y)

    default_m = M.all_metrics(te_y, te_p, threshold=0.5)
    calib_m = M.all_metrics(te_y, te_p, threshold=best_t)

    row = {
        "task": args.task,
        "chosen_threshold": round(best_t, 3),
        "val_mcc_at_threshold": round(best_mcc, 4),
        "test_acc_default": round(default_m["acc"], 4),
        "test_acc_calibrated": round(calib_m["acc"], 4),
        "test_mcc_default": round(default_m["mcc"], 4),
        "test_mcc_calibrated": round(calib_m["mcc"], 4),
        "test_f1_default": round(default_m["f1"], 4),
        "test_f1_calibrated": round(calib_m["f1"], 4),
        "test_auc": round(default_m["auc"], 4),  # threshold-free, unchanged
        "n_test": default_m["n"],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "A2_threshold_calibration.csv")
    header = not os.path.exists(out)
    pd.DataFrame([row]).to_csv(out, mode="a", header=header, index=False)
    print("A2", args.task, "->", row)


if __name__ == "__main__":
    main()
