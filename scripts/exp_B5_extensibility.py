"""B5 core experiment (Reviewer 1 #7, extensibility): attach a 22nd head (Hemolytic
activity) to the trained seed-42 model, freeze everything else (ESM-2, base embedding,
shared Transformer+CNN, and all 21 existing heads), and train ONLY the new head's
parameters. Then:
  1. Evaluate the new head on the held-out Hemolytic test set.
  2. Re-run inference on the 21 ORIGINAL tasks' test sets with this same extended model
     and diff the probabilities against results/C0_seed42_test_probs.csv (already
     computed from the un-extended model) to quantitatively confirm nothing changed --
     this is the actual evidence for R1 #7, not just an assertion.

Usage (from the repository root):
  PYTHONPATH=. python scripts/exp_B5_extensibility.py \
      --checkpoint_dir checkpoints/seed_42/best_model --datasets_dir datasets \
      --out_dir results --c0_probs results/C0_seed42_test_probs.csv
The haemolytic data are read from datasets/hemolytic/ (see prep_hemolytic.py there).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from transformers import EsmTokenizer

import metrics as M
from model_loader import load_model, predict_probs, MAX_LENGTH
from mtl_peptide_classifier import MTLPeptideClassifier, PeptideDataset, SequenceHead, get_all_peptide_tasks


def evaluate_head(model, tokenizer, device, seqs, labels, task_name, batch_size=32):
    probs = predict_probs(model, tokenizer, seqs, tasks=[task_name], batch_size=batch_size, device=device)[task_name]
    probs = np.asarray(probs)
    y = np.asarray(labels)
    return M.all_metrics(y, probs, threshold=0.5), probs


def read_xy(path):
    df = pd.read_csv(path)
    sc = "sequence" if "sequence" in df.columns else ("Sequence" if "Sequence" in df.columns else df.columns[0])
    lc = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[1])
    df = df.dropna(subset=[sc, lc])
    seqs = df[sc].astype(str).tolist()
    labels = df[lc].astype(int).tolist()
    return seqs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--datasets_dir", default="datasets")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--c0_probs", required=True, help="results/C0_seed42_test_probs.csv, for the invariance check")
    ap.add_argument("--hemo_train", default="datasets/hemolytic/22__Hemolytic_activity_train.csv")
    ap.add_argument("--hemo_test", default="datasets/hemolytic/22__Hemolytic_activity_test.csv")
    ap.add_argument("--task_name", default="Hemolytic")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. load the pretrained seed-42 model (21 heads, dropout=0 as model_loader
    #         builds it for inference; re-enabled to 0.3 just for the new head below) ----
    model, tokenizer, _, task_names = load_model(args.checkpoint_dir, device=device)
    print(f"Loaded base model: {len(task_names)} existing heads, feature_dim={model.feature_dim}")
    assert args.task_name not in model.heads, f"{args.task_name} already exists in this checkpoint"

    # ---- 2. attach the new head ----
    new_head = SequenceHead(input_dim=model.feature_dim, num_classes=2, dropout=0.3).to(device)
    model.heads[args.task_name] = new_head

    # ---- 3. freeze everything except the new head ----
    n_total = n_trainable_before = 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(f"heads.{args.task_name}.")
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_new_head = sum(p.numel() for p in new_head.parameters())
    print(f"Total params: {n_total:,}  Trainable (new head only): {n_trainable:,} "
          f"({100*n_trainable/n_total:.4f}%)  New head param count: {n_new_head:,}")
    assert n_trainable == n_new_head, "freeze did not isolate exactly the new head"

    # ---- 4. data: 80/20 train/val split of the (already leak-filtered) Hemolytic
    #         train CSV, seed 42 -- same convention as train_mtl.py; test CSV untouched ----
    train_csv = Path(args.hemo_train)
    test_csv = Path(args.hemo_test)
    full_train = PeptideDataset(str(train_csv), tokenizer, MAX_LENGTH)
    n_val = max(1, int(len(full_train) * args.val_split))
    n_train = len(full_train) - n_val
    rng = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=rng)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Hemolytic data: train={n_train} val={n_val} (from {len(full_train)} post-filter rows)")

    # ---- 5. train only the new head ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(new_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    best_f1, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                shared = model.encode(ids, mask)
            logits = model.heads[args.task_name](shared, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                shared = model.encode(ids, mask)
                logits = model.heads[args.task_name](shared, mask)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels_t = torch.cat(all_labels)
        preds = torch.argmax(logits, dim=-1).numpy()
        y = labels_t.numpy()
        f1 = M.f1(y, preds)
        acc = M.accuracy(y, preds)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  val ACC={acc:.4f} F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in new_head.state_dict().items()}

    print(f"\nBest val F1: {best_f1:.4f}")
    new_head.load_state_dict(best_state)
    model.heads[args.task_name] = new_head.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "B5_hemolytic_head.pt")

    # ---- 6. final test evaluation (held out, untouched until now) ----
    te_seqs, te_labels = read_xy(test_csv)
    te_metrics, te_probs = evaluate_head(model, tokenizer, device, te_seqs, te_labels, args.task_name)
    print(f"\nHemolytic TEST (n={te_metrics['n']}): ACC={te_metrics['acc']:.4f} AUC={te_metrics['auc']:.4f} "
          f"PR_AUC={te_metrics['pr_auc']:.4f} MCC={te_metrics['mcc']:.4f} Sn={te_metrics['sn']:.4f} Sp={te_metrics['sp']:.4f}")

    pd.DataFrame([{
        "task": args.task_name, "n": te_metrics["n"], "n_pos": te_metrics["n_pos"],
        "ACC": round(te_metrics["acc"], 4), "AUC": round(te_metrics["auc"], 4),
        "PR_AUC": round(te_metrics["pr_auc"], 4), "MCC": round(te_metrics["mcc"], 4),
        "Sn": round(te_metrics["sn"], 4), "Sp": round(te_metrics["sp"], 4),
        "n_total_params": n_total, "n_trainable_params": n_trainable,
        "trainable_pct": round(100 * n_trainable / n_total, 4),
    }]).to_csv(out_dir / "B5_extensibility_head_metrics.csv", index=False)

    # ---- 7. invariance check: re-run the 21 ORIGINAL tasks through this SAME extended
    #         model and diff against the already-computed C0 (pre-extension) probabilities ----
    c0 = pd.read_csv(args.c0_probs)
    rows = []
    max_abs_diff_overall = 0.0
    for task_name in task_names:
        prefix = None
        # reuse get_all_peptide_tasks to map task -> csv_prefix
        tc = get_all_peptide_tasks(args.datasets_dir)
        prefix = tc[task_name]["csv_prefix"]
        test_path = Path(args.datasets_dir) / f"{prefix}_test.csv"
        seqs, labels = read_xy(test_path)
        probs = np.asarray(predict_probs(model, tokenizer, seqs, tasks=[task_name],
                                         batch_size=32, device=device)[task_name])
        old = c0[c0["task"] == task_name].sort_values("sequence")
        new_df = pd.DataFrame({"sequence": seqs, "prob": probs}).sort_values("sequence")
        merged = old.merge(new_df, on="sequence", suffixes=("_old", "_new"))
        diff = (merged["prob_old"] - merged["prob_new"]).abs()
        max_abs_diff = float(diff.max()) if len(diff) else float("nan")
        max_abs_diff_overall = max(max_abs_diff_overall, max_abs_diff if not np.isnan(max_abs_diff) else 0.0)
        rows.append({"task": task_name, "n_matched": len(merged), "n_c0": len(old), "n_new": len(new_df),
                     "max_abs_prob_diff": round(max_abs_diff, 8)})
        print(f"  invariance {task_name:20s} n_matched={len(merged):4d}/{len(old):4d}  max|Δprob|={max_abs_diff:.8f}")

    pd.DataFrame(rows).to_csv(out_dir / "B5_invariance_check.csv", index=False)
    print(f"\nMax |Δprob| across all 21 original tasks: {max_abs_diff_overall:.8f} "
          f"({'IDENTICAL (float noise only)' if max_abs_diff_overall < 1e-4 else 'CHANGED -- investigate'})")


if __name__ == "__main__":
    main()
