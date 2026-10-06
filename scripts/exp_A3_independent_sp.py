"""A3 (Reviewer 2 #2) - independent signal-peptide validation.

Tests the Signal_peptide head on an EXTERNAL dataset the model never saw, with negatives
that are confirmed non-signal (the reviewer's concern). You provide two FASTA files that
codex downloads on the server (see SCRIPTS_README for sources, e.g. SP22 / DeepSig /
SignalP-6 benchmark released after Nov-2020):
  --pos_fasta : proteins WITH an experimentally/curated signal peptide
  --neg_fasta : proteins confirmed to have NO signal peptide (cytoplasmic/nuclear/TM)

Fairness controls applied here (all documented in the manuscript):
  1. Input = N-terminal window (default first 60 residues, capped at 126) - a signal
     peptide is an N-terminal feature and the training SP sequences are short.
  2. Leakage removal - drop any external window that is a near-duplicate (4-mer Jaccard
     >= --identity) of any sequence in the training signal-peptide CSVs. (CD-HIT is the
     gold standard; this alignment-free proxy is used because nothing may be pip-installed.)
  3. Report AUC and PR-AUC as primary (class balance differs); ACC/MCC/Sn/Sp at 0.5 on the
     natural ratio and on a balanced subset.

Usage:
  python exp_A3_independent_sp.py --pos_fasta pos.fasta --neg_fasta neg.fasta \
      --checkpoint_dir <model_dir> --train_sp_glob "<repo>/datasets/*Signal_peptides*.csv" \
      --window 60 --identity 0.9 --out_dir <results_dir>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import metrics as M
from model_loader import load_model, predict_probs

STD = set("ARNDCQEGHILKMFPSTWYV")


def read_fasta(path):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def nterm_window(seq, window, cap=126):
    return seq[: min(window, cap)]


def kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def build_train_kmer_index(train_glob, window):
    train_seqs = []
    for f in glob.glob(train_glob):
        df = pd.read_csv(f)
        sc = "sequence" if "sequence" in df.columns else ("Sequence" if "Sequence" in df.columns else df.columns[0])
        for s in df[sc].astype(str):
            s = s.strip().upper()
            if s and s.lower() != "nan":
                train_seqs.append(nterm_window(s, window))
    return [kmers(s) for s in train_seqs]


def max_jaccard(km, index):
    best = 0.0
    for tk in index:
        inter = len(km & tk)
        if inter == 0:
            continue
        j = inter / len(km | tk)
        if j > best:
            best = j
            if best >= 0.999:
                break
    return best


def prepare(seqs, label, window, index, identity):
    rows = []
    dropped_nonstd = dropped_leak = 0
    for s in seqs:
        w = nterm_window(s, window)
        if len(w) < 5 or any(c not in STD for c in w):
            dropped_nonstd += 1
            continue
        if max_jaccard(kmers(w), index) >= identity:
            dropped_leak += 1
            continue
        rows.append((w, label))
    return rows, dropped_nonstd, dropped_leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_fasta", required=True)
    ap.add_argument("--neg_fasta", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--train_sp_glob", required=True,
                    help='glob for training signal-peptide CSVs, e.g. "<repo>/datasets/*Signal_peptides*.csv"')
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--identity", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    index = build_train_kmer_index(args.train_sp_glob, args.window)
    pos_rows, pn, pl = prepare(read_fasta(args.pos_fasta), 1, args.window, index, args.identity)
    neg_rows, nn, nl = prepare(read_fasta(args.neg_fasta), 0, args.window, index, args.identity)
    rows = pos_rows + neg_rows
    if not rows:
        raise SystemExit("No usable sequences after filtering.")

    seqs = [r[0] for r in rows]
    y = np.asarray([r[1] for r in rows])

    model, tokenizer, device, task_names = load_model(args.checkpoint_dir)
    if "Signal_peptide" not in task_names:
        raise SystemExit(f"'Signal_peptide' not in heads: {task_names}")
    p = np.asarray(predict_probs(model, tokenizer, seqs, tasks=["Signal_peptide"],
                                 batch_size=args.batch_size, device=device)["Signal_peptide"])

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({"sequence": seqs, "label": y, "signal_prob": p}).to_csv(
        os.path.join(args.out_dir, "A3_predictions.csv"), index=False)

    natural = M.all_metrics(y, p, threshold=0.5)

    # balanced subset (downsample majority, seed 42)
    rng = np.random.default_rng(42)
    pos_i = np.where(y == 1)[0]; neg_i = np.where(y == 0)[0]
    k = min(len(pos_i), len(neg_i))
    bal_i = np.concatenate([rng.choice(pos_i, k, replace=False), rng.choice(neg_i, k, replace=False)])
    balanced = M.all_metrics(y[bal_i], p[bal_i], threshold=0.5)

    summary = {
        "window": args.window, "identity_cut": args.identity,
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "dropped_nonstd_pos": pn, "dropped_leak_pos": pl,
        "dropped_nonstd_neg": nn, "dropped_leak_neg": nl,
        "AUC": round(natural["auc"], 4), "PR_AUC": round(natural["pr_auc"], 4),
        "ACC_natural": round(natural["acc"], 4), "MCC_natural": round(natural["mcc"], 4),
        "Sn_natural": round(natural["sn"], 4), "Sp_natural": round(natural["sp"], 4),
        "ACC_balanced": round(balanced["acc"], 4), "MCC_balanced": round(balanced["mcc"], 4),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "A3_independent_signalpeptide.csv"), index=False)
    print("A3 done:", summary)
    print("NOTE: report these numbers honestly even if below the in-benchmark 99.3% ACC "
          "(domain shift). Do not tune the data to inflate them.")


if __name__ == "__main__":
    main()
