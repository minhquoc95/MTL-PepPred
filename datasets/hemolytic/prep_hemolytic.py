"""Build the filtered haemolytic dataset used in Section 3.6 from HemoPI2.

Source: HemoPI2 (Rathore et al. 2025, Communications Biology, doi:10.1038/s42003-025-07615-w),
https://webs.iiitd.edu.in/raghava/hemopi2/download.html -> cross_val_dataset.csv (1540 rows,
train) and independent_dataset.csv (386 rows, test). Result: 708 train / 162 test.
  - filter length 4-50, standard residues only
  - drop any sequence that exactly matches a sequence anywhere in the 21 existing
    training/test CSVs (datasets/*_train.csv, datasets/*_test.csv)
  - write datasets/hemolytic/22__Hemolytic_activity_train.csv / _test.csv (sequence,label)

Usage (from the repository root):
  python datasets/hemolytic/prep_hemolytic.py --raw_dir <folder with the two HemoPI2 CSVs>
"""
import argparse
import glob
import os

import pandas as pd

STD = set("ARNDCQEGHILKMFPSTWYV")


def load_hemo(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    seq_col = "SEQUENCE" if "SEQUENCE" in df.columns else "sequence"
    lab_col = "label"
    df = df.dropna(subset=[seq_col, lab_col])
    seqs = df[seq_col].astype(str).str.strip().str.upper().tolist()
    labels = df[lab_col].astype(int).tolist()
    return seqs, labels


def clean(seqs, labels, existing_seqs):
    kept_s, kept_y = [], []
    n_len = n_nonstd = n_overlap = 0
    for s, y in zip(seqs, labels):
        if not (4 <= len(s) <= 50):
            n_len += 1
            continue
        if any(c not in STD for c in s):
            n_nonstd += 1
            continue
        if s in existing_seqs:
            n_overlap += 1
            continue
        kept_s.append(s)
        kept_y.append(y)
    return kept_s, kept_y, n_len, n_nonstd, n_overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="datasets/hemolytic")
    args = ap.parse_args()

    # 1. collect every sequence already used anywhere in the 21 existing tasks
    existing_seqs = set()
    for f in glob.glob("datasets/*_train.csv") + glob.glob("datasets/*_test.csv"):
        df = pd.read_csv(f)
        sc = "sequence" if "sequence" in df.columns else ("Sequence" if "Sequence" in df.columns else df.columns[0])
        for s in df[sc].astype(str):
            s = s.strip().upper()
            if s and s.lower() != "nan":
                existing_seqs.add(s)
    print(f"Existing 21-task sequence pool: {len(existing_seqs)} unique sequences")

    # 2. load + clean HemoPI2 train (cross_val) and test (independent)
    tr_s, tr_y = load_hemo(os.path.join(args.raw_dir, "cross_val_dataset.csv"))
    te_s, te_y = load_hemo(os.path.join(args.raw_dir, "independent_dataset.csv"))
    print(f"Raw: train={len(tr_s)} test={len(te_s)}")

    tr_s2, tr_y2, tr_nlen, tr_nns, tr_nov = clean(tr_s, tr_y, existing_seqs)
    te_s2, te_y2, te_nlen, te_nns, te_nov = clean(te_s, te_y, existing_seqs)

    print(f"\nTrain: kept={len(tr_s2)}  dropped_length={tr_nlen}  dropped_nonstd={tr_nns}  dropped_overlap={tr_nov}")
    print(f"Test:  kept={len(te_s2)}  dropped_length={te_nlen}  dropped_nonstd={te_nns}  dropped_overlap={te_nov}")
    print(f"Train label balance: pos={sum(tr_y2)} neg={len(tr_y2)-sum(tr_y2)}")
    print(f"Test  label balance: pos={sum(te_y2)} neg={len(te_y2)-sum(te_y2)}")

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({"sequence": tr_s2, "label": tr_y2}).to_csv(
        os.path.join(args.out_dir, "22__Hemolytic_activity_train.csv"), index=False)
    pd.DataFrame({"sequence": te_s2, "label": te_y2}).to_csv(
        os.path.join(args.out_dir, "22__Hemolytic_activity_test.csv"), index=False)
    print(f"\nWrote the two CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
