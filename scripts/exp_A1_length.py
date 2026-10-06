"""A1 (Reviewer 2 #6) - sequence-length distribution + 99th percentile.

Pure pandas/numpy, no GPU, no model. Reads every *_train.csv in the datasets folder,
pools the peptide lengths, and writes:
  - A1_lengths_summary.csv : n, mean, median, p90, p95, p99, p99.5, max
  - A1_lengths_hist.csv     : length, count   (for plotting the histogram off-server)

Usage:
  python exp_A1_length.py --datasets_dir <repo>/datasets --out_dir <results_dir>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def seq_col(df):
    for c in ("sequence", "Sequence", "seq", "Seq"):
        if c in df.columns:
            return c
    return df.columns[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--include_test", action="store_true",
                    help="also pool *_test.csv (default: train only, matching the paper)")
    args = ap.parse_args()

    pattern = "*_train.csv" if not args.include_test else "*.csv"
    files = sorted(glob.glob(os.path.join(args.datasets_dir, pattern)))
    if not files:
        raise SystemExit(f"No CSVs matching {pattern} in {args.datasets_dir}")

    lengths = []
    for f in files:
        df = pd.read_csv(f)
        sc = seq_col(df)
        for s in df[sc].astype(str):
            s = s.strip()
            if s and s.lower() != "nan":
                lengths.append(len(s))

    lengths = np.asarray(lengths)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        "n_sequences": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "p99_5": float(np.percentile(lengths, 99.5)),
        "max": int(lengths.max()),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "A1_lengths_summary.csv"), index=False)

    counts = pd.Series(lengths).value_counts().sort_index()
    counts.rename_axis("length").reset_index(name="count").to_csv(
        os.path.join(args.out_dir, "A1_lengths_hist.csv"), index=False)

    print("A1 done. 99th percentile =", summary["p99"], "(MAX_LENGTH used by the model = 128)")
    print(summary)


if __name__ == "__main__":
    main()
