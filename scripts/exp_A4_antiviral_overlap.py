"""A4 - antiviral label-overlap analysis (Section 3.3.2). Analysis only, no GPU.

Measures whether antiviral sequences overlap with the Toxicity, Antimicrobial, Antibacterial
and Antifungal datasets, and whether overlapping antiviral test sequences are harder to
classify.

  1-2. exact and near-duplicate overlap (4-mer Jaccard >= 0.9, the same proxy as
       exp_A3_independent_sp.py), antiviral positives/negatives vs each comparison task's
       positives/negatives, train + test pooled
  3.   label conflicts (read off the same matrix)
  4.   composition: mean length, net charge at pH 7 (K + R + 0.5H - D - E), GRAVY
       (Kyte-Doolittle)
  5.   antiviral TEST set split into overlapping / non-overlapping sequences, ACC/AUC/MCC
       and sensitivity/specificity from the seed-42 probabilities

Usage (from the repository root):
  python scripts/exp_A4_antiviral_overlap.py --datasets_dir datasets \
      --probs results/C0_seed42_test_probs.csv --out_dir results
Writes A4_antiviral_overlap.csv and A4_antiviral_subset_metrics.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

FILES = {
    "Antiviral": "16__AV_Antiviral",
    "Toxicity": "17__Toxicity_2021_Dataset",
    "Antimicrobial": "5__Antimicrobial_activity",
    "Antibacterial": "14__antibacterial_AB",
    "Antifungal": "15__Antifungal_AF",
}
COMPARISON_TASKS = ["Toxicity", "Antimicrobial", "Antibacterial", "Antifungal"]
IDENTITY = 0.9
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def load_task(datasets_dir, prefix):
    rows = []
    for split in ("train", "test"):
        df = pd.read_csv(Path(datasets_dir) / f"{prefix}_{split}.csv")
        sc = "sequence" if "sequence" in df.columns else df.columns[0]
        lc = "label" if "label" in df.columns else df.columns[1]
        df = df.dropna(subset=[sc, lc])
        for s, y in zip(df[sc].astype(str), df[lc].astype(int)):
            s = s.strip().upper()
            if s and s.lower() != "nan":
                rows.append((s, y))
    return rows


def kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def build_index(seqs):
    kmer_sets = [kmers(s) for s in seqs]
    inv = {}
    for i, ks in enumerate(kmer_sets):
        for km in ks:
            inv.setdefault(km, []).append(i)
    return kmer_sets, inv


def max_jaccard(km, kmer_sets, inv):
    # inverted k-mer index: only sequences sharing at least one 4-mer can reach J > 0
    cand = set()
    for k in km:
        cand.update(inv.get(k, ()))
    best = 0.0
    for i in cand:
        tk = kmer_sets[i]
        inter = len(km & tk)
        if inter:
            best = max(best, inter / len(km | tk))
            if best >= 0.999:
                break
    return best


def kind(avir_label, comp_label):
    if avir_label == comp_label == 1:
        return "co_positive"
    if avir_label == 1:
        return "avir_pos_vs_comparison_neg (label conflict)"
    if comp_label == 1:
        return "avir_neg_vs_comparison_pos (label conflict)"
    return "co_negative"


def composition(seqs):
    lens = [len(s) for s in seqs]
    charge = [s.count("K") + s.count("R") + 0.5 * s.count("H") - s.count("D") - s.count("E") for s in seqs]
    gravy = [sum(KD.get(c, 0.0) for c in s) / len(s) for s in seqs if s]
    return round(float(np.mean(lens)), 2), round(float(np.mean(charge)), 3), round(float(np.mean(gravy)), 3)


def subset_metrics(df):
    y, p = df["label"].values, df["prob"].values
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    return {
        "n": len(df), "n_pos": tp + fn, "n_neg": tn + fp,
        "ACC": round(float(accuracy_score(y, pred)), 4),
        "AUC": round(float(roc_auc_score(y, p)), 4) if len(set(y)) == 2 else float("nan"),
        "MCC": round(float(matthews_corrcoef(y, pred)), 4),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="datasets")
    ap.add_argument("--probs", default="results/C0_seed42_test_probs.csv")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    data = {t: load_task(args.datasets_dir, p) for t, p in FILES.items()}
    avir = {lab: [s for s, y in data["Antiviral"] if y == lab] for lab in (1, 0)}
    sets = {t: {lab: set(s for s, y in data[t] if y == lab) for lab in (1, 0)} for t in COMPARISON_TASKS}

    rows = []
    for task in COMPARISON_TASKS:
        for a_lab in (1, 0):
            a_set = set(avir[a_lab])
            for c_lab in (1, 0):
                n = len(a_set & sets[task][c_lab])
                rows.append(["exact", task, a_lab, c_lab, kind(a_lab, c_lab), n, len(avir[a_lab]),
                             round(100 * n / len(avir[a_lab]), 2)])

    overlap = set()
    avir_km = {lab: [(s, kmers(s)) for s in set(avir[lab])] for lab in (1, 0)}
    for task in COMPARISON_TASKS:
        index = {c_lab: build_index(list(sets[task][c_lab])) for c_lab in (1, 0)}
        for a_lab in (1, 0):
            for c_lab in (1, 0):
                kmer_sets, inv = index[c_lab]
                n = 0
                for s, km in avir_km[a_lab]:
                    if max_jaccard(km, kmer_sets, inv) >= IDENTITY:
                        n += 1
                        overlap.add(s)
                rows.append([f"neardup_jaccard>={IDENTITY}", task, a_lab, c_lab, kind(a_lab, c_lab), n,
                             len(avir[a_lab]), round(100 * n / len(avir[a_lab]), 2)])
        print(f"near-duplicate search done: {task}")

    for group, seqs in [("Antiviral_positive", avir[1]),
                        ("Antimicrobial_positive", list(sets["Antimicrobial"][1])),
                        ("Toxicity_positive", list(sets["Toxicity"][1]))]:
        ln, ch, gr = composition(seqs)
        rows.append(["composition", group, "", "", "n/mean_length/mean_charge/mean_gravy", len(seqs), "",
                     f"len={ln} charge={ch} gravy={gr}"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "A4_antiviral_overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "comparison_task", "antiviral_label", "comparison_label", "kind",
                    "n_match", "n_antiviral_subset", "pct_of_antiviral_subset"])
        w.writerows(rows)

    probs = pd.read_csv(args.probs)
    test = probs[probs["task"] == "Antiviral"].copy()
    test["overlap"] = test["sequence"].astype(str).str.strip().str.upper().isin(overlap)
    subsets = [("all_antiviral_test", test), ("overlapping_with_tox_amp_ab_af", test[test["overlap"]]),
               ("non_overlapping", test[~test["overlap"]])]
    with open(out / "A4_antiviral_subset_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "n", "n_pos", "n_neg", "ACC", "AUC", "MCC",
                                          "sensitivity", "specificity"])
        w.writeheader()
        for name, sub in subsets:
            m = subset_metrics(sub)
            w.writerow({"subset": name, **m})
            print(name, m)


if __name__ == "__main__":
    main()
