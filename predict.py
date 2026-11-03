"""Predict all 21 bioactivities for peptide sequences with the pretrained MTL-PepPred model.

The weights are downloaded from https://huggingface.co/minhquoc95/MTL-PepPred on first use.

Examples:
  python predict.py KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK IPP VPP
  python predict.py --input peptides.csv --output predictions.csv
  python predict.py --input peptides.fasta --output predictions.csv

A CSV input needs a column named "sequence". The output has one row per peptide and one
column per task, each the probability (0-1) that the peptide has that activity.
"""
import argparse
import csv
import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import EsmTokenizer

from mtl_peptide_classifier import MTLPeptideClassifier

REPO = "minhquoc95/MTL-PepPred"
MAX_LENGTH = 128
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def read_sequences(args):
    if not args.input:
        return list(args.sequences)
    if args.input.lower().endswith((".fasta", ".fa", ".faa")):
        seqs, cur = [], []
        with open(args.input) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                    cur = []
                elif line:
                    cur.append(line)
        if cur:
            seqs.append("".join(cur))
        return seqs
    with open(args.input, newline="") as fh:
        return [row["sequence"] for row in csv.DictReader(fh)]


def load_model(weights_dir, device):
    for f in ["heads.pt", "shared_backbone.pt", "ablation_config.json", "task_config.json"]:
        hf_hub_download(repo_id=REPO, filename=f, local_dir=weights_dir)
    with open(os.path.join(weights_dir, "ablation_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(weights_dir, "task_config.json")) as fh:
        task_configs = json.load(fh)
    model = MTLPeptideClassifier(
        task_configs=task_configs, hidden_dim=1280,
        esm_ratio=cfg.get("esm_ratio", 0.9),
        num_transformer_layers=cfg.get("num_transformer_layers", 4), dropout=0.0,
        use_transformer=cfg.get("use_transformer", True), use_cnn=cfg.get("use_cnn", True),
        unfreeze_esm=False,
    )
    backbone = torch.load(os.path.join(weights_dir, "shared_backbone.pt"), map_location=device)
    model.base_embed.load_state_dict(backbone["base_embed"])
    if model.use_transformer:
        model.transformer.load_state_dict(backbone["transformer"])
    if model.use_cnn:
        model.cnn.load_state_dict(backbone["cnn"])
        model.layer_norm.load_state_dict(backbone["layer_norm"])
    heads = torch.load(os.path.join(weights_dir, "heads.pt"), map_location=device)
    for name, head in model.heads.items():
        head.load_state_dict(heads[name])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", help="peptide sequences (one-letter amino acid codes)")
    ap.add_argument("--input", help="CSV with a 'sequence' column, or a FASTA file")
    ap.add_argument("--output", help="write predictions to this CSV (default: print them)")
    ap.add_argument("--weights_dir", default="MTL-PepPred-weights", help="where to cache the weights")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    seqs = [s.strip().upper() for s in read_sequences(args) if s.strip()]
    if not seqs:
        ap.error("give sequences on the command line or with --input")
    for s in seqs:
        bad = set(s) - STANDARD
        if bad:
            ap.error(f"{s}: non-standard residues {sorted(bad)}")
        if len(s) > MAX_LENGTH:
            print(f"note: {s[:20]}... is {len(s)} residues; only the first {MAX_LENGTH} are used")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = load_model(args.weights_dir, device)
    tasks = list(model.heads)

    rows = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i:i + args.batch_size]
            enc = tokenizer([" ".join(s) for s in batch], return_tensors="pt",
                            max_length=MAX_LENGTH, padding="max_length", truncation=True)
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            probs = {t: torch.softmax(model(ids, mask, task_name=t), dim=-1)[:, 1].tolist() for t in tasks}
            for j, s in enumerate(batch):
                rows.append({"sequence": s, **{t: round(probs[t][j], 4) for t in tasks}})

    if args.output:
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sequence"] + tasks)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} predictions x {len(tasks)} tasks to {args.output}")
    else:
        for r in rows:
            print(f"\n{r['sequence']}")
            for t in sorted(tasks, key=lambda t: -r[t]):
                print(f"  {t:18s} {r[t]:.3f}")


if __name__ == "__main__":
    main()
