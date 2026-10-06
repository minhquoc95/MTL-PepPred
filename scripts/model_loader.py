"""Reconstruct the trained MTL-PepPred model and run encode-once inference (AMP off).

Follows the official inference recipe (repo README): a checkpoint directory must contain
    heads.pt  shared_backbone.pt  ablation_config.json  task_config.json
These are the files published on Hugging Face (minhquoc95/MTL-PepPred) and produced by the
training script. Point --checkpoint_dir at the server's existing model folder, or at a
local copy of those four files.

Run this from inside the repo working directory (so `mtl_peptide_classifier` is importable).
ESM-2 650M is loaded from the local HuggingFace cache; it is never re-downloaded here.
"""
import json
from pathlib import Path

import torch
from transformers import EsmTokenizer

from mtl_peptide_classifier import MTLPeptideClassifier

ESM_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 128  # matches training; sequences longer than 126 are truncated


def load_model(checkpoint_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cd = Path(checkpoint_dir)
    ablation = json.load(open(cd / "ablation_config.json"))
    task_configs = json.load(open(cd / "task_config.json"))

    model = MTLPeptideClassifier(
        task_configs=task_configs,
        hidden_dim=1280,
        esm_ratio=ablation.get("esm_ratio", 0.9),
        num_transformer_layers=ablation.get("num_transformer_layers", 4),
        dropout=0.0,
        use_transformer=ablation.get("use_transformer", True),
        use_cnn=ablation.get("use_cnn", True),
        unfreeze_esm=ablation.get("unfreeze_esm", False),
    )

    backbone = torch.load(cd / "shared_backbone.pt", map_location=device)
    model.base_embed.load_state_dict(backbone["base_embed"])
    if model.use_transformer and "transformer" in backbone:
        model.transformer.load_state_dict(backbone["transformer"])
    if model.use_cnn and "cnn" in backbone:
        model.cnn.load_state_dict(backbone["cnn"])
        model.layer_norm.load_state_dict(backbone["layer_norm"])

    heads = torch.load(cd / "heads.pt", map_location=device)
    for name, head in model.heads.items():
        if name in heads:
            head.load_state_dict(heads[name])

    model = model.to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained(ESM_NAME)
    return model, tokenizer, device, list(task_configs.keys())


@torch.no_grad()
def predict_probs(model, tokenizer, sequences, tasks=None, batch_size=32,
                  device="cpu", max_length=MAX_LENGTH):
    """Return {task: [positive-class prob per sequence]}.

    The shared backbone is run ONCE per batch (never once per task), exactly as the
    server prediction script does. AMP is intentionally off for paper-grade numbers.
    """
    tasks = tasks or list(model.heads.keys())
    out = {t: [] for t in tasks}
    seqs = [str(s) for s in sequences]
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        toks = [" ".join(list(s)) for s in chunk]
        enc = tokenizer(toks, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        shared = model.encode(ids, mask)  # ONCE
        for t in tasks:
            logits = model.heads[t](shared, mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            out[t].extend(probs)
    return out
