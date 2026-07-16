"""
Upload the MTL Peptide Classifier (21 tasks) to HuggingFace Hub.

Auth: requires `huggingface-cli login` or HF_TOKEN / HUGGING_FACE_HUB_TOKEN in env.
Run:  python3 push_to_huggingface.py
"""

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from mtl_peptide_classifier import get_canonical_peptide_tasks

# Configuration
REPO_ID = "minhquoc95/MTL-PepPred"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "full_model" / "best_model"
RESULTS_DIR = Path(__file__).parent / "checkpoints" / "full_model"

# Inference-only file set (skip 3.87 GB checkpoint.pt — full optimizer state)
INFERENCE_FILES = ["heads.pt", "shared_backbone.pt", "ablation_config.json"]


def load_test_results() -> dict:
    with open(RESULTS_DIR / "test_results.json") as f:
        return json.load(f)


def build_per_task_table(results: dict) -> str:
    rows = ["| Task | ACC | F1 | AUC | MCC |", "|---|---|---|---|---|"]
    for name, m in results["test_metrics"].items():
        rows.append(
            f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | {m['mcc']:.4f} |"
        )
    return "\n".join(rows)


def create_model_card(results: dict) -> str:
    per_task = build_per_task_table(results)
    return f"""---
license: mit
base_model: facebook/esm2_t33_650M_UR50D
tags:
- biology
- peptide
- multi-task-learning
- protein
- classification
---

# MTL Peptide Classifier (21 Tasks)

Multi-Task Learning peptide classifier covering 21 binary peptide-activity tasks. Built on a frozen ESM-2 (650M) backbone with a parallel Transformer + CNN feature extractor and per-task heads, following a PDeepPP-inspired design.

## Held-out Test Set Performance (Averaged across 21 tasks)

| Metric | Value |
|---|---|
| Accuracy | {results['test_avg_acc']*100:.2f}% |
| F1       | {results['test_avg_f1']*100:.2f}% |
| AUC      | {results['test_avg_auc']*100:.2f}% |
| MCC      | {results['test_avg_mcc']*100:.2f}% |

Best Val Avg F1 (used for checkpoint selection): {results['best_val_avg_f1']*100:.2f}%

## Per-Task Test Metrics

{per_task}

## Architecture

- **Shared encoder**: frozen ESM-2 (`facebook/esm2_t33_650M_UR50D`, 650M params) + learnable base embedding, mixed at `esm_ratio=0.9`
- **Feature extraction (parallel)**: 4-layer Transformer + CNN (kernel=7, padding=3) → concatenated to 2560-dim features
- **Heads**: 21 binary classifiers (`2560 → 256 → 128 → 2`) with masked average pooling
- **Loss**: TUM (Task-Uncertainty Multi-task) loss + label smoothing 0.1

## Tasks

| # | Task | Source |
|---|---|---|
| 1 | ACE_inhibitory | UniDL4BioPep |
| 2 | DPPIV_inhibitory | UniDL4BioPep |
| 3 | Bitter | UniDL4BioPep |
| 4 | Umami | UniDL4BioPep |
| 5 | Antimicrobial | UniDL4BioPep |
| 6 | Antimalarial (main) | UniDL4BioPep |
| 7 | Antimalarial_alt | UniDL4BioPep |
| 8 | Quorum_sensing | UniDL4BioPep |
| 9 | Anticancer (main) | UniDL4BioPep |
| 10 | Anticancer_alt | UniDL4BioPep |
| 11 | AntiMRSA | UniDL4BioPep |
| 12 | TTCA | UniDL4BioPep |
| 13 | BBP | UniDL4BioPep |
| 14 | Anti_parasitic | UniDL4BioPep |
| 15 | NeuroPred | UniDL4BioPep |
| 16 | Antibacterial | UniDL4BioPep |
| 17 | Antifungal | UniDL4BioPep |
| 18 | Antiviral | UniDL4BioPep |
| 19 | Toxicity | UniDL4BioPep |
| 20 | Antioxidant | UniDL4BioPep (antioxidant_FRS) |
| 21 | Signal_peptide | local dataset |

## Usage

```python
import json
import os
from huggingface_hub import hf_hub_download
import torch
from transformers import EsmTokenizer

from mtl_peptide_classifier import MTLPeptideClassifier

REPO = "{REPO_ID}"
checkpoint_dir = "MTL-Peptide-Classifier"
os.makedirs(checkpoint_dir, exist_ok=True)

for fname in ["heads.pt", "shared_backbone.pt", "ablation_config.json", "task_config.json"]:
    hf_hub_download(repo_id=REPO, filename=fname, local_dir=checkpoint_dir)

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
with open(f"{{checkpoint_dir}}/task_config.json") as f:
    task_configs = json.load(f)  # no local dataset directory needed

model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=0.9,
    num_transformer_layers=4,
    dropout=0.3,
    use_transformer=True,
    use_cnn=True,
    unfreeze_esm=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = torch.load(f"{{checkpoint_dir}}/shared_backbone.pt", map_location=device)
heads = torch.load(f"{{checkpoint_dir}}/heads.pt", map_location=device)

model.base_embed.load_state_dict(backbone["base_embed"])
if "transformer" in backbone:
    model.transformer.load_state_dict(backbone["transformer"])
if "cnn" in backbone:
    model.cnn.load_state_dict(backbone["cnn"])
    model.layer_norm.load_state_dict(backbone["layer_norm"])
for name, head in model.heads.items():
    if name in heads:
        head.load_state_dict(heads[name])

model = model.to(device).eval()

sequence = "MKWVTFISLLFLFSSAYSRGVFRR"
tokens = " ".join(list(sequence))
inputs = tokenizer(tokens, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
with torch.no_grad():
    logits = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device), task_name="Antimicrobial")
    probs = torch.softmax(logits, dim=-1)
```

## Training

- Base model: `facebook/esm2_t33_650M_UR50D` (frozen)
- Batch size: 16, learning rate: 1e-4, 50 epochs, dropout: 0.3
- 3-way split per task: 80% train / 20% val (checkpoint selection) / held-out test CSV evaluated once
- Mixed precision, gradient clipping 1.0, cosine annealing LR schedule
- TUM loss + label smoothing 0.1

## Files

- `heads.pt` — per-task classifier heads
- `shared_backbone.pt` — base embedding, Transformer, CNN, LayerNorm
- `ablation_config.json` — architecture configuration for reproducibility
- `task_config.json` — canonical 21-task config (name, csv_prefix, num_classes) for rebuilding all heads
- `test_results.json` — held-out test metrics (per task + averages)
- `mtl_peptide_classifier.py` — model code

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
huggingface_hub
numpy
pandas
scikit-learn
```
"""


def upload_to_huggingface():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info.get('name', user_info)}")
    except Exception as e:
        print(f"Not logged in: {e}")
        print("Run: huggingface-cli login   (or export HF_TOKEN=<token>)")
        return

    repo_url = api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    print(f"Repository ready: {repo_url}")

    upload_dir = Path(__file__).parent / "hf_upload_temp"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()

    print("Staging files...")
    for fname in INFERENCE_FILES:
        src = CHECKPOINT_DIR / fname
        if not src.exists():
            print(f"  MISSING: {src}")
            shutil.rmtree(upload_dir)
            return
        shutil.copy(src, upload_dir / fname)
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  + {fname} ({size_mb:.1f} MB)")

    test_results_src = RESULTS_DIR / "test_results.json"
    shutil.copy(test_results_src, upload_dir / "test_results.json")
    print(f"  + test_results.json")

    task_config_path = upload_dir / "task_config.json"
    task_config_path.write_text(json.dumps(get_canonical_peptide_tasks(), indent=2))
    print(f"  + task_config.json")

    code_src = Path(__file__).parent / "mtl_peptide_classifier.py"
    shutil.copy(code_src, upload_dir / "mtl_peptide_classifier.py")
    print(f"  + mtl_peptide_classifier.py")

    results = load_test_results()
    readme_path = upload_dir / "README.md"
    readme_path.write_text(create_model_card(results))
    print(f"  + README.md")

    print("\nUploading to HuggingFace Hub...")
    api.upload_folder(repo_id=REPO_ID, folder_path=str(upload_dir), repo_type="model")

    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}")
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    upload_to_huggingface()
