# MTL Peptide Classifier

Multi-Task Learning (MTL) peptide classifier trained on UniDL4BioPep peptide activity datasets using a PDeepPP-inspired architecture with ESM-2 backbone.

Predicts 21 peptide bioactivities with one set of stored weights and one protein language
model pass per sequence, reducing stored parameters by ~95% relative to 21 separate
single-task models. Joint training helps some tasks and harms others: against single-task
models of identical architecture, 12 of 21 improve by AUC and 9 decline.

- **Pretrained model**: [huggingface.co/minhquoc95/MTL-PepPred](https://huggingface.co/minhquoc95/MTL-PepPred)
- **Datasets**: [`datasets/`](datasets/) — 21 UniDL4BioPep-derived task CSVs (train/test split), plus the filtered haemolytic set used in Section 3.6 in [`datasets/hemolytic/`](datasets/hemolytic/)
- **Analysis scripts and results**: [`scripts/`](scripts/) and [`results/`](results/) — see [Reproducing the published results](#reproducing-the-published-results)

## Architecture

```
Input: Peptide Sequence
         ↓
    ┌─────────────────────────────────────────────────┐
    │              Shared Encoder                      │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  ESM-2 (650M params)                        │ │
    │  │  facebook/esm2_t33_650M_UR50D               │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  Base Embedding (33 aa → 1280 dim)          │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │  Weighted Combination (ESM ratio: 0.9)          │
    │                      ↓                           │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  Parallel Feature Extraction [ablatable]     │ │
    │  │  ├─ Transformer (4 layers, 8 heads)         │ │
    │  │  └─ CNN (kernel=7, padding=3)               │ │
    │  └─────────────────────────────────────────────┘ │
    │                      ↓                           │
    │         Concatenated Features [2560 dim]        │
    └─────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────┐
    │          Task-Specific Heads (21 tasks)          │
    │  ┌─────────────────────────────────────────────┐ │
    │  │  SequenceHead: 2560 → 256 → 128 → 2        │ │
    │  │  - Masked average pooling                   │ │
    │  │  - 2 FC layers with ReLU + Dropout(0.3)     │ │
    │  └─────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────┘
                      ↓
                 Output: Binary logits
```

## Peptide Activity Tasks

1. ACE_inhibitory - ACE inhibitory activity
2. DPPIV_inhibitory - DPPIV inhibitory activity
3. Bitter - Bitter taste peptides
4. Umami - Umami taste peptides
5. Antimicrobial - Antimicrobial activity
6. Antimalarial - Antimalarial activity (main)
7. Antimalarial_alt - Antimalarial activity (alternative)
8. Quorum_sensing - Quorum sensing activity
9. Anticancer_alt - Anticancer activity (alternative)
10. Anticancer - Anticancer activity (main)
11. AntiMRSA - Anti-MRSA strains activity
12. TTCA - Tumour T cell antigens
13. BBP - Blood-Brain Barrier peptides
14. Anti_parasitic - Anti-parasitic peptides
15. Neuropeptide - Neuroprotective peptides
16. Antibacterial - Antibacterial peptides
17. Antifungal - Antifungal peptides
18. Antiviral - Antiviral peptides
19. Toxicity - Toxicity prediction
20. Antioxidant - Antioxidant activity
21. Signal_peptide - Signal peptides

## Results

All figures are the mean ± standard deviation of five independent training runs
(seeds 42, 1, 2, 3 and 4).

| Metric | Mean ± SD |
|---|---|
| Accuracy | 87.43 ± 0.23% |
| AUC | 92.84 ± 0.28% |
| PR-AUC | 92.19 ± 0.43% |
| MCC | 73.73 ± 0.46% |

Per-task values, the single-task comparison, the frozen-versus-fine-tuned comparison and the
external signal-peptide evaluation are in `results/`. Earlier single-run numbers
(89.4% / 94.0% / 78.6%) are superseded and should not be cited.

Pretrained weights: https://huggingface.co/minhquoc95/MTL-PepPred

## Files

- **`mtl_peptide_classifier.py`** - Model architecture and data utilities (supports ablation flags)
- **`train_mtl.py`** - Training script with ablation study CLI
- **`evaluate_mtl_comprehensive.py`** - Comprehensive evaluation script
- **`ablation_report.py`** - Ablation study reporting and comparison
- **`process_signal_peptides.py`** - Signal peptide dataset preprocessing
- **`SignalPeptides_dattaset_balanced.xlsx`** - Balanced signal peptide dataset
- **`datasets/`** - 21 task CSVs (train/test), UniDL4BioPep-derived + local signal peptide data

## Training Configuration

### Default Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 |
| Dropout | 0.3 |
| Weight Decay | 1e-5 |
| LR Schedule | Cosine annealing |
| Gradient Clipping | 1.0 |
| Label Smoothing | 0.1 |
| Mixed Precision | Enabled |
| TUM Loss | Enabled |
| ESM Ratio | 0.9 |
| Transformer Layers | 4 |

## Usage

### Training

```bash
# Default training
python train_mtl.py --batch_size 16 --lr 1e-4 --epochs 50 --dropout 0.3

# Without TUM loss
python train_mtl.py --no_tum

# Custom label smoothing
python train_mtl.py --label_smoothing 0.05
```

### Ablation Studies

The model supports fine-grained ablation via CLI flags. Each variant is saved to its own checkpoint directory named automatically from the active flags.

```bash
# Full model (baseline)
python train_mtl.py

# Without CNN branch
python train_mtl.py --no_cnn

# Without Transformer branch
python train_mtl.py --no_transformer

# Transformer only, 2 layers
python train_mtl.py --no_cnn --transformer_layers 2

# Unfreeze ESM-2 backbone (use lower lr)
python train_mtl.py --unfreeze_esm --lr 1e-5

# ESM ratio 0.5 (equal mix of ESM + base embedding)
python train_mtl.py --esm_ratio 0.5

# Without TUM loss
python train_mtl.py --no_tum

# Custom run name
python train_mtl.py --no_cnn --ablation_name my_experiment
```

### Evaluation

```bash
python evaluate_mtl_comprehensive.py \
    --model_dir "checkpoints/full_model/best_model" \
    --model_name "full_model" \
    --batch_size 8
```

### Ablation Report

```bash
python ablation_report.py --results_dir checkpoints/
```

### Inference

Runs from a clean clone — no local `checkpoints/` or `datasets/` needed. Downloads the pretrained checkpoint from [huggingface.co/minhquoc95/MTL-PepPred](https://huggingface.co/minhquoc95/MTL-PepPred) and reconstructs the exact trained architecture from `ablation_config.json`.

```bash
pip install torch transformers huggingface_hub
```

```python
import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import EsmTokenizer

from mtl_peptide_classifier import MTLPeptideClassifier

REPO = "minhquoc95/MTL-PepPred"
checkpoint_dir = "MTL-Peptide-Classifier"
os.makedirs(checkpoint_dir, exist_ok=True)

for fname in ["heads.pt", "shared_backbone.pt", "ablation_config.json", "task_config.json"]:
    hf_hub_download(repo_id=REPO, filename=fname, local_dir=checkpoint_dir)

with open(f"{checkpoint_dir}/ablation_config.json") as f:
    ablation_cfg = json.load(f)
with open(f"{checkpoint_dir}/task_config.json") as f:
    task_configs = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

model = MTLPeptideClassifier(
    task_configs=task_configs,
    hidden_dim=1280,
    esm_ratio=ablation_cfg.get("esm_ratio", 0.9),
    num_transformer_layers=ablation_cfg.get("num_transformer_layers", 4),
    dropout=0.0,
    use_transformer=ablation_cfg.get("use_transformer", True),
    use_cnn=ablation_cfg.get("use_cnn", True),
    unfreeze_esm=ablation_cfg.get("unfreeze_esm", False),
)

backbone = torch.load(f"{checkpoint_dir}/shared_backbone.pt", map_location=device)
model.base_embed.load_state_dict(backbone["base_embed"])
if model.use_transformer:
    model.transformer.load_state_dict(backbone["transformer"])
if model.use_cnn:
    model.cnn.load_state_dict(backbone["cnn"])
    model.layer_norm.load_state_dict(backbone["layer_norm"])

heads = torch.load(f"{checkpoint_dir}/heads.pt", map_location=device)
for name, head in model.heads.items():
    head.load_state_dict(heads[name])

model = model.to(device).eval()

sequence = "MKWVTFISLLFLFSSAYSRGVFRR"
tokens = " ".join(list(sequence))
inputs = tokenizer(tokens, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

with torch.no_grad():
    logits = model(
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        task_name="Antimicrobial",
    )
    probs = torch.softmax(logits, dim=-1)
    print(probs)
```

## Ablation Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--no_transformer` | Remove shared Transformer encoder | Off |
| `--no_cnn` | Remove shared CNN branch | Off |
| `--unfreeze_esm` | Allow ESM-2 gradients (fine-tuning) | Off |
| `--esm_ratio` | ESM-2 weight in embedding mix (0–1) | 0.9 |
| `--transformer_layers` | Number of shared Transformer layers | 4 |
| `--no_tum` | Disable TUM multi-task loss | Off |
| `--label_smoothing` | Label smoothing factor | 0.1 |
| `--ablation_name` | Custom checkpoint directory name | auto |

Each run saves an `ablation_config.json` alongside the checkpoint for full reproducibility.

## Key Features

- **Ablatable Architecture**: Transformer and CNN branches can be independently disabled via CLI
- **ESM-2 Backbone**: Frozen by default; can be unfrozen for fine-tuning
- **TUM Loss**: Task-Uncertainty Multi-task Loss with learnable per-task log variances
- **Masked Pooling**: Handles variable-length peptide sequences
- **Auto Variant Naming**: Checkpoint directories named automatically from active ablation flags
- **Windows Compatible**: DataLoader `num_workers` auto-set to 0 on Windows

## Reproducing the published results

Run everything from the repository root. Scripts in `scripts/` import `mtl_peptide_classifier`
from the root, so prefix them with `PYTHONPATH=.` (on Windows: `set PYTHONPATH=.`). The
`--seed` and `--only_task` options of `train_mtl.py` are added by `scripts/patch_train_mtl.py`;
apply it once before any training run.

| Manuscript item | Script | Output in `results/` |
|---|---|---|
| Table 4, Figures 2 and 3 — five-run means | `scripts/patch_train_mtl.py`; `train_mtl.py --seed <s>` for s = 42, 1, 2, 3, 4; `scripts/exp_seed_pertask_metrics.py --seed_tag <s>` for each run; `scripts/rebuild_b1_from_c0.py`; `scripts/update_benchmark_resubmission.py`; `scripts/figures/Figure_2_overall_resubmission.py`, `scripts/figures/Figure_3_heatmap_resubmission.py` | `test_results_seed_*.json`, `C0_seed*_pertask_metrics.csv`, `C0_seed*_test_probs.csv`, `B1_multiseed_overall.csv`, `B1_multiseed_pertask.csv`, `Benchmark Summary - resubmission.xlsx` |
| Wilcoxon signed-rank comparison with the baselines | `scripts/wilcoxon_analysis.py` | `MTL_Statistical_Analysis.xlsx` (written when run) |
| Figure 1 — workflow schematic | `scripts/figures/Figure_1_workflow_build.py` (needs `cairosvg`, `Pillow`) | — |
| Figure 4 — ROC and PR curves | `scripts/figures/Figure_4_roc_pr_resubmission.py` | `C0_seed42_test_probs.csv` |
| Table 6 — ablation | `ablation_report.py` | — |
| Table 7 — frozen vs fine-tuned backbone | `train_mtl.py --unfreeze_esm --lr 1e-5` | `results_unfrozen.json`, `test_results_unfrozen.json` |
| Section 3.5, Figure 5, Table S4 — single-task baselines and transfer | `train_mtl.py --only_task <TASK> --seed 42` for each of the 21 tasks, then `scripts/aggregate.py b2`; `scripts/figures/Figure_5_transfer_variance_resubmission.py` | `test_results_single_*.json`, `B2_transfer_deltas.csv` |
| Figure 5a, Table S3 — learned task variances | `scripts/aggregate.py b1` | `B4_task_variances.csv`, `task_variances_seed_*.json` |
| Section 3.6 — adding a 22nd activity (haemolytic) | `datasets/hemolytic/prep_hemolytic.py`, then `scripts/exp_B5_extensibility.py`; single-task reference: `scripts/patch_add_hemo_task.py` and `train_mtl.py --only_task Hemolytic` | `B5_extensibility_head_metrics.csv`, `B5_invariance_check.csv`, `test_results_single_Hemolytic.json` |
| Section 2.5 and 3.3.4, Table 5 — SignalP-6.0 external evaluation | `scripts/exp_A3_independent_sp.py` | `A3_independent_signalpeptide.csv`, `A3_predictions.csv` |
| Table S2 — decision-threshold calibration | `scripts/exp_A2_threshold.py` | `A2_threshold_calibration.csv` |
| Figure S1 — sequence-length distribution | `scripts/exp_A1_length.py`; `scripts/figures/Figure_S1_length_resubmission.py` | `A1_lengths_hist.csv`, `A1_lengths_summary.csv` |
| Section 3.3.2 — antiviral label-overlap check | `scripts/exp_A4_antiviral_overlap.py` | `A4_antiviral_overlap.csv`, `A4_antiviral_subset_metrics.csv` |
| Table S1 — the 47 negative-set source databases | — | `Table_S1_negative_sources.csv` |

Shared helpers: `scripts/metrics.py` (metric definitions) and `scripts/model_loader.py`
(checkpoint loading). Script-by-script notes are in `scripts/SCRIPTS_README.md`.

The haemolytic data in `datasets/hemolytic/` are kept out of `datasets/` on purpose, so that
the default 21-task training run does not pick them up.

## Licence

The code is released under the MIT licence (see `LICENSE`). The MIT licence covers the code
only, not the third-party data: the datasets are redistributed from UniDL4BioPep, Peptipedia,
HemoPI2, SignalP-6.0 and the 47 databases listed in Table S1, each under its own terms. To cite
this work, see `CITATION.cff`.

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
huggingface_hub
esm
numpy
pandas
scikit-learn
tqdm
openpyxl
```

## References

- ESM-2: Lin et al. (2023) - Evolutionary-scale prediction of atomic-level protein structure with a language model
- PDeepPP: Original architecture inspiration
- TUM Loss: Kendall et al. (2018) - Multi-Task Learning Using Uncertainty to Weigh Losses
- UniDL4BioPep: Benchmark dataset for peptide activity prediction
