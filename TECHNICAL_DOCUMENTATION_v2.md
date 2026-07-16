**Technical Documentation: MTL-PepPred**

*Multi-Task Learning Framework for Unified Peptide Bioactivity Prediction*

Version 2.0 --- Updated May 2026

# 1. Project Overview {#project-overview}

## 1.1 Research Objective {#research-objective}

This project implements a deep learning-based multi-task learning (MTL) system for simultaneous classification of 21 distinct peptide bioactivity types. The system leverages a frozen pre-trained protein language model (ESM-2, 650M parameters) with task-specific classification heads to achieve state-of-the-art performance across diverse peptide classification tasks from the extended UniDL4BioPep benchmark, including food-relevant bioactivities such as ACE inhibition, DPPIV inhibition, bitter taste, umami taste, and antimicrobial preservation.

## 1.2 Core Innovation {#core-innovation}

The architecture combines a frozen ESM-2 backbone with a dual-branch feature extractor (Transformer + CNN) and a task-uncertainty multi-task (TUM) loss function, enabling efficient knowledge transfer across 21 diverse peptide bioactivity prediction tasks. Freezing the backbone reduces trainable parameters by \~86% relative to full fine-tuning (from \~754.7M total to \~104.7M trainable) while preventing catastrophic forgetting of pre-trained protein representations.

**Key update from v1.0:** The former Task 20 (Anti-inflammatory, custom, 14,400 train/3,600 test) has been removed and replaced by two tasks: Task 20 (Antioxidant, UniDL4BioPep antioxidant_FRS) and Task 21, a newly curated Signal Peptide dataset (3,413 positive + 3,413 negative sequences; 2,730/683 train/test split per class). All performance metrics, task tables, and dataset descriptions have been updated accordingly.

# 2. System Architecture {#system-architecture}

## 2.1 High-Level Design {#high-level-design}

MTL-PepPred follows a hard parameter sharing design:

![](media/image1.png){width="6.288194444444445in" height="9.0in"}

Input Layer: Peptide amino acid sequence

↓

Shared Encoding Layer (Frozen + Learnable)

├── ESM-2 Pre-trained Encoder (esm2_t33_650M_UR50D, frozen)

├── Learnable Base Embedding (33 tokens → 1280 dimensions)

└── Weighted combination: h = 0.9 × h_ESM + 0.1 × h_learnable

↓

Dual-Branch Parallel Feature Extraction

├── Global Context Branch: 4-layer Transformer (8 attention heads, d_model=1280)

└── Local Pattern Branch: 1D CNN (kernel size 7, covering 7-aa motif windows)

↓

Feature Fusion: Concatenation → 2560-dimensional vector

↓

Task-Specific Classification Heads (21 independent binary classifiers)

├── Masked Average Pooling (handles variable-length sequences)

├── FC1: 2560 → 256 (ReLU + Dropout 0.3)

├── FC2: 256 → 128 (ReLU + Dropout 0.3)

└── Output: 128 → 2 (binary logits)

## 2.2 Key Architectural Components {#key-architectural-components}

### 2.2.1 Frozen Backbone Strategy {#frozen-backbone-strategy}

- ESM-2 Encoder: esm2_t33_650M_UR50D (33 Transformer layers, 1280-dimensional per-residue embeddings)

- Parameter Freeze: All 650M ESM-2 parameters remain frozen during training

- Rationale: Preserves pre-trained evolutionary and physicochemical protein representations; prevents catastrophic forgetting

- Efficiency: Trainable parameters total approximately 104.7M against a 650M frozen backbone (\~86% reduction relative to full fine-tuning of all 754.7M parameters)

- Learnable complement: A 33-token learnable embedding (42K parameters) provides task-specific adaptation

### 2.2.2 Parameter Distribution {#parameter-distribution}

| **Component**                   | **Parameters** | **Trainable** |
|---------------------------------|----------------|---------------|
| ESM-2 Backbone (frozen)         | \~650M         | No            |
| Base Embedding (33 × 1280)      | \~42K          | Yes           |
| Shared Transformer (4 layers)   | \~78.7M        | Yes           |
| CNN Branch (Conv1d + LayerNorm) | \~11.5M        | Yes           |
| Task Heads (21 × MLP)           | \~14.5M        | Yes           |
| **Total Trainable**             | **\~104.7M**   | **Yes**       |

*Note: Earlier documentation (v1.0: 12M; an intermediate v2.0 draft: 52M) undercounted the shared Transformer branch. The verified total trainable parameter count, computed directly from the model architecture in `mtl_peptide_classifier.py`, is \~104.7M --- comprising the shared Transformer (\~78.7M), CNN branch (\~11.5M), 21 task heads (\~14.5M, \~690K each), and learnable base embeddings (\~42K).*

### 2.2.3 Dual-Branch Feature Extraction {#dual-branch-feature-extraction}

- Transformer Branch: Captures long-range dependencies and global sequence context across the full peptide length

- CNN Branch: Extracts local motifs and short-range patterns within a 7-amino-acid window --- critical for food peptide bioactivity (C-terminal Pro/Ile/Trp for ACE inhibition; N-terminal X-Pro for DPPIV inhibition; hydrophobic clusters for bitter taste)

- Fusion: Full concatenation preserves complementary information; downstream task heads learn optimal combinations per bioactivity type

### 2.2.4 Task-Specific Classification Heads {#task-specific-classification-heads}

- 21 independent binary classifiers --- one per bioactivity task

- Identical MLP architecture per head: FC(2560→256, ReLU, Dropout 0.3) → FC(256→128, ReLU, Dropout 0.3) → Output(128→2)

- Masked average pooling over valid (non-padded) positions for variable-length sequence handling

- \~690K parameters per head; \~14.5M total across 21 heads

# 3. Dataset Configuration {#dataset-configuration}

## 3.1 Task Composition {#task-composition}

MTL-PepPred classifies 21 peptide bioactivity types. Tasks 1-19 follow the original UniDL4BioPep benchmark splits. Task 20 (Antioxidant) is drawn from UniDL4BioPep's antioxidant_FRS dataset. Task 21 (Signal Peptide) is a newly contributed dataset.

| **Task ID** | **Bioactivity Type** | **Description**                                                  | **Dataset Source**                                                     |
|-------------|----------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| 1           | ACE Inhibitory       | Angiotensin-converting enzyme inhibition (antihypertensive)      | UniDL4BioPep                                                           |
| 2           | DPPIV Inhibitory     | Dipeptidyl peptidase IV inhibition (antidiabetic)                | UniDL4BioPep                                                           |
| 3           | Bitter Taste         | Bitter peptide identification                                    | UniDL4BioPep                                                           |
| 4           | Umami Taste          | Umami peptide identification                                     | UniDL4BioPep                                                           |
| 5           | Antimicrobial        | General antimicrobial activity                                   | UniDL4BioPep                                                           |
| 6           | Antimalarial (Main)  | Anti-Plasmodium activity --- main dataset (1:15 imbalance)       | UniDL4BioPep                                                           |
| 7           | Antimalarial (Alt)   | Anti-Plasmodium activity --- alternative dataset (1:5 imbalance) | UniDL4BioPep                                                           |
| 8           | Quorum Sensing       | Quorum sensing inhibition                                        | UniDL4BioPep                                                           |
| 9           | Anticancer (Alt)     | Anticancer activity --- alternative dataset                      | UniDL4BioPep                                                           |
| 10          | Anticancer (Main)    | Anticancer activity --- main dataset                             | UniDL4BioPep                                                           |
| 11          | Anti-MRSA            | Anti-MRSA strains activity                                       | UniDL4BioPep                                                           |
| 12          | TTCA                 | Tumour T cell antigens                                           | UniDL4BioPep                                                           |
| 13          | BBP                  | Blood-brain barrier penetrating peptides                         | UniDL4BioPep                                                           |
| 14          | Antiparasitic        | Antiparasitic activity                                           | UniDL4BioPep                                                           |
| 15          | Neuropeptide         | Neuropeptide activity prediction                                 | UniDL4BioPep                                                           |
| 16          | Antibacterial        | Antibacterial activity (food preservation)                       | UniDL4BioPep                                                           |
| 17          | Antifungal           | Antifungal activity                                              | UniDL4BioPep                                                           |
| 18          | Antiviral            | Antiviral activity                                               | UniDL4BioPep                                                           |
| 19          | Toxicity             | Peptide toxicity prediction (safety assessment)                  | UniDL4BioPep                                                           |
| 20          | Antioxidant          | Antioxidant activity                                             | UniDL4BioPep (antioxidant_FRS)                                         |
| 21          | Signal Peptide       | Signal peptide recognition (bioproduction support)               | Our curated dataset 2,730 pos / 2,730 neg train 683 pos / 683 neg test |

*\* Signal Peptide dataset: Positive sequences retrieved from Peptipedia (22,650 entries before filtering). Negative sequences pooled from 47 publicly available peptide databases, filtered by length (4-50 aa), standard amino acid composition, and pairwise identity (CD-HIT, 90% threshold). Final: 3,413 positive + 3,413 negative sequences; 80:20 train/test split.*

## 3.2 Data Format Specification {#data-format-specification}

- File Format: CSV with train/test splits

- Required Columns: sequence (single-letter amino acid code), label (0 or 1)

- Naming Convention: {task_id}\_\_{task_name}\_peptides\_{train,test}.csv

- Split Ratio: 80% training, 20% testing (stratified by label)

- Maximum sequence length: 128 amino acids (covers 99th percentile of combined dataset)

## 3.3 Data Preprocessing Pipeline {#data-preprocessing-pipeline}

1.  Sequence Tokenisation: Convert amino acid sequences to space-separated format for ESM-2 tokeniser (e.g., ACDE → A C D E)

2.  Padding / Truncation: Standardise to maximum length of 128 tokens

3.  Attention Mask Generation: Binary masks distinguish valid tokens from padding positions

4.  Multi-Task Sampling: Task-homogeneous batching --- one task sampled uniformly per iteration; batch drawn exclusively from that task\'s training data

# 4. Training Methodology {#training-methodology}

## 4.1 Training Configuration {#training-configuration}

| **Parameter**               | **Value** | **Rationale**                                               |
|-----------------------------|-----------|-------------------------------------------------------------|
| **Learning Rate**           | 1e-4      | Standard for transfer learning with frozen backbone         |
| **Batch Size**              | 16        | Balances memory constraints and gradient stability          |
| **Epochs**                  | 50        | Sufficient for convergence without overfitting              |
| **Dropout Rate**            | 0.3       | Strong regularisation for diverse tasks                     |
| **Weight Decay**            | 1e-5      | L2 regularisation to prevent overfitting                    |
| **Gradient Clipping**       | 1.0       | Stabilises training, prevents gradient explosion            |
| **Label Smoothing**         | 0.1       | Reduces overconfidence, improves calibration                |
| **Mixed Precision**         | FP16      | Reduces GPU memory usage, accelerates training              |
| **ESM-2 Weight (alpha)**    | 0.9       | Balances pre-trained ESM-2 features vs learnable embeddings |
| **Loss Weight Init (s\^t)** | 0         | Equal initial task priority across all 21 tasks             |
| **Max Sequence Length**     | 128       | Covers 99th percentile of combined multi-task dataset       |

## 4.2 Loss Function: Task-Uncertainty Multi-Task (TUM) Loss {#loss-function-task-uncertainty-multi-task-tum-loss}

### 4.2.1 Base Loss {#base-loss}

- Cross-Entropy Loss with label smoothing (ε = 0.1)

- Formula: L_CE = CrossEntropy(y_hat, y; ε = 0.1)

### 4.2.2 TUM Loss (Kendall et al., 2018) {#tum-loss-kendall-et-al.-2018}

- Purpose: Dynamically balances loss contributions across 21 heterogeneous tasks differing in difficulty, dataset size, and class imbalance

- Mechanism: Learnable task-specific log-variance parameters s\^t, initialised to 0

- Formula per task t: L\^t_weighted = (1 / 2σ\^t²) × L\^t_CE + log(σ\^t), where σ\^t = exp(s\^t)

- Effect: Tasks with higher learned variance (harder/smaller/more imbalanced) receive proportionally lower loss weight

- Regularisation: The log(σ\^t) term prevents task weights from collapsing to zero

- Relevance: Particularly important for food bioactivity tasks --- umami, bitter, and quorum sensing datasets are orders of magnitude smaller than antimicrobial datasets

*Note: This loss function was labelled \'TIM Loss\' (Threshold-Independent Multi-task) in v1.0 documentation. The correct name used throughout the manuscript is TUM Loss (Task-Uncertainty Multi-task Loss) following Kendall et al. (2018).*

## 4.3 Optimisation Strategy {#optimisation-strategy}

- Optimizer: AdamW (Adam with decoupled weight decay; Loshchilov & Hutter, 2019)

- Learning Rate Schedule: Cosine annealing

- Gradient Clipping: Max norm = 1.0 for training stability

- Mixed Precision: FP16 for forward pass to reduce GPU memory consumption

## 4.4 Training Process {#training-process}

5.  Task Sampling: Uniformly sample one of 21 tasks for each batch iteration

6.  Batch Assembly: Draw batch of 16 sequences from selected task\'s training data

7.  Forward Pass: Encode sequence → dual-branch feature extraction → task head prediction

8.  Loss Computation: Compute task-specific TUM-weighted cross-entropy loss

9.  Backward Pass: Compute gradients for task head and shared Transformer/CNN components only (ESM-2 frozen)

10. Parameter Update: Apply AdamW with gradient clipping and cosine LR schedule

11. Checkpoint Selection: Save model checkpoint achieving highest average F1-score across all 21 tasks on a held-out validation set (20% of the training data, randomly sampled with fixed seed 42), following the UniDL4BioPep benchmark evaluation protocol (Du et al., 2023). The test set is used exclusively for final one-shot evaluation after training is complete.

# 5. Performance Results {#performance-results}

## 5.1 Overall Performance Metrics {#overall-performance-metrics}

Held-out test set performance for the current 21-task checkpoint, verified against the published model card at [huggingface.co/minhquoc95/MTL-PepPred](https://huggingface.co/minhquoc95/MTL-PepPred):

| **Metric**              | **Value**  | **Interpretation**                                           |
|--------------------------|-----------|-----------------------------------------------------------------|
| Average Accuracy (ACC)  | **87.37%** | Correct classification rate across all 21 tasks              |
| Average F1              | **84.80%** | Harmonic mean of precision/recall                             |
| Average AUC-ROC         | **92.82%** | Area under ROC curve --- excellent discrimination             |
| Average MCC             | **73.42%** | Matthews Correlation Coefficient --- balanced class measure   |

Best Val Avg F1 (used for checkpoint selection, 20% validation split): 83.43%

*Note: PR-AUC is not reported for this checkpoint. `train_mtl.py`'s validation/test evaluation (which `test_results.json` and the HF model card are built from) only tracks ACC/F1/AUC/MCC; `evaluate_mtl_comprehensive.py` computes PR-AUC but has not yet been re-run against this checkpoint.*

## 5.2 Top Performing Tasks {#top-performing-tasks}

*Pending regeneration.* The previous per-task ranking table (Anti-MRSA, Antimalarial main, Signal Peptide, ...) was measured on a 20-task checkpoint trained before the test-set-leakage fix in checkpoint selection (see Appendix, item 6) and before the Antioxidant task (21st) was added, so it does not describe the model reported in §5.1. Re-run `evaluate_mtl_comprehensive.py` on the current checkpoint and repopulate this table from its per-task output rather than carrying the old 20-task numbers forward.

## 5.3 Performance Distribution Analysis {#performance-distribution-analysis}

*Pending regeneration* for the same reason as §5.2 — the high/moderate/challenging task buckets previously listed here were derived from the pre-leakage-fix, 20-task run and are not verified for the current 21-task checkpoint.

## 5.4 Ablation Study Summary {#ablation-study-summary}

*Removed — unverified.* The ablation table previously here (ESM ratio, No Transformer, No CNN, etc.) was measured on the pre-leakage-fix 20-task pipeline and has not been re-run since the checkpoint-selection fix or the addition of the Antioxidant task. Re-run the ablation variants with the current `train_mtl.py` (see README "Ablation Studies") and regenerate this table with `ablation_report.py --results_dir checkpoints/` once complete; do not carry the old numbers forward, as they were produced under a since-fixed, leakage-biased checkpoint-selection methodology.

# 6. Technical Implementation Details {#technical-implementation-details}

## 6.1 Sequence Encoding {#sequence-encoding}

- Tokenisation: Space-separated amino acids (e.g., ACDEFGH → A C D E F G H)

- Maximum Length: 128 amino acids (covers 99th percentile of dataset)

- Vocabulary: Standard 20 amino acids + ESM-2 special tokens (padding, mask, etc.)

## 6.2 Variable-Length Handling {#variable-length-handling}

- Attention Masking: Binary masks distinguish valid tokens from padding

- Masked Average Pooling: h_pooled = Σ(H_i × M_i) / Σ(M_i), where M is the binary mask

## 6.3 Memory Optimisation {#memory-optimisation}

- Mixed Precision Training: FP16 for forward pass, FP32 for gradient accumulation

- Frozen Backbone: Zero gradient computation for ESM-2 parameters

- Checkpointing: Saves shared encoder weights + 21 task-specific head weights separately

## 6.4 Computational Requirements {#computational-requirements}

- GPU: RTX 4060 Ti 16GB (minimum 8GB VRAM recommended)

- Training Time: \~9 hours for 50 epochs across 21 tasks (\~140K+ total sequences) --- *figure carried over from the 20-task run; not re-measured after the Antioxidant task was added*

- Inference Speed: \~25 sequences/second at batch size 16

# 7. Model Deployment {#model-deployment}

## 7.1 Checkpoint Structure {#checkpoint-structure}

> checkpoints/best_model/
>
> ├── checkpoint.pt \# Full model state (shared encoder + all heads)
>
> ├── heads.pt \# Task-specific head weights (21 heads)
>
> ├── shared_backbone.pt \# Shared Transformer + CNN encoder weights
>
> ├── training_history.json \# Training metrics per epoch
>
> └── results.json \# Final test set performance (all 21 tasks)

## 7.2 Inference Pipeline {#inference-pipeline}

12. Load Pre-trained Components: ESM-2 tokeniser and MTL-PepPred checkpoint

13. Initialise Model: Reconstruct architecture with trained weights

14. Tokenise Input: Convert peptide sequence to space-separated format; truncate/pad to 128

15. Task Selection: Specify which bioactivity task(s) to predict (1--21)

16. Forward Pass: ESM-2 encoding → dual-branch extraction → task head → logits

17. Probability Extraction: Apply softmax to logits for class probabilities

18. Decision: Binary classification at default threshold 0.5 (adjustable per task)

Multi-task inference: A single forward pass through the shared encoder produces embeddings used by all 21 task heads simultaneously, enabling screening of all 21 bioactivities in a single computation.

# 8. Key Contributions {#key-contributions}

## 8.1 Scientific Contributions {#scientific-contributions}

19. Most comprehensive unified MTL framework for peptide bioactivity prediction: 21 tasks spanning food enzyme inhibition (ACE, DPPIV), taste modulation (bitter, umami), antimicrobial preservation, and safety assessment

20. Frozen backbone strategy: Demonstrates that preserving pre-trained ESM-2 representations (\~86% parameter reduction) achieves performance comparable to full fine-tuning

21. TUM Loss adaptation: Successfully applies task-uncertainty weighting (Kendall et al., 2018) to large-scale multi-task peptide prediction with heterogeneous datasets

22. Signal Peptide Dataset: New curated benchmark dataset of 3,413 positive and 3,413 negative sequences from 47 sources, achieving 100% AUC across all evaluated architectures and relevant to computational identification of secretion signals for recombinant bioproduction of food-grade bioactive peptide ingredients

## 8.2 Engineering Contributions {#engineering-contributions}

23. Modular hard-parameter sharing architecture: New bioactivity tasks can be added by training an additional head without retraining shared components

24. \~95% storage reduction relative to 21 independent single-task models

25. Comprehensive evaluation: ACC, AUC, PR-AUC, and MCC metrics for thorough assessment including class-imbalanced tasks

26. Reproducibility: Complete training history, checkpoint management, and publicly released datasets

## 8.3 Performance Summary {#performance-summary}

27. Achieves 87.37% average ACC and 92.82% average AUC across 21 diverse tasks (verified against the published HF model card; see §5.1)

28. *Pending re-verification:* comparison against UniDL4BioPep per-task results (previously "matches or exceeds on 12/20 tasks; highest AUC on 9/20 tasks") was measured on the pre-leakage-fix 20-task run and has not been recomputed for the current 21-task checkpoint.

29. *Pending re-verification:* comparison against dedicated task-specific tools (previously "surpasses on 12/16 tasks with available comparisons") is likewise unverified for the current checkpoint.

30. *Pending re-verification:* the largest-gain claim (previously "Antimalarial-main, +9.0% AUC vs UniDL4BioPep") depended on the same pre-leakage-fix run and has not been recomputed.

# 9. Future Directions {#future-directions}

## 9.1 Potential Enhancements {#potential-enhancements}

31. Lightweight task-specific adapter modules (e.g., LoRA) for selective performance improvement on priority tasks without retraining shared components

32. Integration of gastrointestinal stability predictors and bioavailability modelling to bridge sequence-level prediction with functional food efficacy

33. Extension of benchmark with additional food-specific bioactivities: antioxidant, anti-inflammatory, opioid activities

34. Multi-label classification framework to model multifunctional peptides (e.g., lactoferricin sequences with concurrent antimicrobial and antihypertensive properties)

## 9.2 Research Opportunities {#research-opportunities}

35. Few-shot learning for rapid extension to novel bioactivity types with minimal experimental data

36. Interpretability tools: Saliency mapping and amino acid position importance analysis for structure-activity relationship insights

37. Hierarchical task clustering exploiting biological taxonomy (e.g., antimicrobial → antibacterial / antifungal / antiviral)

38. Sequence generation: Generative modelling for de novo food bioactive peptide design

# 10. Conclusion {#conclusion}

MTL-PepPred demonstrates the effectiveness of multi-task learning with frozen pre-trained backbones for comprehensive peptide bioactivity prediction. The framework achieves 87.37% average accuracy and 92.82% mean AUC across 21 diverse peptide classification tasks using a single unified deployable model (§5.1), establishing a robust computational foundation for food-derived bioactive peptide discovery. *The CNN-branch ablation claim from v1.0/early v2.0 drafts (−31.3% ACC without CNN) has been removed pending re-verification on the current 21-task, leakage-fixed pipeline (§5.4) — it should not be cited until re-run.* Model unification reduces storage requirements by approximately 95% relative to 21 independent single-task models, with a single forward pass enabling simultaneous screening of all 21 bioactivity categories.

# Appendix: Change Log from v1.0 {#appendix-change-log-from-v1.0}

| **\#** | **Item**               | **v1.0 (old)**                              | **v2.0 (corrected)**                                      |
|--------|------------------------|---------------------------------------------|-----------------------------------------------------------|
| **1**  | Task 20                | Anti-inflammatory (14,400/3,600 train/test) | Removed. Replaced by Task 20 = Antioxidant (UniDL4BioPep antioxidant_FRS) and Task 21 = Signal Peptide (2,730/683 train/test per class) |
| **2**  | Task 9 / Task 10 order | 9=Anticancer Main, 10=Anticancer Alt        | 9=Anticancer Alt, 10=Anticancer Main                      |
| **3**  | Task 15 name           | Neuroprotective                             | Neuropeptide                                              |
| **4**  | Trainable parameters   | \~12M                                       | \~104.7M (verified breakdown in Table 2.2; an intermediate draft's \~52M figure undercounted the shared Transformer) |
| **5**  | Loss function name     | TIM Loss                                    | TUM Loss (Kendall et al., 2018)                           |
| **6**  | Checkpoint selection   | test set F1-score (methodologically incorrect) | validation F1-score (20% of training data, seed 42); test set reserved for final one-shot evaluation |
| **7**  | GPU                    | RTX 4070 Ti 16GB                            | RTX 4060 Ti 16GB                                          |
| **8**  | Task count             | 20 tasks                                    | 21 tasks (Antioxidant + Signal Peptide added; Anti-inflammatory removed) |
| **9**  | Overall ACC / AUC      | 88.89% / 94.10%                             | 87.37% / 92.82% (§5.1; verified against current 21-task checkpoint on HuggingFace) |
