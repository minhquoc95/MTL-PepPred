# scripts — analysis behind the resubmitted manuscript

Every number in the manuscript's tables and figures comes from one of these scripts; the
outputs they wrote are in `../results/`. The README's "Reproducing the published results"
section maps each table and figure to its script.

Run everything from the repository root. The scripts import `mtl_peptide_classifier` from the
root, so set `PYTHONPATH=.` (Windows: `set PYTHONPATH=.`). `--checkpoint_dir` is a folder
holding `heads.pt`, `shared_backbone.pt`, `ablation_config.json` and `task_config.json` — a
training output (`checkpoints/<run>/best_model`) or a download from
https://huggingface.co/minhquoc95/MTL-PepPred.

## Files

| Script | Purpose | GPU? |
|---|---|---|
| `metrics.py` | numpy metrics (ACC/AUC/PR-AUC/MCC/Sn/Sp/BACC); PR-AUC is average precision, identical to scikit-learn's `average_precision_score` | no |
| `model_loader.py` | rebuild the model from a checkpoint folder; encode-once inference with mixed precision off | yes (inference) |
| `patch_train_mtl.py` | adds `--seed` and `--only_task` to `train_mtl.py` and saves the learned TUM variances | no |
| `exp_seed_pertask_metrics.py` | per-task metrics and per-sequence test probabilities for one trained run (`C0_seed<s>_*.csv`) | yes (inference) |
| `rebuild_b1_from_c0.py` | five-run means and SDs (`B1_multiseed_*.csv`) from the five `C0_seed*_pertask_metrics.csv` | no |
| `aggregate.py` | `b1`: learned task variances (`B4_task_variances.csv`); `b2`: multi-task vs single-task deltas (`B2_transfer_deltas.csv`) | no |
| `update_benchmark_resubmission.py` | Table 4 workbook: baseline rows from `Benchmark Summary - new.xlsx`, MTL-PepPred rows from the five C0 files | no |
| `wilcoxon_analysis.py` | Wilcoxon signed-rank comparison of MTL-PepPred with the baselines in Table 4 | no |
| `exp_A1_length.py` | sequence-length distribution (Figure S1) | no |
| `exp_A2_threshold.py` | decision-threshold calibration (Table S2) | yes |
| `exp_A3_independent_sp.py` | external SignalP-6.0 evaluation of the signal-peptide head (Table 5) | yes |
| `exp_A4_antiviral_overlap.py` | antiviral label-overlap analysis (Section 3.3.2) | no |
| `exp_B5_extensibility.py` | attach and train a 22nd (haemolytic) head with everything else frozen, then check the 21 original tasks are unchanged (Section 3.6) | yes |
| `patch_add_hemo_task.py` | registers the haemolytic task, needed only for its single-task baseline | no |
| `figures/` | Figures 1–5 and S1 | no |

The haemolytic dataset and the script that builds it from HemoPI2 are in
`../datasets/hemolytic/` (`prep_hemolytic.py`).

## Run order

```
# 0. once: add --seed / --only_task to train_mtl.py
python scripts/patch_train_mtl.py

# Table 4, Figures 2-3: five runs, then per-run evaluation and aggregation
for s in 42 1 2 3 4; do
  python train_mtl.py --seed $s --ablation_name seed_$s
  PYTHONPATH=. python scripts/exp_seed_pertask_metrics.py \
      --checkpoint_dir checkpoints/seed_$s/best_model --datasets_dir datasets \
      --out_dir results --seed_tag $s
done
# aggregate.py b1 writes B4_task_variances.csv and also a B1_multiseed_* pair from the
# training-time evaluation; rebuild_b1_from_c0.py must run after it to replace that pair
python scripts/aggregate.py b1 --glob "checkpoints/seed_*/test_results.json" \
    --variances_glob "checkpoints/seed_*/task_variances.json" --out_dir results
python scripts/rebuild_b1_from_c0.py --results_dir results
python scripts/update_benchmark_resubmission.py

# Table 7: frozen vs fine-tuned backbone
python train_mtl.py --unfreeze_esm --lr 1e-5 --ablation_name unfrozen

# Section 3.5, Table S4: single-task baselines for all 21 tasks
for t in $(python -c "import json;print(' '.join(json.load(open('checkpoints/seed_42/best_model/task_config.json'))))"); do
  python train_mtl.py --only_task $t --seed 42 --ablation_name single_$t
done
python scripts/aggregate.py b2 --mtl checkpoints/seed_42/test_results.json \
    --single_glob "checkpoints/single_*/test_results.json" --out_dir results

# Section 3.6: haemolytic extensibility
python datasets/hemolytic/prep_hemolytic.py --raw_dir <folder with the two HemoPI2 CSVs>
PYTHONPATH=. python scripts/exp_B5_extensibility.py \
    --checkpoint_dir checkpoints/seed_42/best_model --datasets_dir datasets \
    --out_dir results --c0_probs results/C0_seed42_test_probs.csv

# Figure S1, Table S2, Table 5, Section 3.3.2
python scripts/exp_A1_length.py --datasets_dir datasets --out_dir results
PYTHONPATH=. python scripts/exp_A2_threshold.py --task Bitter \
    --train_csv datasets/3__Bitter_train.csv --test_csv datasets/3__Bitter_test.csv \
    --checkpoint_dir checkpoints/seed_42/best_model --out_dir results
PYTHONPATH=. python scripts/exp_A2_threshold.py --task Anti_parasitic \
    --train_csv datasets/12__APP__Anti-parasitic_train.csv \
    --test_csv datasets/12__APP__Anti-parasitic_test.csv \
    --checkpoint_dir checkpoints/seed_42/best_model --out_dir results
PYTHONPATH=. python scripts/exp_A3_independent_sp.py --pos_fasta pos.fasta --neg_fasta neg.fasta \
    --checkpoint_dir checkpoints/seed_42/best_model \
    --train_sp_glob "datasets/*Signal_peptides*.csv" --window 60 --identity 0.9 --out_dir results
python scripts/exp_A4_antiviral_overlap.py --datasets_dir datasets \
    --probs results/C0_seed42_test_probs.csv --out_dir results
```

For Table 5, `pos.fasta` / `neg.fasta` are the signal-peptide and confirmed non-signal-peptide
proteins of the SignalP-6.0 benchmark set (`benchmark_set_sp5.fasta`, DTU Health Tech).

## Notes on specific result files

- `results/B5_extensibility_head_metrics.csv`: its `PR_AUC` column (0.8606) was computed with
  the earlier trapezoidal-rule `pr_auc()`, not with the average precision used everywhere else
  in `results/`. The test-set probabilities of that run were not saved, so it cannot be
  recomputed without re-running `exp_B5_extensibility.py`; the manuscript does not report a
  haemolytic PR-AUC.
- `results/C1_comparison_claims.csv` (manuscript comparison claims checked against the
  Table 4 baselines and `B1_multiseed_pertask.csv`) was produced by ad hoc analysis code that
  is not included here.
- The single-task baselines and `test_results_seed_*.json` come from `train_mtl.py`'s own
  end-of-training evaluation (mixed precision on). Table 4 uses the mixed-precision-off
  re-evaluation in `C0_seed*_pertask_metrics.csv`; the two agree to within 0.005 on every
  task and seed.
