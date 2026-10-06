# codex_scripts — ready-made scripts for the resubmission experiments

These scripts let the agent **run** the experiments without editing the model code by hand.
Read `..\CODEX_SERVER_RUNBOOK.md` for the rules (no-leak, no-`pip install`, disk safety,
STOP conditions) and `..\README.md` for the scientific goal + the results log to fill in.

Copy this whole folder into the repo working directory on the server (so `import
mtl_peptide_classifier` works), then run the scripts from there.

## Files
| Script | Purpose | GPU? |
|---|---|---|
| `metrics.py` | numpy metrics (ACC/AUC/PR-AUC/MCC/Sn/Sp), no scikit-learn | no |
| `model_loader.py` | rebuild the model + encode-once inference (AMP off) | yes (inference) |
| `exp_A1_length.py` | A1 length distribution + 99th percentile | no |
| `exp_A2_threshold.py` | A2 threshold calibration (Bitter, Anti_parasitic) | yes |
| `exp_A3_independent_sp.py` | A3 independent signal-peptide validation | yes |
| `patch_train_mtl.py` | adds `--seed`, `--only_task`, saves `task_variances.json` | no |
| `aggregate.py` | B1 mean±SD + B4 variances; B2 transfer deltas | no |

`--checkpoint_dir` = a folder holding `heads.pt`, `shared_backbone.pt`,
`ablation_config.json`, `task_config.json` (the server's existing model folder, or a local
copy / HF download). `<repo>` = the cloned repo dir. `<results_dir>` = your scratch results
folder (copy it back to `Resubmission\results\` at the end).

## Self-check first (2 min)
```
python -c "import torch,transformers,numpy,pandas;print('core ok')"
python -c "import sklearn" ; python -c "import tqdm"     # note if either errors (fallbacks exist)
python train_mtl.py --help                                # confirm real flag names
```

## Run order

### Phase A (cheap; do first)
```
# A1 - no GPU
python exp_A1_length.py --datasets_dir <repo>/datasets --out_dir <results_dir>

# A2 - run once per task
python exp_A2_threshold.py --task Bitter \
    --train_csv <repo>/datasets/3__Bitter_train.csv \
    --test_csv  <repo>/datasets/3__Bitter_test.csv \
    --checkpoint_dir <model_dir> --out_dir <results_dir>
python exp_A2_threshold.py --task Anti_parasitic \
    --train_csv <repo>/datasets/12__APP__Anti-parasitic_train.csv \
    --test_csv  <repo>/datasets/12__APP__Anti-parasitic_test.csv \
    --checkpoint_dir <model_dir> --out_dir <results_dir>
# (confirm the exact CSV filenames with: ls <repo>/datasets)

# A3 - download the external benchmark first (on the server), then:
python exp_A3_independent_sp.py --pos_fasta pos.fasta --neg_fasta neg.fasta \
    --checkpoint_dir <model_dir> \
    --train_sp_glob "<repo>/datasets/*Signal_peptides*.csv" \
    --window 60 --identity 0.9 --out_dir <results_dir>
```
**A3 data sources** (download on the server; pick ONE positive + confirmed-negative set):
SP22 (Swiss-Prot SP proteins with ECO:0000269/0000305 released after Nov-2020), or the
DeepSig / SignalP-6 benchmark (`deepsig.biocomp.unibo.it` / the SignalP-6 data release).
Positives = signal-peptide proteins; negatives = confirmed non-SP (cytoplasmic/nuclear/TM).
Report the numbers honestly even if below the in-benchmark 99.3% ACC (domain shift).

### Phase B (training; server authorised — still obey the STOP conditions)
```
python patch_train_mtl.py            # adds --seed, --only_task, task_variances.json
python train_mtl.py --help           # verify --seed and --only_task now appear

# B1 - 5 seeds (each run: evaluate, keep test_results.json + task_variances.json,
#      then DELETE the multi-GB checkpoint.pt for that run)
for s in 42 1 2 3 4; do
  python train_mtl.py --seed $s --ablation_name seed_$s
  # (training already runs the one-shot test at the end and writes test_results.json)
  rm -f checkpoints/seed_$s/best_model/checkpoint.pt
done
python aggregate.py b1 --glob "checkpoints/seed_*/test_results.json" \
    --variances_glob "checkpoints/seed_*/task_variances.json" --out_dir <results_dir>

# B3 - frozen vs fine-tuned (one run; high memory; delete checkpoint after)
python train_mtl.py --unfreeze_esm --lr 1e-5 --ablation_name unfrozen
rm -f checkpoints/unfrozen/best_model/checkpoint.pt

# B2 - single-task baselines: priority six first, then the rest if disk/time allow
for t in Antimalarial ACE_inhibitory Antioxidant TTCA DPPIV_inhibitory Toxicity ; do
  python train_mtl.py --only_task $t --ablation_name single_$t
  rm -f checkpoints/single_$t/best_model/checkpoint.pt
done
python aggregate.py b2 --mtl checkpoints/seed_42/test_results.json \
    --single_glob "checkpoints/single_*/test_results.json" --out_dir <results_dir>
```
(Use the exact task-name keys from `task_config.json`. `Antimalarial` = the main dataset.)

## Notes / guardrails
- If `patch_train_mtl.py` aborts, the upstream file changed: do the equivalent edit by hand
  and tell the user which anchor was missing. Do not guess.
- If `sklearn`/`tqdm` are missing, do **not** install: `metrics.py` already avoids sklearn;
  stub tqdm if the training script imports it (`tqdm = lambda x, **k: x`).
- Never point any output, log, or filename at server credentials or identifiers.
- After each experiment, write the numbers into `..\README.md` §4 and append `..\README.md`
  §6, and copy the CSV/JSON back to `Resubmission\results\`.
