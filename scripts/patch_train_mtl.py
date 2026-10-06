"""Idempotent, self-verifying patcher for train_mtl.py (patches P1-P3).

Adds, without you editing the file by hand:
  P1  --seed N            : replaces the four hard-coded seed(42) calls with args.seed
  P2  --only_task NAME    : restricts training to one task (for the single-task baselines)
  P3  saves task_variances.json (the learned TUM per-task sigma^2) after training

Run it INSIDE the repo working directory:
  python patch_train_mtl.py            # apply
  python patch_train_mtl.py --revert   # restore from the .bak backup

It makes a train_mtl.py.bak backup, is safe to run twice (skips already-applied parts),
and ABORTS LOUDLY if an anchor it expects is not found - so if the upstream file changed,
it reports instead of producing wrong code. If it aborts, do the equivalent edit by hand
and tell the user which anchor was missing.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

FILE = Path("train_mtl.py")


def die(msg):
    print("PATCH ABORTED:", msg)
    print("-> The upstream train_mtl.py differs from what this patcher expects. "
          "Apply the equivalent change by hand and report the missing anchor to the user.")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not FILE.exists():
        die(f"{FILE} not found - run this inside the repo working directory.")

    if args.revert:
        bak = Path("train_mtl.py.bak")
        if not bak.exists():
            die("no train_mtl.py.bak to revert from.")
        shutil.copy(bak, FILE)
        print("Reverted train_mtl.py from backup.")
        return

    src = FILE.read_text(encoding="utf-8")
    if "# [PATCH:applied]" in src:
        print("Patches already applied - nothing to do.")
        return

    shutil.copy(FILE, "train_mtl.py.bak")
    changed = []

    # ---- P1a: argparse --seed and --only_task, inserted before parse_args ----
    if "args = parser.parse_args()" not in src:
        die("anchor 'args = parser.parse_args()' not found.")
    inject = (
        '    parser.add_argument("--seed", type=int, default=42, help="[PATCH] random seed")\n'
        '    parser.add_argument("--only_task", type=str, default="", help="[PATCH] train a single task only")\n'
        "    args = parser.parse_args()"
    )
    src = src.replace("    args = parser.parse_args()", inject, 1)
    changed.append("added --seed and --only_task")

    # ---- P1b: seed values 42 -> args.seed ----
    replaced_any = False
    for pat in ("torch.manual_seed(42)", "np.random.seed(42)", "random.seed(42)"):
        newpat = pat.replace("42", "args.seed")
        if pat in src:
            src = src.replace(pat, newpat)
            replaced_any = True
    # the val-split generator:  torch.Generator().manual_seed(42)
    if ".manual_seed(42)" in src:
        src = src.replace(".manual_seed(42)", ".manual_seed(args.seed)")
        replaced_any = True
    if not replaced_any:
        die("no seed(42) calls found to parameterise (looked for torch/np/random seed(42)).")
    changed.append("seeds -> args.seed")

    # ---- P2: filter task_configs to a single task ----
    anchor = "task_configs = get_all_peptide_tasks(config.data_dir)"
    if anchor not in src:
        die(f"anchor '{anchor}' not found for --only_task.")
    filt = (
        anchor + "\n"
        "    if args.only_task:\n"
        "        if args.only_task not in task_configs:\n"
        "            raise SystemExit(f\"--only_task {args.only_task} not in {list(task_configs)}\")\n"
        "        task_configs = {args.only_task: task_configs[args.only_task]}\n"
        "        config.ablation_name = config.ablation_name or f\"single_{args.only_task}\"\n"
    )
    src = src.replace(anchor, filt, 1)
    changed.append("--only_task filter")

    # ---- P3: save learned TUM per-task variances after training ----
    anchor2 = "history = trainer.train()"
    if anchor2 not in src:
        die(f"anchor '{anchor2}' not found for the variance dump.")
    dump = (
        anchor2 + "\n"
        "    # [PATCH] save learned TUM per-task variances (sigma^2 = exp(log_var))\n"
        "    try:\n"
        "        _loss_mod = getattr(trainer, 'tim_loss', None) or getattr(trainer, 'tum_loss', None)\n"
        "        if _loss_mod is not None and hasattr(_loss_mod, 'log_vars'):\n"
        "            import json as _json\n"
        "            _sig2 = torch.exp(_loss_mod.log_vars.detach().cpu()).tolist()\n"
        "            _names = list(train_loader.task_names)\n"
        "            _vpath = Path(config.output_dir) / config.get_variant_name() / 'task_variances.json'\n"
        "            _vpath.parent.mkdir(parents=True, exist_ok=True)\n"
        "            _json.dump({n: s for n, s in zip(_names, _sig2)}, open(_vpath, 'w'), indent=2)\n"
        "            print('[PATCH] saved task variances ->', _vpath)\n"
        "    except Exception as _e:\n"
        "        print('[PATCH] WARN could not save task variances:', _e)\n"
    )
    src = src.replace(anchor2, dump, 1)
    changed.append("save task_variances.json")

    src = "# [PATCH:applied] --seed/--only_task/task_variances added by patch_train_mtl.py\n" + src
    FILE.write_text(src, encoding="utf-8")
    print("Patched train_mtl.py (backup at train_mtl.py.bak):")
    for c in changed:
        print("  -", c)
    print("Verify with: python train_mtl.py --help   (should now list --seed and --only_task)")


if __name__ == "__main__":
    main()
