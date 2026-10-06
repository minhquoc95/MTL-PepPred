"""Register the haemolytic task in PEPTIDE_TASK_PREFIXES of mtl_peptide_classifier.py, needed
only for the single-task haemolytic baseline (train_mtl.py --only_task Hemolytic, after copying
datasets/hemolytic/*.csv into datasets/). Run from the repository root. Idempotent.
"""
from pathlib import Path

f = Path("mtl_peptide_classifier.py")
src = f.read_text(encoding="utf-8")
anchor = '    "19__Signal_peptides": "Signal_peptide",\n'
new_line = '    "22__Hemolytic_activity": "Hemolytic",\n'
if new_line in src:
    print("Already patched.")
else:
    if anchor not in src:
        raise SystemExit("PATCH ABORTED: anchor not found")
    src = src.replace(anchor, anchor + new_line, 1)
    f.write_text(src, encoding="utf-8")
    print("Patched: added Hemolytic to PEPTIDE_TASK_PREFIXES")
