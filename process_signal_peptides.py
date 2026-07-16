"""
Process Signal Peptide dataset for MTL training.
Splits the dataset into train and test sets following UniDL4BioPep format.

Task 21: Signal_peptide (see TECHNICAL_DOCUMENTATION_v2.md §3.1). The CSV
filename prefix "19__Signal_peptides" is a historical artifact of the order
tasks were added and must NOT be changed — it matches the prefix registered
in mtl_peptide_classifier.py's PEPTIDE_TASK_PREFIXES and the files already
committed under datasets/.

Input: SignalPeptides_dattaset_balanced.xlsx (NOT tracked in this repo as of
commit fd7cf13 — it was removed after being processed). The processed output,
datasets/19__Signal_peptides_{train,test}.csv, is already committed, so this
script only needs to be re-run if regenerating the split from raw data.

Dataset provenance (see TECHNICAL_DOCUMENTATION_v2.md §3.1 footnote):
positive sequences retrieved from Peptipedia (22,650 entries before
filtering); negative sequences pooled from 47 publicly available peptide
databases, filtered by length (4-50 aa), standard amino acid composition,
and pairwise identity (CD-HIT, 90% threshold). Final: 3,413 positive +
3,413 negative sequences. No single public download URL exists for this
already-curated/balanced set — request the raw sources above and rebuild
via the CD-HIT filtering pipeline described in the documentation if the
xlsx file is unavailable.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Read the dataset
script_dir = Path(__file__).parent
input_file = script_dir / "SignalPeptides_dattaset_balanced.xlsx"
datasets_dir = script_dir / "datasets"

if not input_file.exists():
    raise FileNotFoundError(
        f"{input_file.name} not found. This raw file is not tracked in git "
        f"(removed at commit fd7cf13); see the module docstring for dataset "
        f"provenance and how to obtain/rebuild it. If you only need the "
        f"already-processed splits, use datasets/19__Signal_peptides_train.csv "
        f"and datasets/19__Signal_peptides_test.csv, which are already committed."
    )

# Create datasets directory if not exists
datasets_dir.mkdir(exist_ok=True)

# Load data
df = pd.read_excel(input_file)
print(f"Loaded {len(df)} sequences from {input_file.name}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['Label'].value_counts()}")

# Rename columns to match expected format (lowercase)
df = df.rename(columns={'Sequence': 'sequence', 'Label': 'label'})

# --- Data quality checks -----------------------------------------------

n_before = len(df)
dupe_mask = df.duplicated(subset='sequence', keep='first')
n_dupes = int(dupe_mask.sum())
if n_dupes:
    print(f"\n[WARN] Found {n_dupes} duplicate sequence(s); dropping duplicates.")
    df = df[~dupe_mask].reset_index(drop=True)
print(f"Duplicate check: {n_before} -> {len(df)} sequences after dedup")

invalid_mask = df['sequence'].apply(
    lambda s: not isinstance(s, str) or len(s) == 0 or not set(s.upper()) <= VALID_AMINO_ACIDS
)
n_invalid = int(invalid_mask.sum())
if n_invalid:
    print(f"[WARN] Found {n_invalid} sequence(s) with non-standard characters; dropping.")
    df = df[~invalid_mask].reset_index(drop=True)
print(f"Sequence validity check: {len(df)} sequences remain with only standard amino acids (ACDEFGHIKLMNPQRSTVWY)")

class_counts = df['label'].value_counts()
if len(class_counts) != 2:
    raise ValueError(f"Expected exactly 2 classes, found {len(class_counts)}: {class_counts.to_dict()}")
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Class counts after cleaning: {class_counts.to_dict()} (imbalance ratio {imbalance_ratio:.2f}:1)")
if imbalance_ratio > 1.5:
    print(f"[WARN] Class imbalance ratio {imbalance_ratio:.2f}:1 exceeds 1.5:1 — dataset is no longer balanced after cleaning.")

# -------------------------------------------------------------------------

# Split into train and test (80/20 split like other datasets)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Use prefix consistent with existing datasets/ files and
# mtl_peptide_classifier.py's PEPTIDE_TASK_PREFIXES (Task 21, historical
# prefix "19__" — see module docstring)
prefix = "19__Signal_peptides"

train_file = datasets_dir / f"{prefix}_train.csv"
test_file = datasets_dir / f"{prefix}_test.csv"

# Save with expected column names (lowercase)
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\nCreated training set: {train_file.name} ({len(train_df)} sequences)")
print(f"Created test set: {test_file.name} ({len(test_df)} sequences)")
print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
