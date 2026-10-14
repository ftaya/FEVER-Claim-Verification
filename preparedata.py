# ============================================================
# 0. IMPORTS ET CONFIGURATION
# ============================================================
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FILENAME = Path("train.jsonl")


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
def load_records(path: Path) -> list:
    """Charge un fichier JSON standard ou JSON Lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data] if "claim" in data else list(data.values())
        raise ValueError("Unexpected JSON structure")

    except (json.JSONDecodeError, ValueError):
        # Format JSON Lines : un enregistrement par ligne
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


df = pd.DataFrame(load_records(FILENAME))
print("Dataset loaded successfully!")


# ============================================================
# 2. APERÇU GÉNÉRAL
# ============================================================
print("Shape:", df.shape)
print("Number of examples:", len(df))
print("Number of columns:", len(df.columns))
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 records:")
print(df.head(5)) 

print("\nFirst complete record:")
print(df.iloc[0].to_dict())


# ============================================================
# 3. QUALITÉ DES DONNÉES
# ============================================================
print("Missing values per column:")
print(df.isnull().sum())

if "claim" in df.columns:
    print("\nMissing claims:", df["claim"].isna().sum())
    print("Duplicate claims:", df["claim"].duplicated().sum())

if "id" in df.columns:
    print("Duplicate IDs:", df["id"].duplicated().sum())


# ============================================================
# 4. ANALYSE DES LABELS
# ============================================================
if "label" in df.columns:
    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nUnique labels:", df["label"].unique())

    df["label"].value_counts().plot(
        kind="bar",
        rot=0,
        title="Distribution of FEVER Labels",
    )
    plt.xlabel("Label")
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.show()
else:
    print("No label column found.")


# ============================================================
# 5. STRUCTURE DES PREUVES (EVIDENCE)
# ============================================================
if "evidence" in df.columns:
    print("\nEvidence type:", type(df["evidence"].iloc[0]))
    print("First evidence example:")
    print(df["evidence"].iloc[0])

    # Un exemple par label
    for label in df["label"].unique():
        example = df[df["label"] == label].iloc[0]
        print(f"\n--- {label} ---")
        print("Claim:   ", example["claim"])
        print("Evidence:", example["evidence"])


# ============================================================
# 6. VÉRIFICATION DU FICHIER BRUT
# ============================================================
size_mb = FILENAME.stat().st_size / (1024 ** 2)
print(f"\n{FILENAME.name} - {size_mb:.2f} MB")

with open(FILENAME, "r", encoding="utf-8") as f:
    first_record = json.loads(next(f))

print("Claim:   ", first_record["claim"])
print("Label:   ", first_record["label"])
print("Evidence:", first_record["evidence"])