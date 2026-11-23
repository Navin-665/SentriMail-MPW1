import os
import pandas as pd
import numpy as np

PROCESSED_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1\model_training\data\processed"

train_path = os.path.join(PROCESSED_DIR, "train.parquet")
val_path = os.path.join(PROCESSED_DIR, "val.parquet")
test_path = os.path.join(PROCESSED_DIR, "test.parquet")

print("=" * 60)
print("INSPECTING EXISTING T5 BASELINE DATASET (PARQUET SPLITS)")
print("=" * 60)

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

print(f"• Train Set: {len(train_df):,} rows")
print(f"• Val Set:   {len(val_df):,} rows")
print(f"• Test Set:  {len(test_df):,} rows")
print(f"• Total Baseline Dataset: {len(train_df) + len(val_df) + len(test_df):,} rows\n")

print("Columns in T5 Dataset:", train_df.columns.tolist())

print("\n--- Source Breakdown in Train Split ---")
if "source" in train_df.columns:
    print(train_df["source"].value_counts(dropna=False))

print("\n--- Category Breakdown in Train Split ---")
if "category" in train_df.columns:
    print(train_df["category"].value_counts(dropna=False))

print("\n--- Response Target Breakdown ---")
has_resp = train_df["response"].notna() & (train_df["response"].astype(str).str.strip() != "")
print(f"Rows with Response Target: {has_resp.sum():,} ({has_resp.sum()/len(train_df)*100:.1f}%)")
print(f"Rows without Response Target (Classification only): {(~has_resp).sum():,} ({(~has_resp).sum()/len(train_df)*100:.1f}%)")

print("=" * 60 + "\n")
