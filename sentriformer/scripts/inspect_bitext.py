import sys
from datasets import load_dataset
import pandas as pd

print("=" * 60)
print("INSPECTING BITEXT DATASET...")
print("=" * 60)

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
df = ds.to_pandas()

print(f"• Total Bitext Examples: {len(df):,}")
print("• Columns:", df.columns.tolist())

print("\n--- Intent / Category Distribution ---")
if "category" in df.columns:
    print(df["category"].value_counts())
elif "intent" in df.columns:
    print(df["intent"].value_counts().head(10))

print("\n--- Text & Response Length Stats ---")
inst_lens = df["instruction"].astype(str).str.len()
resp_lens = df["response"].astype(str).str.len()
print(f"Instruction Chars: Mean = {inst_lens.mean():.1f}, Median = {inst_lens.median():.1f}, Max = {inst_lens.max()}")
print(f"Response Chars:    Mean = {resp_lens.mean():.1f}, Median = {resp_lens.median():.1f}, Max = {resp_lens.max()}")

print("=" * 60 + "\n")
