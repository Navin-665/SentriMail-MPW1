"""
==========================================================================
  [M9.3.1] Dialogue SFT Dataset Inspection & Distribution Analysis
==========================================================================
  Inspects dialogue_train, dialogue_val, and dialogue_test:
  - Row counts and source distributions (TWCS vs Bitext vs others)
  - Token length distributions for 'context' (Input) and 'response' (Target)
  - Real customer support dialogue examples
==========================================================================
"""
import os
import sys
import pyarrow.parquet as pq
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

sys.stdout.reconfigure(encoding="utf-8")

TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def inspect_dialogue_splits():
    print("=" * 80)
    print("      [M9.3.1] DIALOGUE SFT DATASET INSPECTION & TOKEN LENGTH AUDIT")
    print("=" * 80)
    
    splits = ["train", "val", "test"]
    split_tables = {}
    
    for s in splits:
        path = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", f"dialogue_{s}.parquet")
        t = pq.read_table(path)
        split_tables[s] = t
        print(f"\n📂 dialogue_{s}.parquet:")
        print(f"  - Total Dialogue Pairs: {t.num_rows:,}")
        print(f"  - Columns: {t.column_names}")
        
        sources = t["source"].to_pylist()
        src_counts = Counter(sources)
        print(f"  - Data Sources Breakdown:")
        for src, count in src_counts.most_common():
            print(f"      • {src:15s}: {count:>8,} ({count/len(sources)*100:>5.1f}%)")
            
    # Sample token length analysis on training data
    train_t = split_tables["train"]
    sample_size = min(10000, train_t.num_rows)
    sub = train_t.slice(0, sample_size)
    
    contexts = sub["context"].to_pylist()
    responses = sub["response"].to_pylist()
    
    ctx_lengths = [len(tok.encode(c, add_special_tokens=False)) for c in contexts]
    resp_lengths = [len(tok.encode(r, add_special_tokens=False)) for r in responses]
    
    print("\n" + "-" * 80)
    print(f"📊 Token Length Distribution (Sampled across {sample_size:,} pairs):")
    print("-" * 80)
    print(f"  Context Input Tokens (Encoder):")
    print(f"    - Mean:   {np.mean(ctx_lengths):.1f} tokens")
    print(f"    - Median: {np.median(ctx_lengths):.1f} tokens")
    print(f"    - 95th %: {np.percentile(ctx_lengths, 95):.1f} tokens")
    print(f"    - Max:    {np.max(ctx_lengths)} tokens")
    print()
    print(f"  Response Target Tokens (Decoder):")
    print(f"    - Mean:   {np.mean(resp_lengths):.1f} tokens")
    print(f"    - Median: {np.median(resp_lengths):.1f} tokens")
    print(f"    - 95th %: {np.percentile(resp_lengths, 95):.1f} tokens")
    print(f"    - Max:    {np.max(resp_lengths)} tokens")
    
    print("\n" + "-" * 80)
    print("📝 Representative Real-World Dialogue Samples:")
    print("-" * 80)
    for i in [0, 50, 1500, 5000]:
        if i < len(contexts):
            src = sub["source"][i].as_py()
            print(f"\n[Sample {i+1} | Source: {src}]")
            print(f"  📥 Customer Context:")
            print(f"     \"{contexts[i]}\"")
            print(f"  📤 Agent Response:")
            print(f"     \"{responses[i]}\"")
            
    print("\n" + "=" * 80)


if __name__ == "__main__":
    inspect_dialogue_splits()
