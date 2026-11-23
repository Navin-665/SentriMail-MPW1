import os
import sys
import time
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

CFPB_PATH = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1\rawdata\complaints.csv"

CFPB_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "billing",
    "Credit reporting": "billing",
    "Debt collection": "billing",
    "Mortgage": "billing",
    "Credit card or prepaid card": "billing",
    "Credit card": "billing",
    "Prepaid card": "billing",
    "Checking or savings account": "billing",
    "Bank account or service": "billing",
    "Money transfer, virtual currency, or money service": "billing",
    "Money transfers": "billing",
    "Vehicle loan or lease": "billing",
    "Consumer Loan": "billing",
    "Student loan": "billing",
    "Payday loan, title loan, or personal loan": "billing",
    "Payday loan": "billing",
    "Other financial service": "other",
}

print("=" * 60)
print("INSPECTING CFPB DATASET (FAST VECTORIZED)...")
print("=" * 60)
t0 = time.time()

chunk_size = 500_000
usecols = ["Product", "Sub-product", "Issue", "Sub-issue", "Consumer complaint narrative", "Company", "Company response to consumer"]

total_rows = 0
total_narratives = 0
narratives_gte_10 = 0
narratives_gte_50 = 0

product_counter = Counter()
subproduct_counter = Counter()
issue_counter = Counter()
canonical_counter = Counter()

char_lens = []
word_lens = []

# Deduplication set
seen_texts = set()
total_duplicates = 0

chunk_idx = 0
for chunk in pd.read_csv(CFPB_PATH, usecols=usecols, chunksize=chunk_size, low_memory=False):
    chunk_idx += 1
    total_rows += len(chunk)
    
    # Counts
    for p in chunk["Product"].dropna():
        product_counter[p] += 1
    for sp in chunk["Sub-product"].dropna():
        subproduct_counter[sp] += 1
    for iss in chunk["Issue"].dropna():
        issue_counter[iss] += 1
        
    narr_mask = chunk["Consumer complaint narrative"].notna()
    narr_df = chunk[narr_mask]
    n_cnt = len(narr_df)
    total_narratives += n_cnt
    
    if n_cnt > 0:
        texts = narr_df["Consumer complaint narrative"].astype(str).str.strip()
        lens = texts.str.len()
        words = texts.str.split().str.len()
        
        narratives_gte_10 += (lens >= 10).sum()
        gte_50_mask = lens >= 50
        narratives_gte_50 += gte_50_mask.sum()
        
        # Category breakdown for narrative rows
        for prod in narr_df["Product"].dropna():
            c = CFPB_MAP.get(prod, "other")
            canonical_counter[c] += 1
            
        # Sampling for length stats
        sample_k = min(len(lens), 5000)
        sample_idx = np.random.choice(len(lens), sample_k, replace=False)
        char_lens.extend(lens.iloc[sample_idx].tolist())
        word_lens.extend(words.iloc[sample_idx].tolist())
        
        # Deduplication using set operations
        gte_50_texts = texts[gte_50_mask].values
        for t in gte_50_texts:
            if t in seen_texts:
                total_duplicates += 1
            else:
                seen_texts.add(t)

    print(f"  Processed {total_rows:,} rows ({total_narratives:,} narratives, {len(seen_texts):,} unique) in {time.time()-t0:.1f}s...")

print(f"\nFinished CFPB scan in {time.time()-t0:.1f}s")

char_arr = np.array(char_lens)
word_arr = np.array(word_lens)
est_token_arr = word_arr * 1.35

print("\n" + "=" * 30 + " CFPB SUMMARY " + "=" * 30)
print(f"• Total Rows in raw CSV: {total_rows:,}")
print(f"• Usable Narratives (non-null): {total_narratives:,} ({total_narratives/total_rows*100:.1f}%)")
print(f"  - Narratives >= 10 chars: {narratives_gte_10:,}")
print(f"  - Narratives >= 50 chars: {narratives_gte_50:,}")
print(f"\n• DEDUPLICATION & USEFUL EXAMPLES:")
print(f"  - Unique Narratives (>=50 chars): {len(seen_texts):,}")
print(f"  - Duplicate Narratives (>=50 chars): {total_duplicates:,} ({total_duplicates/max(narratives_gte_50, 1)*100:.2f}% duplicate rate)")
print(f"  - Total Useful Labelled Examples (unique, >=50 chars): {len(seen_texts):,}")

print(f"\n• TAXONOMY HIERARCHY & CATEGORIES:")
print(f"  - Unique Raw Products: {len(product_counter)}")
print(f"  - Unique Raw Sub-products: {len(subproduct_counter)}")
print(f"  - Unique Raw Issues: {len(issue_counter)}")
print(f"  - Unique Mapped SentriMail Canonical Categories: {len(canonical_counter)}")

print(f"\n• TOP 10 RAW PRODUCTS (All rows):")
for p, c in product_counter.most_common(10):
    print(f"    {p}: {c:,} ({c/total_rows*100:.2f}%)")

print(f"\n• CANONICAL CATEGORY DISTRIBUTION (Narrative rows):")
tot_can = sum(canonical_counter.values())
for cat, c in canonical_counter.most_common():
    print(f"    {cat}: {c:,} ({c/tot_can*100:.2f}%)")

print(f"\n• COMPLAINT LENGTH STATS (Characters):")
print(f"    Mean: {char_arr.mean():.1f} | Median: {np.median(char_arr):.1f} | Std: {char_arr.std():.1f}")
print(f"    Min: {char_arr.min()} | Max: {char_arr.max():,}")
print(f"    75th Pct: {np.percentile(char_arr, 75):.1f} | 90th Pct: {np.percentile(char_arr, 90):.1f} | 95th Pct: {np.percentile(char_arr, 95):.1f} | 99th Pct: {np.percentile(char_arr, 99):.1f}")

print(f"\n• COMPLAINT LENGTH STATS (Words):")
print(f"    Mean: {word_arr.mean():.1f} | Median: {np.median(word_arr):.1f} | Std: {word_arr.std():.1f}")
print(f"    Min: {word_arr.min()} | Max: {word_arr.max():,}")
print(f"    75th Pct: {np.percentile(word_arr, 75):.1f} | 90th Pct: {np.percentile(word_arr, 90):.1f} | 95th Pct: {np.percentile(word_arr, 95):.1f} | 99th Pct: {np.percentile(word_arr, 99):.1f}")

print(f"\n• ESTIMATED SUBWORD TOKEN COUNTS (BPE / T5 WordPiece ~1.35x words):")
print(f"    Mean Tokens: ~{est_token_arr.mean():.0f}")
print(f"    Median Tokens: ~{np.median(est_token_arr):.0f}")
print(f"    75th Pct: ~{np.percentile(est_token_arr, 75):.0f} tokens")
print(f"    90th Pct: ~{np.percentile(est_token_arr, 90):.0f} tokens")
print(f"    95th Pct: ~{np.percentile(est_token_arr, 95):.0f} tokens")
print(f"    % Narratives Exceeding 256 tokens: {(est_token_arr > 256).sum() / len(est_token_arr) * 100:.1f}%")
print(f"    % Narratives Exceeding 512 tokens: {(est_token_arr > 512).sum() / len(est_token_arr) * 100:.1f}%")
print("=" * 60 + "\n")
