import os
import sys
import time
import re
import gc
import pandas as pd
import numpy as np
from collections import Counter
import pyarrow.csv as pv
import pyarrow.compute as pc

# Ensure unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

CFPB_PATH = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1\rawdata\complaints.csv"
TWCS_PATH = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1\rawdata\twcs.csv"

# Category mapping definition for SentriMail
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

def map_category(raw_cat):
    if not raw_cat or not isinstance(raw_cat, str):
        return "other"
    clean = raw_cat.strip()
    if clean in CFPB_MAP:
        return CFPB_MAP[clean]
    if clean.upper() in CFPB_MAP:
        return CFPB_MAP[clean.upper()]
    if clean.lower() in CFPB_MAP:
        return CFPB_MAP[clean.lower()]
    return "other"

def inspect_twcs():
    print("=" * 60)
    print("INSPECTING TWCS DATASET...")
    print("=" * 60)
    t0 = time.time()
    
    # 1. Read TWCS
    df = pd.read_csv(TWCS_PATH, low_memory=False)
    print(f"Loaded TWCS: {len(df):,} total rows in {time.time()-t0:.2f}s")
    
    df["inbound"] = df["inbound"].astype(str).str.upper() == "TRUE"
    df["text"] = df["text"].astype(str)
    
    total_tweets = len(df)
    inbound_df = df[df["inbound"]].copy()
    outbound_df = df[~df["inbound"]].copy()
    
    total_inbound = len(inbound_df)
    total_outbound = len(outbound_df)
    
    # Unique entities
    unique_customer_authors = inbound_df["author_id"].nunique()
    unique_brands = outbound_df["author_id"].nunique()
    
    # 2. Conversation structures
    # Top-of-thread: inbound tweets where in_response_to_tweet_id is null
    top_of_thread = inbound_df[inbound_df["in_response_to_tweet_id"].isna()].copy()
    mid_thread_inbound = inbound_df[inbound_df["in_response_to_tweet_id"].notna()].copy()
    
    # Conversations initiated by customers
    total_conversations = len(top_of_thread)
    
    # Response Tweet IDs mapping
    # Note: response_tweet_id can be comma-separated list of IDs
    top_of_thread["first_reply_id"] = top_of_thread["response_tweet_id"].astype(str).str.split(",").str[0].str.strip()
    top_of_thread["first_reply_id_num"] = pd.to_numeric(top_of_thread["first_reply_id"], errors="coerce")
    
    outbound_subset = outbound_df[["tweet_id", "author_id", "text", "created_at"]].dropna(subset=["tweet_id"]).copy()
    outbound_subset["tweet_id_num"] = pd.to_numeric(outbound_subset["tweet_id"], errors="coerce")
    
    # Vectorized merge
    print("Merging customer inquiries with brand replies...")
    merged = pd.merge(
        top_of_thread,
        outbound_subset,
        left_on="first_reply_id_num",
        right_on="tweet_id_num",
        suffixes=("_customer", "_brand"),
        how="inner"
    )
    
    total_paired = len(merged)
    unpaired_top_of_thread = total_conversations - total_paired
    
    # 3. Conversation length metrics (turns and characters)
    # Estimate conversation turns across dataset
    # Grouping by root tweet or tracking chains
    # Approximate turns: 1 initial customer + 1 brand reply + mid-thread turns
    # Distribution of mid-thread replies per conversation
    # Character and word length analysis
    c_texts = merged["text_customer"].astype(str)
    b_texts = merged["text_brand"].astype(str)
    
    c_words = c_texts.str.split().str.len()
    b_words = b_texts.str.split().str.len()
    c_chars = c_texts.str.len()
    b_chars = b_texts.str.len()
    
    # 4. Response quality analysis
    # Filter definitions
    MIN_COMPLAINT_WORDS = 4
    MIN_RESPONSE_WORDS = 3
    BOILERPLATE_PATTERN = r"\bDM\b|direct message|reach out to us|send us a DM|link in bio|please call|check out this link|http[s]?://"
    
    is_short = (c_words < MIN_COMPLAINT_WORDS) | (b_words < MIN_RESPONSE_WORDS)
    is_boilerplate = b_texts.str.contains(BOILERPLATE_PATTERN, case=False, regex=True)
    
    # Clean text: remove @mentions
    clean_c = c_texts.str.replace(r"@\w+", "", regex=True).str.strip()
    clean_b = b_texts.str.replace(r"@\w+", "", regex=True).str.strip()
    is_empty_after_clean = (clean_c.str.len() < 10) | (clean_b.str.len() < 10)
    
    valid_pairs_mask = (~is_short) & (~is_boilerplate) & (~is_empty_after_clean)
    usable_pairs_count = valid_pairs_mask.sum()
    
    # Top brands
    top_brands = outbound_df["author_id"].value_counts().head(10)
    
    print("\n" + "=" * 30 + " TWCS SUMMARY " + "=" * 30)
    print(f"• Total Tweets: {total_tweets:,}")
    print(f"  - Inbound (Customer): {total_inbound:,} ({total_inbound/total_tweets*100:.1f}%)")
    print(f"  - Outbound (Brand): {total_outbound:,} ({total_outbound/total_tweets*100:.1f}%)")
    print(f"• Unique Customer Authors: {unique_customer_authors:,}")
    print(f"• Unique Brand Handles: {unique_brands:,}")
    print(f"\n• CONVERSATIONS:")
    print(f"  - Total Inbound Conversations Initiated (Top-of-Thread): {total_conversations:,}")
    print(f"  - Mid-thread Follow-up Inbound Tweets: {len(mid_thread_inbound):,}")
    print(f"  - Conversations with Direct Brand Reply: {total_paired:,} ({total_paired/total_conversations*100:.1f}% response rate)")
    print(f"  - Conversations with No/Missing Brand Reply: {unpaired_top_of_thread:,} ({unpaired_top_of_thread/total_conversations*100:.1f}%)")
    
    print(f"\n• AVERAGE LENGTHS (for Paired Interactions):")
    print(f"  - Customer Complaint Chars: Mean = {c_chars.mean():.1f}, Median = {np.median(c_chars):.1f}, Max = {c_chars.max():,}")
    print(f"  - Customer Complaint Words: Mean = {c_words.mean():.1f}, Median = {np.median(c_words):.1f}, Max = {c_words.max():,}")
    print(f"  - Brand Response Chars:     Mean = {b_chars.mean():.1f}, Median = {np.median(b_chars):.1f}, Max = {b_chars.max():,}")
    print(f"  - Brand Response Words:     Mean = {b_words.mean():.1f}, Median = {np.median(b_words):.1f}, Max = {b_words.max():,}")
    
    print(f"\n• RESPONSE QUALITY & DEFLECTION:")
    print(f"  - Dropped Short (<4 complaint / <3 response words): {is_short.sum():,} ({is_short.sum()/total_paired*100:.1f}%)")
    print(f"  - Boilerplate / Channel Deflections (DM, Phone, URL): {is_boilerplate.sum():,} ({is_boilerplate.sum()/total_paired*100:.1f}%)")
    print(f"  - Empty / Too Short after @mention strip (<10 chars): {is_empty_after_clean.sum():,} ({is_empty_after_clean.sum()/total_paired*100:.1f}%)")
    
    print(f"\n• USABLE TEXT PAIRS:")
    print(f"  - Total High-Quality Clean Pairs (ready for training): {usable_pairs_count:,} ({usable_pairs_count/total_paired*100:.1f}% of pairs)")
    
    print(f"\n• TOP ACTIVE BRANDS:")
    for b, c in top_brands.items():
        print(f"    @{b}: {c:,} responses")
    print("=" * 60 + "\n")
    
    del df, inbound_df, outbound_df, top_of_thread, merged, outbound_subset
    gc.collect()

def inspect_cfpb():
    print("=" * 60)
    print("INSPECTING CFPB DATASET...")
    print("=" * 60)
    t0 = time.time()
    
    # Read CFPB in chunks
    chunk_size = 200_000
    usecols = ["Product", "Sub-product", "Issue", "Consumer complaint narrative"]
    
    total_rows = 0
    total_narratives = 0
    narratives_gte_10 = 0
    narratives_gte_50 = 0
    
    product_counter = Counter()
    canonical_counter = Counter()
    
    # Word count and char length stats accumulator
    char_lens = []
    word_lens = []
    
    # Duplicate tracking via hash set
    import hashlib
    seen_hashes = set()
    duplicate_count = 0
    useful_labelled_count = 0
    
    print("Processing CFPB CSV in streaming chunks...")
    chunk_idx = 0
    for chunk in pd.read_csv(CFPB_PATH, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk_idx += 1
        total_rows += len(chunk)
        
        # Count all products
        for p in chunk["Product"].dropna():
            product_counter[p] += 1
            
        # Non-null narratives
        narr_mask = chunk["Consumer complaint narrative"].notna()
        narr_df = chunk[narr_mask]
        n_cnt = len(narr_df)
        total_narratives += n_cnt
        
        if n_cnt > 0:
            texts = narr_df["Consumer complaint narrative"].astype(str)
            lens = texts.str.len()
            words = texts.str.split().str.len()
            
            gte_10 = (lens >= 10).sum()
            gte_50 = (lens >= 50).sum()
            narratives_gte_10 += gte_10
            narratives_gte_50 += gte_50
            
            # Canonical categories
            for prod in narr_df["Product"].dropna():
                c = map_category(prod)
                canonical_counter[c] += 1
                
            # Sample for length distribution (take 5%)
            sample_k = min(len(lens), 4000)
            sample_idx = np.random.choice(len(lens), sample_k, replace=False)
            char_lens.extend(lens.iloc[sample_idx].tolist())
            word_lens.extend(words.iloc[sample_idx].tolist())
            
            # Check duplicates on strings >= 50 chars
            gte_50_df = narr_df[lens >= 50]
            for _, r in gte_50_df.iterrows():
                txt = r["Consumer complaint narrative"].strip()
                h = hashlib.md5(txt.encode('utf-8')).hexdigest()
                if h in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(h)
                    useful_labelled_count += 1
                    
        if chunk_idx % 5 == 0:
            print(f"  Processed {total_rows:,} rows ({total_narratives:,} narratives) in {time.time()-t0:.1f}s...")

    print(f"\nFinished CFPB scan in {time.time()-t0:.2f}s")
    
    char_arr = np.array(char_lens)
    word_arr = np.array(word_lens)
    
    # Token estimation:
    # Rule of thumb: 1 word ~ 1.3 - 1.4 tokens (T5/WordPiece/BPE)
    # Plus let's compute directly on the word array
    est_token_arr = word_arr * 1.35
    
    print("\n" + "=" * 30 + " CFPB SUMMARY " + "=" * 30)
    print(f"• Total Rows: {total_rows:,}")
    print(f"• Total Usable Narratives (non-null): {total_narratives:,} ({total_narratives/total_rows*100:.1f}% of all complaints)")
    print(f"  - Narratives >= 10 chars: {narratives_gte_10:,}")
    print(f"  - Narratives >= 50 chars: {narratives_gte_50:,}")
    print(f"• Duplicate Rate:")
    print(f"  - Unique Narratives (>=50 chars): {len(seen_hashes):,}")
    print(f"  - Duplicate Narratives: {duplicate_count:,} ({duplicate_count/max(narratives_gte_50, 1)*100:.2f}% duplicate rate)")
    print(f"• Useful Labelled Examples (deduplicated, >=50 chars, labelled): {useful_labelled_count:,}")
    
    print(f"\n• CATEGORIES & TAXONOMY:")
    print(f"  - Unique Raw CFPB Products: {len(product_counter)}")
    print(f"  - Unique Canonical Categories Mapped: {len(canonical_counter)}")
    
    print(f"\n• RAW PRODUCT DISTRIBUTION (All complaints):")
    for p, c in product_counter.most_common():
        print(f"    {p}: {c:,} ({c/total_rows*100:.2f}%)")
        
    print(f"\n• CANONICAL CATEGORY DISTRIBUTION (Complaints with Narratives):")
    tot_can = sum(canonical_counter.values())
    for cat, c in canonical_counter.most_common():
        print(f"    {cat}: {c:,} ({c/tot_can*100:.2f}%)")
        
    print(f"\n• COMPLAINT LENGTH STATS (Characters):")
    print(f"    Mean: {char_arr.mean():.1f} chars")
    print(f"    Median: {np.median(char_arr):.1f} chars")
    print(f"    Std Dev: {char_arr.std():.1f} chars")
    print(f"    Min: {char_arr.min()} chars")
    print(f"    Max: {char_arr.max():,} chars")
    print(f"    75th Percentile: {np.percentile(char_arr, 75):.1f} chars")
    print(f"    90th Percentile: {np.percentile(char_arr, 90):.1f} chars")
    print(f"    95th Percentile: {np.percentile(char_arr, 95):.1f} chars")
    print(f"    99th Percentile: {np.percentile(char_arr, 99):.1f} chars")

    print(f"\n• COMPLAINT LENGTH STATS (Words):")
    print(f"    Mean: {word_arr.mean():.1f} words")
    print(f"    Median: {np.median(word_arr):.1f} words")
    print(f"    Std Dev: {word_arr.std():.1f} words")
    print(f"    Min: {word_arr.min()} words")
    print(f"    Max: {word_arr.max():,} words")
    print(f"    75th Percentile: {np.percentile(word_arr, 75):.1f} words")
    print(f"    90th Percentile: {np.percentile(word_arr, 90):.1f} words")
    print(f"    95th Percentile: {np.percentile(word_arr, 95):.1f} words")
    print(f"    99th Percentile: {np.percentile(word_arr, 99):.1f} words")

    print(f"\n• ESTIMATED TOKEN COUNT STATS (Subwords / BPE):")
    print(f"    Mean: ~{est_token_arr.mean():.0f} tokens")
    print(f"    Median: ~{np.median(est_token_arr):.0f} tokens")
    print(f"    90th Percentile: ~{np.percentile(est_token_arr, 90):.0f} tokens")
    print(f"    95th Percentile: ~{np.percentile(est_token_arr, 95):.0f} tokens")
    print(f"    % Exceeding 256 tokens: {(est_token_arr > 256).sum() / len(est_token_arr) * 100:.1f}%")
    print(f"    % Exceeding 512 tokens: {(est_token_arr > 512).sum() / len(est_token_arr) * 100:.1f}%")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    inspect_twcs()
    inspect_cfpb()
