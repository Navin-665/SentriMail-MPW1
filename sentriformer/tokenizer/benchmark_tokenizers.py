"""
SentriTokenizer Comprehensive Benchmark Suite
Evaluates 16k vs 24k vs 32k vocabularies across all 4 datasets on:
- Tokens / word, Tokens / complaint, Tokens / response
- Characters / token & Compression ratio
- UNK rate & Vocabulary utilization
- Context window coverage (256, 384, 512, 768 tokens)
- Embedding parameter cost (for 10-20M Transformer design)
"""
import os
import sys
import json
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
DATA_DIR = os.path.join(ROOT_DIR, "sentriformer", "data")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "sentriformer", "tokenizer")
VAL_JSONL = os.path.join(DATA_DIR, "val.jsonl")


def load_evaluation_samples(val_jsonl: str, max_per_source: int = 5000) -> Dict[str, List[Dict[str, str]]]:
    """Loads a balanced set of held-out validation records from all 4 sources."""
    samples = {"cfpb": [], "twcs": [], "customer_support": [], "bitext": []}
    
    with open(val_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            src = rec.get("source", "")
            if src in samples and len(samples[src]) < max_per_source:
                samples[src].append({
                    "text": rec.get("text", ""),
                    "response": rec.get("response", "") if rec.get("response") else ""
                })
                
    return samples


def benchmark_single_tokenizer(tokenizer_path: str, samples_by_source: Dict[str, List[Dict[str, str]]], vocab_name: str) -> Dict[str, Any]:
    """Runs rigorous empirical benchmarking on a single trained tokenizer."""
    print(f"\n--- Benchmarking {vocab_name} ({tokenizer_path}) ---")
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    unk_id = tok.unk_token_id
    
    results = {
        "vocab_size": len(tok),
        "per_source": {},
        "overall": {},
    }
    
    all_token_lengths = []
    all_char_lengths = []
    all_word_counts = []
    all_unk_counts = 0
    all_total_tokens = 0
    
    # Context coverage cutoffs
    cutoffs = [256, 384, 512, 768]
    
    for src, sample_list in samples_by_source.items():
        src_tokens = []
        src_chars = []
        src_words = []
        src_unks = 0
        src_tot_toks = 0
        
        for item in sample_list:
            t = item["text"]
            if not t:
                continue
            enc = tok.encode(t, add_special_tokens=True)
            num_tok = len(enc)
            num_char = len(t)
            num_word = len(t.split())
            
            # Count UNKs
            num_unk = enc.count(unk_id)
            src_unks += num_unk
            src_tot_toks += num_tok
            
            src_tokens.append(num_tok)
            src_chars.append(num_char)
            src_words.append(num_word)
            
            # Also evaluate response if present
            r = item.get("response")
            if r:
                r_enc = tok.encode(r, add_special_tokens=True)
                src_tokens.append(len(r_enc))
                src_chars.append(len(r))
                src_words.append(len(r.split()))
                src_unks += r_enc.count(unk_id)
                src_tot_toks += len(r_enc)

        tok_arr = np.array(src_tokens)
        char_arr = np.array(src_chars)
        word_arr = np.array(src_words)
        
        results["per_source"][src] = {
            "samples_evaluated": len(sample_list),
            "total_tokens": int(src_tot_toks),
            "avg_tokens_per_text": round(float(np.mean(tok_arr)), 2),
            "avg_tokens_per_word": round(float(np.sum(tok_arr) / max(1, np.sum(word_arr))), 2),
            "chars_per_token": round(float(np.sum(char_arr) / max(1, np.sum(tok_arr))), 2),
            "compression_ratio": round(float(np.sum(char_arr) / max(1, np.sum(tok_arr))), 2),
            "unk_rate_pct": round(float(src_unks / max(1, src_tot_toks) * 100), 4),
        }
        
        all_token_lengths.extend(src_tokens)
        all_char_lengths.extend(src_chars)
        all_word_counts.extend(src_words)
        all_unk_counts += src_unks
        all_total_tokens += src_tot_toks

    all_tok_arr = np.array(all_token_lengths)
    all_char_arr = np.array(all_char_lengths)
    all_word_arr = np.array(all_word_counts)
    
    # Global metrics
    results["overall"] = {
        "total_tokens_evaluated": int(all_total_tokens),
        "avg_tokens_per_text": round(float(np.mean(all_tok_arr)), 2),
        "median_tokens_per_text": round(float(np.median(all_tok_arr)), 2),
        "avg_tokens_per_word": round(float(np.sum(all_tok_arr) / max(1, np.sum(all_word_arr))), 2),
        "chars_per_token": round(float(np.sum(all_char_arr) / max(1, np.sum(all_tok_arr))), 2),
        "compression_ratio": round(float(np.sum(all_char_arr) / max(1, np.sum(all_tok_arr))), 2),
        "unk_rate_pct": round(float(all_unk_counts / max(1, all_total_tokens) * 100), 4),
    }
    
    # Coverage at context thresholds
    results["context_coverage"] = {}
    for c in cutoffs:
        pct = float(np.sum(all_tok_arr <= c) / len(all_tok_arr) * 100)
        results["context_coverage"][f"{c}_tokens"] = round(pct, 2)
        
    # Model Parameter Embedding Cost Calculation
    # For d_model = 256 and d_model = 384:
    results["embedding_param_cost"] = {
        "d_model_256": f"{len(tok) * 256 / 1e6:.2f}M params ({len(tok) * 256:,} weights)",
        "d_model_384": f"{len(tok) * 384 / 1e6:.2f}M params ({len(tok) * 384:,} weights)",
        "d_model_512": f"{len(tok) * 512 / 1e6:.2f}M params ({len(tok) * 512:,} weights)",
    }
    
    return results


def main():
    print("=" * 70)
    print("         SENTRITOKENIZER BENCHMARK EVALUATION (16k vs 24k vs 32k)")
    print("=" * 70)
    
    print("Loading held-out validation samples across all 4 datasets...")
    samples = load_evaluation_samples(VAL_JSONL, max_per_source=5000)
    for k, v in samples.items():
        print(f"  - {k}: {len(v):,} records")
        
    vocab_dirs = [
        ("16k", os.path.join(TOKENIZER_DIR, "vocab_16k")),
        ("24k", os.path.join(TOKENIZER_DIR, "vocab_24k")),
        ("32k", os.path.join(TOKENIZER_DIR, "vocab_32k")),
    ]
    
    all_benchmarks = {}
    for name, path in vocab_dirs:
        if os.path.exists(path):
            all_benchmarks[name] = benchmark_single_tokenizer(path, samples, name)
            
    # Save benchmark results to JSON
    out_json = os.path.join(TOKENIZER_DIR, "tokenizer_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)
    print(f"\nSaved raw benchmark metrics to: {out_json}")
    
    # Print formatted comparative table
    print("\n" + "=" * 80)
    print("                      COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'16k Vocab':<15} | {'24k Vocab':<15} | {'32k Vocab':<15}")
    print("-" * 80)
    
    for metric, key in [
        ("Vocabulary Size", "vocab_size"),
        ("Avg Tokens / Text", "avg_tokens_per_text"),
        ("Avg Tokens / Word", "avg_tokens_per_word"),
        ("Chars / Token (Compression)", "chars_per_token"),
        ("UNK Rate (%)", "unk_rate_pct"),
    ]:
        v16 = all_benchmarks.get("16k", {}).get("overall", {}).get(key, all_benchmarks.get("16k", {}).get(key, "N/A"))
        v24 = all_benchmarks.get("24k", {}).get("overall", {}).get(key, all_benchmarks.get("24k", {}).get(key, "N/A"))
        v32 = all_benchmarks.get("32k", {}).get("overall", {}).get(key, all_benchmarks.get("32k", {}).get(key, "N/A"))
        print(f"{metric:<30} | {str(v16):<15} | {str(v24):<15} | {str(v32):<15}")
        
    print("-" * 80)
    print("CONTEXT WINDOW COVERAGE (% of texts retained without truncation):")
    for c in ["256_tokens", "384_tokens", "512_tokens", "768_tokens"]:
        v16 = all_benchmarks.get("16k", {}).get("context_coverage", {}).get(c, "N/A")
        v24 = all_benchmarks.get("24k", {}).get("context_coverage", {}).get(c, "N/A")
        v32 = all_benchmarks.get("32k", {}).get("context_coverage", {}).get(c, "N/A")
        print(f"  {c:<28} | {str(v16) + '%':<15} | {str(v24) + '%':<15} | {str(v32) + '%':<15}")
        
    print("-" * 80)
    print("EMBEDDING MATRIX COST (Weight Budget for 10-20M Model):")
    for d in ["d_model_256", "d_model_384", "d_model_512"]:
        v16 = all_benchmarks.get("16k", {}).get("embedding_param_cost", {}).get(d, "N/A")
        v24 = all_benchmarks.get("24k", {}).get("embedding_param_cost", {}).get(d, "N/A")
        v32 = all_benchmarks.get("32k", {}).get("embedding_param_cost", {}).get(d, "N/A")
        print(f"  {d:<28} | {str(v16):<15} | {str(v24):<15} | {str(v32):<15}")
    print("=" * 80)


if __name__ == "__main__":
    main()
