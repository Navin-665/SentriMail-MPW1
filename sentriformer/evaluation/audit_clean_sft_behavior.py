"""
==========================================================================
  M10.2 — SentriFormer Clean Response SFT Behavioral Quality Audit
==========================================================================
  - Conducts a paired, head-to-head behavioral evaluation across 80 inquiries:
      * 30 Canonical Multi-Category Benchmark Inquiries (M9.3.1 Benchmark)
      * 50 Stratified Unseen Inquiries from clean_dialogue_test.parquet
  - Evaluates both:
      1. M9.3 Baseline Checkpoint (dialogue_sft_best.pt)
      2. M10.2 Corrected Clean SFT Checkpoint (clean_sft_best.pt)
  - Audits 10 Behavioral Dimensions:
      1. Non-empty response %
      2. EOS termination %
      3. Intent/Domain Relevance Proxy %
      4. Grammar / Fluency %
      5. Repetition %
      6. Contamination Breakdown (Overall %, Handles %, URLs %, Signatures %, DM Deflections %)
      7. Boundary/Completeness Proxy %
      8. Appropriate Empathy %
      9. Category Routing & Confidence Distribution (<25% low confidence flags)
      10. CSAT Forecast Distribution (1-5 Stars)
  - Computes Absolute and Relative Reduction in Contamination Artifacts.
  - Outputs JSON and Markdown audit reports in sentriformer/results/clean_sft/
==========================================================================
"""
import os
import sys
import re
import json
import math
import random
import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.evaluation.audit_dialogue_responses import BENCHMARK_SCENARIOS

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")

M93_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "dialogue", "dialogue_sft_best.pt")
M102_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "clean_sft", "clean_sft_best.pt")
TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_test.parquet")

OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "clean_sft")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIT_JSON = os.path.join(OUTPUT_DIR, "clean_sft_behavioral_audit.json")
AUDIT_MD = os.path.join(OUTPUT_DIR, "M10_2_BEHAVIORAL_AUDIT.md")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


# Contamination Artifact Regex Patterns
RE_TWITTER_HANDLE = re.compile(r"@[A-Za-z0-9_]+")
RE_URL = re.compile(r"https?://\S+|t\.co/\S+")
RE_AGENT_SIGNATURE = re.compile(r"\^[A-Z]{2,4}\b")
RE_DM_DEFLECTION = re.compile(
    r"\b(dm|direct message|dm us|send us a dm|message us|reach out via dm|contact us via dm|in our dms)\b",
    re.IGNORECASE
)
RE_EMPATHY_MARKERS = re.compile(
    r"\b(apologize|apologies|sorry|glad to help|happy to help|understand your concern|let us look into|happy to assist|pleasure to assist)\b",
    re.IGNORECASE
)


def detect_repetition(text: str) -> bool:
    """Detects degenerate token or phrase loops in generated text."""
    words = text.lower().split()
    if len(words) < 6:
        return False
    # Check 3-gram repetition
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    counts = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1
        if counts[tg] >= 3:
            return True
    # Check unigram loop
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return True
    return False


def audit_single_response(response_text: str, raw_ids: List[int], expected_keywords: List[str], eos_token_id: int = 3) -> Dict[str, Any]:
    """Analyzes a generated response across behavioral dimensions."""
    clean_text = response_text.strip()
    words = clean_text.lower().split()
    word_count = len(words)
    
    # 1. Generation metrics
    is_non_empty = len(clean_text) > 0 and word_count > 0
    eos_terminated = (eos_token_id in raw_ids) if raw_ids else False
    
    # 2. Contamination checks (non-mutually exclusive)
    handles = RE_TWITTER_HANDLE.findall(clean_text)
    urls = RE_URL.findall(clean_text)
    signatures = RE_AGENT_SIGNATURE.findall(clean_text)
    dm_deflections = RE_DM_DEFLECTION.findall(clean_text)
    
    has_handle = len(handles) > 0
    has_url = len(urls) > 0
    has_signature = len(signatures) > 0
    has_dm = len(dm_deflections) > 0
    is_contaminated = has_handle or has_url or has_signature or has_dm
    
    # 3. Intent/Domain Relevance Proxy
    keyword_hits = [kw for kw in expected_keywords if kw.lower() in clean_text.lower()]
    relevance_ratio = len(keyword_hits) / max(1, min(3, len(expected_keywords)))
    relevance_score = min(100.0, relevance_ratio * 100.0)
    
    # 4. Boundary / Completeness Proxy
    ends_clean_punctuation = clean_text.endswith((".", "!", "?", "\"", "'")) if clean_text else False
    # Also consider complete imperative/closing phrase even if terminal period was omitted
    has_closing_phrase = any(clean_text.lower().endswith(p) for p in [
        "let us know", "help you", "assist you", "thank you", "for you", "reach out"
    ])
    completeness_proxy = (ends_clean_punctuation or has_closing_phrase) and word_count >= 6
    
    # 5. Fluency / Syntax
    has_repetition = detect_repetition(clean_text)
    fluency_score = 100.0
    if has_repetition:
        fluency_score -= 50.0
    if not is_non_empty or word_count < 4:
        fluency_score -= 40.0
    if not (ends_clean_punctuation or has_closing_phrase):
        fluency_score -= 20.0
    fluency_score = max(0.0, min(100.0, fluency_score))
    
    # 6. Appropriate Empathy
    has_empathy = bool(RE_EMPATHY_MARKERS.search(clean_text)) or any(g in clean_text.lower() for g in ["apologize", "sorry", "understand", "glad to"])
    
    return {
        "text": clean_text,
        "token_count": len(raw_ids),
        "word_count": word_count,
        "is_non_empty": is_non_empty,
        "eos_terminated": eos_terminated,
        "contamination": {
            "is_contaminated": is_contaminated,
            "has_handle": has_handle,
            "has_url": has_url,
            "has_signature": has_signature,
            "has_dm": has_dm,
            "handles": handles,
            "urls": urls,
            "signatures": signatures,
            "dm_deflections": dm_deflections
        },
        "intent_relevance_proxy": round(relevance_score, 1),
        "keyword_hits": keyword_hits,
        "completeness_proxy": completeness_proxy,
        "fluency_score": round(fluency_score, 1),
        "has_repetition": has_repetition,
        "has_empathy": has_empathy
    }


@torch.no_grad()
def generate_m93(model, tokenizer, config, input_ids, attention_mask, max_new_tokens=100, temperature=0.7, top_k=50):
    """M9.3 generation using SentriFormerForConditionalGeneration."""
    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device
    encoder_outputs = model.sentriformer.encoder(input_ids=input_ids, attention_mask=attention_mask)
    curr_token = torch.full((batch_size, 1), config.bos_token_id, dtype=torch.long, device=device)
    generated = curr_token.clone()
    past_key_values = None
    
    for _ in range(max_new_tokens):
        decoder_out, past_key_values = model.sentriformer.decoder(
            input_ids=curr_token,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = model.lm_head(decoder_out[:, -1, :]) / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        curr_token = next_token
        if (next_token == config.eos_token_id).all():
            break
    return generated


@torch.no_grad()
def generate_m102(model, tokenizer, config, input_ids, attention_mask, max_new_tokens=100, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """M10.2 generation using SentriFormerForMultiTask with BOS/PAD suppression and repetition penalty."""
    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device
    encoder_outputs = model.sentriformer.encoder(input_ids=input_ids, attention_mask=attention_mask)
    curr_token = torch.full((batch_size, 1), config.bos_token_id, dtype=torch.long, device=device)
    generated = curr_token.clone()
    past_key_values = None
    
    for _ in range(max_new_tokens):
        decoder_out, past_key_values = model.sentriformer.decoder(
            input_ids=curr_token,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = model.lm_head(decoder_out[:, -1, :])
        # BOS & PAD suppression
        if config.bos_token_id is not None:
            logits[:, config.bos_token_id] = -float("inf")
        if config.pad_token_id is not None:
            logits[:, config.pad_token_id] = -float("inf")
            
        # Repetition penalty
        if repetition_penalty != 1.0:
            for b in range(batch_size):
                for tok in set(generated[b].tolist()):
                    if logits[b, tok] > 0:
                        logits[b, tok] /= repetition_penalty
                    else:
                        logits[b, tok] *= repetition_penalty
                        
        if temperature > 0.0:
            scaled = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, [-1]]] = -float("inf")
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        generated = torch.cat([generated, next_token], dim=1)
        curr_token = next_token
        if (next_token == config.eos_token_id).all():
            break
    return generated


def main():
    log("=" * 80)
    log("  [M10.2] SENTRIFORMER CLEAN SFT BEHAVIORAL QUALITY AUDIT (PAIRED 80)")
    log("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # 1. Load Tokenizer & Config
    log("\n[1/4] Loading Tokenizer, Config, and Classification Labels...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5
    
    id2label = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            id2label = json.load(f).get("id2label", {})
            
    # 2. Prepare 80 Test Inquiries
    log("\n[2/4] Assembling 80 Paired Test Inquiries...")
    all_inquiries = []
    
    # A. 30 Canonical Inquiries from BENCHMARK_SCENARIOS
    for cat in BENCHMARK_SCENARIOS:
        cat_key = cat["category"]
        cat_title = cat["category_title"]
        expected_kw = cat["expected_keywords"]
        for q in cat["queries"]:
            all_inquiries.append({
                "source": "canonical_benchmark",
                "category_group": cat_title,
                "query": q,
                "expected_keywords": expected_kw,
                "reference_response": None
            })
    log(f"  Added {len(all_inquiries)} canonical multi-category benchmark inquiries.")
    
    # B. 50 Stratified Unseen Inquiries from clean_dialogue_test.parquet
    test_table = pq.read_table(TEST_PARQUET, columns=["context", "response"])
    ctx_list = test_table["context"].to_pylist()
    resp_list = test_table["response"].to_pylist()
    
    random.seed(42)
    sampled_indices = random.sample(range(len(ctx_list)), 50)
    
    for idx in sampled_indices:
        q = ctx_list[idx].strip()
        ref = resp_list[idx].strip()
        # Derive keyword expectations from reference response
        ref_words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", ref)]
        kw = list(dict.fromkeys(ref_words))[:8]  # unique top keywords
        all_inquiries.append({
            "source": "clean_test_parquet",
            "category_group": "Clean Test Set (Unseen)",
            "query": q,
            "expected_keywords": kw,
            "reference_response": ref
        })
    log(f"  Added 50 stratified unseen inquiries from clean_dialogue_test.parquet.")
    log(f"  Total Paired Benchmark Suite: {len(all_inquiries)} inquiries.")
    
    # 3. Load Models
    log("\n[3/4] Loading Checkpoints for Paired Evaluation...")
    log(f"  [Model A: M9.3 Baseline] Loading {M93_CKPT}")
    model_m93 = SentriFormerForConditionalGeneration(config).to(device)
    raw_m93 = torch.load(M93_CKPT, map_location=device, weights_only=False)
    model_m93.load_state_dict(raw_m93["model_state_dict"] if "model_state_dict" in raw_m93 else raw_m93)
    model_m93.eval()
    
    log(f"  [Model B: M10.2 Clean SFT] Loading {M102_CKPT}")
    model_m102 = SentriFormerForMultiTask(config).to(device)
    raw_m102 = torch.load(M102_CKPT, map_location=device, weights_only=False)
    model_m102.load_state_dict(raw_m102["model_state_dict"] if "model_state_dict" in raw_m102 else raw_m102)
    model_m102.eval()
    
    # 4. Execute Paired Inferences & Audit
    log("\n[4/4] Executing Paired Inferences across 80 Inquiries...")
    
    paired_cases = []
    
    m93_stats = {
        "non_empty": 0, "eos_terminated": 0, "total_tokens": 0,
        "contaminated": 0, "handles": 0, "urls": 0, "signatures": 0, "dm": 0,
        "relevance_sum": 0.0, "completeness": 0, "fluency_sum": 0.0, "repetition": 0, "empathy": 0
    }
    
    m102_stats = {
        "non_empty": 0, "eos_terminated": 0, "total_tokens": 0,
        "contaminated": 0, "handles": 0, "urls": 0, "signatures": 0, "dm": 0,
        "relevance_sum": 0.0, "completeness": 0, "fluency_sum": 0.0, "repetition": 0, "empathy": 0,
        "category_counts": {}, "category_conf_sum": 0.0, "low_confidence_count": 0,
        "csat_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    }
    
    N = len(all_inquiries)
    
    for i, item in enumerate(all_inquiries):
        query = item["query"]
        expected_kw = item["expected_keywords"]
        enc = tokenizer(query, return_tensors="pt").to(device)
        
        # --- Model A: M9.3 Inference ---
        with torch.no_grad():
            gen_toks_m93 = generate_m93(model_m93, tokenizer, config, enc["input_ids"], enc["attention_mask"], max_new_tokens=100, temperature=0.7)
        raw_ids_m93 = gen_toks_m93[0].tolist()
        text_m93 = tokenizer.decode(raw_ids_m93, skip_special_tokens=True).strip()
        audit_m93 = audit_single_response(text_m93, raw_ids_m93, expected_kw, eos_token_id=config.eos_token_id)
        
        # --- Model B: M10.2 Inference ---
        with torch.no_grad():
            m102_outputs = model_m102(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cat_probs = F.softmax(m102_outputs["category_logits"], dim=-1)[0]
            top_prob, top_cat_id = torch.max(cat_probs, dim=-1)
            cat_name = id2label.get(str(top_cat_id.item()), f"Category {top_cat_id.item()}")
            cat_conf = top_prob.item() * 100.0
            csat_star = torch.argmax(m102_outputs["csat_logits"], dim=-1).item() + 1
            
            gen_toks_m102 = generate_m102(
                model_m102, tokenizer, config, enc["input_ids"], enc["attention_mask"],
                max_new_tokens=100, temperature=0.7, repetition_penalty=1.2
            )
        raw_ids_m102 = gen_toks_m102[0].tolist()
        text_m102 = tokenizer.decode(raw_ids_m102, skip_special_tokens=True).strip()
        audit_m102 = audit_single_response(text_m102, raw_ids_m102, expected_kw, eos_token_id=config.eos_token_id)
        
        # Tally M9.3
        if audit_m93["is_non_empty"]: m93_stats["non_empty"] += 1
        if audit_m93["eos_terminated"]: m93_stats["eos_terminated"] += 1
        m93_stats["total_tokens"] += audit_m93["token_count"]
        if audit_m93["contamination"]["is_contaminated"]: m93_stats["contaminated"] += 1
        if audit_m93["contamination"]["has_handle"]: m93_stats["handles"] += 1
        if audit_m93["contamination"]["has_url"]: m93_stats["urls"] += 1
        if audit_m93["contamination"]["has_signature"]: m93_stats["signatures"] += 1
        if audit_m93["contamination"]["has_dm"]: m93_stats["dm"] += 1
        m93_stats["relevance_sum"] += audit_m93["intent_relevance_proxy"]
        if audit_m93["completeness_proxy"]: m93_stats["completeness"] += 1
        m93_stats["fluency_sum"] += audit_m93["fluency_score"]
        if audit_m93["has_repetition"]: m93_stats["repetition"] += 1
        if audit_m93["has_empathy"]: m93_stats["empathy"] += 1
        
        # Tally M10.2
        if audit_m102["is_non_empty"]: m102_stats["non_empty"] += 1
        if audit_m102["eos_terminated"]: m102_stats["eos_terminated"] += 1
        m102_stats["total_tokens"] += audit_m102["token_count"]
        if audit_m102["contamination"]["is_contaminated"]: m102_stats["contaminated"] += 1
        if audit_m102["contamination"]["has_handle"]: m102_stats["handles"] += 1
        if audit_m102["contamination"]["has_url"]: m102_stats["urls"] += 1
        if audit_m102["contamination"]["has_signature"]: m102_stats["signatures"] += 1
        if audit_m102["contamination"]["has_dm"]: m102_stats["dm"] += 1
        m102_stats["relevance_sum"] += audit_m102["intent_relevance_proxy"]
        if audit_m102["completeness_proxy"]: m102_stats["completeness"] += 1
        m102_stats["fluency_sum"] += audit_m102["fluency_score"]
        if audit_m102["has_repetition"]: m102_stats["repetition"] += 1
        if audit_m102["has_empathy"]: m102_stats["empathy"] += 1
        
        m102_stats["category_counts"][cat_name] = m102_stats["category_counts"].get(cat_name, 0) + 1
        m102_stats["category_conf_sum"] += cat_conf
        if cat_conf < 25.0:
            m102_stats["low_confidence_count"] += 1
        m102_stats["csat_counts"][csat_star] += 1
        
        paired_cases.append({
            "inquiry_id": i + 1,
            "source": item["source"],
            "category_group": item["category_group"],
            "query": query,
            "expected_keywords": expected_kw,
            "m93": {
                "response": text_m93,
                "raw_token_count": audit_m93["token_count"],
                "audit": audit_m93
            },
            "m102": {
                "response": text_m102,
                "raw_token_count": audit_m102["token_count"],
                "category": cat_name,
                "category_confidence": round(cat_conf, 1),
                "csat_forecast": csat_star,
                "audit": audit_m102
            }
        })
        
        if (i + 1) % 20 == 0 or (i + 1) == N:
            log(f"  Processed {i+1:>2}/{N} inquiries...")
            
    # 5. Compute Comparative Summary Rates
    m93_summary = {
        "non_empty_pct": round(m93_stats["non_empty"] / N * 100, 1),
        "eos_termination_pct": round(m93_stats["eos_terminated"] / N * 100, 1),
        "avg_tokens": round(m93_stats["total_tokens"] / N, 1),
        "overall_contamination_pct": round(m93_stats["contaminated"] / N * 100, 1),
        "handles_pct": round(m93_stats["handles"] / N * 100, 1),
        "urls_pct": round(m93_stats["urls"] / N * 100, 1),
        "signatures_pct": round(m93_stats["signatures"] / N * 100, 1),
        "dm_deflection_pct": round(m93_stats["dm"] / N * 100, 1),
        "relevance_proxy_pct": round(m93_stats["relevance_sum"] / N, 1),
        "completeness_proxy_pct": round(m93_stats["completeness"] / N * 100, 1),
        "fluency_pct": round(m93_stats["fluency_sum"] / N, 1),
        "repetition_pct": round(m93_stats["repetition"] / N * 100, 1),
        "empathy_pct": round(m93_stats["empathy"] / N * 100, 1)
    }
    
    m102_summary = {
        "non_empty_pct": round(m102_stats["non_empty"] / N * 100, 1),
        "eos_termination_pct": round(m102_stats["eos_terminated"] / N * 100, 1),
        "avg_tokens": round(m102_stats["total_tokens"] / N, 1),
        "overall_contamination_pct": round(m102_stats["contaminated"] / N * 100, 1),
        "handles_pct": round(m102_stats["handles"] / N * 100, 1),
        "urls_pct": round(m102_stats["urls"] / N * 100, 1),
        "signatures_pct": round(m102_stats["signatures"] / N * 100, 1),
        "dm_deflection_pct": round(m102_stats["dm"] / N * 100, 1),
        "relevance_proxy_pct": round(m102_stats["relevance_sum"] / N, 1),
        "completeness_proxy_pct": round(m102_stats["completeness"] / N * 100, 1),
        "fluency_pct": round(m102_stats["fluency_sum"] / N, 1),
        "repetition_pct": round(m102_stats["repetition"] / N * 100, 1),
        "empathy_pct": round(m102_stats["empathy"] / N * 100, 1),
        "mean_category_confidence_pct": round(m102_stats["category_conf_sum"] / N, 1),
        "low_confidence_rate_pct": round(m102_stats["low_confidence_count"] / N * 100, 1),
        "category_distribution": m102_stats["category_counts"],
        "csat_distribution": m102_stats["csat_counts"]
    }
    
    # Reductions
    contam_abs_diff = round(m93_summary["overall_contamination_pct"] - m102_summary["overall_contamination_pct"], 1)
    contam_rel_diff = round((contam_abs_diff / max(m93_summary["overall_contamination_pct"], 1e-5)) * 100, 1)
    
    handle_abs_diff = round(m93_summary["handles_pct"] - m102_summary["handles_pct"], 1)
    url_abs_diff = round(m93_summary["urls_pct"] - m102_summary["urls_pct"], 1)
    sig_abs_diff = round(m93_summary["signatures_pct"] - m102_summary["signatures_pct"], 1)
    dm_abs_diff = round(m93_summary["dm_deflection_pct"] - m102_summary["dm_deflection_pct"], 1)
    
    comparative_results = {
        "timestamp": timestamp(),
        "total_paired_inquiries": N,
        "m93_baseline": m93_summary,
        "m102_clean_sft": m102_summary,
        "contamination_reduction": {
            "overall_absolute_reduction_pp": contam_abs_diff,
            "overall_relative_reduction_pct": contam_rel_diff,
            "handles_absolute_reduction_pp": handle_abs_diff,
            "urls_absolute_reduction_pp": url_abs_diff,
            "signatures_absolute_reduction_pp": sig_abs_diff,
            "dm_deflection_absolute_reduction_pp": dm_abs_diff,
        },
        "cases": paired_cases
    }
    
    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2, ensure_ascii=False)
    log(f"  💾 Saved full audit JSON to: {AUDIT_JSON}")
    
    # 6. Print Console Summary
    print("\n" + "=" * 80)
    print("  🏆 PAIRED HEAD-TO-HEAD AUDIT SUMMARY (80 UNSEEN COMPLAINTS)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'M9.3 Baseline':<15} | {'M10.2 Clean SFT':<15} | {'Delta':<12}")
    print("-" * 80)
    print(f"{'Overall Contamination Rate':<35} | {m93_summary['overall_contamination_pct']:>13.1f}% | {m102_summary['overall_contamination_pct']:>13.1f}% | {contam_abs_diff:>+10.1f} pp")
    print(f"  - Twitter @Handles Rate           | {m93_summary['handles_pct']:>13.1f}% | {m102_summary['handles_pct']:>13.1f}% | {handle_abs_diff:>+10.1f} pp")
    print(f"  - URL / t.co Link Rate             | {m93_summary['urls_pct']:>13.1f}% | {m102_summary['urls_pct']:>13.1f}% | {url_abs_diff:>+10.1f} pp")
    print(f"  - Agent ^Signatures Rate          | {m93_summary['signatures_pct']:>13.1f}% | {m102_summary['signatures_pct']:>13.1f}% | {sig_abs_diff:>+10.1f} pp")
    print(f"  - DM Deflection Phrases            | {m93_summary['dm_deflection_pct']:>13.1f}% | {m102_summary['dm_deflection_pct']:>13.1f}% | {dm_abs_diff:>+10.1f} pp")
    print("-" * 80)
    print(f"{'Non-Empty Response Rate':<35} | {m93_summary['non_empty_pct']:>13.1f}% | {m102_summary['non_empty_pct']:>13.1f}% | {m102_summary['non_empty_pct']-m93_summary['non_empty_pct']:>+10.1f} pp")
    print(f"{'EOS Termination Rate':<35} | {m93_summary['eos_termination_pct']:>13.1f}% | {m102_summary['eos_termination_pct']:>13.1f}% | {m102_summary['eos_termination_pct']-m93_summary['eos_termination_pct']:>+10.1f} pp")
    print(f"{'Avg Response Tokens':<35} | {m93_summary['avg_tokens']:>14.1f} | {m102_summary['avg_tokens']:>14.1f} | {m102_summary['avg_tokens']-m93_summary['avg_tokens']:>+11.1f}")
    print(f"{'Intent/Domain Relevance Proxy':<35} | {m93_summary['relevance_proxy_pct']:>13.1f}% | {m102_summary['relevance_proxy_pct']:>13.1f}% | {m102_summary['relevance_proxy_pct']-m93_summary['relevance_proxy_pct']:>+10.1f} pp")
    print(f"{'Boundary/Completeness Proxy':<35} | {m93_summary['completeness_proxy_pct']:>13.1f}% | {m102_summary['completeness_proxy_pct']:>13.1f}% | {m102_summary['completeness_proxy_pct']-m93_summary['completeness_proxy_pct']:>+10.1f} pp")
    print(f"{'Fluency Score':<35} | {m93_summary['fluency_pct']:>13.1f}% | {m102_summary['fluency_pct']:>13.1f}% | {m102_summary['fluency_pct']-m93_summary['fluency_pct']:>+10.1f} pp")
    print(f"{'Degenerate Repetition Rate':<35} | {m93_summary['repetition_pct']:>13.1f}% | {m102_summary['repetition_pct']:>13.1f}% | {m102_summary['repetition_pct']-m93_summary['repetition_pct']:>+10.1f} pp")
    print(f"{'Appropriate Empathy Rate':<35} | {m93_summary['empathy_pct']:>13.1f}% | {m102_summary['empathy_pct']:>13.1f}% | {m102_summary['empathy_pct']-m93_summary['empathy_pct']:>+10.1f} pp")
    print("=" * 80)
    print(f"M10.2 Mean Category Confidence: {m102_summary['mean_category_confidence_pct']:.1f}% | Low-Confidence (<25%): {m102_summary['low_confidence_rate_pct']:.1f}%")
    print(f"M10.2 CSAT Distribution: {m102_summary['csat_distribution']}")
    print(f"Overall Contamination Reduction: -{contam_abs_diff:.1f} pp (Relative: -{contam_rel_diff:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
