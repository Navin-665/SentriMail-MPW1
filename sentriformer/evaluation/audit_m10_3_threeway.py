"""
==========================================================================
  M10.3 — Three-Way SentriFormer Behavioral Quality Audit
==========================================================================
  - Same 80 paired inquiries used in M10.2 audit:
      * 30 Canonical Multi-Category Benchmark Inquiries
      * 50 Stratified Unseen Inquiries from clean_dialogue_test.parquet
  - Evaluates three checkpoints side-by-side:
      1. M9.3 Baseline (dialogue_sft_best.pt)
      2. M10.2 Clean SFT (clean_sft_best.pt)
      3. M10.3 Refined SFT (m10_3_sft_best.pt)
  - Core 10 behavioral dimensions (same as M10.2 audit):
      1. Non-empty response %
      2. EOS termination %
      3. Intent/Domain Relevance Proxy %
      4. Grammar / Fluency %
      5. Degenerate Repetition %
      6. Contamination Breakdown (Overall, Handles, URLs, Signatures, DM)
      7. Boundary/Completeness Proxy %
      8. Appropriate Empathy %
      9. Category Routing & Confidence
      10. CSAT Forecast Distribution
  - NEW: 4 M10.3 Safety/Quality Gates:
      S1. Unsupported Personal Names (fabricated names not in input)
      S2. Invented URLs (any URL not present in original input)
      S3. Invented Contact Info (fabricated phone/email/contact details)
      S4. Orphaned Channel Phrases ("at:", "here:", "website:", "via:" with
          missing destination)
  - Outputs JSON + Markdown audit reports
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
M103_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "m10_3_sft", "m10_3_sft_best.pt")
TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_test.parquet")

OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "m10_3_audit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIT_JSON = os.path.join(OUTPUT_DIR, "m10_3_threeway_audit.json")
AUDIT_MD = os.path.join(OUTPUT_DIR, "M10_3_BEHAVIORAL_AUDIT.md")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Contamination Artifact Regex Patterns (reused from M10.2 audit)
# ══════════════════════════════════════════════════════════════════════════════
RE_TWITTER_HANDLE = re.compile(r"@[A-Za-z0-9_]+")
RE_URL = re.compile(r"https?://\S+|t\.co/\S+")
RE_AGENT_SIGNATURE = re.compile(r"\^[A-Z]{2,4}\b")
RE_DM_DEFLECTION = re.compile(
    r"\b(dm|direct message|dm us|send us a dm|message us|reach out via dm|contact us via dm|in our dms)\b",
    re.IGNORECASE
)
RE_EMPATHY_MARKERS = re.compile(
    r"\b(apologize|apologies|sorry|glad to help|happy to help|understand your concern|"
    r"let us look into|happy to assist|pleasure to assist)\b",
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════════════════
# NEW M10.3 Safety Gate Regex Patterns
# ══════════════════════════════════════════════════════════════════════════════

# S1. Unsupported Personal Names — common first names the model might hallucinate
# We detect "Hey/Hi/Hello <CapitalizedName>" patterns not present in the input
RE_GREETING_NAME = re.compile(
    r"\b(?:hey|hi|hello|dear|thanks?)\s*,?\s+([A-Z][a-z]{2,15})\b"
)

# S2. Invented URLs — any URL-like pattern in the response
RE_ANY_URL = re.compile(
    r"(?:https?://|www\.)\S+|[a-z0-9-]+\.\w{2,4}/\S*"
)

# S3. Invented Contact Information — phone numbers, email addresses
RE_PHONE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|"
    r"\b1-8\d{2}[-.\s]\d{3}[-.\s]\d{4}\b|"
    r"\b\d{3}[-.\s]\d{4}\b"
)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# S4. Orphaned Channel Phrases — "at:", "here:", "website:", "via:" where destination is missing
RE_ORPHANED_PHRASE = re.compile(
    r"(?:reach\s+(?:out\s+)?(?:to\s+)?us\s+at\s*[:.]?\s*$|"
    r"(?:contact|visit|check)\s+(?:us\s+)?(?:at|here)\s*[:.]?\s*$|"
    r"\b(?:at|here|website|via)\s*:\s*(?:[.!?\s]|$)|"
    r"\bhere\s*:\s*(?:If|if|and|so|We|we|[A-Z])|"
    r"\bwebsite\s*:\s*(?:If|if|and|so|We|we|[A-Z])|"
    r"(?:our\s+website|our\s+page|our\s+site)\s*:\s*(?:[.!?\s]|$))",
    re.MULTILINE | re.IGNORECASE
)


def detect_repetition(text: str) -> bool:
    """Detects degenerate token or phrase loops in generated text."""
    words = text.lower().split()
    if len(words) < 6:
        return False
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    counts = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1
        if counts[tg] >= 3:
            return True
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return True
    return False


def safety_check_response(response_text: str, input_text: str) -> Dict[str, Any]:
    """Runs the 4 M10.3 safety/quality gates on a single response."""
    clean = response_text.strip()
    input_lower = input_text.lower()

    # S1. Unsupported Personal Names
    greeting_matches = RE_GREETING_NAME.findall(clean)
    # Filter out names that actually appear in the customer input
    unsupported_names = [n for n in greeting_matches if n.lower() not in input_lower]
    has_unsupported_name = len(unsupported_names) > 0

    # S2. Invented URLs
    response_urls = RE_ANY_URL.findall(clean)
    input_urls = RE_ANY_URL.findall(input_text)
    input_url_set = set(u.lower() for u in input_urls)
    invented_urls = [u for u in response_urls if u.lower() not in input_url_set]
    has_invented_url = len(invented_urls) > 0

    # S3. Invented Contact Information
    phones = RE_PHONE.findall(clean)
    emails = RE_EMAIL.findall(clean)
    # Filter out any that are present in the input
    input_phones = set(RE_PHONE.findall(input_text))
    input_emails = set(e.lower() for e in RE_EMAIL.findall(input_text))
    invented_phones = [p for p in phones if p not in input_phones]
    invented_emails = [e for e in emails if e.lower() not in input_emails]
    has_invented_contact = len(invented_phones) > 0 or len(invented_emails) > 0

    # S4. Orphaned Channel Phrases
    orphaned_matches = RE_ORPHANED_PHRASE.findall(clean)
    has_orphaned_phrase = len(orphaned_matches) > 0

    # Overall safety
    has_any_safety_issue = (
        has_unsupported_name or has_invented_url or
        has_invented_contact or has_orphaned_phrase
    )

    return {
        "has_any_safety_issue": has_any_safety_issue,
        "unsupported_names": unsupported_names,
        "has_unsupported_name": has_unsupported_name,
        "invented_urls": invented_urls,
        "has_invented_url": has_invented_url,
        "invented_phones": invented_phones,
        "invented_emails": invented_emails,
        "has_invented_contact": has_invented_contact,
        "orphaned_phrases": orphaned_matches,
        "has_orphaned_phrase": has_orphaned_phrase,
    }


def audit_single_response(
    response_text: str,
    raw_ids: List[int],
    expected_keywords: List[str],
    input_text: str,
    eos_token_id: int = 3,
    run_safety: bool = False
) -> Dict[str, Any]:
    """Analyzes a generated response across behavioral dimensions + optional safety gates."""
    clean_text = response_text.strip()
    words = clean_text.lower().split()
    word_count = len(words)

    # 1. Generation metrics
    is_non_empty = len(clean_text) > 0 and word_count > 0
    eos_terminated = (eos_token_id in raw_ids) if raw_ids else False

    # 2. Contamination checks
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
    has_empathy = bool(RE_EMPATHY_MARKERS.search(clean_text)) or any(
        g in clean_text.lower() for g in ["apologize", "sorry", "understand", "glad to"]
    )

    result = {
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

    # M10.3 Safety Gates (only for models that should be clean)
    if run_safety:
        result["safety"] = safety_check_response(clean_text, input_text)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Generation Functions
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_m93(model, tokenizer, config, input_ids, attention_mask,
                 max_new_tokens=100, temperature=0.7, top_k=50):
    """M9.3 generation using SentriFormerForConditionalGeneration."""
    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device
    encoder_outputs = model.sentriformer.encoder(
        input_ids=input_ids, attention_mask=attention_mask
    )
    curr_token = torch.full(
        (batch_size, 1), config.bos_token_id, dtype=torch.long, device=device
    )
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
def generate_multitask(model, tokenizer, config, input_ids, attention_mask,
                       max_new_tokens=100, temperature=0.7, top_k=50,
                       repetition_penalty=1.2):
    """Generation using SentriFormerForMultiTask (for both M10.2 and M10.3)."""
    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device
    encoder_outputs = model.sentriformer.encoder(
        input_ids=input_ids, attention_mask=attention_mask
    )
    curr_token = torch.full(
        (batch_size, 1), config.bos_token_id, dtype=torch.long, device=device
    )
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


# ══════════════════════════════════════════════════════════════════════════════
# Statistics Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_stats():
    """Creates a fresh stats accumulator."""
    return {
        "non_empty": 0, "eos_terminated": 0, "total_tokens": 0,
        "contaminated": 0, "handles": 0, "urls": 0, "signatures": 0, "dm": 0,
        "relevance_sum": 0.0, "completeness": 0, "fluency_sum": 0.0,
        "repetition": 0, "empathy": 0,
        "category_counts": {}, "category_conf_sum": 0.0, "low_confidence_count": 0,
        "csat_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        # M10.3 safety counters
        "safety_unsupported_name": 0,
        "safety_invented_url": 0,
        "safety_invented_contact": 0,
        "safety_orphaned_phrase": 0,
        "safety_any_issue": 0,
    }


def tally(stats, audit, cat_name=None, cat_conf=None, csat_star=None):
    """Accumulates audit results into stats dict."""
    if audit["is_non_empty"]:
        stats["non_empty"] += 1
    if audit["eos_terminated"]:
        stats["eos_terminated"] += 1
    stats["total_tokens"] += audit["token_count"]
    if audit["contamination"]["is_contaminated"]:
        stats["contaminated"] += 1
    if audit["contamination"]["has_handle"]:
        stats["handles"] += 1
    if audit["contamination"]["has_url"]:
        stats["urls"] += 1
    if audit["contamination"]["has_signature"]:
        stats["signatures"] += 1
    if audit["contamination"]["has_dm"]:
        stats["dm"] += 1
    stats["relevance_sum"] += audit["intent_relevance_proxy"]
    if audit["completeness_proxy"]:
        stats["completeness"] += 1
    stats["fluency_sum"] += audit["fluency_score"]
    if audit["has_repetition"]:
        stats["repetition"] += 1
    if audit["has_empathy"]:
        stats["empathy"] += 1

    if cat_name is not None:
        stats["category_counts"][cat_name] = stats["category_counts"].get(cat_name, 0) + 1
    if cat_conf is not None:
        stats["category_conf_sum"] += cat_conf
        if cat_conf < 25.0:
            stats["low_confidence_count"] += 1
    if csat_star is not None:
        stats["csat_counts"][csat_star] += 1

    # Safety tallies
    if "safety" in audit:
        s = audit["safety"]
        if s["has_unsupported_name"]:
            stats["safety_unsupported_name"] += 1
        if s["has_invented_url"]:
            stats["safety_invented_url"] += 1
        if s["has_invented_contact"]:
            stats["safety_invented_contact"] += 1
        if s["has_orphaned_phrase"]:
            stats["safety_orphaned_phrase"] += 1
        if s["has_any_safety_issue"]:
            stats["safety_any_issue"] += 1


def summarize(stats, N):
    """Computes percentage-level summary from raw stats."""
    s = {
        "non_empty_pct": round(stats["non_empty"] / N * 100, 1),
        "eos_termination_pct": round(stats["eos_terminated"] / N * 100, 1),
        "avg_tokens": round(stats["total_tokens"] / N, 1),
        "overall_contamination_pct": round(stats["contaminated"] / N * 100, 1),
        "handles_pct": round(stats["handles"] / N * 100, 1),
        "urls_pct": round(stats["urls"] / N * 100, 1),
        "signatures_pct": round(stats["signatures"] / N * 100, 1),
        "dm_deflection_pct": round(stats["dm"] / N * 100, 1),
        "relevance_proxy_pct": round(stats["relevance_sum"] / N, 1),
        "completeness_proxy_pct": round(stats["completeness"] / N * 100, 1),
        "fluency_pct": round(stats["fluency_sum"] / N, 1),
        "repetition_pct": round(stats["repetition"] / N * 100, 1),
        "empathy_pct": round(stats["empathy"] / N * 100, 1),
    }
    if stats["category_conf_sum"] > 0:
        s["mean_category_confidence_pct"] = round(stats["category_conf_sum"] / N, 1)
        s["low_confidence_rate_pct"] = round(stats["low_confidence_count"] / N * 100, 1)
        s["category_distribution"] = stats["category_counts"]
        s["csat_distribution"] = stats["csat_counts"]

    # Safety summary
    s["safety_unsupported_name_pct"] = round(stats["safety_unsupported_name"] / N * 100, 1)
    s["safety_invented_url_pct"] = round(stats["safety_invented_url"] / N * 100, 1)
    s["safety_invented_contact_pct"] = round(stats["safety_invented_contact"] / N * 100, 1)
    s["safety_orphaned_phrase_pct"] = round(stats["safety_orphaned_phrase"] / N * 100, 1)
    s["safety_any_issue_pct"] = round(stats["safety_any_issue"] / N * 100, 1)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Main Audit
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 80)
    log("  [M10.3] THREE-WAY BEHAVIORAL QUALITY AUDIT (M9.3 vs M10.2 vs M10.3)")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ── 1. Load Tokenizer & Config ────────────────────────────────────────────
    log("\n[1/5] Loading Tokenizer, Config, and Classification Labels...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5

    id2label = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            id2label = json.load(f).get("id2label", {})

    # ── 2. Assemble 80 Paired Inquiries ──────────────────────────────────────
    log("\n[2/5] Assembling 80 Paired Test Inquiries...")
    all_inquiries = []

    # A. 30 Canonical Inquiries
    for cat in BENCHMARK_SCENARIOS:
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

    # B. 50 Stratified Unseen Inquiries (same seed=42 as M10.2 audit)
    test_table = pq.read_table(TEST_PARQUET, columns=["context", "response"])
    ctx_list = test_table["context"].to_pylist()
    resp_list = test_table["response"].to_pylist()

    random.seed(42)
    sampled_indices = random.sample(range(len(ctx_list)), 50)

    for idx in sampled_indices:
        q = ctx_list[idx].strip()
        ref = resp_list[idx].strip()
        ref_words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", ref)]
        kw = list(dict.fromkeys(ref_words))[:8]
        all_inquiries.append({
            "source": "clean_test_parquet",
            "category_group": "Clean Test Set (Unseen)",
            "query": q,
            "expected_keywords": kw,
            "reference_response": ref
        })
    log(f"  Added 50 stratified unseen inquiries from clean_dialogue_test.parquet.")
    log(f"  Total Paired Benchmark Suite: {len(all_inquiries)} inquiries.")

    # ── 3. Load All Three Models ─────────────────────────────────────────────
    log("\n[3/5] Loading Three Checkpoints for Three-Way Evaluation...")

    log(f"  [Model A: M9.3 Baseline] Loading {M93_CKPT}")
    model_m93 = SentriFormerForConditionalGeneration(config).to(device)
    raw_m93 = torch.load(M93_CKPT, map_location=device, weights_only=False)
    model_m93.load_state_dict(
        raw_m93["model_state_dict"] if "model_state_dict" in raw_m93 else raw_m93
    )
    model_m93.eval()

    log(f"  [Model B: M10.2 Clean SFT] Loading {M102_CKPT}")
    model_m102 = SentriFormerForMultiTask(config).to(device)
    raw_m102 = torch.load(M102_CKPT, map_location=device, weights_only=False)
    model_m102.load_state_dict(
        raw_m102["model_state_dict"] if "model_state_dict" in raw_m102 else raw_m102
    )
    model_m102.eval()

    log(f"  [Model C: M10.3 Refined SFT] Loading {M103_CKPT}")
    model_m103 = SentriFormerForMultiTask(config).to(device)
    raw_m103 = torch.load(M103_CKPT, map_location=device, weights_only=False)
    model_m103.load_state_dict(
        raw_m103["model_state_dict"] if "model_state_dict" in raw_m103 else raw_m103
    )
    model_m103.eval()

    # ── 4. Execute Three-Way Inferences ──────────────────────────────────────
    log("\n[4/5] Executing Three-Way Inferences across 80 Inquiries...")

    m93_stats = make_stats()
    m102_stats = make_stats()
    m103_stats = make_stats()
    paired_cases = []
    N = len(all_inquiries)

    for i, item in enumerate(all_inquiries):
        query = item["query"]
        expected_kw = item["expected_keywords"]
        enc = tokenizer(query, return_tensors="pt").to(device)
        iids, amask = enc["input_ids"], enc["attention_mask"]

        # ── Model A: M9.3 ───────────────────────────────────────────────────
        gen_m93 = generate_m93(model_m93, tokenizer, config, iids, amask,
                               max_new_tokens=100, temperature=0.7)
        raw_m93_ids = gen_m93[0].tolist()
        text_m93 = tokenizer.decode(raw_m93_ids, skip_special_tokens=True).strip()
        audit_m93 = audit_single_response(
            text_m93, raw_m93_ids, expected_kw, query,
            eos_token_id=config.eos_token_id, run_safety=True
        )
        tally(m93_stats, audit_m93)

        # ── Model B: M10.2 ──────────────────────────────────────────────────
        m102_out = model_m102(input_ids=iids, attention_mask=amask)
        cat_probs_102 = F.softmax(m102_out["category_logits"], dim=-1)[0]
        top_prob_102, top_cat_id_102 = torch.max(cat_probs_102, dim=-1)
        cat_name_102 = id2label.get(str(top_cat_id_102.item()), f"Cat{top_cat_id_102.item()}")
        cat_conf_102 = top_prob_102.item() * 100.0
        csat_102 = torch.argmax(m102_out["csat_logits"], dim=-1).item() + 1

        gen_m102 = generate_multitask(model_m102, tokenizer, config, iids, amask,
                                       max_new_tokens=100, temperature=0.7,
                                       repetition_penalty=1.2)
        raw_m102_ids = gen_m102[0].tolist()
        text_m102 = tokenizer.decode(raw_m102_ids, skip_special_tokens=True).strip()
        audit_m102 = audit_single_response(
            text_m102, raw_m102_ids, expected_kw, query,
            eos_token_id=config.eos_token_id, run_safety=True
        )
        tally(m102_stats, audit_m102, cat_name_102, cat_conf_102, csat_102)

        # ── Model C: M10.3 ──────────────────────────────────────────────────
        m103_out = model_m103(input_ids=iids, attention_mask=amask)
        cat_probs_103 = F.softmax(m103_out["category_logits"], dim=-1)[0]
        top_prob_103, top_cat_id_103 = torch.max(cat_probs_103, dim=-1)
        cat_name_103 = id2label.get(str(top_cat_id_103.item()), f"Cat{top_cat_id_103.item()}")
        cat_conf_103 = top_prob_103.item() * 100.0
        csat_103 = torch.argmax(m103_out["csat_logits"], dim=-1).item() + 1

        gen_m103 = generate_multitask(model_m103, tokenizer, config, iids, amask,
                                       max_new_tokens=100, temperature=0.7,
                                       repetition_penalty=1.2)
        raw_m103_ids = gen_m103[0].tolist()
        text_m103 = tokenizer.decode(raw_m103_ids, skip_special_tokens=True).strip()
        audit_m103 = audit_single_response(
            text_m103, raw_m103_ids, expected_kw, query,
            eos_token_id=config.eos_token_id, run_safety=True
        )
        tally(m103_stats, audit_m103, cat_name_103, cat_conf_103, csat_103)

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
                "category": cat_name_102,
                "category_confidence": round(cat_conf_102, 1),
                "csat_forecast": csat_102,
                "audit": audit_m102
            },
            "m103": {
                "response": text_m103,
                "raw_token_count": audit_m103["token_count"],
                "category": cat_name_103,
                "category_confidence": round(cat_conf_103, 1),
                "csat_forecast": csat_103,
                "audit": audit_m103
            }
        })

        if (i + 1) % 10 == 0 or (i + 1) == N:
            log(f"  Processed {i+1:>2}/{N} inquiries...")

    # ── 5. Compute Summaries ─────────────────────────────────────────────────
    log("\n[5/5] Computing Comparative Summary...")

    m93_summary = summarize(m93_stats, N)
    m102_summary = summarize(m102_stats, N)
    m103_summary = summarize(m103_stats, N)

    comparative_results = {
        "timestamp": timestamp(),
        "audit_version": "M10.3 Three-Way",
        "total_paired_inquiries": N,
        "m93_baseline": m93_summary,
        "m102_clean_sft": m102_summary,
        "m103_refined_sft": m103_summary,
        "cases": paired_cases
    }

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2, ensure_ascii=False)
    log(f"  💾 Saved full audit JSON to: {AUDIT_JSON}")

    # ── 6. Generate Markdown Report ──────────────────────────────────────────
    md_lines = []
    md_lines.append("# M10.3 SentriFormer Three-Way Behavioral Quality Audit\n")
    md_lines.append(f"**Evaluation Date**: {datetime.date.today()}")
    md_lines.append(f"**Audited Models**:")
    md_lines.append(f"- **M9.3 Baseline**: `dialogue_sft_best.pt` (raw Twitter customer support)")
    md_lines.append(f"- **M10.2 Clean SFT**: `clean_sft_best.pt` (615K clean pairs, corrected causal SDPA & EOS)")
    md_lines.append(f"- **M10.3 Refined SFT**: `m10_3_sft_best.pt` (530K refined pairs, orphan repair, dedup, 4-tier templates)")
    md_lines.append(f"**Dataset**: {N} Paired Inquiries (30 Canonical + 50 Stratified Unseen)\n")
    md_lines.append("---\n")

    # Core comparison table
    md_lines.append("## 1. Paired Head-to-Head Metrics (N=80)\n")
    md_lines.append("| Metric | M9.3 | M10.2 | M10.3 | Δ M10.2→M10.3 |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    rows = [
        ("**Overall Contamination**", "overall_contamination_pct", "%", "pp"),
        ("  @Handles", "handles_pct", "%", "pp"),
        ("  URLs / t.co", "urls_pct", "%", "pp"),
        ("  ^Signatures", "signatures_pct", "%", "pp"),
        ("  DM Deflection", "dm_deflection_pct", "%", "pp"),
        ("**Non-Empty Response**", "non_empty_pct", "%", "pp"),
        ("**EOS Termination**", "eos_termination_pct", "%", "pp"),
        ("**Avg Response Tokens**", "avg_tokens", "", ""),
        ("**Intent/Domain Relevance Proxy**", "relevance_proxy_pct", "%", "pp"),
        ("**Boundary/Completeness**", "completeness_proxy_pct", "%", "pp"),
        ("**Fluency Score**", "fluency_pct", "%", "pp"),
        ("**Degenerate Repetition**", "repetition_pct", "%", "pp"),
        ("**Empathy Rate**", "empathy_pct", "%", "pp"),
    ]

    for label, key, suffix, delta_suffix in rows:
        v93 = m93_summary.get(key, 0)
        v102 = m102_summary.get(key, 0)
        v103 = m103_summary.get(key, 0)
        delta = v103 - v102
        sign = "+" if delta >= 0 else ""
        md_lines.append(
            f"| {label} | {v93}{suffix} | {v102}{suffix} | {v103}{suffix} | "
            f"{sign}{delta:.1f} {delta_suffix} |"
        )

    md_lines.append("")

    # Safety table
    md_lines.append("## 2. M10.3 Safety / Quality Gates (N=80)\n")
    md_lines.append("| Safety Gate | M9.3 | M10.2 | M10.3 | Δ M10.2→M10.3 |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    safety_rows = [
        ("**S1. Unsupported Personal Names**", "safety_unsupported_name_pct"),
        ("**S2. Invented URLs**", "safety_invented_url_pct"),
        ("**S3. Invented Contact Info**", "safety_invented_contact_pct"),
        ("**S4. Orphaned Channel Phrases**", "safety_orphaned_phrase_pct"),
        ("**Overall Safety Issue Rate**", "safety_any_issue_pct"),
    ]

    for label, key in safety_rows:
        v93 = m93_summary.get(key, 0)
        v102 = m102_summary.get(key, 0)
        v103 = m103_summary.get(key, 0)
        delta = v103 - v102
        sign = "+" if delta >= 0 else ""
        md_lines.append(
            f"| {label} | {v93}% | {v102}% | {v103}% | {sign}{delta:.1f} pp |"
        )

    md_lines.append("")

    # M10.3 routing/CSAT
    if "mean_category_confidence_pct" in m103_summary:
        md_lines.append("## 3. M10.3 Category Routing & CSAT\n")
        md_lines.append(f"- **Mean Category Confidence**: {m103_summary['mean_category_confidence_pct']}%")
        md_lines.append(f"- **Low Confidence (<25%) Rate**: {m103_summary['low_confidence_rate_pct']}%")
        md_lines.append(f"- **CSAT Forecast Distribution**: {m103_summary['csat_distribution']}")
        md_lines.append(f"- **Top Categories**: {json.dumps(dict(sorted(m103_summary['category_distribution'].items(), key=lambda x: -x[1])[:5]))}")
        md_lines.append("")

    # Sample responses (first 6)
    md_lines.append("## 4. Sample Three-Way Response Comparisons\n")
    for case in paired_cases[:6]:
        md_lines.append(f"### Case {case['inquiry_id']} — {case['category_group']}\n")
        md_lines.append(f"> **Customer**: \"{case['query']}\"\n")
        md_lines.append(f"| Model | Response | Tokens | Contaminated | Relevance | Safety |")
        md_lines.append(f"| :--- | :--- | :---: | :---: | :---: | :---: |")

        for tag in ["m93", "m102", "m103"]:
            c = case[tag]
            resp = c["response"][:120].replace("|", "\\|").replace("\n", " ")
            tok = c["raw_token_count"]
            cont = "⛔" if c["audit"]["contamination"]["is_contaminated"] else "✅"
            rel = f"{c['audit']['intent_relevance_proxy']}%"
            if "safety" in c["audit"]:
                safe = "⚠️" if c["audit"]["safety"]["has_any_safety_issue"] else "✅"
            else:
                safe = "—"
            md_lines.append(f"| **{tag.upper()}** | {resp}… | {tok} | {cont} | {rel} | {safe} |")
        md_lines.append("")

    md_report = "\n".join(md_lines)
    with open(AUDIT_MD, "w", encoding="utf-8") as f:
        f.write(md_report)
    log(f"  📄 Saved Markdown audit report to: {AUDIT_MD}")

    # ── 7. Console Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  🏆 THREE-WAY BEHAVIORAL AUDIT SUMMARY (80 PAIRED INQUIRIES)")
    print("=" * 100)
    hdr = f"{'Metric':<36} | {'M9.3':>10} | {'M10.2':>10} | {'M10.3':>10} | {'Δ 10.2→10.3':>12}"
    print(hdr)
    print("-" * 100)

    console_rows = [
        ("Overall Contamination", "overall_contamination_pct", "%"),
        ("  @Handles", "handles_pct", "%"),
        ("  URLs / t.co", "urls_pct", "%"),
        ("  ^Signatures", "signatures_pct", "%"),
        ("  DM Deflection", "dm_deflection_pct", "%"),
        ("Non-Empty Response", "non_empty_pct", "%"),
        ("EOS Termination", "eos_termination_pct", "%"),
        ("Avg Response Tokens", "avg_tokens", ""),
        ("Intent Relevance Proxy", "relevance_proxy_pct", "%"),
        ("Boundary/Completeness", "completeness_proxy_pct", "%"),
        ("Fluency Score", "fluency_pct", "%"),
        ("Degenerate Repetition", "repetition_pct", "%"),
        ("Empathy Rate", "empathy_pct", "%"),
    ]

    for label, key, sfx in console_rows:
        v93 = m93_summary.get(key, 0)
        v102 = m102_summary.get(key, 0)
        v103 = m103_summary.get(key, 0)
        d = v103 - v102
        sign = "+" if d >= 0 else ""
        print(f"{label:<36} | {v93:>9.1f}{sfx} | {v102:>9.1f}{sfx} | {v103:>9.1f}{sfx} | {sign}{d:>10.1f} pp")

    print("-" * 100)
    print("  M10.3 SAFETY / QUALITY GATES")
    print("-" * 100)
    safety_console = [
        ("S1. Unsupported Names", "safety_unsupported_name_pct"),
        ("S2. Invented URLs", "safety_invented_url_pct"),
        ("S3. Invented Contact Info", "safety_invented_contact_pct"),
        ("S4. Orphaned Channel Phrases", "safety_orphaned_phrase_pct"),
        ("Overall Safety Issue Rate", "safety_any_issue_pct"),
    ]
    for label, key in safety_console:
        v93 = m93_summary.get(key, 0)
        v102 = m102_summary.get(key, 0)
        v103 = m103_summary.get(key, 0)
        d = v103 - v102
        sign = "+" if d >= 0 else ""
        print(f"{label:<36} | {v93:>9.1f}% | {v102:>9.1f}% | {v103:>9.1f}% | {sign}{d:>10.1f} pp")

    print("=" * 100)

    if "mean_category_confidence_pct" in m103_summary:
        print(f"M10.3 Mean Category Confidence: {m103_summary['mean_category_confidence_pct']:.1f}% | "
              f"Low-Confidence (<25%): {m103_summary['low_confidence_rate_pct']:.1f}%")
        print(f"M10.3 CSAT Distribution: {m103_summary['csat_distribution']}")

    print("=" * 100)
    log(f"\n✅ M10.3 Three-Way Behavioral Audit Complete!")


if __name__ == "__main__":
    main()
