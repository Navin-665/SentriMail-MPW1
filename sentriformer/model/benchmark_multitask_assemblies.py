"""
==========================================================================
  M9.5 — Multi-Task Checkpoint Assembly & Zero-Shot Benchmarking
==========================================================================
  Audits 3 candidate assembly strategies against individual baseline test metrics:
  - Strategy A: M8 Clean Backbone + M9.2 Category Head + M9.4 CSAT Head + M8 Decoder
  - Strategy B: M8 Clean Backbone + M9.2 Category Head + M9.4 CSAT Head + M9.3 Dialogue Decoder
  - Strategy C: M9.2 Classifier Backbone + M9.2 Category Head + M9.4 CSAT Head + M9.3 Dialogue Decoder

  Evaluates on:
  - Category Test Set (classification_test.parquet)
  - CSAT Test Set (csat_test.parquet)
  - Dialogue Test Set (dialogue_test.parquet - sample subset for speed)
==========================================================================
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator, evaluate_dialogue

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
M8_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "checkpoint_best.pt")
M92_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "classification", "classifier_best.pt")
M93_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "dialogue", "dialogue_sft_best.pt")
M94_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "csat", "csat_best.pt")

CLASS_TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_test.parquet")
CSAT_TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_test.parquet")
DIAL_TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_test.parquet")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "multitask")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sd(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data.get("model_state_dict", data)


def assemble_model(strategy: str, config: SentriFormerConfig, device: torch.device):
    """Assembles a SentriFormerForMultiTask model according to specified strategy."""
    model = SentriFormerForMultiTask(config)
    assembled_sd = {}
    
    m8_sd = load_sd(M8_CKPT)
    m92_sd = load_sd(M92_CKPT)
    m93_sd = load_sd(M93_CKPT)
    m94_sd = load_sd(M94_CKPT)
    
    if strategy == "A":
        # Strategy A: Pure M8 backbone + M9.2 Category Head + M9.4 CSAT Head
        # Backbone & LM Head from M8
        for k, v in m8_sd.items():
            assembled_sd[k] = v.clone()
        # Category head from M9.2
        for k in m92_sd:
            if k.startswith("category_head."):
                assembled_sd[k] = m92_sd[k].clone()
        # CSAT head from M9.4
        for k in m94_sd:
            if k.startswith("csat_head."):
                assembled_sd[k] = m94_sd[k].clone()
                
    elif strategy == "B":
        # Strategy B: M8 Encoder & Embeddings + M9.3 Decoder + M9.2 Category + M9.4 CSAT
        for k, v in m8_sd.items():
            if k.startswith("sentriformer.encoder.") or k.startswith("sentriformer.shared_embeddings."):
                assembled_sd[k] = v.clone()
        for k, v in m93_sd.items():
            if k.startswith("sentriformer.decoder.") or k == "lm_head.weight":
                assembled_sd[k] = v.clone()
        for k in m92_sd:
            if k.startswith("category_head."):
                assembled_sd[k] = m92_sd[k].clone()
        for k in m94_sd:
            if k.startswith("csat_head."):
                assembled_sd[k] = m94_sd[k].clone()
                
    elif strategy == "C":
        # Strategy C: M9.2 Classifier Backbone + M9.3 Decoder + M9.2 Category + M9.4 CSAT
        for k, v in m92_sd.items():
            if not k.startswith("csat_head."):
                assembled_sd[k] = v.clone()
        for k, v in m93_sd.items():
            if k.startswith("sentriformer.decoder."):
                assembled_sd[k] = v.clone()
        for k in m94_sd:
            if k.startswith("csat_head."):
                assembled_sd[k] = m94_sd[k].clone()
                
    # Tie embeddings if configured
    if config.tie_word_embeddings and "sentriformer.shared_embeddings.weight" in assembled_sd:
        assembled_sd["sentriformer.encoder.embed_tokens.weight"] = assembled_sd["sentriformer.shared_embeddings.weight"]
        assembled_sd["sentriformer.decoder.embed_tokens.weight"] = assembled_sd["sentriformer.shared_embeddings.weight"]
        assembled_sd["lm_head.weight"] = assembled_sd["sentriformer.shared_embeddings.weight"]

    model.load_state_dict(assembled_sd, strict=False)
    model.to(device)
    model.eval()
    return model, assembled_sd


def evaluate_assembly(model, class_loader, csat_loader, dial_loader, device):
    """Evaluates the assembled model across all 3 test sets."""
    model.eval()
    
    # 1. Category Evaluation
    all_preds_cat, all_tgts_cat = [], []
    with torch.no_grad():
        for batch in class_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["category_logits"], dim=-1).cpu().numpy()
            all_preds_cat.extend(preds)
            all_tgts_cat.extend(labels.cpu().numpy())
    cat_acc = accuracy_score(all_tgts_cat, all_preds_cat)
    cat_wf1 = f1_score(all_tgts_cat, all_preds_cat, average="weighted", zero_division=0)
    
    # 2. CSAT Evaluation
    all_preds_csat, all_tgts_csat = [], []
    with torch.no_grad():
        for batch in csat_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["csat_logits"], dim=-1).cpu().numpy()
            all_preds_csat.extend(preds)
            all_tgts_csat.extend(labels.cpu().numpy())
    csat_acc = accuracy_score(all_tgts_csat, all_preds_csat)
    preds_scores = [p + 1 for p in all_preds_csat]
    tgts_scores = [t + 1 for t in all_tgts_csat]
    csat_w1 = sum(abs(p - t) <= 1 for p, t in zip(preds_scores, tgts_scores)) / len(tgts_scores)
    csat_mae = mean_absolute_error(tgts_scores, preds_scores)
    
    # 3. Dialogue Evaluation (on subset)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(dial_loader):
            if i >= 150:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, generation_labels=labels)
                loss = outputs["generation_loss"]
            active_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens
    dial_loss = total_loss / max(total_tokens, 1)
    dial_ppl = math.exp(min(dial_loss, 20.0))
    
    return {
        "category_accuracy": cat_acc,
        "category_weighted_f1": cat_wf1,
        "csat_accuracy": csat_acc,
        "csat_within_1": csat_w1,
        "csat_mae": csat_mae,
        "dialogue_loss": dial_loss,
        "dialogue_ppl": dial_ppl
    }


def main():
    print("=" * 80)
    print("  [M9.5] SENTRIFORMER MULTI-TASK ASSEMBLY AUDIT & ZERO-SHOT BENCHMARK")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5
    
    # Load test datasets
    print("\nLoading Test Datasets (Sampled for rapid zero-shot benchmarking)...")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    label2id = labels_meta["label2id"]
    
    class_ds = ClassificationParquetDataset(CLASS_TEST_PARQUET, tokenizer, label2id=label2id, max_seq_length=128, limit=5000)
    csat_ds = CSATDataset(CSAT_TEST_PARQUET, tokenizer, max_len=128)
    dial_ds = DialogueSFTDataset(DIAL_TEST_PARQUET, tokenizer, max_ctx_len=128, max_resp_len=256, limit=5000)
    
    class_collator = ClassificationDataCollator(pad_token_id=config.pad_token_id, max_seq_length=128)
    csat_collator = CSATCollator(pad_token_id=config.pad_token_id)
    dial_collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)
    
    class_loader = DataLoader(class_ds, batch_size=64, shuffle=False, collate_fn=class_collator)
    csat_loader = DataLoader(csat_ds, batch_size=64, shuffle=False, collate_fn=csat_collator)
    dial_loader = DataLoader(dial_ds, batch_size=32, shuffle=False, collate_fn=dial_collator)
    
    strategies = {
        "Strategy A (M8 Backbone + M9.2 Cat + M9.4 CSAT)": "A",
        "Strategy B (M8 Enc/Emb + M9.3 Dec + M9.2 Cat + M9.4 CSAT)": "B",
        "Strategy C (M9.2 Classifier Backbone + M9.3 Dec + M9.4 CSAT)": "C",
    }
    
    results = {}
    print("\n" + "-" * 95)
    print(f"{'Strategy':<55} | {'Cat Acc':<8} | {'CSAT ±1':<8} | {'CSAT MAE':<9} | {'Dial PPL':<8}")
    print("-" * 95)
    
    for strat_name, strat_code in strategies.items():
        model, assembled_sd = assemble_model(strat_code, config, device)
        m = evaluate_assembly(model, class_loader, csat_loader, dial_loader, device)
        results[strat_name] = m
        print(f"{strat_name:<55} | {m['category_accuracy']*100:>6.2f}% | {m['csat_within_1']*100:>6.2f}% | {m['csat_mae']:>7.3f} | {m['dialogue_ppl']:>7.2f}")
        
    print("-" * 95)
    
    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, "assembly_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Assembly benchmark saved to: {json_path}")


if __name__ == "__main__":
    main()
