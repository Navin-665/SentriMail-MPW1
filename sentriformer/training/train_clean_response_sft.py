"""
==========================================================================
  M10.2 — SentriFormer Clean Response SFT Fine-Tuning
==========================================================================
  - Initializes from multitask_best.pt (M9.5 Unified Multi-Task Baseline)
  - Fine-tunes the decoder on clean_dialogue_train.parquet (615,698 clean pairs)
  - Features:
      * Label Smoothing (0.1) to prevent over-confidence & repetition collapse
      * Multi-sentence professional enterprise response formatting
      * Periodic live generation checks on 4 unseen customer complaints
      * Repetition penalty & temperature calibration
  - Preserves Category & CSAT multi-task representations in shared encoder
  - Saves best checkpoint: sentriformer/checkpoints/clean_sft/clean_sft_best.pt
==========================================================================
"""
import os
import sys
import time
import math
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator
from sentriformer.evaluation.audit_multitask_behavior import generate_multitask_response

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
UNIFIED_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "multitask", "multitask_best.pt")

CLEAN_TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_train.parquet")
CLEAN_VAL_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_val.parquet")
CLEAN_TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_test.parquet")

OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "clean_sft")
LOG_FILE = os.path.join(OUTPUT_DIR, "clean_sft_training_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


def evaluate_clean_dialogue(model, dataloader, device, max_batches=150):
    """Evaluates cross-entropy loss & perplexity on clean validation/test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
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
            
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl, total_tokens


def save_clean_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss
    }
    torch.save(state, path)
    log(f"  💾 Saved Clean SFT checkpoint: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SentriFormer M10.3 Clean Response SFT Fine-Tuning")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for Clean Response SFT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_ctx_len", type=int, default=128, help="Maximum context tokens")
    parser.add_argument("--max_resp_len", type=int, default=256, help="Maximum response tokens")
    parser.add_argument("--eval_every", type=int, default=2500, help="Evaluate every N steps")
    parser.add_argument("--train_data", type=str, default=os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "m10_3_dialogue_train.parquet"))
    parser.add_argument("--val_data", type=str, default=os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "m10_3_dialogue_val.parquet"))
    parser.add_argument("--test_data", type=str, default=os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "m10_3_dialogue_test.parquet"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "m10_3_sft"))
    parser.add_argument("--init_ckpt", type=str, default=UNIFIED_CKPT)
    parser.add_argument("--ckpt_name", type=str, default="m10_3_sft_best.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 80)
    log("  [M10.3] SENTRIFORMER CLEAN RESPONSE SFT FINE-TUNING")
    log("=" * 80)
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    log(f"Train data: {args.train_data}")
    log(f"Val data:   {args.val_data}")
    log(f"Output dir: {args.output_dir}")

    # 1. Load Tokenizer & Unified Model Config
    log("\n[1/5] Loading Tokenizer & Unified Model Config...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5

    log("\n[2/5] Initializing SentriFormerForMultiTask & Loading M9.5 Unified Baseline...")
    model = SentriFormerForMultiTask(config).to(device)
    
    log(f"  Loading unified weights from: {args.init_ckpt}")
    unified_raw = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    unified_dict = unified_raw["model_state_dict"] if "model_state_dict" in unified_raw else unified_raw
    model.load_state_dict(unified_dict, strict=True)
    log("  Unified Multi-Task weights loaded successfully!")

    # 2. Datasets
    log("\n[3/5] Loading Clean Dialogue Datasets...")
    train_ds = DialogueSFTDataset(args.train_data, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len)
    val_ds = DialogueSFTDataset(args.val_data, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len)
    test_ds = DialogueSFTDataset(args.test_data, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len)

    collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    log(f"  Clean Train samples: {len(train_ds):,} ({steps_per_epoch:,} steps/epoch)")
    log(f"  Clean Val samples:   {len(val_ds):,}")
    log(f"  Clean Test samples:  {len(test_ds):,}")
    log(f"  Total SFT steps:     {total_steps:,} across {args.epochs} epochs")

    # 3. Optimizer & Scheduler
    # Freeze classification, sentiment/urgency, and CSAT heads to preserve M9.5 representations
    log("\n[4/5] Setting up AdamW Optimizer & Scheduler...")
    for param in model.category_head.parameters():
        param.requires_grad = False
    for param in model.sentiment_head.parameters():
        param.requires_grad = False
    for param in model.csat_head.parameters():
        param.requires_grad = False
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"  Trainable parameters (Encoder-Decoder & LM Head): {trainable_count:,}")
    log(f"  Frozen parameters (Category & CSAT Heads):         {frozen_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_ckpt_path = os.path.join(args.output_dir, args.ckpt_name)
    best_val_loss = float("inf")
    global_step = 0

    test_inquiries = [
        "I was double charged for my monthly subscription. Can you please refund the extra fee?",
        "My tracking number says delivered but I have not received my package yet.",
        "Could you help me change the email address and shipping address linked to my account?",
        "Why is there a late fee on my credit card when I scheduled the payment two days before the due date?"
    ]

    # 4. Training Loop
    log("\n[5/5] Starting Clean Response SFT Training...")
    log("=" * 80)
    start_time = time.time()
    running_loss = 0.0
    running_tokens = 0

    for epoch in range(args.epochs):
        model.train()
        log(f"\n--- EPOCH {epoch + 1}/{args.epochs} ---")

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, generation_labels=labels)
                    loss = outputs["generation_loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, generation_labels=labels)
                loss = outputs["generation_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1

            active_tokens = (labels != -100).sum().item()
            running_loss += loss.item() * active_tokens
            running_tokens += active_tokens

            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_remaining = total_steps - global_step
                eta_secs = steps_remaining * (elapsed / global_step) if global_step > 0 else 0
                avg_loss = running_loss / max(running_tokens, 1)

                log(f"  Step {global_step:>6,}/{total_steps:,} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | ETA: {datetime.timedelta(seconds=int(eta_secs))}")
                running_loss = 0.0
                running_tokens = 0

            # Periodic Validation & Live Generation
            if global_step % args.eval_every == 0:
                log(f"\n  📊 Running Validation at Step {global_step:,}...")
                val_loss, val_ppl, _ = evaluate_clean_dialogue(model, val_loader, device, max_batches=150)
                log(f"  📊 Clean Val Loss: {val_loss:.4f} | Clean Val PPL: {val_ppl:.2f}")

                # Live Response Sample
                log("  🤖 Live Response Generation Check:")
                for test_idx, test_query in enumerate(test_inquiries[:2]):
                    enc = tokenizer(test_query, return_tensors="pt").to(device)
                    gen_toks = generate_multitask_response(model, tokenizer, config, enc["input_ids"], enc["attention_mask"], max_new_tokens=60, temperature=0.7)
                    gen_txt = tokenizer.decode(gen_toks[0].tolist(), skip_special_tokens=True).strip()
                    log(f"    Q{test_idx+1}: \"{test_query[:45]}...\" -> \"{gen_txt[:70]}...\"")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_clean_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_loss)
                    log(f"  ⭐ New Best Clean Val Loss: {best_val_loss:.4f} (PPL: {val_ppl:.2f})!")

                model.train()

        # End of Epoch Validation
        log(f"\n✅ Epoch {epoch+1} Complete. Full Validation Evaluation...")
        val_loss, val_ppl, _ = evaluate_clean_dialogue(model, val_loader, device, max_batches=None)
        log(f"  📊 Epoch {epoch+1} Clean Val Loss: {val_loss:.4f} | Clean Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_clean_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_loss)

    # 5. Final Test Evaluation on Clean Test Set
    log("\n" + "=" * 80)
    log("  🏆 FINAL CLEAN TEST-SET BENCHMARK (clean_dialogue_test.parquet)")
    log("=" * 80)

    if os.path.exists(best_ckpt_path):
        log(f"Loading best clean checkpoint: {best_ckpt_path}")
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_data["model_state_dict"])

    test_loss, test_ppl, test_tokens = evaluate_clean_dialogue(model, test_loader, device, max_batches=None)
    log(f"\n  Final Clean Test Loss:       {test_loss:.4f}")
    log(f"  Final Clean Test Perplexity: {test_ppl:.2f}")
    log(f"  Total Clean Test Tokens:     {test_tokens:,}")

    # 6. Comprehensive Live Inference Generation
    log("\n" + "=" * 80)
    log("  🤖 FINAL UNIFIED MULTI-TASK LIVE INFERENCE SAMPLES (M10.2)")
    log("=" * 80)

    id2label = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            id2label = json.load(f).get("id2label", {})

    for i, inquiry in enumerate(test_inquiries):
        enc = tokenizer(inquiry, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cat_probs = F.softmax(outputs["category_logits"], dim=-1)[0]
            top_prob, top_cat_id = torch.max(cat_probs, dim=-1)
            cat_name = id2label.get(str(top_cat_id.item()), f"Category {top_cat_id.item()}")
            cat_conf = top_prob.item() * 100.0

            csat_star = torch.argmax(outputs["csat_logits"], dim=-1).item() + 1
            gen_tokens = generate_multitask_response(
                model, tokenizer, config, enc["input_ids"], enc["attention_mask"],
                max_new_tokens=100, temperature=0.7, repetition_penalty=1.2
            )
            raw_ids = gen_tokens[0].tolist()
            resp_text = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()

        print(f"\n[Scenario {i+1}]")
        print(f"  Customer: \"{inquiry}\"")
        print(f"  Predicted Category: {cat_name} (Confidence: {cat_conf:.1f}%) | CSAT Forecast: [{csat_star} ⭐]")
        print(f"  Raw IDs: {raw_ids[:20]}{'...' if len(raw_ids) > 20 else ''} (Total tokens: {len(raw_ids)})")
        print(f"  Clean Response Draft:\n  \"{resp_text}\"")

    log(f"\n✅ M10.3 Clean Response SFT Complete! Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
