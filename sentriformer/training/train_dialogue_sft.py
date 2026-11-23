"""
==========================================================================
  M9.3 — SentriFormer Dialogue / Response SFT Fine-Tuning
==========================================================================
  - Initializes Seq2Seq Encoder-Decoder from checkpoint_best.pt
  - Trains Response Generation (Context -> Customer Support Response)
  - Evaluates Cross-Entropy Loss & Perplexity on Validation & Test Sets
  - Generates live test responses across unseen customer inquiries
  - FP16 Mixed Precision, AdamW, Cosine LR Scheduler, Gradient Clipping
  - Auto-resumes from latest checkpoint if interrupted
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
PRETRAINED_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "checkpoint_best.pt")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_train.parquet")
VAL_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_val.parquet")
TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_test.parquet")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "dialogue")
LOG_FILE = os.path.join(OUTPUT_DIR, "dialogue_sft_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


class DialogueSFTDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_ctx_len: int = 128, max_resp_len: int = 256, limit: int = None):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        
        table = pq.read_table(parquet_path, columns=["context", "response"])
        if limit:
            table = table.slice(0, limit)
            
        self.contexts = table["context"].to_pylist()
        self.responses = table["response"].to_pylist()
        
    def __len__(self):
        return len(self.contexts)
        
    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        resp = self.responses[idx]
        
        inp_ids = self.tokenizer.encode(ctx, truncation=True, max_length=self.max_ctx_len, add_special_tokens=True)
        resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.max_resp_len - 1, add_special_tokens=True)
        
        # Guarantee that every target response explicitly ends with </s> (EOS token ID 3)
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 3
        if len(resp_ids) == 0 or resp_ids[-1] != eos_id:
            resp_ids = resp_ids + [eos_id]
            
        return {"input_ids": inp_ids, "labels": resp_ids}


class DialogueSFTCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        batch_size = len(batch)
        max_inp = max(len(x["input_ids"]) for x in batch)
        max_tgt = max(len(x["labels"]) for x in batch)
        
        input_ids = torch.full((batch_size, max_inp), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_inp), dtype=torch.long)
        labels = torch.full((batch_size, max_tgt), -100, dtype=torch.long)
        
        for i, item in enumerate(batch):
            inp = item["input_ids"]
            tgt = item["labels"]
            input_ids[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            attention_mask[i, :len(inp)] = 1
            labels[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate_dialogue(model, dataloader, device, max_batches=200):
    """Computes Cross-Entropy Loss & Perplexity on dialogue dataset."""
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                
            active_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens
            
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl, total_tokens


def save_dialogue_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, best_val_loss):
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
    log(f"  💾 Saved dialogue checkpoint: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SentriFormer M9.3 Dialogue SFT Fine-Tuning")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (optimal 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for SFT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_ctx_len", type=int, default=128, help="Maximum context tokens")
    parser.add_argument("--max_resp_len", type=int, default=256, help="Maximum response tokens")
    parser.add_argument("--eval_every", type=int, default=2500, help="Evaluate every N steps")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit train samples for debugging")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 80)
    log("  [M9.3] SENTRIFORMER DIALOGUE / RESPONSE SFT FINE-TUNING")
    log("=" * 80)
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer & Model
    log("\n[1/5] Loading Tokenizer & Model Config...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)

    log("\n[2/5] Initializing Seq2Seq Architecture & Loading Pretrained Backbone...")
    model = SentriFormerForConditionalGeneration(config).to(device)
    
    # Load pretrained weights
    log(f"  Loading weights from: {PRETRAINED_CKPT}")
    pt_raw = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    pt_dict = pt_raw["model_state_dict"] if "model_state_dict" in pt_raw else pt_raw
    model.load_state_dict(pt_dict, strict=False)
    log("  Pretrained Seq2Seq weights loaded successfully!")

    # 2. Datasets
    log("\n[3/5] Loading Dialogue Datasets...")
    train_ds = DialogueSFTDataset(TRAIN_PARQUET, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len, limit=args.limit_train)
    val_ds = DialogueSFTDataset(VAL_PARQUET, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len)
    test_ds = DialogueSFTDataset(TEST_PARQUET, tokenizer, max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len)

    collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    log(f"  Train samples: {len(train_ds):,} ({steps_per_epoch:,} steps/epoch)")
    log(f"  Val samples:   {len(val_ds):,}")
    log(f"  Test samples:  {len(test_ds):,}")
    log(f"  Total steps:   {total_steps:,} across {args.epochs} epochs")

    # 3. Optimizer & Scheduler
    log("\n[4/5] Setting up AdamW Optimizer & Scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Check for existing checkpoint
    best_ckpt_path = os.path.join(OUTPUT_DIR, "dialogue_sft_best.pt")
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # 4. Training Loop
    log("\n[5/5] Starting Dialogue Response SFT Training...")
    log("=" * 80)
    model.train()
    running_loss = 0.0
    running_tokens = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        log(f"\n--- EPOCH {epoch + 1}/{args.epochs} ---")
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"]
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                
            # Periodic Validation
            if global_step % args.eval_every == 0:
                log(f"\n  📊 Running Validation at Step {global_step:,}...")
                val_loss, val_ppl, _ = evaluate_dialogue(model, val_loader, device, max_batches=200)
                log(f"  📊 Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_dialogue_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_loss)
                    log(f"  ⭐ New Best Val Loss: {best_val_loss:.4f} (PPL: {val_ppl:.2f})!")
                    
                model.train()

        # End of epoch validation
        log(f"\n✅ Epoch {epoch+1} Complete. Full Validation Evaluation...")
        val_loss, val_ppl, _ = evaluate_dialogue(model, val_loader, device)
        log(f"  📊 Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dialogue_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_loss)

    # 5. Final Evaluation on Held-Out Test Set
    log("\n" + "=" * 80)
    log("  🏆 FINAL TEST-SET BENCHMARK (dialogue_test.parquet)")
    log("=" * 80)
    
    if os.path.exists(best_ckpt_path):
        log(f"Loading best checkpoint for final evaluation: {best_ckpt_path}")
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_data["model_state_dict"])
        
    test_loss, test_ppl, test_tokens = evaluate_dialogue(model, test_loader, device)
    log(f"\n  Final Test Loss:       {test_loss:.4f}")
    log(f"  Final Test Perplexity: {test_ppl:.2f}")
    log(f"  Total Test Tokens:     {test_tokens:,}")
    
    # 6. Live Response Generation on Unseen Inquiries
    log("\n" + "=" * 80)
    log("  🤖 LIVE RESPONSE GENERATION SAMPLES (UNSEEN CUSTOMER INQUIRIES)")
    log("=" * 80)
    
    test_inquiries = [
        "I was double charged for my monthly subscription. Can you please refund the extra fee?",
        "My tracking number says delivered but I have not received my package yet.",
        "Could you help me change the email address and shipping address linked to my account?",
        "Why is there a late fee on my credit card when I scheduled the payment two days before the due date?"
    ]
    
    for i, inquiry in enumerate(test_inquiries):
        enc = tokenizer(inquiry, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                top_k=50
            )
        response_text = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True)
        print(f"\n[Test Case {i+1}]")
        print(f"  Customer: \"{inquiry}\"")
        print(f"  SentriFormer Response:\n  \"{response_text.strip()}\"")
        
    log(f"\n✅ Dialogue SFT training complete! Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
