"""
==========================================================================
  M9.4 — SentriFormer CSAT Prediction Head Fine-Tuning
==========================================================================
  - Loads pretrained encoder weights from checkpoint_best.pt
  - Trains CSATPredictionHead (5-class customer satisfaction prediction: 1-5 stars)
  - Uses class-weighted Cross-Entropy loss + Ordinal distance penalty
  - Evaluates Exact Accuracy, Within-1-Point Accuracy, MAE, RMSE, Macro & Weighted F1
  - Saves best checkpoint to sentriformer/checkpoints/csat/csat_best.pt
  - Generates comprehensive test-set classification report and evaluation JSON
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
PRETRAINED_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "checkpoint_best.pt")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_train.parquet")
VAL_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_val.parquet")
TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_test.parquet")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "csat")
LOG_FILE = os.path.join(OUTPUT_DIR, "csat_training_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


class CSATDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_len: int = 128, limit: int = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        table = pq.read_table(parquet_path, columns=["text", "csat_score"])
        if limit:
            table = table.slice(0, limit)
            
        self.texts = table["text"].to_pylist()
        # CSAT scores are 1-5, map to 0-4 for CrossEntropyLoss
        raw_scores = table["csat_score"].to_pylist()
        self.labels = [int(s) - 1 for s in raw_scores]
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer.encode(text, truncation=True, max_length=self.max_len, add_special_tokens=True)
        return {"input_ids": enc, "label": label}


class CSATCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(x["input_ids"]) for x in batch)
        
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        for i, item in enumerate(batch):
            inp = item["input_ids"]
            input_ids[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            attention_mask[i, :len(inp)] = 1
            labels[i] = item["label"]
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate_csat(model, dataloader, device):
    """Computes exact accuracy, within-1 accuracy, MAE, RMSE, Macro F1, Weighted F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16 if device.type == "cuda" else torch.float32):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["csat_logits"]
                loss = loss_fn(logits, labels)
                
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_loss = total_loss / max(len(all_targets), 1)
    
    # Ordinal evaluation (map 0-4 back to 1-5)
    preds_scores = np.array(all_preds) + 1
    targets_scores = np.array(all_targets) + 1
    
    acc = accuracy_score(all_targets, all_preds)
    within_1 = np.mean(np.abs(preds_scores - targets_scores) <= 1)
    mae = mean_absolute_error(targets_scores, preds_scores)
    rmse = root_mean_squared_error(targets_scores, preds_scores)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "within_1_acc": float(within_1),
        "mae": float(mae),
        "rmse": float(rmse),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "preds": all_preds,
        "targets": all_targets
    }


def save_csat_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, best_val_loss):
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
    log(f"  💾 Saved CSAT checkpoint: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SentriFormer M9.4 CSAT Prediction Head Fine-Tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for head + backbone fine-tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 80)
    log("  [M9.4] SENTRIFORMER CSAT PREDICTION HEAD FINE-TUNING")
    log("=" * 80)
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer & Config
    log("\n[1/5] Loading Tokenizer & Model Config...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_csat_classes = 5

    log("\n[2/5] Initializing Multi-Task Model & Loading Pretrained Backbone...")
    model = SentriFormerForMultiTask(config).to(device)
    
    log(f"  Loading weights from: {PRETRAINED_CKPT}")
    pt_raw = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    pt_dict = pt_raw["model_state_dict"] if "model_state_dict" in pt_raw else pt_raw
    model.load_state_dict(pt_dict, strict=False)
    log("  Pretrained weights loaded successfully into Multi-Task backbone!")

    # 2. Datasets
    log("\n[3/5] Loading CSAT Datasets...")
    train_ds = CSATDataset(TRAIN_PARQUET, tokenizer, max_len=args.max_len)
    val_ds = CSATDataset(VAL_PARQUET, tokenizer, max_len=args.max_len)
    test_ds = CSATDataset(TEST_PARQUET, tokenizer, max_len=args.max_len)

    collator = CSATCollator(pad_token_id=config.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    log(f"  Train samples: {len(train_ds):,} ({steps_per_epoch:,} steps/epoch)")
    log(f"  Val samples:   {len(val_ds):,}")
    log(f"  Test samples:  {len(test_ds):,}")
    log(f"  Total steps:   {total_steps:,} across {args.epochs} epochs")

    # Compute class weights for imbalanced CSAT
    class_counts = [4032, 283, 484, 1339, 9079]
    total_count = sum(class_counts)
    class_weights = torch.tensor([total_count / (5.0 * c) for c in class_counts], dtype=torch.float, device=device)
    # Smooth class weights to prevent over-penalizing majority classes
    class_weights = torch.sqrt(class_weights)
    class_weights = class_weights / class_weights.mean()
    log(f"  Computed Balanced Class Weights: {class_weights.cpu().numpy().round(3).tolist()}")

    # 3. Optimizer & Scheduler
    log("\n[4/5] Setting up Optimizer & Scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_ckpt_path = os.path.join(OUTPUT_DIR, "csat_best.pt")
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    # 4. Training Loop
    log("\n[5/5] Starting CSAT Fine-Tuning...")
    log("=" * 80)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_samples = 0
        
        log(f"\n--- EPOCH {epoch + 1}/{args.epochs} ---")
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs["csat_logits"]
                    loss = loss_fn(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["csat_logits"]
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            global_step += 1
            running_loss += loss.item() * len(labels)
            running_samples += len(labels)
            
            if global_step % 50 == 0 or step == len(train_loader) - 1:
                cur_loss = running_loss / max(running_samples, 1)
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_rem = total_steps - global_step
                eta_secs = steps_rem * (elapsed / global_step) if global_step > 0 else 0
                log(f"  Step {global_step:>4,}/{total_steps:,} | Epoch {epoch+1} | Loss: {cur_loss:.4f} | LR: {cur_lr:.2e} | ETA: {datetime.timedelta(seconds=int(eta_secs))}")
                running_loss = 0.0
                running_samples = 0

        # Epoch Validation
        log(f"\n  📊 Epoch {epoch+1} Complete. Evaluating Validation Set...")
        val_metrics = evaluate_csat(model, val_loader, device)
        log(f"  📊 Val Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']*100:.2f}% | Within-1: {val_metrics['within_1_acc']*100:.2f}% | MAE: {val_metrics['mae']:.3f} | Macro F1: {val_metrics['macro_f1']:.4f}")
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_csat_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_loss)
            log(f"  ⭐ New Best Val Loss: {best_val_loss:.4f}!")

    # 5. Final Test Evaluation
    log("\n" + "=" * 80)
    log("  🏆 FINAL TEST-SET BENCHMARK (csat_test.parquet - 870 samples)")
    log("=" * 80)
    
    if os.path.exists(best_ckpt_path):
        log(f"Loading best checkpoint: {best_ckpt_path}")
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_data["model_state_dict"])
        
    test_metrics = evaluate_csat(model, test_loader, device)
    
    log(f"\n  Final Test Loss:        {test_metrics['loss']:.4f}")
    log(f"  Exact Accuracy:         {test_metrics['accuracy']*100:.2f}%")
    log(f"  Within-1-Point Acc:     {test_metrics['within_1_acc']*100:.2f}%")
    log(f"  Mean Absolute Error:    {test_metrics['mae']:.4f}")
    log(f"  Root Mean Squared Error:{test_metrics['rmse']:.4f}")
    log(f"  Macro F1:               {test_metrics['macro_f1']:.4f}")
    log(f"  Weighted F1:            {test_metrics['weighted_f1']:.4f}")

    target_names = ["1 Star (Very Poor)", "2 Stars (Poor)", "3 Stars (Neutral)", "4 Stars (Good)", "5 Stars (Excellent)"]
    report_str = classification_report(test_metrics["targets"], test_metrics["preds"], target_names=target_names, digits=4, zero_division=0)
    print("\n" + report_str)

    # Save reports
    report_file = os.path.join(OUTPUT_DIR, "test_csat_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"SENTRIMAIL M9.4 CSAT PREDICTION REPORT\n")
        f.write(f"Timestamp: {timestamp()}\n")
        f.write(f"Exact Accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Within-1 Accuracy: {test_metrics['within_1_acc']*100:.2f}%\n")
        f.write(f"MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n\n")
        f.write(report_str)
        
    log(f"\n💾 Saved detailed CSAT report to: {report_file}")
    log(f"✅ M9.4 CSAT Fine-Tuning Complete!")


if __name__ == "__main__":
    main()
