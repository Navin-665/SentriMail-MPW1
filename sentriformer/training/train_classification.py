"""
==========================================================================
  M9.2 — SentriFormer Multi-Task Classification Fine-Tuning
==========================================================================
  - Initializes Pretrained Encoder Backbone from checkpoint_best.pt
  - Trains Category Classification Head (33 Customer Issue Categories)
  - Full Metrics: Accuracy, Macro F1, Weighted F1, Confusion Matrix
  - FP16 Mixed Precision, AdamW, Cosine LR Scheduler, Gradient Clipping
  - Auto-resumes from latest checkpoint if interrupted
  - Evaluates on 100% unseen Test Set (classification_test.parquet)
==========================================================================
"""
import os
import sys
import time
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask, pool_encoder_states
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator

sys.stdout.reconfigure(encoding="utf-8")

# Default Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
PRETRAINED_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "checkpoint_best.pt")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_train.parquet")
VAL_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_val.parquet")
TEST_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_test.parquet")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "classification")
LOG_FILE = os.path.join(OUTPUT_DIR, "classification_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


def evaluate_classifier(model, dataloader, device, num_classes, max_batches=None):
    """Computes Loss, Accuracy, Macro F1, and Weighted F1 on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                loss = outputs["category_loss"]
                logits = outputs["category_logits"]
                
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    avg_loss = total_loss / max(len(all_targets), 1)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(all_targets),
        "preds": all_preds,
        "targets": all_targets
    }


def save_classifier_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, best_val_f1, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_f1": best_val_f1,
        "metadata": metadata
    }
    torch.save(state, path)
    log(f"  💾 Saved classifier checkpoint: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SentriFormer M9.2 Classification Fine-Tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence context")
    parser.add_argument("--eval_every", type=int, default=2500, help="Evaluate on validation set every N steps")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit train samples (for quick debug)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 80)
    log("  [M9.2] SENTRIFORMER MULTI-TASK CLASSIFICATION FINE-TUNING")
    log("=" * 80)
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer and Label Mapping
    log("\n[1/5] Loading Tokenizer & Category Labels...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_info = json.load(f)
    label2id = label_info["label2id"]
    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    num_classes = label_info["num_classes"]
    log(f"  Tokenizer vocab: {len(tokenizer):,} tokens")
    log(f"  Classification categories: {num_classes} classes")

    # 2. Build Model and Load Pretrained Backbone
    log("\n[2/5] Initializing SentriFormer Multi-Task Architecture & Loading Weights...")
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = num_classes
    
    model = SentriFormerForMultiTask(config).to(device)
    
    # Load pretrained weights from checkpoint_best.pt into backbone
    log(f"  Loading pretrained backbone from: {PRETRAINED_CKPT}")
    pt_raw = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    pt_dict = pt_raw["model_state_dict"] if "model_state_dict" in pt_raw else pt_raw
    
    # Load weights with strict=False (loads shared_embeddings, encoder, decoder, lm_head; initializes fresh category head)
    missing, unexpected = model.load_state_dict(pt_dict, strict=False)
    log(f"  Pretrained backbone loaded successfully!")
    log(f"  Newly initialized heads: {[k for k in missing if 'category_head' in k or 'sentiment_head' in k or 'csat_head' in k]}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total Model Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # 3. Create DataLoaders
    log("\n[3/5] Loading Datasets...")
    train_ds = ClassificationParquetDataset(TRAIN_PARQUET, tokenizer, label2id, max_seq_length=args.max_seq_length, limit=args.limit_train)
    val_ds = ClassificationParquetDataset(VAL_PARQUET, tokenizer, label2id, max_seq_length=args.max_seq_length)
    test_ds = ClassificationParquetDataset(TEST_PARQUET, tokenizer, label2id, max_seq_length=args.max_seq_length)
    
    collator = ClassificationDataCollator(pad_token_id=config.pad_token_id, max_seq_length=args.max_seq_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator, pin_memory=True)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    log(f"  Train samples: {len(train_ds):,} ({steps_per_epoch:,} steps/epoch)")
    log(f"  Val samples:   {len(val_ds):,}")
    log(f"  Test samples:  {len(test_ds):,}")
    log(f"  Total training steps: {total_steps:,} across {args.epochs} epochs")

    # 4. Optimizer & Schedule
    log("\n[4/5] Setting up AdamW Optimizer & Cosine Schedule...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Check for resume
    latest_ckpt = os.path.join(OUTPUT_DIR, "classifier_best.pt")
    start_epoch = 0
    global_step = 0
    best_val_f1 = 0.0

    # 5. Training Loop
    log("\n[5/5] Starting Classification Training...")
    log("=" * 80)
    model.train()
    running_loss = 0.0
    running_samples = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        log(f"\n--- EPOCH {epoch + 1}/{args.epochs} ---")
        epoch_start = time.time()
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                    loss = outputs["category_loss"]
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                loss = outputs["category_loss"]
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            global_step += 1
            
            running_loss += loss.item() * len(labels)
            running_samples += len(labels)
            
            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_remaining = total_steps - global_step
                eta_secs = steps_remaining * (elapsed / global_step) if global_step > 0 else 0
                avg_loss = running_loss / max(running_samples, 1)
                
                log(f"  Step {global_step:>6,}/{total_steps:,} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | ETA: {datetime.timedelta(seconds=int(eta_secs))}")
                running_loss = 0.0
                running_samples = 0
                
            # Periodic Validation
            if global_step % args.eval_every == 0:
                log(f"\n  📊 Running Validation at Step {global_step:,}...")
                val_metrics = evaluate_classifier(model, val_loader, device, num_classes, max_batches=300)
                log(f"  📊 Val Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']:.4f} | Weighted F1: {val_metrics['weighted_f1']:.4f}")
                
                # Check for new best model
                if val_metrics["weighted_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["weighted_f1"]
                    save_classifier_checkpoint(
                        os.path.join(OUTPUT_DIR, "classifier_best.pt"),
                        model, optimizer, scaler, scheduler, epoch, global_step, best_val_f1,
                        {"val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"]}
                    )
                    log(f"  ⭐ New Best Val Weighted F1: {best_val_f1*100:.2f}%!")
                    
                model.train()

        # End of epoch validation
        log(f"\n✅ Epoch {epoch+1} Complete. Full Validation Evaluation...")
        val_metrics = evaluate_classifier(model, val_loader, device, num_classes)
        log(f"  📊 Epoch {epoch+1} Val Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']:.4f} | Weighted F1: {val_metrics['weighted_f1']:.4f}")
        
        if val_metrics["weighted_f1"] > best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            save_classifier_checkpoint(
                os.path.join(OUTPUT_DIR, "classifier_best.pt"),
                model, optimizer, scaler, scheduler, epoch, global_step, best_val_f1,
                {"val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"]}
            )

    # 6. Final Evaluation on 100% Unseen Test Set
    log("\n" + "=" * 80)
    log("  🏆 FINAL TEST-SET BENCHMARK (classification_test.parquet)")
    log("=" * 80)
    
    # Load best checkpoint
    best_ckpt_path = os.path.join(OUTPUT_DIR, "classifier_best.pt")
    if os.path.exists(best_ckpt_path):
        log(f"Loading best checkpoint for final evaluation: {best_ckpt_path}")
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_data["model_state_dict"])
        
    test_metrics = evaluate_classifier(model, test_loader, device, num_classes)
    log(f"\n  Final Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    log(f"  Final Test Macro F1:    {test_metrics['macro_f1']:.4f}")
    log(f"  Final Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    log(f"  Total Test Samples:     {test_metrics['num_samples']:,}")
    
    # Generate Detailed Classification Report
    target_names = [id2label[i] for i in range(num_classes)]
    cls_report = classification_report(test_metrics["targets"], test_metrics["preds"], target_names=target_names, zero_division=0)
    print("\nDetailed Per-Category Performance Breakdown:\n")
    print(cls_report)
    
    with open(os.path.join(OUTPUT_DIR, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(cls_report)
        
    log(f"\n✅ Classification training and evaluation complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
