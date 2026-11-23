"""
==========================================================================
  M9.5 — SentriFormer Unified Multi-Task Joint Calibration Training
==========================================================================
  - Loads clean pretrained backbone from M8 checkpoint_best.pt
  - Initializes Category Head (33 classes), CSAT Head (5 classes), and Dialogue Decoder
  - Simultaneously optimizes:
      L_total = 1.0 * L_category + 0.1 * L_csat + 0.5 * L_dialogue
  - Balanced joint batching across:
      1. CFPB Product Categories (classification_train.parquet)
      2. CSAT Satisfaction Ratings (csat_train.parquet)
      3. Dialogue Response Pairs (dialogue_train.parquet)
  - Evaluates all 3 tasks periodically on held-out validation sets:
      * Category Accuracy & Weighted F1
      * CSAT Within-±1 Accuracy & MAE
      * Dialogue Loss & Perplexity
  - Saves best unified checkpoint: sentriformer/checkpoints/multitask/multitask_best.pt
==========================================================================
"""
import os
import sys
import time
import math
import json
import datetime
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator

sys.stdout.reconfigure(encoding="utf-8")

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
PRETRAINED_M8 = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "checkpoint_best.pt")

CLASS_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_train.parquet")
CLASS_VAL = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_val.parquet")
CLASS_TEST = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_test.parquet")

CSAT_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_train.parquet")
CSAT_VAL = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_val.parquet")
CSAT_TEST = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_test.parquet")

DIAL_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_train.parquet")
DIAL_VAL = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_val.parquet")
DIAL_TEST = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_test.parquet")

OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "multitask")
LOG_FILE = os.path.join(OUTPUT_DIR, "multitask_training_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


def evaluate_multitask_validation(model, class_loader, csat_loader, dial_loader, device, max_batches=100):
    """Joint Validation Evaluation across Category, CSAT, and Dialogue."""
    model.eval()
    
    # 1. Category Evaluation
    all_preds_cat, all_tgts_cat = [], []
    cat_loss_tot = 0.0
    with torch.no_grad():
        for i, batch in enumerate(class_loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                cat_loss_tot += outputs["category_loss"].item() * len(labels)
                preds = torch.argmax(outputs["category_logits"], dim=-1).cpu().numpy()
            all_preds_cat.extend(preds)
            all_tgts_cat.extend(labels.cpu().numpy())
    cat_acc = accuracy_score(all_tgts_cat, all_preds_cat)
    cat_wf1 = f1_score(all_tgts_cat, all_preds_cat, average="weighted", zero_division=0)
    cat_loss = cat_loss_tot / max(len(all_tgts_cat), 1)

    # 2. CSAT Evaluation
    all_preds_csat, all_tgts_csat = [], []
    csat_loss_tot = 0.0
    with torch.no_grad():
        for i, batch in enumerate(csat_loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, csat_labels=labels)
                csat_loss_tot += outputs["csat_loss"].item() * len(labels)
                preds = torch.argmax(outputs["csat_logits"], dim=-1).cpu().numpy()
            all_preds_csat.extend(preds)
            all_tgts_csat.extend(labels.cpu().numpy())
    csat_acc = accuracy_score(all_tgts_csat, all_preds_csat)
    preds_scores = [p + 1 for p in all_preds_csat]
    tgts_scores = [t + 1 for t in all_tgts_csat]
    csat_w1 = sum(abs(p - t) <= 1 for p, t in zip(preds_scores, tgts_scores)) / max(len(tgts_scores), 1)
    csat_mae = mean_absolute_error(tgts_scores, preds_scores)
    csat_loss = csat_loss_tot / max(len(all_tgts_csat), 1)

    # 3. Dialogue Evaluation
    total_dial_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(dial_loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, generation_labels=labels)
                loss = outputs["generation_loss"]
            active_tokens = (labels != -100).sum().item()
            total_dial_loss += loss.item() * active_tokens
            total_tokens += active_tokens
    dial_loss = total_dial_loss / max(total_tokens, 1)
    dial_ppl = math.exp(min(dial_loss, 20.0))

    # Composite Validation Metric (lower is better)
    # loss_composite = 1.0 * cat_loss + 0.1 * csat_loss + 0.5 * dial_loss
    composite_score = (1.0 * cat_loss) + (0.1 * csat_loss) + (0.5 * dial_loss)

    return {
        "composite_loss": composite_score,
        "category_accuracy": cat_acc,
        "category_weighted_f1": cat_wf1,
        "category_loss": cat_loss,
        "csat_accuracy": csat_acc,
        "csat_within_1": csat_w1,
        "csat_mae": csat_mae,
        "csat_loss": csat_loss,
        "dialogue_loss": dial_loss,
        "dialogue_ppl": dial_ppl
    }


def save_multitask_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, best_val_score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_score": best_val_score
    }
    torch.save(state, path)
    log(f"  💾 Saved Unified Multi-Task checkpoint: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SentriFormer M9.5 Unified Multi-Task Joint Calibration")
    parser.add_argument("--epochs", type=int, default=2, help="Number of joint calibration epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for joint calibration")
    parser.add_argument("--batch_cat", type=int, default=16, help="Category batch size per step")
    parser.add_argument("--batch_csat", type=int, default=8, help="CSAT batch size per step")
    parser.add_argument("--batch_dial", type=int, default=16, help="Dialogue batch size per step")
    parser.add_argument("--eval_every", type=int, default=2500, help="Run validation every N steps")
    parser.add_argument("--steps_per_epoch", type=int, default=25000, help="Max joint steps per epoch")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 80)
    log("  [M9.5] SENTRIFORMER UNIFIED MULTI-TASK JOINT CALIBRATION")
    log("=" * 80)
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer & Model
    log("\n[1/5] Loading Tokenizer, Label Mapping & Model Config...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    label2id = labels_meta["label2id"]

    log("\n[2/5] Initializing SentriFormerForMultiTask & Loading M8 Pretrained Foundation...")
    model = SentriFormerForMultiTask(config).to(device)

    # Load M8 Pretrained Foundation
    log(f"  Foundation Weights: {PRETRAINED_M8}")
    m8_raw = torch.load(PRETRAINED_M8, map_location=device, weights_only=False)
    m8_dict = m8_raw["model_state_dict"] if "model_state_dict" in m8_raw else m8_raw
    model.load_state_dict(m8_dict, strict=False)
    log("  Pretrained M8 weights loaded successfully into Shared Backbone & Decoder!")

    # 2. Datasets
    log("\n[3/5] Setting Up Multi-Task Data Pipelines...")
    # Train Datasets
    class_train_ds = ClassificationParquetDataset(CLASS_TRAIN, tokenizer, label2id=label2id, max_seq_length=128)
    csat_train_ds = CSATDataset(CSAT_TRAIN, tokenizer, max_len=128)
    dial_train_ds = DialogueSFTDataset(DIAL_TRAIN, tokenizer, max_ctx_len=128, max_resp_len=256)

    # Val Datasets
    class_val_ds = ClassificationParquetDataset(CLASS_VAL, tokenizer, label2id=label2id, max_seq_length=128)
    csat_val_ds = CSATDataset(CSAT_VAL, tokenizer, max_len=128)
    dial_val_ds = DialogueSFTDataset(DIAL_VAL, tokenizer, max_ctx_len=128, max_resp_len=256)

    # Test Datasets
    class_test_ds = ClassificationParquetDataset(CLASS_TEST, tokenizer, label2id=label2id, max_seq_length=128)
    csat_test_ds = CSATDataset(CSAT_TEST, tokenizer, max_len=128)
    dial_test_ds = DialogueSFTDataset(DIAL_TEST, tokenizer, max_ctx_len=128, max_resp_len=256)

    # Collators
    class_collator = ClassificationDataCollator(pad_token_id=config.pad_token_id, max_seq_length=128)
    csat_collator = CSATCollator(pad_token_id=config.pad_token_id)
    dial_collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)

    # Loaders
    class_train_loader = DataLoader(class_train_ds, batch_size=args.batch_cat, shuffle=True, collate_fn=class_collator, pin_memory=True)
    csat_train_loader = DataLoader(csat_train_ds, batch_size=args.batch_csat, shuffle=True, collate_fn=csat_collator, pin_memory=True)
    dial_train_loader = DataLoader(dial_train_ds, batch_size=args.batch_dial, shuffle=True, collate_fn=dial_collator, pin_memory=True)

    class_val_loader = DataLoader(class_val_ds, batch_size=64, shuffle=False, collate_fn=class_collator, pin_memory=True)
    csat_val_loader = DataLoader(csat_val_ds, batch_size=64, shuffle=False, collate_fn=csat_collator, pin_memory=True)
    dial_val_loader = DataLoader(dial_val_ds, batch_size=32, shuffle=False, collate_fn=dial_collator, pin_memory=True)

    class_test_loader = DataLoader(class_test_ds, batch_size=64, shuffle=False, collate_fn=class_collator, pin_memory=True)
    csat_test_loader = DataLoader(csat_test_ds, batch_size=64, shuffle=False, collate_fn=csat_collator, pin_memory=True)
    dial_test_loader = DataLoader(dial_test_ds, batch_size=32, shuffle=False, collate_fn=dial_collator, pin_memory=True)

    total_steps = args.steps_per_epoch * args.epochs
    log(f"  Category Train: {len(class_train_ds):,} | Val: {len(class_val_ds):,} | Test: {len(class_test_ds):,}")
    log(f"  CSAT Train:     {len(csat_train_ds):,} | Val: {len(csat_val_ds):,} | Test: {len(csat_test_ds):,}")
    log(f"  Dialogue Train: {len(dial_train_ds):,} | Val: {len(dial_val_ds):,} | Test: {len(dial_test_ds):,}")
    log(f"  Steps per Epoch: {args.steps_per_epoch:,} | Total Joint Steps: {total_steps:,} across {args.epochs} epochs")

    # 3. Optimizer & Scheduler
    log("\n[4/5] Setting up AdamW Optimizer & Scheduler (LR: 5e-5, Warmup: 500 steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_ckpt_path = os.path.join(OUTPUT_DIR, "multitask_best.pt")
    best_val_score = float("inf")
    global_step = 0

    # 4. Joint Training Loop
    log("\n[5/5] Starting Unified Multi-Task Joint Calibration...")
    log("=" * 80)
    start_time = time.time()
    
    # Infinitely repeating iterators for CSAT & Dialogue to interleave with main loop
    csat_iter = itertools.cycle(csat_train_loader)
    dial_iter = itertools.cycle(dial_train_loader)
    class_iter = itertools.cycle(class_train_loader)

    for epoch in range(args.epochs):
        model.train()
        running_cat_loss = 0.0
        running_csat_loss = 0.0
        running_dial_loss = 0.0
        running_tot_loss = 0.0
        step_in_epoch = 0

        log(f"\n--- EPOCH {epoch + 1}/{args.epochs} ---")

        while step_in_epoch < args.steps_per_epoch:
            optimizer.zero_grad(set_to_none=True)

            # 1. Fetch multi-task batches
            batch_cat = next(class_iter)
            batch_csat = next(csat_iter)
            batch_dial = next(dial_iter)

            # Move to device
            cat_ids = batch_cat["input_ids"].to(device)
            cat_mask = batch_cat["attention_mask"].to(device)
            cat_labels = batch_cat["labels"].to(device)

            csat_ids = batch_csat["input_ids"].to(device)
            csat_mask = batch_csat["attention_mask"].to(device)
            csat_labels = batch_csat["labels"].to(device)

            dial_ids = batch_dial["input_ids"].to(device)
            dial_mask = batch_dial["attention_mask"].to(device)
            dial_labels = batch_dial["labels"].to(device)

            # 2. Forward & Loss computation
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    # Category Forward
                    out_cat = model(input_ids=cat_ids, attention_mask=cat_mask, category_labels=cat_labels)
                    loss_cat = out_cat["category_loss"]

                    # CSAT Forward
                    out_csat = model(input_ids=csat_ids, attention_mask=csat_mask, csat_labels=csat_labels)
                    loss_csat = out_csat["csat_loss"]

                    # Dialogue Forward
                    out_dial = model(input_ids=dial_ids, attention_mask=dial_mask, generation_labels=dial_labels)
                    loss_dial = out_dial["generation_loss"]

                    # Combined Weighted Loss
                    total_loss = (1.0 * loss_cat) + (0.1 * loss_csat) + (0.5 * loss_dial)

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out_cat = model(input_ids=cat_ids, attention_mask=cat_mask, category_labels=cat_labels)
                loss_cat = out_cat["category_loss"]
                out_csat = model(input_ids=csat_ids, attention_mask=csat_mask, csat_labels=csat_labels)
                loss_csat = out_csat["csat_loss"]
                out_dial = model(input_ids=dial_ids, attention_mask=dial_mask, generation_labels=dial_labels)
                loss_dial = out_dial["generation_loss"]
                total_loss = (1.0 * loss_cat) + (0.1 * loss_csat) + (0.5 * loss_dial)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1
            step_in_epoch += 1

            running_cat_loss += loss_cat.item()
            running_csat_loss += loss_csat.item()
            running_dial_loss += loss_dial.item()
            running_tot_loss += total_loss.item()

            if global_step % 100 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_rem = total_steps - global_step
                eta_secs = steps_rem * (elapsed / global_step) if global_step > 0 else 0

                c_loss = running_cat_loss / 100
                s_loss = running_csat_loss / 100
                d_loss = running_dial_loss / 100
                t_loss = running_tot_loss / 100

                log(f"  Step {global_step:>6,}/{total_steps:,} | Ep {epoch+1} | Total: {t_loss:.4f} [Cat: {c_loss:.4f}, CSAT: {s_loss:.4f}, Dial: {d_loss:.4f}] | LR: {cur_lr:.2e} | ETA: {datetime.timedelta(seconds=int(eta_secs))}")
                running_cat_loss, running_csat_loss, running_dial_loss, running_tot_loss = 0.0, 0.0, 0.0, 0.0

            # Periodic Validation
            if global_step % args.eval_every == 0:
                log(f"\n  📊 Running Joint Multi-Task Validation at Step {global_step:,}...")
                v = evaluate_multitask_validation(model, class_val_loader, csat_val_loader, dial_val_loader, device, max_batches=100)
                log(f"  📊 Val Metrics -> Cat Acc: {v['category_accuracy']*100:.2f}% (WF1: {v['category_weighted_f1']:.4f}) | CSAT ±1: {v['csat_within_1']*100:.2f}% (MAE: {v['csat_mae']:.3f}) | Dial PPL: {v['dialogue_ppl']:.2f} | Composite Loss: {v['composite_loss']:.4f}")

                if v["composite_loss"] < best_val_score:
                    best_val_score = v["composite_loss"]
                    save_multitask_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_score)
                    log(f"  ⭐ New Best Joint Validation Score: {best_val_score:.4f}!")

                model.train()

        # End of Epoch Full Validation
        log(f"\n✅ Epoch {epoch+1} Complete. Running Full Validation...")
        v = evaluate_multitask_validation(model, class_val_loader, csat_val_loader, dial_val_loader, device, max_batches=None)
        log(f"  📊 Epoch {epoch+1} Full Val -> Cat Acc: {v['category_accuracy']*100:.2f}% | CSAT ±1: {v['csat_within_1']*100:.2f}% (MAE: {v['csat_mae']:.3f}) | Dial PPL: {v['dialogue_ppl']:.2f}")

        if v["composite_loss"] < best_val_score:
            best_val_score = v["composite_loss"]
            save_multitask_checkpoint(best_ckpt_path, model, optimizer, scaler, scheduler, epoch, global_step, best_val_score)

    # 5. Final Test Evaluation across all 3 held-out test sets
    log("\n" + "=" * 80)
    log("  🏆 FINAL UNIFIED TEST-SET BENCHMARK (Held-Out Test Sets)")
    log("=" * 80)

    if os.path.exists(best_ckpt_path):
        log(f"Loading best checkpoint: {best_ckpt_path}")
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_data["model_state_dict"])

    test_metrics = evaluate_multitask_validation(model, class_test_loader, csat_test_loader, dial_test_loader, device, max_batches=None)
    log(f"\n  Final Test Category Accuracy:    {test_metrics['category_accuracy']*100:.2f}%")
    log(f"  Final Test Category Weighted F1: {test_metrics['category_weighted_f1']:.4f}")
    log(f"  Final Test CSAT Exact Acc:       {test_metrics['csat_accuracy']*100:.2f}%")
    log(f"  Final Test CSAT Within-±1 Acc:   {test_metrics['csat_within_1']*100:.2f}%")
    log(f"  Final Test CSAT MAE:             {test_metrics['csat_mae']:.4f}")
    log(f"  Final Test Dialogue Loss:        {test_metrics['dialogue_loss']:.4f}")
    log(f"  Final Test Dialogue Perplexity:  {test_metrics['dialogue_ppl']:.2f}")

    # Save summary report
    report_file = os.path.join(OUTPUT_DIR, "final_multitask_test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    log(f"\n💾 Saved Multi-Task Test Report to: {report_file}")
    log(f"🎉 M9.5 Unified Multi-Task Joint Calibration Complete!")


if __name__ == "__main__":
    main()
