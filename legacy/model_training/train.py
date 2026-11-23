"""
Training script for the SentriMail unified multi-task model (SentriModel).

Trains all 4 tasks jointly:
  1. Sentiment classification (2 classes)
  2. Emotion classification (7 classes)
  3. Category classification (7 classes, masked for unlabelled sources)
  4. Response generation (Autoregressive T5 decoder, masked for narrative-only sources)

Features:
  - Automatic Mixed Precision (AMP) on CUDA for maximum throughput
  - Dynamic sequence padding per batch via collate_fn
  - Class-weighted multi-task loss with task loss logging
  - Learning rate warmup with linear decay
  - Validation evaluation with Macro-F1 scores at the end of each epoch
  - Best model checkpoint saving
  - Mid-epoch checkpointing every CHECKPOINT_EVERY_STEPS steps
  - Full resume support: --resume loads the latest checkpoint and continues
    from the exact epoch+step where training was interrupted
"""

import argparse
import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import config
from dataset import SentriDataset, build_collate_fn
from losses import compute_class_weights, compute_multitask_loss
from model import SentriModel

LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
BEST_CHECKPOINT_NAME = "best_model.pt"


def save_checkpoint(path, epoch, step, model, optimizer, scheduler, scaler, best_val_loss, is_best=False):
    """Saves a full resumable checkpoint — model weights, optimizer state,
    scheduler state, scaler state, and the current position (epoch+step).
    Called both mid-epoch (every CHECKPOINT_EVERY_STEPS) and after each
    epoch's validation, so the worst-case loss on interruption is ~2.5 min
    of training at 3.1 it/s, not a full 1.9-hour epoch."""
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "sentiment_labels": config.SENTIMENT_LABELS,
        "emotion_labels": config.EMOTION_LABELS,
        "category_labels": config.CATEGORY_LABELS,
        "backbone_model_name": config.BACKBONE_MODEL_NAME,
    }
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), BEST_CHECKPOINT_NAME)
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Loads a previously saved checkpoint and returns the epoch and step
    to resume from. The step is the *next* step to execute — if training
    saved at step 500, this returns step=500 so the loop skips steps 0–499."""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    resume_epoch = ckpt["epoch"]
    resume_step = ckpt["step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print(f"  Restored epoch {resume_epoch}, step {resume_step}, best_val_loss={best_val_loss:.4f}")
    return resume_epoch, resume_step, best_val_loss


def evaluate(model, val_loader, class_weights, device):
    """Runs evaluation over the validation dataset and computes:
      - Average multi-task loss & individual task losses
      - Accuracy and Macro-F1 for Sentiment, Emotion, and Category
    """
    model.eval()
    total_val_loss = 0.0
    loss_accumulators = {
        "sentiment_loss": 0.0,
        "emotion_loss": 0.0,
        "category_loss": 0.0,
        "generation_loss": 0.0,
    }

    sentiment_preds, sentiment_trues = [], []
    emotion_preds, emotion_trues = [], []
    category_preds, category_trues = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_labels = batch["response_labels"].to(device)

            # Batch labels to device
            batch_labels = {
                "sentiment_labels": batch["sentiment_labels"].to(device),
                "has_sentiment": batch["has_sentiment"].to(device),
                "emotion_labels": batch["emotion_labels"].to(device),
                "has_emotion": batch["has_emotion"].to(device),
                "category_labels": batch["category_labels"].to(device),
                "has_category": batch["has_category"].to(device),
                "response_labels": response_labels,
                "has_response": batch["has_response"].to(device),
            }

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(input_ids, attention_mask, response_labels)
                val_loss, loss_dict = compute_multitask_loss(outputs, batch_labels, class_weights)

            total_val_loss += val_loss.item()
            for key in loss_accumulators:
                loss_accumulators[key] += loss_dict[key]

            # Classification predictions
            has_sent = batch["has_sentiment"].tolist()
            s_preds = torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().tolist()
            s_trues = batch["sentiment_labels"].tolist()
            for pred, true, valid in zip(s_preds, s_trues, has_sent):
                if valid:
                    sentiment_preds.append(pred)
                    sentiment_trues.append(true)

            has_emot = batch["has_emotion"].tolist()
            e_preds = torch.argmax(outputs["emotion_logits"], dim=-1).cpu().tolist()
            e_trues = batch["emotion_labels"].tolist()
            for pred, true, valid in zip(e_preds, e_trues, has_emot):
                if valid:
                    emotion_preds.append(pred)
                    emotion_trues.append(true)

            # Category only for rows with has_category=True
            has_cat = batch["has_category"].tolist()
            c_preds = torch.argmax(outputs["category_logits"], dim=-1).cpu().tolist()
            c_trues = batch["category_labels"].tolist()
            for pred, true, valid in zip(c_preds, c_trues, has_cat):
                if valid:
                    category_preds.append(pred)
                    category_trues.append(true)

    num_batches = len(val_loader)
    avg_loss = total_val_loss / max(num_batches, 1)

    metrics = {
        "val_total_loss": avg_loss,
        "sentiment_f1": f1_score(sentiment_trues, sentiment_preds, average="macro", zero_division=0) if sentiment_trues else 0.0,
        "sentiment_acc": accuracy_score(sentiment_trues, sentiment_preds) if sentiment_trues else 0.0,
        "emotion_f1": f1_score(emotion_trues, emotion_preds, average="macro", zero_division=0) if emotion_trues else 0.0,
        "emotion_acc": accuracy_score(emotion_trues, emotion_preds) if emotion_trues else 0.0,
        "category_f1": f1_score(category_trues, category_preds, average="macro", zero_division=0) if category_trues else 0.0,
        "category_acc": accuracy_score(category_trues, category_preds) if category_trues else 0.0,
    }
    for key in loss_accumulators:
        metrics[f"val_{key}"] = loss_accumulators[key] / max(num_batches, 1)

    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SentriMail Multi-Task Model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Peak learning rate")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Target device (default auto)")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for fast smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_checkpoint.pt")
    args = parser.parse_args()

    # 1. Device Resolution
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 2. Tokenizer & Datasets
    print(f"Loading tokenizer: {config.BACKBONE_MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(config.BACKBONE_MODEL_NAME)

    print(f"Loading training data from {config.TRAIN_PARQUET_PATH}...")
    train_dataset = SentriDataset(config.TRAIN_PARQUET_PATH)
    print(f"Loading validation data from {config.VAL_PARQUET_PATH}...")
    val_dataset = SentriDataset(config.VAL_PARQUET_PATH)

    collate_fn = build_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    batches_per_epoch = args.max_batches or len(train_loader)
    print(f"Train samples: {len(train_dataset):,} ({len(train_loader):,} batches, using {batches_per_epoch:,} per epoch)")
    print(f"Val samples:   {len(val_dataset):,} ({len(val_loader):,} batches)")

    # 3. Compute Class Weights for Imbalanced Classes
    print(f"Computing class weights ({config.CLASS_WEIGHT_METHOD}, beta={config.CLASS_WEIGHT_BETA})...")
    class_weights = {
        "sentiment": compute_class_weights(
            train_dataset.dataframe["sentiment"], config.SENTIMENT_LABELS,
            method=config.CLASS_WEIGHT_METHOD, beta=config.CLASS_WEIGHT_BETA,
        ).to(device),
        "emotion": compute_class_weights(
            train_dataset.dataframe["emotion"], config.EMOTION_LABELS,
            method=config.CLASS_WEIGHT_METHOD, beta=config.CLASS_WEIGHT_BETA,
        ).to(device),
        "category": compute_class_weights(
            train_dataset.dataframe["category"].dropna(), config.CATEGORY_LABELS,
            method=config.CLASS_WEIGHT_METHOD, beta=config.CLASS_WEIGHT_BETA,
        ).to(device),
    }
    print("  Category class weights:", {
        label: round(w.item(), 3) for label, w in zip(config.CATEGORY_LABELS, class_weights["category"])
    })

    # 4. Model, Optimizer, Scheduler & Scaler
    print(f"Initializing SentriModel (backbone: {config.BACKBONE_MODEL_NAME})...")
    model = SentriModel(config.BACKBONE_MODEL_NAME).to(device)

    total_steps = batches_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 1
    resume_step = 0  # 0 means start of epoch, >0 means skip that many batches

    # 5. Resume from checkpoint if requested
    latest_ckpt_path = os.path.join(config.CHECKPOINT_DIR, LATEST_CHECKPOINT_NAME)
    if args.resume and os.path.exists(latest_ckpt_path):
        ckpt_epoch, ckpt_step, best_val_loss = load_checkpoint(
            latest_ckpt_path, model, optimizer, scheduler, scaler, device
        )
        # If the checkpoint was saved at the END of an epoch (step == batches_per_epoch),
        # we start from the next epoch at step 0. Otherwise we resume mid-epoch.
        if ckpt_step >= batches_per_epoch:
            start_epoch = ckpt_epoch + 1
            resume_step = 0
            print(f"  Epoch {ckpt_epoch} was fully completed. Starting from epoch {start_epoch}.")
        else:
            start_epoch = ckpt_epoch
            resume_step = ckpt_step
            print(f"  Resuming epoch {start_epoch} from step {resume_step}/{batches_per_epoch}.")
    elif args.resume:
        print("No checkpoint found — starting fresh training.")

    # Also save tokenizer once upfront
    tokenizer.save_pretrained(config.CHECKPOINT_DIR)

    # 6. Training Loop
    print("\n" + "=" * 65)
    print("STARTING MULTI-TASK TRAINING")
    print(f"Epochs: {start_epoch}–{args.epochs} | Batch Size: {args.batch_size} | Total Steps: {total_steps:,}")
    print(f"Mid-epoch checkpoints every {config.CHECKPOINT_EVERY_STEPS} steps")
    print("=" * 65 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        steps_in_epoch = 0

        # How many steps to skip if resuming mid-epoch
        skip_steps = resume_step if epoch == start_epoch else 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            total=batches_per_epoch,
            initial=skip_steps,
        )

        for step, batch in enumerate(progress_bar):
            # Skip already-processed batches on resume
            if step < skip_steps:
                continue

            if args.max_batches and step >= args.max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_labels = batch["response_labels"].to(device)

            batch_labels = {
                "sentiment_labels": batch["sentiment_labels"].to(device),
                "has_sentiment": batch["has_sentiment"].to(device),
                "emotion_labels": batch["emotion_labels"].to(device),
                "has_emotion": batch["has_emotion"].to(device),
                "category_labels": batch["category_labels"].to(device),
                "has_category": batch["has_category"].to(device),
                "response_labels": response_labels,
                "has_response": batch["has_response"].to(device),
            }

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(input_ids, attention_mask, response_labels)
                loss, loss_breakdown = compute_multitask_loss(outputs, batch_labels, class_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            steps_in_epoch += 1
            avg_loss = running_loss / steps_in_epoch
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "sent": f"{loss_breakdown['sentiment_loss']:.2f}",
                "emot": f"{loss_breakdown['emotion_loss']:.2f}",
                "cat": f"{loss_breakdown['category_loss']:.2f}",
                "gen": f"{loss_breakdown['generation_loss']:.2f}",
                "lr": f"{current_lr:.2e}",
            })

            # Mid-epoch checkpoint — worst case you lose CHECKPOINT_EVERY_STEPS
            # steps (~2.5 min at 3.1 it/s with 500-step interval), not 1.9 hours
            if steps_in_epoch > 0 and steps_in_epoch % config.CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(
                    latest_ckpt_path, epoch, step + 1,
                    model, optimizer, scheduler, scaler, best_val_loss,
                )
                tqdm.write(f"  [checkpoint] saved at epoch {epoch}, step {step + 1}")

        epoch_duration = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch} Complete in {epoch_duration/60:.1f} min | Train Loss: {avg_loss:.4f} ---")

        # Validation Phase
        print("Running validation evaluation...")
        val_metrics = evaluate(model, val_loader, class_weights, device)

        print(f"Validation Results (Epoch {epoch}):")
        print(f"  Total Loss:   {val_metrics['val_total_loss']:.4f}")
        print(f"  Sentiment:    Macro-F1 = {val_metrics['sentiment_f1']:.4f} | Acc = {val_metrics['sentiment_acc']:.4f}")
        print(f"  Emotion:      Macro-F1 = {val_metrics['emotion_f1']:.4f} | Acc = {val_metrics['emotion_acc']:.4f}")
        print(f"  Category:     Macro-F1 = {val_metrics['category_f1']:.4f} | Acc = {val_metrics['category_acc']:.4f}")
        print(f"  Losses: Sent={val_metrics['val_sentiment_loss']:.3f} | Emot={val_metrics['val_emotion_loss']:.3f} | Cat={val_metrics['val_category_loss']:.3f} | Gen={val_metrics['val_generation_loss']:.3f}")

        # End-of-epoch checkpoint (always saved as latest, and as best if val improved)
        is_new_best = val_metrics["val_total_loss"] < best_val_loss
        if is_new_best:
            best_val_loss = val_metrics["val_total_loss"]

        save_checkpoint(
            latest_ckpt_path, epoch, batches_per_epoch,
            model, optimizer, scheduler, scaler, best_val_loss,
            is_best=is_new_best,
        )

        if is_new_best:
            print(f"  --> New best! Saved best_model.pt (Val Loss: {best_val_loss:.4f})")
        else:
            print(f"  --> Val loss {val_metrics['val_total_loss']:.4f} did not improve from {best_val_loss:.4f}")

        print("-" * 65 + "\n")

    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model: {os.path.join(config.CHECKPOINT_DIR, BEST_CHECKPOINT_NAME)}")
    print(f"Latest checkpoint: {os.path.join(config.CHECKPOINT_DIR, LATEST_CHECKPOINT_NAME)}")


if __name__ == "__main__":
    main()
