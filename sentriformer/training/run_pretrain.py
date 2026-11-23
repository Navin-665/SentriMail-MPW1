"""
==========================================================================
  M8.6 — SentriFormer Full Pretraining (3 Epochs, T5 Span Corruption)
==========================================================================
  ✅ Checkpoint every 10,000 steps
  ✅ Validation evaluation every 10,000 steps
  ✅ Save final checkpoint
  ✅ Resume capability (optimizer, scaler, scheduler, step, epoch state)
  ✅ Loss logging (JSON + console)
  ✅ FP16 mixed precision
  ✅ Batch size 64
  ✅ Gradient clipping (max_norm=1.0)
  ✅ Cosine LR schedule with 2000-step warmup
==========================================================================
"""
import os
import sys
import math
import time
import json
import datetime
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Project root and imports
# ---------------------------------------------------------------------------
ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration
from sentriformer.training.pretrain_dataset import PretrainParquetDataset, PretrainDataCollator

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# HYPERPARAMETERS (frozen for M8.6)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 2000
MAX_GRAD_NORM = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPS = 1e-8

# Checkpointing & Logging
CHECKPOINT_EVERY = 10_000       # Save checkpoint every N steps
VALIDATE_EVERY = 10_000         # Run validation every N steps
LOG_EVERY = 100                 # Print training loss every N steps
VAL_BATCHES = 500               # Number of validation batches per eval

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "train.parquet")
VAL_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "val.parquet")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "sentriformer", "checkpoints")
LOG_FILE = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "training_log.jsonl")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------
def run_validation(model, val_loader, device, max_batches=500):
    """Run validation and return average loss + token counts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

            # Count active decoder tokens for proper averaging
            active_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens
            total_batches += 1

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, total_batches, total_tokens


# ---------------------------------------------------------------------------
# Checkpoint Save / Load
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model,
    optimizer,
    scaler,
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_losses: list,
    val_losses: list,
    lr_history: list,
):
    """Save a full resumable checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses[-5000:],  # Keep last 5000 entries to limit file size
        "val_losses": val_losses,
        "lr_history": lr_history[-5000:],
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "num_epochs": NUM_EPOCHS,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
    }
    torch.save(checkpoint, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"  💾 Checkpoint saved: {path} ({size_mb:.1f} MB)")


def load_checkpoint(path: str, model, optimizer, scaler, scheduler, device):
    """Load a checkpoint and restore full training state."""
    log(f"  📂 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint.get("best_val_loss", float("inf")),
        checkpoint.get("train_losses", []),
        checkpoint.get("val_losses", []),
        checkpoint.get("lr_history", []),
    )


def find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent checkpoint file for auto-resume."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            try:
                step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
                checkpoints.append((step, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue

    # Also check for final checkpoint
    final = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    if os.path.exists(final):
        checkpoints.append((float("inf"), final))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


# ---------------------------------------------------------------------------
# Append-Only JSON Log
# ---------------------------------------------------------------------------
def append_log_entry(entry: dict):
    """Append a single JSON line to the training log."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    log("=" * 75)
    log("  M8.6 — SentriFormer Full Pretraining (3 Epochs)")
    log("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # -----------------------------------------------------------------------
    # 1. Load tokenizer & create model
    # -----------------------------------------------------------------------
    log("\n[1/5] Loading tokenizer and building model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    config = SentriFormerConfig(
        vocab_size=len(tokenizer),
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        attention_heads=8,
        head_dim=32,
        d_ff=1024,
        max_seq_length=MAX_SEQ_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )

    model = SentriFormerForConditionalGeneration(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # -----------------------------------------------------------------------
    # 2. Create dataloaders
    # -----------------------------------------------------------------------
    log("\n[2/5] Creating dataloaders...")
    log(f"  Train: {TRAIN_PARQUET}")
    log(f"  Val:   {VAL_PARQUET}")

    train_dataset = PretrainParquetDataset(
        parquet_path=TRAIN_PARQUET,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    train_collator = PretrainDataCollator(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3.0,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = PretrainParquetDataset(
        parquet_path=VAL_PARQUET,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    val_collator = PretrainDataCollator(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3.0,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=0,
        pin_memory=True,
    )

    num_train_samples = len(train_dataset)
    steps_per_epoch = num_train_samples // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    log(f"  Training samples:  {num_train_samples:,}")
    log(f"  Steps per epoch:   {steps_per_epoch:,}")
    log(f"  Total steps:       {total_steps:,}")
    log(f"  Batch size:        {BATCH_SIZE}")
    log(f"  Epochs:            {NUM_EPOCHS}")

    # -----------------------------------------------------------------------
    # 3. Optimizer, scheduler, scaler
    # -----------------------------------------------------------------------
    log("\n[3/5] Setting up optimizer, scheduler, and scaler...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    log(f"  Optimizer:    AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    log(f"  Scheduler:    Cosine with {WARMUP_STEPS} warmup steps")
    log(f"  Mixed Prec:   FP16 (GradScaler enabled)")
    log(f"  Grad Clip:    max_norm={MAX_GRAD_NORM}")

    # -----------------------------------------------------------------------
    # 4. Resume from checkpoint if available
    # -----------------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    lr_history = []
    resume_step_in_epoch = 0

    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        log(f"\n[RESUME] Found checkpoint: {latest_ckpt}")
        (
            start_epoch,
            global_step,
            best_val_loss,
            train_losses,
            val_losses,
            lr_history,
        ) = load_checkpoint(latest_ckpt, model, optimizer, scaler, scheduler, device)

        # Calculate which step within the epoch we need to skip to
        resume_step_in_epoch = global_step % steps_per_epoch
        if resume_step_in_epoch == 0 and global_step > 0:
            # Completed full epoch, start next one
            start_epoch = global_step // steps_per_epoch
            resume_step_in_epoch = 0

        log(f"  Resuming from epoch {start_epoch}, global step {global_step}")
        log(f"  Will skip {resume_step_in_epoch} steps in epoch {start_epoch}")
        log(f"  Best val loss so far: {best_val_loss:.6f}")
    else:
        log("\n[START] No checkpoint found. Starting fresh training.")

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    log("\n[4/5] Starting training loop...")
    log("=" * 75)

    # Log hyperparameters
    append_log_entry({
        "event": "training_start",
        "timestamp": timestamp(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": num_train_samples,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "start_epoch": start_epoch,
        "resume_step": global_step,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "num_epochs": NUM_EPOCHS,
        },
    })

    model.train()
    training_start_time = time.time()
    epoch_start_time = time.time()

    # Running loss for LOG_EVERY smoothing
    running_loss = 0.0
    running_tokens = 0
    running_enc_tokens = 0
    running_dec_tokens = 0

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            log(f"\n{'─' * 75}")
            log(f"  EPOCH {epoch + 1}/{NUM_EPOCHS}")
            log(f"{'─' * 75}")
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0
            epoch_tokens = 0
            epoch_steps = 0

            for step_in_epoch, batch in enumerate(train_loader):
                # Skip already-processed steps when resuming
                if epoch == start_epoch and step_in_epoch < resume_step_in_epoch:
                    continue

                # ---- Forward + Backward ----
                optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs["loss"]

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                global_step += 1

                # ---- Token accounting ----
                enc_active = attention_mask.sum().item()
                dec_active = (labels != -100).sum().item()
                step_tokens = enc_active + dec_active
                loss_val = loss.item()

                running_loss += loss_val * dec_active  # Weight by decoder tokens
                running_tokens += dec_active
                running_enc_tokens += enc_active
                running_dec_tokens += dec_active
                epoch_loss_sum += loss_val * dec_active
                epoch_tokens += dec_active
                epoch_steps += 1

                # ---- Console logging ----
                if global_step % LOG_EVERY == 0:
                    avg_loss = running_loss / max(running_tokens, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - training_start_time
                    steps_remaining = total_steps - global_step
                    tokens_per_sec = (running_enc_tokens + running_dec_tokens) / (LOG_EVERY * (elapsed / global_step)) if global_step > 0 else 0

                    # Simple ETA
                    if global_step > 0:
                        secs_per_step = elapsed / (global_step - (start_epoch * steps_per_epoch + resume_step_in_epoch) if epoch == start_epoch else global_step - start_epoch * steps_per_epoch)
                        eta_secs = steps_remaining * (elapsed / (global_step - (start_epoch * steps_per_epoch + resume_step_in_epoch))) if (global_step - (start_epoch * steps_per_epoch + resume_step_in_epoch)) > 0 else 0
                        eta_str = str(datetime.timedelta(seconds=int(eta_secs)))
                    else:
                        eta_str = "N/A"

                    log(
                        f"  Step {global_step:>7,}/{total_steps:,} | "
                        f"Epoch {epoch+1} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad: {grad_norm:.2f} | "
                        f"ETA: {eta_str}"
                    )

                    # Append to log
                    train_losses.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 6),
                        "lr": current_lr,
                        "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                        "enc_tokens": running_enc_tokens,
                        "dec_tokens": running_dec_tokens,
                    })
                    lr_history.append({"step": global_step, "lr": current_lr})

                    append_log_entry({
                        "event": "train_step",
                        "timestamp": timestamp(),
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 6),
                        "lr": current_lr,
                        "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                    })

                    # Reset running averages
                    running_loss = 0.0
                    running_tokens = 0
                    running_enc_tokens = 0
                    running_dec_tokens = 0

                # ---- Validation ----
                if global_step % VALIDATE_EVERY == 0:
                    log(f"\n  📊 Running validation at step {global_step:,}...")
                    val_loss, val_batches, val_tokens = run_validation(
                        model, val_loader, device, max_batches=VAL_BATCHES
                    )
                    log(f"  📊 Val Loss: {val_loss:.6f} ({val_batches} batches, {val_tokens:,} tokens)")

                    val_losses.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "val_loss": round(val_loss, 6),
                        "val_batches": val_batches,
                        "val_tokens": val_tokens,
                    })

                    append_log_entry({
                        "event": "validation",
                        "timestamp": timestamp(),
                        "step": global_step,
                        "epoch": epoch + 1,
                        "val_loss": round(val_loss, 6),
                        "val_batches": val_batches,
                        "val_tokens": val_tokens,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
                        save_checkpoint(
                            best_path, model, optimizer, scaler, scheduler,
                            epoch, global_step, best_val_loss,
                            train_losses, val_losses, lr_history,
                        )
                        log(f"  ⭐ New best val loss: {best_val_loss:.6f}")

                    model.train()

                # ---- Periodic checkpoint ----
                if global_step % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}.pt")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scaler, scheduler,
                        epoch, global_step, best_val_loss,
                        train_losses, val_losses, lr_history,
                    )

            # ---- End of epoch ----
            epoch_elapsed = time.time() - epoch_start_time
            epoch_avg_loss = epoch_loss_sum / max(epoch_tokens, 1)
            log(f"\n  ✅ Epoch {epoch + 1} complete in {datetime.timedelta(seconds=int(epoch_elapsed))}")
            log(f"     Avg train loss: {epoch_avg_loss:.6f}")
            log(f"     Steps: {epoch_steps:,}")

            append_log_entry({
                "event": "epoch_end",
                "timestamp": timestamp(),
                "epoch": epoch + 1,
                "avg_train_loss": round(epoch_avg_loss, 6),
                "epoch_steps": epoch_steps,
                "elapsed_seconds": round(epoch_elapsed, 1),
            })

            # Epoch-end validation
            log(f"\n  📊 Running end-of-epoch validation...")
            val_loss, val_batches, val_tokens = run_validation(
                model, val_loader, device, max_batches=VAL_BATCHES
            )
            log(f"  📊 Epoch {epoch + 1} Val Loss: {val_loss:.6f}")

            val_losses.append({
                "step": global_step,
                "epoch": epoch + 1,
                "val_loss": round(val_loss, 6),
                "val_batches": val_batches,
                "val_tokens": val_tokens,
                "epoch_end": True,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
                save_checkpoint(
                    best_path, model, optimizer, scaler, scheduler,
                    epoch, global_step, best_val_loss,
                    train_losses, val_losses, lr_history,
                )
                log(f"  ⭐ New best val loss: {best_val_loss:.6f}")

            model.train()

            # Reset resume offset for subsequent epochs
            resume_step_in_epoch = 0

        # -----------------------------------------------------------------------
        # 6. Save final checkpoint & summary
        # -----------------------------------------------------------------------
        log("\n[5/5] Saving final checkpoint...")
        final_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
        save_checkpoint(
            final_path, model, optimizer, scaler, scheduler,
            NUM_EPOCHS - 1, global_step, best_val_loss,
            train_losses, val_losses, lr_history,
        )

        # Also save just the model weights for easy loading
        model_weights_path = os.path.join(CHECKPOINT_DIR, "sentriformer_base_weights.pt")
        torch.save(model.state_dict(), model_weights_path)
        log(f"  💾 Model weights saved: {model_weights_path}")

        # Save config
        config_path = os.path.join(CHECKPOINT_DIR, "config.json")
        config.save_pretrained(config_path)
        log(f"  💾 Config saved: {config_path}")

        total_elapsed = time.time() - training_start_time
        log("\n" + "=" * 75)
        log("  🎉 PRETRAINING COMPLETE!")
        log("=" * 75)
        log(f"  Total time:      {datetime.timedelta(seconds=int(total_elapsed))}")
        log(f"  Total steps:     {global_step:,}")
        log(f"  Best val loss:   {best_val_loss:.6f}")
        log(f"  Final checkpoint: {final_path}")
        log(f"  Best checkpoint:  {os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pt')}")
        log(f"  Training log:     {LOG_FILE}")
        log("=" * 75)

        append_log_entry({
            "event": "training_complete",
            "timestamp": timestamp(),
            "total_steps": global_step,
            "total_seconds": round(total_elapsed, 1),
            "best_val_loss": round(best_val_loss, 6),
        })

    except KeyboardInterrupt:
        log("\n⚠️  Training interrupted by user!")
        emergency_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}_interrupted.pt")
        save_checkpoint(
            emergency_path, model, optimizer, scaler, scheduler,
            epoch, global_step, best_val_loss,
            train_losses, val_losses, lr_history,
        )
        log(f"  Emergency checkpoint saved. Resume with same command.")
        append_log_entry({
            "event": "interrupted",
            "timestamp": timestamp(),
            "step": global_step,
            "epoch": epoch + 1,
        })

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        emergency_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}_error.pt")
        try:
            save_checkpoint(
                emergency_path, model, optimizer, scaler, scheduler,
                epoch, global_step, best_val_loss,
                train_losses, val_losses, lr_history,
            )
            log(f"  Emergency checkpoint saved at step {global_step}. Resume with same command.")
        except Exception as save_err:
            log(f"  ❌ Could not save emergency checkpoint: {save_err}")

        append_log_entry({
            "event": "error",
            "timestamp": timestamp(),
            "step": global_step,
            "epoch": epoch + 1 if 'epoch' in dir() else 0,
            "error": str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
