"""
==========================================================================
  M9.5.1 — Multi-Task Throughput Benchmark (RTX 3060 Laptop GPU)
==========================================================================
  Tests simultaneous multi-task batch execution (Category + CSAT + Dialogue):
  - Measures Step time (ms)
  - Throughput (multi-task samples/sec)
  - VRAM Consumption Peak (GB)
  - Exact estimated runtime for joint calibration
==========================================================================
"""
import os
import sys
import time
import json
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator

sys.stdout.reconfigure(encoding="utf-8")

TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")

CLASS_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_train.parquet")
CSAT_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_train.parquet")
DIAL_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_train.parquet")


def benchmark_multitask():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  [M9.5.1] MULTI-TASK JOINT CALIBRATION THROUGHPUT BENCHMARK")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label2id = json.load(f)["label2id"]

    # Load small benchmark datasets
    class_ds = ClassificationParquetDataset(CLASS_TRAIN, tokenizer, label2id=label2id, max_seq_length=128, limit=500)
    csat_ds = CSATDataset(CSAT_TRAIN, tokenizer, max_len=128, limit=500)
    dial_ds = DialogueSFTDataset(DIAL_TRAIN, tokenizer, max_ctx_len=128, max_resp_len=256, limit=500)

    class_collator = ClassificationDataCollator(pad_token_id=config.pad_token_id, max_seq_length=128)
    csat_collator = CSATCollator(pad_token_id=config.pad_token_id)
    dial_collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)

    batch_configs = [
        {"cat": 16, "csat": 8, "dial": 16, "name": "Config 1 (16 Cat + 8 CSAT + 16 Dial = 40 samples/step)"},
        {"cat": 32, "csat": 16, "dial": 32, "name": "Config 2 (32 Cat + 16 CSAT + 32 Dial = 80 samples/step)"},
    ]

    print("\nBenchmarking multi-task forward + backward passes...\n")
    print(f"{'Configuration':<50} | {'Step Time':<12} | {'Throughput':<16} | {'VRAM Peak':<10} | {'Est. 25k Steps'}")
    print("-" * 105)

    for cfg in batch_configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        class_loader = DataLoader(class_ds, batch_size=cfg["cat"], shuffle=True, collate_fn=class_collator)
        csat_loader = DataLoader(csat_ds, batch_size=cfg["csat"], shuffle=True, collate_fn=csat_collator)
        dial_loader = DataLoader(dial_ds, batch_size=cfg["dial"], shuffle=True, collate_fn=dial_collator)

        model = SentriFormerForMultiTask(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        class_iter = iter(class_loader)
        csat_iter = iter(csat_loader)
        dial_iter = iter(dial_loader)

        # Warmup (3 steps)
        for _ in range(3):
            b_cat = next(class_iter)
            b_csat = next(csat_iter)
            b_dial = next(dial_iter)

            cat_ids, cat_mask, cat_lbl = b_cat["input_ids"].to(device), b_cat["attention_mask"].to(device), b_cat["labels"].to(device)
            csat_ids, csat_mask, csat_lbl = b_csat["input_ids"].to(device), b_csat["attention_mask"].to(device), b_csat["labels"].to(device)
            dial_ids, dial_mask, dial_lbl = b_dial["input_ids"].to(device), b_dial["attention_mask"].to(device), b_dial["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                o_cat = model(input_ids=cat_ids, attention_mask=cat_mask, category_labels=cat_lbl)
                o_csat = model(input_ids=csat_ids, attention_mask=csat_mask, csat_labels=csat_lbl)
                o_dial = model(input_ids=dial_ids, attention_mask=dial_mask, generation_labels=dial_lbl)
                loss = (1.0 * o_cat["category_loss"]) + (0.1 * o_csat["csat_loss"]) + (0.5 * o_dial["generation_loss"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Timed run (20 steps)
        t0 = time.time()
        steps_done = 0
        total_samples = 0
        samples_per_step = cfg["cat"] + cfg["csat"] + cfg["dial"]

        for _ in range(20):
            try:
                b_cat = next(class_iter)
                b_csat = next(csat_iter)
                b_dial = next(dial_iter)
            except StopIteration:
                class_iter = iter(class_loader)
                csat_iter = iter(csat_loader)
                dial_iter = iter(dial_loader)
                b_cat = next(class_iter)
                b_csat = next(csat_iter)
                b_dial = next(dial_iter)

            cat_ids, cat_mask, cat_lbl = b_cat["input_ids"].to(device), b_cat["attention_mask"].to(device), b_cat["labels"].to(device)
            csat_ids, csat_mask, csat_lbl = b_csat["input_ids"].to(device), b_csat["attention_mask"].to(device), b_csat["labels"].to(device)
            dial_ids, dial_mask, dial_lbl = b_dial["input_ids"].to(device), b_dial["attention_mask"].to(device), b_dial["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                o_cat = model(input_ids=cat_ids, attention_mask=cat_mask, category_labels=cat_lbl)
                o_csat = model(input_ids=csat_ids, attention_mask=csat_mask, csat_labels=csat_lbl)
                o_dial = model(input_ids=dial_ids, attention_mask=dial_mask, generation_labels=dial_lbl)
                loss = (1.0 * o_cat["category_loss"]) + (0.1 * o_csat["csat_loss"]) + (0.5 * o_dial["generation_loss"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            steps_done += 1
            total_samples += samples_per_step

        dur = time.time() - t0
        step_time_ms = (dur / steps_done) * 1000
        throughput = total_samples / dur
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        est_25k_steps_min = (25000 * (step_time_ms / 1000)) / 60

        print(f"{cfg['name']:<50} | {step_time_ms:>8.1f} ms | {throughput:>10.1f} s/s | {peak_vram:>7.2f} GB | {est_25k_steps_min:>8.1f} min ({est_25k_steps_min/60:.2f} h)")

    print("-" * 105)
    print("\n✅ M9.5.1 Throughput Benchmark Complete!")


if __name__ == "__main__":
    benchmark_multitask()
