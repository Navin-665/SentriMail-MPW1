"""
==========================================================================
  [M9.2.1] Classification Throughput Benchmark & Compute Budget
==========================================================================
  Measures real-world training steps/sec and samples/sec for
  Encoder + Multi-Head Classification on NVIDIA RTX 3060.
==========================================================================
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")


def benchmark_classification():
    print("=" * 75)
    print("  [M9.2.1] CLASSIFICATION TRAINING THROUGHPUT BENCHMARK (RTX 3060)")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1"))
    
    with open(os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json"), "r", encoding="utf-8") as f:
        label_info = json.load(f)
    label2id = label_info["label2id"]
    
    config = SentriFormerConfig.from_pretrained(os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json"))
    config.num_categories = len(label2id)
    
    model = SentriFormerForMultiTask(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    
    train_ds = ClassificationParquetDataset(
        os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_train.parquet"),
        tokenizer=tok,
        label2id=label2id,
        max_seq_length=512,
        limit=20000
    )
    collator = ClassificationDataCollator(pad_token_id=config.pad_token_id, max_seq_length=512)
    
    for bs in [64, 128]:
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collator)
        it = iter(loader)
        
        # 1. Warmup
        for _ in range(10):
            batch = next(it)
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                loss = outputs["category_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # 2. Benchmark 100 steps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_samples = 0
        
        for _ in range(100):
            batch = next(it)
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, category_labels=labels)
                loss = outputs["category_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_samples += len(labels)
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        steps_per_sec = 100 / elapsed
        samples_per_sec = total_samples / elapsed
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        print(f"\nResults @ Batch Size = {bs}:")
        print(f"  - Steps/Sec:     {steps_per_sec:.2f} step/s ({1000/steps_per_sec:.1f} ms/step)")
        print(f"  - Throughput:    {samples_per_sec:,.1f} samples/sec")
        print(f"  - Peak VRAM:     {vram_mb:.1f} MB ({(6144-vram_mb)/1024:.2f} GB Free)")
        
        total_train_samples = 2_219_168
        epoch_sec = total_train_samples / samples_per_sec
        print(f"  - 1 Full Epoch (2.22M records): {epoch_sec/60:.1f} minutes ({epoch_sec/3600:.2f} hours)")
        print(f"  - 3 Full Epochs:                {(epoch_sec*3)/60:.1f} minutes ({((epoch_sec*3)/3600):.2f} hours)")


if __name__ == "__main__":
    benchmark_classification()
