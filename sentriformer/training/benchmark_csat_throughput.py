"""
==========================================================================
  M9.4 — CSAT Throughput Benchmark (RTX 3060 Laptop GPU)
==========================================================================
  Tests batch sizes 16, 32, 64, 128 to evaluate:
  - Throughput (samples/sec)
  - VRAM consumption (GB)
  - Estimated time per epoch across 15,217 training samples
==========================================================================
"""
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.train_csat import CSATDataset, CSATCollator

sys.stdout.reconfigure(encoding="utf-8")

TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "csat_train.parquet")


def benchmark_csat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  [M9.4] CSAT PREDICTION THROUGHPUT BENCHMARK")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_csat_classes = 5
    
    dataset = CSATDataset(TRAIN_PARQUET, tokenizer, max_len=128, limit=1000)
    collator = CSATCollator(pad_token_id=config.pad_token_id)
    
    batch_sizes = [16, 32, 64, 128]
    print(f"\nEvaluating batch sizes on {len(dataset)} benchmark samples...\n")
    print(f"{'Batch Size':<12} | {'Throughput (samples/s)':<24} | {'VRAM Peak (GB)':<16} | {'Est. Time / Epoch (15.2k)'}")
    print("-" * 80)
    
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = SentriFormerForMultiTask(config).to(device)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collator)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Warmup
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs["csat_logits"], labels)
            loss.backward()
            break
            
        # Timed run
        t0 = time.time()
        samples_done = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs["csat_logits"], labels)
            loss.backward()
            samples_done += len(labels)
            
        dur = time.time() - t0
        throughput = samples_done / dur
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        time_per_epoch_sec = 15217 / throughput
        
        print(f"{bs:<12} | {throughput:>18.1f} s/s | {peak_vram:>12.2f} GB | {time_per_epoch_sec:>10.1f}s ({time_per_epoch_sec/60:.2f} min)")
        
    print("\n✅ Benchmark Complete!")


if __name__ == "__main__":
    benchmark_csat()
