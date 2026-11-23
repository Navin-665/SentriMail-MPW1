"""
==========================================================================
  [M9.3.2] Dialogue SFT Throughput Benchmark (NVIDIA RTX 3060)
==========================================================================
  Measures real-world training steps/sec, token throughput, and memory
  for Seq2Seq Dialogue Response SFT (Context -> Response).
==========================================================================
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")


class DialogueParquetDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_ctx_len: int = 128, max_resp_len: int = 256, limit: int = 20000):
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
        resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.max_resp_len, add_special_tokens=True)
        return {"input_ids": inp_ids, "labels": resp_ids}


class DialogueCollator:
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


def benchmark_dialogue():
    print("=" * 80)
    print("  [M9.3.2] DIALOGUE SFT TRAINING THROUGHPUT BENCHMARK (RTX 3060)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1"))
    config = SentriFormerConfig.from_pretrained(os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json"))
    
    model = SentriFormerForConditionalGeneration(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    
    train_ds = DialogueParquetDataset(
        os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "dialogue_train.parquet"),
        tokenizer=tok,
        max_ctx_len=128,
        max_resp_len=256,
        limit=20000
    )
    collator = DialogueCollator(pad_token_id=config.pad_token_id)
    
    total_train_samples = 893_896
    
    for bs in [32, 64]:
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # 2. Benchmark 100 steps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_samples = 0
        total_active_tokens = 0
        
        for _ in range(100):
            batch = next(it)
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_samples += len(labels)
            total_active_tokens += attention_mask.sum().item() + (labels != -100).sum().item()
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        steps_per_sec = 100 / elapsed
        samples_per_sec = total_samples / elapsed
        tokens_per_sec = total_active_tokens / elapsed
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        epoch_sec = total_train_samples / samples_per_sec
        
        print(f"\nThroughput @ Batch Size = {bs}:")
        print(f"  - Training Steps/Sec:     {steps_per_sec:.2f} step/s ({1000/steps_per_sec:.1f} ms/step)")
        print(f"  - Sample Throughput:      {samples_per_sec:,.1f} samples/sec")
        print(f"  - Active Token Rate:      {tokens_per_sec:,.1f} tokens/sec")
        print(f"  - Peak VRAM Allocated:    {vram_mb:.1f} MB ({(6144-vram_mb)/1024:.2f} GB Free Headroom)")
        print(f"  - 1 Full Epoch (893k pairs): {epoch_sec/60:.1f} minutes ({epoch_sec/3600:.2f} hours)")
        print(f"  - 3 Full Epochs:             {(epoch_sec*3)/60:.1f} minutes ({((epoch_sec*3)/3600):.2f} hours)")
        
    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_dialogue()
