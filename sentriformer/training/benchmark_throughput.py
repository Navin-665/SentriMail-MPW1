"""
SentriFormer GPU Throughput Benchmark & Training Time Calculator (M8.3, M8.4, M8.5)
Tests:
1. Tiny Pretraining Convergence (100 steps on real corpus data)
2. GPU Memory & Throughput Benchmark (Batch size 16, 32, 64)
3. Full Pretraining Duration & Compute Calculations for 3.18M Records (~750M Tokens)
"""
import os
import sys
import time
import json
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration
from sentriformer.training.train_pretrain import create_pretrain_dataloader, run_training_step


def test_tiny_pretrain_convergence(device: torch.device):
    print("\n" + "=" * 70)
    print(" [M8.3] TINY PRETRAINING TEST (100 Real Steps with AdamW & AMP)")
    print("=" * 70)
    
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    train_parquet = os.path.join(ROOT_DIR, "sentriformer", "data", "train.parquet")
    
    config = SentriFormerConfig(
        vocab_size=len(tok),
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        attention_heads=8,
        head_dim=32,
        d_ff=1024,
        max_seq_length=512,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        unk_token_id=tok.unk_token_id,
    )
    
    model = SentriFormerForConditionalGeneration(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    loader = create_pretrain_dataloader(
        parquet_path=train_parquet,
        tokenizer=tok,
        batch_size=16,
        max_seq_length=256,
        limit=5000,
        shuffle=True
    )
    
    model.train()
    step = 0
    losses = []
    
    t0 = time.time()
    for batch in loader:
        optimizer.zero_grad()
        loss = run_training_step(model, batch, scaler, device, use_amp=True)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        losses.append(loss.item())
        step += 1
        
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:3d}/100 | Loss: {loss.item():.4f} | Avg (last 10): {sum(losses[-10:])/len(losses[-10:]):.4f}")
            
        if step >= 100:
            break
            
    elapsed = time.time() - t0
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    
    print(f"\n  - 100 Steps Completed in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")
    print(f"  - Initial Span Corruption Loss: {initial_loss:.4f}")
    print(f"  - Final Span Corruption Loss:   {final_loss:.4f} (Drop: {initial_loss - final_loss:.4f})")
    assert final_loss < initial_loss, "Pretraining loss did not decrease!"
    print("  => M8.3 Tiny Pretraining Test: PASSED")
    return model


def benchmark_gpu_throughput(device: torch.device):
    print("\n" + "=" * 70)
    print(" [M8.4] GPU THROUGHPUT & MEMORY BENCHMARK")
    print("=" * 70)
    
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    train_parquet = os.path.join(ROOT_DIR, "sentriformer", "data", "train.parquet")
    
    config = SentriFormerConfig(
        vocab_size=len(tok),
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        attention_heads=8,
        head_dim=32,
        d_ff=1024,
        max_seq_length=512,
    )
    
    model = SentriFormerForConditionalGeneration(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    results = {}
    batch_candidates = [16, 32, 64] if device.type == "cuda" else [16]
    
    for bs in batch_candidates:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        loader = create_pretrain_dataloader(
            parquet_path=train_parquet,
            tokenizer=tok,
            batch_size=bs,
            max_seq_length=256,
            limit=10000,
            shuffle=False
        )
        
        # Warmup 5 batches
        it = iter(loader)
        for _ in range(5):
            batch = next(it)
            optimizer.zero_grad()
            loss = run_training_step(model, batch, scaler, device, use_amp=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
        # Benchmark 30 batches
        total_samples = 0
        total_tokens = 0
        t0 = time.time()
        
        for _ in range(30):
            batch = next(it)
            optimizer.zero_grad()
            loss = run_training_step(model, batch, scaler, device, use_amp=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            b_size = batch["input_ids"].shape[0]
            in_toks = batch["attention_mask"].sum().item()
            out_toks = (batch["labels"] != -100).sum().item()
            
            total_samples += b_size
            total_tokens += (in_toks + out_toks)
            
        dur = time.time() - t0
        samples_per_sec = total_samples / dur
        tokens_per_sec = total_tokens / dur
        
        peak_vram_mb = 0
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        results[bs] = {
            "batch_size": bs,
            "samples_per_sec": round(samples_per_sec, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "seconds_per_100k_samples": round(100_000 / samples_per_sec, 1)
        }
        
        print(f"  Batch Size {bs:2d} -> {samples_per_sec:6.1f} samples/s | {tokens_per_sec:7.1f} tokens/s | Peak VRAM: {peak_vram_mb:6.1f} MB")
        
    return results


def calculate_full_training_time(throughput_results: Dict[int, Any]):
    print("\n" + "=" * 70)
    print(" [M8.5] ESTIMATED FULL PRETRAINING TIME & COMPUTE BUDGET")
    print("=" * 70)
    
    total_records = 3_184_956  # Frozen train split
    est_total_tokens = 750_000_000
    
    # Select best batch size
    best_bs = max(throughput_results.keys(), key=lambda b: throughput_results[b]["samples_per_sec"])
    best_rate = throughput_results[best_bs]["samples_per_sec"]
    tok_rate = throughput_results[best_bs]["tokens_per_sec"]
    
    sec_1_epoch = total_records / best_rate
    hours_1_epoch = sec_1_epoch / 3600
    
    sec_3_epochs = sec_1_epoch * 3
    hours_3_epochs = sec_3_epochs / 3600
    
    print(f"  Total Frozen Training Examples: {total_records:,}")
    print(f"  Estimated Token Budget:         {est_total_tokens:,} tokens")
    print(f"  Optimal Batch Size:             {best_bs}")
    print(f"  Measured GPU Throughput:        {best_rate:.1f} samples/sec ({tok_rate:.1f} tokens/sec)")
    print()
    print(f"  --- Time Estimates ---")
    print(f"  1 Full Epoch (3.18M examples):  {hours_1_epoch:.2f} hours  ({sec_1_epoch/60:.1f} minutes)")
    print(f"  3 Full Epochs (9.55M examples): {hours_3_epochs:.2f} hours")
    print()
    print(f"  Recommended Full Training Strategy:")
    print(f"  - 1 Epoch (~{hours_1_epoch:.1f}h) gives complete domain representation over all 3.18M complaints & dialogues.")
    print("=" * 70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    test_tiny_pretrain_convergence(device)
    throughput_results = benchmark_gpu_throughput(device)
    calculate_full_training_time(throughput_results)


if __name__ == "__main__":
    main()
