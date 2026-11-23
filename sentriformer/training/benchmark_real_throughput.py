"""
[M8.5.1] Rigorous Real-Token Throughput Benchmark & Compute Budget Analysis
Accurately measures active Encoder + Decoder tokens processed per second
across representative multi-length batches from train.parquet.
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration
from sentriformer.training.train_pretrain import create_pretrain_dataloader, run_training_step


def benchmark_real_workload(device: torch.device):
    print("=" * 75)
    print(" [M8.5.1] RIGOROUS REAL-TOKEN THROUGHPUT BENCHMARK (NVIDIA RTX 3060)")
    print("=" * 75)
    
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
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    
    # Test batch sizes: 32, 64 with full context (max_seq_length=512)
    batch_sizes = [32, 64] if device.type == "cuda" else [16]
    benchmark_results = {}
    
    for bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        print(f"\nEvaluating Real Pretraining Workload @ Batch Size = {bs}, Max Context = 512...")
        
        loader = create_pretrain_dataloader(
            parquet_path=train_parquet,
            tokenizer=tok,
            batch_size=bs,
            max_seq_length=512,
            limit=20000,
            shuffle=True
        )
        
        # 1. Warmup 10 steps to compile CUDA graphs and initialize PyTorch memory pools
        it = iter(loader)
        for _ in range(10):
            batch = next(it)
            optimizer.zero_grad()
            loss = run_training_step(model, batch, scaler, device, use_amp=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
        # 2. Benchmark 100 actual training steps with exact token accounting
        total_encoder_tokens = 0
        total_decoder_tokens = 0
        total_samples = 0
        total_steps = 100
        losses = []
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for step in range(total_steps):
            batch = next(it)
            optimizer.zero_grad()
            loss = run_training_step(model, batch, scaler, device, use_amp=True)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Exact active (non-padded) token accounting
            enc_active = batch["attention_mask"].sum().item()
            dec_active = (batch["labels"] != -100).sum().item()
            
            total_encoder_tokens += enc_active
            total_decoder_tokens += dec_active
            total_samples += batch["input_ids"].shape[0]
            losses.append(loss.item())
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        total_active_tokens = total_encoder_tokens + total_decoder_tokens
        steps_per_sec = total_steps / elapsed
        samples_per_sec = total_samples / elapsed
        enc_tok_per_sec = total_encoder_tokens / elapsed
        dec_tok_per_sec = total_decoder_tokens / elapsed
        total_tok_per_sec = total_active_tokens / elapsed
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0
        
        avg_enc_len = total_encoder_tokens / total_samples
        avg_dec_len = total_decoder_tokens / total_samples
        
        benchmark_results[bs] = {
            "batch_size": bs,
            "steps_per_sec": round(steps_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 2),
            "avg_encoder_seq_len": round(avg_enc_len, 1),
            "avg_decoder_seq_len": round(avg_dec_len, 1),
            "encoder_tokens_per_sec": round(enc_tok_per_sec, 1),
            "decoder_tokens_per_sec": round(dec_tok_per_sec, 1),
            "total_tokens_per_sec": round(total_tok_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "loss_start_100": round(losses[0], 4),
            "loss_end_100": round(losses[-1], 4),
        }
        
        print(f"  Throughput Metrics (Batch {bs}):")
        print(f"    - Training Steps/Sec:     {steps_per_sec:6.2f} step/s ({1000/steps_per_sec:5.1f} ms/step)")
        print(f"    - Sample Throughput:      {samples_per_sec:6.2f} samples/s")
        print(f"    - Avg Active Sequence:    Encoder={avg_enc_len:.1f} tok, Decoder={avg_dec_len:.1f} tok (Total {avg_enc_len+avg_dec_len:.1f} tok/sample)")
        print(f"    - Encoder Processing:     {enc_tok_per_sec:8.1f} tokens/sec")
        print(f"    - Decoder Processing:     {dec_tok_per_sec:8.1f} tokens/sec")
        print(f"    - TOTAL TOKEN THROUGHPUT: {total_tok_per_sec:8.1f} ACTIVE TOKENS/SEC")
        print(f"    - Peak VRAM Allocated:    {peak_vram_mb:6.1f} MB ({(6144-peak_vram_mb)/1024:.2f} GB Free Headroom)")

    return benchmark_results


def print_rigorous_duration_table(benchmark_results):
    print("\n" + "=" * 75)
    print("                    EMPIRICAL PRETRAINING RUNTIME AUDIT")
    print("=" * 75)
    
    total_records = 3_184_956
    est_corpus_tokens = 750_000_000
    
    # Use batch size 64 metrics
    bs = 64 if 64 in benchmark_results else list(benchmark_results.keys())[0]
    res = benchmark_results[bs]
    
    tok_rate = res["total_tokens_per_sec"]
    sample_rate = res["samples_per_sec"]
    
    # 1. Token-Based Calculation
    sec_750m_tokens = est_corpus_tokens / tok_rate
    hours_750m_tokens = sec_750m_tokens / 3600
    
    # 2. Sample-Based Calculation
    sec_full_epoch = total_records / sample_rate
    hours_full_epoch = sec_full_epoch / 3600
    
    print(f"Active Hardware:             NVIDIA GeForce RTX 3060 (Laptop)")
    print(f"Optimal Batch Size:          {bs} (Peak VRAM: {res['peak_vram_mb']:.1f} MB)")
    print(f"Real Active Token Rate:      {tok_rate:,.1f} tokens/second")
    print(f"Real Record Processing Rate: {sample_rate:,.1f} samples/second")
    print("-" * 75)
    print(f"{'Target Workload / Checkpoint':<35} | {'Tokens / Records':<20} | {'Exact Expected Time'}")
    print("-" * 75)
    
    checkpoints = [
        ("10 Million Token Checkpoint", 10_000_000, 10_000_000 / tok_rate),
        ("50 Million Token Checkpoint", 50_000_000, 50_000_000 / tok_rate),
        ("100 Million Token Checkpoint", 100_000_000, 100_000_000 / tok_rate),
        ("250 Million Token Checkpoint", 250_000_000, 250_000_000 / tok_rate),
        ("500 Million Token Checkpoint", 500_000_000, 500_000_000 / tok_rate),
        ("1 Full Epoch (3.18M Records)", total_records, sec_full_epoch),
        ("3 Full Epochs (9.55M Records)", total_records * 3, sec_full_epoch * 3),
    ]
    
    for name, count, seconds in checkpoints:
        if seconds < 3600:
            time_str = f"{seconds/60:5.1f} minutes"
        else:
            time_str = f"{seconds/3600:5.2f} hours ({seconds/60:.0f} mins)"
        print(f"  {name:<33} | {count:>18,} | {time_str}")
        
    print("=" * 75)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = benchmark_real_workload(device)
    print_rigorous_duration_table(results)


if __name__ == "__main__":
    main()
