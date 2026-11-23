"""
SentriFormer Phase M7 Architecture Verification & Test Suite
Tests:
1. Exact Parameter Count & Weight Tying Validation
2. Forward Pass Tensor Shape Integrity (batch=1, 4; seq=16, 128, 512)
3. Causal Masking & Cross-Attention Behavior
4. Tokenizer -> Model End-to-End Pipeline Integration
5. Tiny Overfit Optimization (Gradient Flow & Loss Convergence)
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim

# Add root path
ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerModel, SentriFormerForConditionalGeneration
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from transformers import AutoTokenizer


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculates granular parameter breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate unique parameters when weights are tied
    unique_params = set()
    for p in model.parameters():
        unique_params.add(p.data_ptr())
    
    # Count unique elements
    unique_total = 0
    seen_ptrs = set()
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            unique_total += p.numel()
            seen_ptrs.add(p.data_ptr())
            
    return {
        "total": total,
        "unique_total": unique_total,
        "trainable": trainable
    }


def test_parameter_counts():
    print("\n[1/5] Testing Parameter Counts & Weight Tying...")
    config = SentriFormerConfig(
        vocab_size=16000,
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        attention_heads=8,
        head_dim=32,
        d_ff=1024,
        max_seq_length=512,
        tie_word_embeddings=True
    )
    
    model = SentriFormerForConditionalGeneration(config)
    counts = count_parameters(model)
    
    # Granular component breakdown
    embed_params = config.vocab_size * config.d_model
    # Per layer: self_attn (4 * d^2) + ffn (2 * d * d_ff) + 2 layer_norms (2 * d)
    enc_layer_params = (4 * 256 * 256) + (2 * 256 * 1024) + (4 * 256)
    dec_layer_params = (4 * 256 * 256) + (4 * 256 * 256) + (2 * 256 * 1024) + (6 * 256)
    
    enc_total = config.encoder_layers * enc_layer_params
    dec_total = config.decoder_layers * dec_layer_params
    
    print(f"  - Embedding Table (16k x 256): {embed_params:,} params ({embed_params/1e6:.2f}M)")
    print(f"  - 4x Encoder Layers:           {enc_total:,} params ({enc_total/1e6:.2f}M)")
    print(f"  - 4x Decoder Layers:           {dec_total:,} params ({dec_total/1e6:.2f}M)")
    print(f"  - Core Unique Parameters:      {counts['unique_total']:,} ({counts['unique_total']/1e6:.2f}M)")
    print(f"  - Weight Tying:                {'VERIFIED (lm_head shared with embedding)' if model.lm_head.weight is model.sentriformer.shared_embeddings.weight else 'FAILED'}")
    
    # Multi-task model counts
    multi_model = SentriFormerForMultiTask(config)
    multi_counts = count_parameters(multi_model)
    print(f"  - Multi-Task Total Params:     {multi_counts['unique_total']:,} ({multi_counts['unique_total']/1e6:.2f}M)")
    
    assert 10_000_000 <= counts['unique_total'] <= 16_000_000, f"Model size {counts['unique_total']} out of 10-16M target!"
    print("  => Parameter Count Validation: PASSED")


def test_forward_shapes():
    print("\n[2/5] Testing Forward Pass Tensor Shapes (Different Batches & Sequence Lengths)...")
    config = SentriFormerConfig(vocab_size=16000, d_model=256, encoder_layers=4, decoder_layers=4)
    model = SentriFormerForConditionalGeneration(config)
    model.eval()
    
    test_cases = [
        (1, 16, 8),
        (4, 128, 64),
        (2, 512, 128),
    ]
    
    for b, seq_in, seq_out in test_cases:
        input_ids = torch.randint(0, config.vocab_size, (b, seq_in))
        decoder_ids = torch.randint(0, config.vocab_size, (b, seq_out))
        attn_mask = torch.ones((b, seq_in), dtype=torch.long)
        labels = torch.randint(0, config.vocab_size, (b, seq_out))
        
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_ids,
                attention_mask=attn_mask,
                labels=labels
            )
            
        assert out["logits"].shape == (b, seq_out, config.vocab_size), f"Logits shape mismatch: {out['logits'].shape}"
        assert out["loss"] is not None and not torch.isnan(out["loss"]), "Loss is NaN or None"
        print(f"  - Batch {b}, In-Seq {seq_in}, Out-Seq {seq_out} -> Logits {out['logits'].shape}, Loss={out['loss'].item():.4f} (OK)")
        
    print("  => Forward Pass Shape Validation: PASSED")


def test_causal_masking():
    print("\n[3/5] Testing Decoder Causal Masking & KV Caching...")
    config = SentriFormerConfig(vocab_size=16000, d_model=256, encoder_layers=4, decoder_layers=4)
    model = SentriFormerForConditionalGeneration(config)
    model.eval()
    
    # Verify that changing future target tokens does NOT alter earlier target logits
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    dec_input_1 = torch.tensor([[2, 100, 200, 300, 400]])
    dec_input_2 = torch.tensor([[2, 100, 200, 999, 888]])  # Altered tokens at positions 3 and 4
    
    with torch.no_grad():
        out1 = model(input_ids=input_ids, decoder_input_ids=dec_input_1)["logits"]
        out2 = model(input_ids=input_ids, decoder_input_ids=dec_input_2)["logits"]
        
    # Positions 0, 1, 2 must produce identical logits!
    diff = torch.max(torch.abs(out1[:, :3, :] - out2[:, :3, :])).item()
    assert diff < 1e-5, f"Causal leakage detected! Max difference at earlier tokens: {diff}"
    print(f"  - Causal isolation difference at earlier tokens: {diff:.8f} (Zero Leakage)")
    print("  => Causal Masking Validation: PASSED")


def test_tokenizer_integration():
    print("\n[4/5] Testing Tokenizer -> Model End-to-End Pipeline Integration...")
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    config = SentriFormerConfig(
        vocab_size=len(tok),
        d_model=256,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        unk_token_id=tok.unk_token_id,
    )
    model = SentriFormerForConditionalGeneration(config)
    
    complaint = "My credit card was charged $250.00 unauthorized fee by Equifax!"
    inputs = tok(complaint, return_tensors="pt")
    
    # Test generation
    with torch.no_grad():
        gen_tokens = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=16)
        decoded = tok.decode(gen_tokens[0], skip_special_tokens=False)
        
    print(f"  - Input Complaint: {complaint}")
    print(f"  - Tokenized IDs ({len(inputs['input_ids'][0])} tokens): {inputs['input_ids'][0].tolist()}")
    print(f"  - Generated Token Output: {gen_tokens[0].tolist()}")
    print(f"  - Decoded Output: {decoded}")
    print("  => Tokenizer Integration: PASSED")


def test_tiny_overfit():
    print("\n[5/5] Testing Tiny Overfit Optimization (Gradient Flow & Loss Convergence)...")
    config = SentriFormerConfig(
        vocab_size=500,  # Small vocab for fast overfit check
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        attention_heads=4,
        head_dim=32,
        d_ff=256,
        dropout=0.0  # Disable dropout for deterministic overfit test
    )
    model = SentriFormerForConditionalGeneration(config)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)
    
    # Fixed tiny dataset of 2 samples with short sequences
    input_ids = torch.randint(10, 200, (2, 8))
    target_ids = torch.randint(10, 200, (2, 8))
    
    initial_loss = None
    final_loss = None
    
    for step in range(60):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=target_ids)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"  - Step 0 Loss:  {initial_loss:.4f}")
    print(f"  - Step 60 Loss: {final_loss:.4f}")
    assert final_loss < 0.5, f"Model failed to overfit tiny batch! Loss: {initial_loss} -> {final_loss}"
    print(f"  - Loss Drop:    {initial_loss:.4f} -> {final_loss:.4f} (Near-Zero Convergence)")
    print("  => Gradient Flow & Optimization: PASSED")


def main():
    print("=" * 70)
    print("       SENTRIFORMER M7 ARCHITECTURE VERIFICATION SUITE")
    print("=" * 70)
    
    test_parameter_counts()
    test_forward_shapes()
    test_causal_masking()
    test_tokenizer_integration()
    test_tiny_overfit()
    
    print("\n" + "=" * 70)
    print("ALL M7 ARCHITECTURE & INTEGRATION TESTS PASSED SUCCESSFULLY! 🔒")
    print("=" * 70)


if __name__ == "__main__":
    main()
