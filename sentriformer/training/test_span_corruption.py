"""
Verification & Test Suite for T5 Span Corruption (M8.2)
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from typing import List
import torch
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.training.span_corruption import SpanCorruptor, corrupt_tokens_t5
from sentriformer.training.pretrain_dataset import PretrainDataCollator


def test_span_corruption_logic():
    print("\n[1/3] Testing T5 Span Corruption Algorithm on Real Complaints...")
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    sample_text = (
        "I was charged an unauthorized overdraft fee of $35.00 on my checking account "
        "even though I had sufficient funds deposited yesterday. I called customer service "
        "and the agent refused to refund the charges."
    )
    
    tokens = tok.encode(sample_text, add_special_tokens=False)
    print(f"  - Original text ({len(tokens)} tokens): '{sample_text}'")
    
    corruptor = SpanCorruptor(
        noise_density=0.15,
        mean_noise_span_length=3.0,
        extra_ids_start=tok.convert_tokens_to_ids("<extra_id_0>"),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id
    )
    
    corrupted_ids, target_ids = corruptor.corrupt(tokens)
    
    corrupted_decoded = tok.decode(corrupted_ids)
    target_decoded = tok.decode(target_ids)
    
    print(f"\n  - Corrupted Encoder Input ({len(corrupted_ids)} tokens):")
    print(f"    '{corrupted_decoded}'")
    print(f"\n  - Decoder Target Labels ({len(target_ids)} tokens):")
    print(f"    '{target_decoded}'")
    
    # Assertions
    assert "<extra_id_0>" in corrupted_decoded, "Missing <extra_id_0> in corrupted encoder input!"
    assert "<extra_id_0>" in target_decoded, "Missing <extra_id_0> in decoder target labels!"
    assert target_ids[-1] == tok.eos_token_id, "Target labels must terminate with EOS token </s>!"
    assert len(corrupted_ids) <= len(tokens) + 2, "Corrupted length should be shorter or comparable to original!"
    print("\n  => Span Corruption Logic: PASSED")


def test_sentinel_order_and_reconstruction():
    print("\n[2/3] Verifying Sentinel Alignment & Span Integrity...")
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    extra_0 = tok.convert_tokens_to_ids("<extra_id_0>")
    
    # Run 100 stochastic sequences to verify sentinel monotonicity
    for test_idx in range(100):
        length = torch.randint(20, 100, (1,)).item()
        dummy_tokens = list(range(1000, 1000 + length))
        
        corr_ids, tgt_ids = corrupt_tokens_t5(
            dummy_tokens,
            noise_density=0.15,
            mean_noise_span_length=3.0,
            extra_ids_start=extra_0
        )
        
        # Extract sentinels from corrupted input and targets
        corr_sentinels = [t for t in corr_ids if extra_0 <= t < extra_0 + 100]
        tgt_sentinels = [t for t in tgt_ids if extra_0 <= t < extra_0 + 100]
        
        # Sentinels in input and target must match exactly in order
        assert corr_sentinels == tgt_sentinels, f"Sentinel mismatch: {corr_sentinels} vs {tgt_sentinels}"
        # Sentinels must be strictly increasing: 0, 1, 2, ...
        expected = [extra_0 + i for i in range(len(corr_sentinels))]
        assert corr_sentinels == expected, f"Sentinels not strictly sequential: {corr_sentinels}"
        
    print("  - Tested 100 randomized sequences: All sentinels strictly aligned & sequential.")
    print("  => Sentinel Order & Alignment: PASSED")


def test_collator_and_batch_shapes():
    print("\n[3/3] Testing PretrainDataCollator with Dynamic Batch Padding...")
    tok_path = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    collator = PretrainDataCollator(tok, max_seq_length=512)
    
    # Create variable-length batch
    batch_raw = [
        tok.encode("My credit card was charged twice.", add_special_tokens=False),
        tok.encode("I called support and waited for 45 minutes on hold without any response from an agent.", add_special_tokens=False),
        tok.encode("Refund not received.", add_special_tokens=False),
        tok.encode("Fraud on my account reported last Monday. Please resolve immediately.", add_special_tokens=False),
    ]
    
    batch = collator(batch_raw)
    
    print(f"  - Batch size: {len(batch_raw)}")
    print(f"  - input_ids shape:      {batch['input_ids'].shape} (dtype={batch['input_ids'].dtype})")
    print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  - labels shape:         {batch['labels'].shape}")
    
    assert batch['input_ids'].shape == batch['attention_mask'].shape
    assert (batch['labels'] == -100).any(), "Labels must contain -100 ignore_index for padded tokens!"
    print("  => Collator & Batch Shapes: PASSED")


def main():
    print("=" * 70)
    print("      T5 SPAN CORRUPTION PIPELINE VERIFICATION SUITE (M8.2)")
    print("=" * 70)
    
    test_span_corruption_logic()
    test_sentinel_order_and_reconstruction()
    test_collator_and_batch_shapes()
    
    print("\n" + "=" * 70)
    print("ALL M8.2 SPAN CORRUPTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
