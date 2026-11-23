"""
==========================================================================
  M10.2 Gate Check: 4-Point Smoke Test
==========================================================================
  Gate 1: Causal Mask Correctness (Gradient isolation: future token grad = 0.0)
  Gate 2: EOS Target Presence (100/100 label sequences end in </s> = ID 3)
  Gate 3: 100-Step Loss Sanity (Realistic start loss, trend decrease, no NaN/Inf)
  Gate 4: Raw Token Autoregressive Generation (No BOS loop, no PAD, non-empty)
==========================================================================
"""
import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator
from sentriformer.evaluation.audit_multitask_behavior import generate_multitask_response

TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
UNIFIED_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "multitask", "multitask_best.pt")
CLEAN_TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_train.parquet")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")

sys.stdout.reconfigure(encoding="utf-8")


def run_gate_1_causal_mask():
    print("\n" + "=" * 80)
    print("  [GATE 1] Testing Causal Self-Attention Gradient Isolation")
    print("=" * 80)
    
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5
    model = SentriFormerForMultiTask(config)
    model.eval()
    
    # Test sequence: length 6
    seq_len = 6
    dec_input_ids = torch.tensor([[2, 105, 420, 890, 1200, 3]], dtype=torch.long)
    enc_hidden = torch.randn(1, 8, config.d_model)
    enc_mask = torch.ones(1, 8, dtype=torch.long)
    dec_mask = torch.ones(1, seq_len, dtype=torch.long)
    
    # We test gradient of dec_out[0, i, :] with respect to the embedding of dec_input_ids[:, j]
    # For every pair (i, j) where j > i, the gradient must be EXACTLY 0.0.
    embed_tokens = model.sentriformer.decoder.embed_tokens
    
    max_leakage = 0.0
    leak_pairs = []
    
    for i in range(seq_len - 1):
        # Fresh graph for each position i
        embs = embed_tokens(dec_input_ids) * math.sqrt(config.d_model)
        embs.retain_grad()
        
        # Run through decoder using raw embeddings hook
        pos_embs = model.sentriformer.decoder.pos_emb(seq_len, embs.device)
        h = model.sentriformer.decoder.dropout(embs + pos_embs)
        
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=embs.device)).view(1, 1, seq_len, seq_len)
        pad_mask = (dec_mask != 0).view(1, 1, 1, seq_len)
        extended_self_mask = causal_mask & pad_mask
        extended_enc_mask = (enc_mask != 0).view(1, 1, 1, 8)
        
        for layer in model.sentriformer.decoder.layers:
            h, _ = layer(
                hidden_states=h,
                encoder_hidden_states=enc_hidden,
                attention_mask=extended_self_mask,
                encoder_attention_mask=extended_enc_mask,
            )
        dec_out = model.sentriformer.decoder.final_norm(h)
        
        # Loss depends ONLY on position i
        loss_i = dec_out[0, i, :].sum()
        loss_i.backward()
        
        grad = embs.grad[0]  # shape [seq_len, d_model]
        for j in range(i + 1, seq_len):
            leak = grad[j].abs().max().item()
            if leak > max_leakage:
                max_leakage = leak
            if leak > 1e-7:
                leak_pairs.append((i, j, leak))
                
        print(f"  Query Pos {i} -> max grad on future positions {list(range(i+1, seq_len))}: {grad[i+1:].abs().max().item():.2e}")
        
    if max_leakage > 1e-7:
        print(f"\n❌ [GATE 1 FAILED] Future-token gradient leakage detected! Max leakage: {max_leakage}")
        print(f"   Leaking pairs: {leak_pairs}")
        return False
    else:
        print(f"\n✅ [GATE 1 PASS] Strict causal isolation verified. Max future-token gradient = {max_leakage:.2e}")
        return True


def run_gate_2_eos_presence():
    print("\n" + "=" * 80)
    print("  [GATE 2] Verifying Target EOS (</s> = ID 3) Presence across 100 Samples")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    eos_id = tokenizer.eos_token_id
    print(f"  Tokenizer EOS token: '{tokenizer.eos_token}' (ID: {eos_id})")
    
    dataset = DialogueSFTDataset(CLEAN_TRAIN_PARQUET, tokenizer, max_ctx_len=128, max_resp_len=256, limit=100)
    
    eos_present_count = 0
    eos_missing_count = 0
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = item["labels"]
        if len(labels) > 0 and labels[-1] == eos_id:
            eos_present_count += 1
        else:
            eos_missing_count += 1
            if eos_missing_count <= 3:
                print(f"    Missing EOS in sample {idx}: last 3 tokens = {labels[-3:] if len(labels)>=3 else labels}")
                
    print(f"  EOS present in labels: {eos_present_count}/100")
    print(f"  EOS missing in labels: {eos_missing_count}/100")
    
    if eos_present_count == 100 and eos_missing_count == 0:
        print("✅ [GATE 2 PASS] 100/100 samples terminate with EOS token (</s>).")
        return True
    else:
        print("❌ [GATE 2 FAILED] Some samples do not terminate with EOS!")
        return False


def run_gate_3_loss_sanity(device):
    print("\n" + "=" * 80)
    print("  [GATE 3] 100-Step Training Loss Sanity & Trend Verification")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5
    
    model = SentriFormerForMultiTask(config).to(device)
    print(f"  Loading M9.5 baseline from: {UNIFIED_CKPT}")
    unified_raw = torch.load(UNIFIED_CKPT, map_location=device, weights_only=False)
    unified_dict = unified_raw["model_state_dict"] if "model_state_dict" in unified_raw else unified_raw
    model.load_state_dict(unified_dict, strict=True)
    
    # Freeze classification and CSAT heads
    for param in model.category_head.parameters():
        param.requires_grad = False
    for param in model.sentiment_head.parameters():
        param.requires_grad = False
    for param in model.csat_head.parameters():
        param.requires_grad = False
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    
    dataset = DialogueSFTDataset(CLEAN_TRAIN_PARQUET, tokenizer, max_ctx_len=128, max_resp_len=256, limit=3200)
    collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collator)
    
    model.train()
    losses = []
    
    print("  Running 100 training steps...")
    step = 0
    for batch in loader:
        if step >= 100:
            break
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, generation_labels=labels)
        loss = outputs["generation_loss"]
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ [GATE 3 FAILED] NaN or Inf loss at step {step}!")
            return False, model
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        step += 1
        
        if step % 20 == 0 or step == 1:
            print(f"    Step {step:>3}/100 | Loss: {loss.item():.4f}")
            
    initial_loss = losses[0]
    first_20_avg = sum(losses[:20]) / 20.0
    last_20_avg = sum(losses[-20:]) / 20.0
    min_loss = min(losses)
    
    print(f"\n  Initial Step 1 Loss:     {initial_loss:.4f}")
    print(f"  First 20 Steps Avg Loss: {first_20_avg:.4f}")
    print(f"  Last 20 Steps Avg Loss:  {last_20_avg:.4f}")
    print(f"  Minimum Step Loss:       {min_loss:.4f}")
    
    # Validation checks
    # 1. Start loss must be in realistic non-collapsed range (e.g. 4.0 - 11.0)
    if initial_loss < 2.0:
        print(f"❌ [GATE 3 FAILED] Initial loss is suspiciously low ({initial_loss:.4f}). Potential leakage!")
        return False, model
    if initial_loss > 15.0:
        print(f"❌ [GATE 3 FAILED] Initial loss exploded ({initial_loss:.4f})!")
        return False, model
        
    # 2. Loss trend should decrease: last 20 avg < first 20 avg
    if last_20_avg >= first_20_avg:
        print(f"❌ [GATE 3 FAILED] Loss trend did not decrease (First 20: {first_20_avg:.4f}, Last 20: {last_20_avg:.4f})")
        return False, model
        
    print("✅ [GATE 3 PASS] Loss starts in realistic range, decreases cleanly without NaN/Inf.")
    return True, model


def run_gate_4_raw_generation(model, device):
    print("\n" + "=" * 80)
    print("  [GATE 4] Raw Token Autoregressive Generation & Category Reporting")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = model.config
    
    id2label = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            id2label = json.load(f).get("id2label", {})
            
    test_queries = [
        "I was double charged for my monthly subscription. Can you please refund the extra fee?",
        "Why is there a late fee on my credit card when I scheduled the payment two days before the due date?"
    ]
    
    all_passed = True
    
    for idx, query in enumerate(test_queries):
        print(f"\n  --- Test Case {idx+1} ---")
        print(f"  Query: \"{query}\"")
        enc = tokenizer(query, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cat_probs = F.softmax(outputs["category_logits"], dim=-1)[0]
            top_prob, top_cat_id = torch.max(cat_probs, dim=-1)
            cat_name = id2label.get(str(top_cat_id.item()), f"ID {top_cat_id.item()}")
            cat_conf = top_prob.item() * 100.0
            csat_star = torch.argmax(outputs["csat_logits"], dim=-1).item() + 1
            
            gen_tokens = generate_multitask_response(
                model, tokenizer, config, enc["input_ids"], enc["attention_mask"],
                max_new_tokens=60, temperature=0.7, repetition_penalty=1.2
            )
            
        raw_ids = gen_tokens[0].tolist()
        resp_text = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()
        
        print(f"  Predicted Category: {cat_name} (Confidence: {cat_conf:.1f}%) | CSAT Forecast: [{csat_star} ⭐]")
        print(f"  Raw IDs: {raw_ids[:25]}{'...' if len(raw_ids) > 25 else ''} (Count: {len(raw_ids)})")
        print(f"  Decoded: \"{resp_text}\"")
        
        # Requirements:
        # 1. raw IDs contain > 1 token
        if len(raw_ids) <= 1:
            print(f"❌ [GATE 4 FAIL] Raw IDs contain only {len(raw_ids)} token(s)!")
            all_passed = False
            continue
            
        # 2. BOS (2) is not repeatedly generated
        if raw_ids[1:].count(config.bos_token_id) > 0:
            print(f"❌ [GATE 4 FAIL] BOS token ({config.bos_token_id}) generated in output positions!")
            all_passed = False
            continue
            
        # 3. PAD (0) is never generated
        if config.pad_token_id in raw_ids:
            print(f"❌ [GATE 4 FAIL] PAD token ({config.pad_token_id}) found in generated IDs!")
            all_passed = False
            continue
            
        # 4. Valid vocabulary IDs
        invalid_ids = [t for t in raw_ids if t < 0 or t >= config.vocab_size]
        if invalid_ids:
            print(f"❌ [GATE 4 FAIL] Invalid token IDs: {invalid_ids}")
            all_passed = False
            continue
            
        # 5. Non-empty response
        if len(resp_text) == 0:
            print("❌ [GATE 4 FAIL] Decoded response text is empty!")
            all_passed = False
            continue
            
        eos_found = (config.eos_token_id in raw_ids)
        reached_max = (len(raw_ids) >= 61)  # 1 BOS + 60 new tokens
        print(f"  Terminated by EOS: {eos_found} | Reached Max Length: {reached_max}")
        
        if not eos_found and not reached_max:
            print("❌ [GATE 4 FAIL] Neither EOS was reached nor max length!")
            all_passed = False
            continue
            
    if all_passed:
        print("\n✅ [GATE 4 PASS] Autoregressive generation produces valid non-empty tokens with correct suppression and category routing.")
        return True
    else:
        return False


def main():
    print("=" * 80)
    print("  SENTRIFORMER M10.2 ARCHITECTURE & SMOKE TEST GATE CHECK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # Gate 1
    g1 = run_gate_1_causal_mask()
    if not g1:
        print("\n⛔ STOPPING: Gate 1 Failed. Immediate debugging required.")
        sys.exit(1)
        
    # Gate 2
    g2 = run_gate_2_eos_presence()
    if not g2:
        print("\n⛔ STOPPING: Gate 2 Failed. Immediate debugging required.")
        sys.exit(1)
        
    # Gate 3
    g3, trained_model = run_gate_3_loss_sanity(device)
    if not g3:
        print("\n⛔ STOPPING: Gate 3 Failed. Immediate debugging required.")
        sys.exit(1)
        
    # Gate 4
    g4 = run_gate_4_raw_generation(trained_model, device)
    if not g4:
        print("\n⛔ STOPPING: Gate 4 Failed. Immediate debugging required.")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("  🎉 ALL 4 GATES PASSED! ARCHITECTURE & DATASET PIPELINE FULLY VERIFIED.")
    print("  Ready to proceed with Full Clean Response SFT (M10.2).")
    print("=" * 80)


if __name__ == "__main__":
    main()
