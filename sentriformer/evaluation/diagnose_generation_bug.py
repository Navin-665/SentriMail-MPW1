"""
==========================================================================
  M10.2 Diagnostic Script: Generation, Tokenizer, Labels & Category Audit
==========================================================================
"""
import os
import sys
import json
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator

TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
CLEAN_TRAIN_PARQUET = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_train.parquet")
CLEAN_SFT_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "clean_sft", "clean_sft_best.pt")
MULTITASK_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "multitask", "multitask_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def diagnose():
    print("=" * 80)
    print("  [DIAGNOSTIC] M10.2 GENERATION & CATEGORY PIPELINE AUDIT")
    print("=" * 80)
    print(f"Device: {device}")
    
    # 1. Inspect Tokenizer Special Tokens
    print("\n--- 1. TOKENIZER & CONFIG SPECIAL TOKENS ---")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5
    
    print(f"Config:    PAD={config.pad_token_id}, BOS={config.bos_token_id}, EOS={config.eos_token_id}, UNK={config.unk_token_id}")
    print(f"Tokenizer: PAD={tokenizer.pad_token_id} ('{tokenizer.pad_token}'), "
          f"BOS={tokenizer.bos_token_id} ('{tokenizer.bos_token}'), "
          f"EOS={tokenizer.eos_token_id} ('{tokenizer.eos_token}'), "
          f"UNK={tokenizer.unk_token_id} ('{tokenizer.unk_token}')")
    print(f"Vocab size: Config={config.vocab_size}, Tokenizer={len(tokenizer)}")
    
    # 2. Inspect Dataset and Collator for One Example
    print("\n--- 2. DATASET SAMPLE & COLLATOR LABEL MASKING ---")
    ds = DialogueSFTDataset(CLEAN_TRAIN_PARQUET, tokenizer, max_ctx_len=128, max_resp_len=256, limit=5)
    sample0 = ds[0]
    print(f"Sample 0 raw context:  \"{ds.contexts[0][:80]}...\"")
    print(f"Sample 0 raw response: \"{ds.responses[0][:80]}...\"")
    print(f"Sample 0 input_ids ({len(sample0['input_ids'])} tokens): {sample0['input_ids'][:15]}...")
    print(f"Sample 0 labels    ({len(sample0['labels'])} tokens): {sample0['labels'][:15]}...")
    print(f"Decoded input_ids: \"{tokenizer.decode(sample0['input_ids'])}\"")
    print(f"Decoded labels:    \"{tokenizer.decode(sample0['labels'])}\"")
    
    collator = DialogueSFTCollator(pad_token_id=config.pad_token_id)
    batch = collator([ds[0], ds[1]])
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape:    {batch['labels'].shape}")
    print(f"Batch labels row 0 first 20: {batch['labels'][0][:20].tolist()}")
    active_tokens = (batch['labels'][0] != -100).sum().item()
    pad_tokens = (batch['labels'][0] == -100).sum().item()
    print(f"Active tokens in row 0: {active_tokens}, Ignored (-100): {pad_tokens}")
    
    # 3. Inspect Category ID 28 and Mapping
    print("\n--- 3. CATEGORY ID MAPPING ---")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    id2label = {int(k): v for k, v in labels_meta["id2label"].items()}
    print(f"Category 28 is: \"{id2label.get(28, 'UNKNOWN')}\"")
    print(f"Total categories in mapping: {len(id2label)}")
    print("Sample category IDs:")
    for cid in [0, 1, 2, 3, 28, 32]:
        print(f"  ID {cid:2d} -> {id2label.get(cid)}")
        
    # 4. Load Checkpoint and Test Forward Pass Loss
    print("\n--- 4. CHECKPOINT FORWARD PASS LOSS CALCULATION ---")
    model = SentriFormerForMultiTask(config).to(device)
    ckpt = torch.load(CLEAN_SFT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        b_in = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        b_lbl = batch["labels"].to(device)
        out = model(input_ids=b_in, attention_mask=b_mask, generation_labels=b_lbl)
        print(f"Computed generation_loss on batch: {out['generation_loss'].item():.6f}")
        
        # Check decoder outputs and shift_right in forward
        print(f"Generation logits shape: {out['generation_logits'].shape}")
        pred_toks = torch.argmax(out['generation_logits'], dim=-1)
        print(f"First 15 predicted tokens on target positions: {pred_toks[0][:15].tolist()}")
        print(f"First 15 target labels:                        {b_lbl[0][:15].tolist()}")
        
    # 5. Step-by-Step Raw Autoregressive Generation Trace
    print("\n--- 5. STEP-BY-STEP GENERATION TRACE (PROMPT 1) ---")
    prompt = "I was double charged for my monthly subscription. Can you please refund the extra fee?"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Prompt: \"{prompt}\"")
    print(f"Encoded input_ids ({enc['input_ids'].shape}): {enc['input_ids'][0].tolist()}")
    
    with torch.no_grad():
        # Category logits check
        out_cat = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cat_probs = torch.softmax(out_cat["category_logits"], dim=-1)[0]
        top3_cat = torch.topk(cat_probs, 3)
        print("\nTop 3 Category Predictions:")
        for prob, cid in zip(top3_cat.values, top3_cat.indices):
            print(f"  Category {cid.item()} ({id2label.get(cid.item())}): {prob.item()*100:.2f}%")
            
        # Raw Step-by-step Decoder Trace
        encoder_outputs = model.sentriformer.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        
        # Test generation starting from BOS (token 2) vs other tokens
        curr_token = torch.full((1, 1), config.bos_token_id, dtype=torch.long, device=device)
        print(f"\nStarting token: ID {curr_token.item()} ('{tokenizer.decode([curr_token.item()])}')")
        
        past_key_values = None
        generated_ids = [curr_token.item()]
        
        print("\nFirst 10 generation steps:")
        for step in range(10):
            dec_out, past_key_values = model.sentriformer.decoder(
                input_ids=curr_token,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=enc["attention_mask"],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = model.lm_head(dec_out[:, -1, :])
            probs = torch.softmax(logits, dim=-1)[0]
            top5 = torch.topk(probs, 5)
            
            top1_id = top5.indices[0].item()
            top1_prob = top5.values[0].item()
            
            print(f"  Step {step+1}: top token={top1_id} ('{tokenizer.decode([top1_id])}', prob={top1_prob*100:.2f}%)")
            print(f"         top 5: {[(idx.item(), tokenizer.decode([idx.item()]), f'{p.item()*100:.1f}%') for p, idx in zip(top5.values, top5.indices)]}")
            
            generated_ids.append(top1_id)
            curr_token = torch.tensor([[top1_id]], dtype=torch.long, device=device)
            
            if top1_id == config.eos_token_id:
                print(f"  -> EOS reached at step {step+1}!")
                break
                
        print(f"\nFull Generated Token IDs: {generated_ids}")
        print(f"Decoded with skip_special_tokens=False: \"{tokenizer.decode(generated_ids, skip_special_tokens=False)}\"")
        print(f"Decoded with skip_special_tokens=True:  \"{tokenizer.decode(generated_ids, skip_special_tokens=True)}\"")

if __name__ == "__main__":
    diagnose()
