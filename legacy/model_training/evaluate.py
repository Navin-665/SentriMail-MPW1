"""
Comprehensive evaluation and interactive inference script for SentriModel.

Features:
  1. Full benchmark evaluation on test.parquet (42,291 rows)
  2. Out-of-domain generalization benchmark on validation_ecommerce.parquet (1,336 rows)
  3. Per-class Precision, Recall, Macro-F1, and Confusion Matrices
  4. Interactive live demo mode for real-time email complaint triage
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

import config
from dataset import SentriDataset, build_collate_fn
from model import SentriModel
from response_safety import sanitize_or_fallback, contains_unsafe_content
from category_facts import get_category_fallback, route_with_confidence_gate


def load_trained_model(checkpoint_path="checkpoints/best_model.pt", device="cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first via train.py!")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    tokenizer = T5Tokenizer.from_pretrained(config.CHECKPOINT_DIR if os.path.exists(os.path.join(config.CHECKPOINT_DIR, "tokenizer_config.json")) else config.BACKBONE_MODEL_NAME)
    model = SentriModel(checkpoint.get("backbone_model_name", config.BACKBONE_MODEL_NAME))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_metrics", {}).get("val_total_loss", None))
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A"
    print(f"Loaded SentriModel (Epoch {checkpoint.get('epoch', 'N/A')}, Val Loss: {val_loss_str})")
    return model, tokenizer


def evaluate_dataset(model, tokenizer, parquet_path, dataset_name="Test Set", device="cuda"):
    print("\n" + "=" * 65)
    print(f"EVALUATING: {dataset_name} ({parquet_path})")
    print("=" * 65)

    dataset = SentriDataset(parquet_path)
    collate_fn = build_collate_fn(tokenizer)
    loader = DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    sentiment_preds, sentiment_trues = [], []
    emotion_preds, emotion_trues = [], []
    category_preds, category_trues = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type=device.type if isinstance(device, torch.device) else "cuda", enabled=(str(device) != "cpu")):
                outputs = model(input_ids, attention_mask)

            has_sent = batch["has_sentiment"].tolist()
            s_p = torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().tolist()
            s_t = batch["sentiment_labels"].tolist()
            for pred, true, valid in zip(s_p, s_t, has_sent):
                if valid:
                    sentiment_preds.append(pred)
                    sentiment_trues.append(true)

            has_emot = batch["has_emotion"].tolist()
            e_p = torch.argmax(outputs["emotion_logits"], dim=-1).cpu().tolist()
            e_t = batch["emotion_labels"].tolist()
            for pred, true, valid in zip(e_p, e_t, has_emot):
                if valid:
                    emotion_preds.append(pred)
                    emotion_trues.append(true)

            has_cat = batch["has_category"].tolist()
            c_p = torch.argmax(outputs["category_logits"], dim=-1).cpu().tolist()
            c_t = batch["category_labels"].tolist()
            for pred, true, valid in zip(c_p, c_t, has_cat):
                if valid:
                    category_preds.append(pred)
                    category_trues.append(true)

    if sentiment_trues:
        print(f"\n--- 1. Sentiment Classification ({len(sentiment_trues):,} samples) ---")
        print(classification_report(sentiment_trues, sentiment_preds, target_names=config.SENTIMENT_LABELS, digits=4, zero_division=0))
    else:
        print("\n--- 1. Sentiment Classification --- (No sentiment labels present in this dataset)")

    if emotion_trues:
        print(f"\n--- 2. Emotion Classification ({len(emotion_trues):,} samples) ---")
        print(classification_report(emotion_trues, emotion_preds, target_names=config.EMOTION_LABELS, digits=4, zero_division=0))
    else:
        print("\n--- 2. Emotion Classification --- (No emotion labels present in this dataset)")

    if category_trues:
        print(f"\n--- 3. Category Classification ({len(category_trues):,} samples) ---")
        print(classification_report(category_trues, category_preds, target_names=config.CATEGORY_LABELS, digits=4, zero_division=0))
        
        cm = confusion_matrix(category_trues, category_preds, labels=list(range(len(config.CATEGORY_LABELS))))
        print("\n--- Category Confusion Matrix (Rows: True, Cols: Pred) ---")
        header = f"{'True / Pred':<18} " + " ".join(f"{lbl[:6]:>7}" for lbl in config.CATEGORY_LABELS)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = f"{config.CATEGORY_LABELS[i]:<18} " + " ".join(f"{count:>7d}" for count in row)
            print(row_str)
    else:
        print("\n--- 3. Category Classification --- (No category labels present in this dataset)")


def interactive_demo(model, tokenizer, device="cuda"):
    print("\n" + "=" * 65)
    print("SENTRIMAIL INTERACTIVE MULTI-TASK TRIAGE DEMO")
    print("Type a customer complaint or email (or 'exit' to quit)")
    print("=" * 65 + "\n")

    while True:
        try:
            text = input("Enter email/complaint > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text or text.lower() in ("exit", "quit", "q"):
            break

        inputs = tokenizer(text, return_tensors="pt", max_length=config.MAX_INPUT_TOKENS, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            
            # Softmax confidences
            s_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)[0]
            e_probs = torch.softmax(outputs["emotion_logits"], dim=-1)[0]
            c_probs = torch.softmax(outputs["category_logits"], dim=-1)[0]

            s_idx = torch.argmax(s_probs).item()
            e_idx = torch.argmax(e_probs).item()

            # Confidence gating
            predicted_cat, cat_confidence, needs_human_review = route_with_confidence_gate(
                c_probs, config.CATEGORY_LABELS, threshold=config.CONFIDENCE_THRESHOLD
            )

            # Autoregressive generation
            gen_ids = model.generate_response(inputs["input_ids"], inputs["attention_mask"], max_new_tokens=config.MAX_RESPONSE_TOKENS)
            raw_response_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # Safety filter & verified grounding
            fallback_text = get_category_fallback(predicted_cat)
            is_unsafe = contains_unsafe_content(raw_response_text)
            safe_response_text = sanitize_or_fallback(raw_response_text, fallback_text)

        print("\n" + "-" * 50)
        print(f"  Sentiment:  {config.SENTIMENT_LABELS[s_idx].upper()} ({s_probs[s_idx]*100:.1f}% confidence)")
        print(f"  Emotion:    {config.EMOTION_LABELS[e_idx].upper()} ({e_probs[e_idx]*100:.1f}% confidence)")
        
        routing_flag = " [LOW CONFIDENCE -> Flagged for Human Review]" if needs_human_review else ""
        print(f"  Category:   {predicted_cat.upper()} ({cat_confidence*100:.1f}% confidence){routing_flag}")
        
        if is_unsafe:
            print(f"  AI Reply:   \"{safe_response_text}\" [Safety Filter: Fallback to Verified Template]")
            print(f"  (Raw Gen):  \"{raw_response_text}\" [BLOCKED: Unsafe URL/Sign-off]")
        else:
            print(f"  AI Reply:   \"{safe_response_text}\"")
        print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SentriModel Checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--interactive", action="store_true", help="Launch interactive demo loop")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_trained_model(args.checkpoint, device)

    if args.interactive:
        interactive_demo(model, tokenizer, device)
    else:
        # 1. Test Split
        evaluate_dataset(model, tokenizer, config.TEST_PARQUET_PATH, dataset_name="Test Split (42,291 rows)", device=device)
        # 2. Out-of-domain E-commerce validation benchmark
        evaluate_dataset(model, tokenizer, config.ECOMMERCE_VALIDATION_PATH, dataset_name="Held-Out E-commerce Benchmark (1,336 rows)", device=device)


if __name__ == "__main__":
    main()
