"""
M10.3.1 Comprehensive Diagnostic: Classifier Analysis & Causal Attribution
- Analyzes Category Classifier (Top-1, Top-3, Confidence, Calibration, Accuracy)
- Causal Attribution of Decoder Failures (Classifier Error vs Decoder Error)
- Comprehensive Breakdown on the 80 Paired Audit Inquiries
"""
import os
import sys
import re
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask

# Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "sentriformer", "tokenizer", "sentritokenizer_v1")
CONFIG_PATH = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "config.json")
LABELS_PATH = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "classification_labels.json")
M103_CKPT = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "m10_3_sft", "m10_3_sft_best.pt")
AUDIT_JSON = os.path.join(ROOT_DIR, "sentriformer", "results", "m10_3_audit", "m10_3_threeway_audit.json")

# Ground truth mappings for the 10 canonical benchmark groups to valid classification categories
GROUP_TO_EXPECTED_CATS = {
    "Double Charge & Overbilling": ["Credit card", "Credit card or prepaid card", "Checking or savings account", "Bank account or service", "Payments related", "Refund Related"],
    "Refund Request / Disputed Fee": ["Refund Related", "Returns", "Checking or savings account", "Bank account or service", "Credit card", "Credit card or prepaid card", "Payments related"],
    "Delivery Problem / Lost Package": ["Order Related", "Returns", "Product Queries"],
    "Account & Address Update": ["Bank account or service", "Checking or savings account", "Credit card", "App/website", "Onboarding related", "Returns"],
    "Billing & Late Fee Dispute": ["Credit card", "Credit card or prepaid card", "Checking or savings account", "Bank account or service", "Payments related", "Debt collection"],
    "Subscription / Service Cancellation": ["Cancellation", "Order Related", "Returns", "Payments related"],
    "Card Problem / Declined Payment": ["Credit card", "Credit card or prepaid card", "Checking or savings account", "Bank account or service", "Payments related"],
    "Fraudulent / Unauthorized Transaction": ["Credit card", "Credit card or prepaid card", "Checking or savings account", "Bank account or service", "Debt collection", "Money transfers"],
    "Password & Login Access Reset": ["App/website", "Bank account or service", "Onboarding related"],
    "Order Status & Processing Delay": ["Order Related", "Returns", "Product Queries"]
}

def main():
    print("=" * 80)
    print("  M10.3.1 — CLASSIFIER CALIBRATION & CAUSAL ATTRIBUTION DIAGNOSTIC")
    print("=" * 80)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = SentriFormerConfig.from_pretrained(CONFIG_PATH)
    config.num_categories = 33
    config.num_csat_classes = 5

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    id2label = labels_meta["id2label"]
    label2id = labels_meta["label2id"]

    # Load M10.3 model
    print("Loading M10.3 checkpoint...")
    model = SentriFormerForMultiTask(config).to(device)
    raw = torch.load(M103_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw)
    model.eval()

    # Load previous audit data
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    cases = audit_data.get("cases", [])
    print(f"Loaded {len(cases)} audit cases.\n")

    top1_hits = 0
    top3_hits = 0
    evaluated_count = 0
    confidences = []
    top1_names = {}

    causal_records = []

    # Vocative name regex: detects ", Name." or ", Name!" or ", Name,"
    RE_VOCATIVE_NAME = re.compile(r",\s+([A-Z][a-z]{2,15})[.,!?]")

    for c in cases:
        qid = c["inquiry_id"]
        group = c["category_group"]
        q = c["query"]
        m103_info = c["m103"]
        resp = m103_info["response"]
        audit = m103_info["audit"]

        # Run forward pass for logits
        enc = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cat_logits = out["category_logits"][0] # [33]
            probs = F.softmax(cat_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=3)
        top1_cat = id2label[str(topk_indices[0].item())]
        top1_prob = topk_probs[0].item() * 100.0
        top2_cat = id2label[str(topk_indices[1].item())]
        top2_prob = topk_probs[1].item() * 100.0
        top3_cat = id2label[str(topk_indices[2].item())]
        top3_prob = topk_probs[2].item() * 100.0

        confidences.append(top1_prob)
        top1_names[top1_cat] = top1_names.get(top1_cat, 0) + 1

        # Check accuracy if group is canonical
        expected_cats = GROUP_TO_EXPECTED_CATS.get(group, None)
        is_top1_correct = None
        is_top3_correct = None

        if expected_cats is not None:
            evaluated_count += 1
            is_top1_correct = (top1_cat in expected_cats)
            is_top3_correct = any(id2label[str(idx.item())] in expected_cats for idx in topk_indices)
            if is_top1_correct: top1_hits += 1
            if is_top3_correct: top3_hits += 1

        # Check response quality
        is_useful_or_correct = audit["intent_relevance_proxy"] >= 33.3 and not audit.get("safety", {}).get("has_orphaned_phrase", False)

        # Vocative name check
        vocative_matches = RE_VOCATIVE_NAME.findall(resp)
        unsupported_vocative = [m for m in vocative_matches if m.lower() not in q.lower()]

        # Causal Attribution
        # If canonical:
        if expected_cats is not None:
            if is_top1_correct:
                if is_useful_or_correct:
                    attribution = "Joint Success (Classifier OK + Decoder OK)"
                else:
                    attribution = "Decoder Failure (Classifier OK, but Decoder Fumbled)"
            else:
                if is_useful_or_correct:
                    attribution = "Serendipitous (Classifier Wrong, but Decoder OK)"
                else:
                    attribution = "Classifier Failure / Joint Miss (Classifier Wrong + Decoder Fumbled)"
        else:
            attribution = "Unseen Test Case"

        causal_records.append({
            "id": qid,
            "group": group,
            "query": q,
            "response": resp,
            "top1_cat": top1_cat,
            "top1_prob": top1_prob,
            "top2_cat": top2_cat,
            "top2_prob": top2_prob,
            "top3_cat": top3_cat,
            "top3_prob": top3_prob,
            "expected_cats": expected_cats,
            "is_top1_correct": is_top1_correct,
            "is_top3_correct": is_top3_correct,
            "relevance": audit["intent_relevance_proxy"],
            "attribution": attribution,
            "unsupported_vocatives": unsupported_vocative
        })

    # Summary Statistics
    print("=== PART A: CLASSIFIER CALIBRATION & ACCURACY (Canonical Cases N=30) ===")
    print(f"Top-1 Domain Accuracy: {top1_hits}/{evaluated_count} ({top1_hits/evaluated_count*100:.1f}%)")
    print(f"Top-3 Domain Accuracy: {top3_hits}/{evaluated_count} ({top3_hits/evaluated_count*100:.1f}%)")
    print(f"Mean Top-1 Confidence: {sum(confidences)/len(confidences):.2f}%")
    print(f"Min Top-1 Confidence:  {min(confidences):.2f}%")
    print(f"Max Top-1 Confidence:  {max(confidences):.2f}%")
    print("\nTop Predicted Categories across all 80 queries:")
    for cat, count in sorted(top1_names.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  {cat:40s}: {count:2d} ({count/len(cases)*100:.1f}%)")

    print("\n=== PART B: CAUSAL ATTRIBUTION ON CANONICAL INQUIRIES (N=30) ===")
    attr_counts = {}
    for r in causal_records[:30]:
        a = r["attribution"]
        attr_counts[a] = attr_counts.get(a, 0) + 1
    for a, cnt in attr_counts.items():
        print(f"  {a:60s}: {cnt:2d} ({cnt/30*100:.1f}%)")

    print("\n=== PART C: VOCATIVE NAME HALLUCINATIONS CAUGHT ===")
    voc_caught = [r for r in causal_records if r["unsupported_vocatives"]]
    print(f"Found {len(voc_caught)} cases with mid-sentence unsupported vocatives:")
    for r in voc_caught:
        print(f"  Case {r['id']}: Names={r['unsupported_vocatives']} | Text: \"{r['response']}\"")

    # Save detailed diagnostic JSON
    diag_out = os.path.join(ROOT_DIR, "sentriformer", "results", "m10_3_audit", "m10_3_1_classifier_attribution.json")
    with open(diag_out, "w", encoding="utf-8") as f:
        json.dump(causal_records, f, indent=2)
    print(f"\nDetailed diagnostic saved to: {diag_out}")

if __name__ == "__main__":
    main()
