# Model card — SentriFormer T3 (multi-task trained, 2026-11-08)

**File:** `wsp_runs/FINAL_T3/sentriformer_t3.pt` · SHA256 `8d01c4fc…6a67d80` · 11.65M parameters (architecture unchanged)
**Lineage:** M8 pretrain → … → M10.3 (`checkpoints/m10_3_sft/m10_3_sft_best.pt`, untouched) → Phase-0 WSP head realignment →
**Phase-T run T3**: 18,000 steps of joint multi-task training (category, CSAT, sentiment, urgency, dialogue), step 16,000 selected
on val by a selection score frozen before training. Training: 70 min on RTX 3060 Laptop, peak VRAM 2.2 GB.

## Measured performance — held-out TEST sets, evaluated once after val selection
| Capability | Metric | **T3** | Phase-0 | Original M10.3 | Status |
|---|---|---|---|---|---|
| Category (33) | accuracy / macro F1 | **72.4% / 0.353** | 63.7% / 0.205 | 39.6% / 0.078 | learned (M9.5 ref 74.0% / 0.317) |
| | ECE / conf<0.25 | 0.058 / 0.7% | 0.069 / 2.2% | 0.165 / 63% | calibrated |
| CSAT (1-5) | exact / ±1 / MAE / macro F1 | 73.3% / 82.2% / 0.71 / **0.361** | 73.9% / 82.3% / 0.72 / 0.311 | 66.9% / 73.7% / 1.03 / 0.278 | learned; never predicts 2★ |
| Sentiment (neg/neu/pos) | macro F1 / acc | **0.742 / 87.2%** | untrained | untrained | learned |
| | human-labelled GoEmotions subset | 0.531 / 61.6% | — | — | modest outside complaints |
| Urgency (Low..Immediate) | val: Spearman / MAE vs mean baseline | 0.15 (CI −0.04…0.23) / 0.73 vs 0.67 | ≈0 | ≈0 | **NOT validated** (failed val criteria; test n=202: Spearman 0.28) |
| Replies (m10_4 test, 300 greedy) | teacher-forced loss / ROUGE-L | 2.78 / 0.135 | 2.98 / 0.130 | 2.98 / 0.130 | improved slightly |
| | dangling channel phrase (corrected detector, val) | **6.3%** | 21.7% | 21.7% | improved, not solved |

## What it does well
- Routes CFPB financial complaints confidently and correctly (behaviour suite: 20/21 vs 10/21 for M10.3), including
  e-commerce refund/delivery complaints that Phase-0 misrouted.
- Sentiment on complaint text agrees 91% with the DistilBERT teacher it was distilled from.
- Calibrated confidence; `needs_human_triage` fires when top-1 < 0.25.

## Limitations (measured)
- **Replies are still generic acknowledgements** ("reach out through our Customer Support Portal"); ~34% are deflection-style,
  6.3% still contain a dangling "send us a note at so…". They rarely address the specific problem. Not suitable for unsupervised sending.
- **Urgency is not usable**: the head was trained on only 932 rows and failed validation.
- CSAT is trained on e-commerce support data; on CFPB complaints it often predicts 5★ for clearly angry texts (behaviour suite cases 6, 7, 20).
- Sentiment labels are mostly teacher pseudo-labels; "neutral" comes almost only from Reddit (GoEmotions) text.
- Single-turn only: the model sees just the latest customer message (128 tokens for replies, 512 for routing).
- English only; 13 rare classes and CFPB-era duplicate labels limit macro F1.

## Use
`python -I wsp/inference.py "complaint"` (T3 is now the default). Regression tests: `FINAL_T3/regression_tests.json` (5/5 pass).
