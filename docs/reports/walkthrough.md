# Walkthrough: M10.2 Architectural Fixes & Behavioral Quality Audit

## Overview
We resolved the fundamental root causes of the M10.2 generation failure (causal attention masking, EOS token labeling, and generation suppression), verified all fixes across a strict 4-gate smoke test, and conducted a rigorous **80-inquiry paired behavioral quality audit** directly comparing the baseline (`M9.3 dialogue_sft_best.pt`) against the corrected model (`M10.2 clean_sft_best.pt`).

---

## 1. Architectural Fixes & Verification

1. **Decoder Causal Masking**:
   - Replaced flawed conditional logic (`is_causal = True if self_past is None and attention_mask is None else False`) with explicit lower-triangular causal masks combined with key padding masks (`causal_mask & pad_mask`).
   - Strictly enforced `is_causal=False` whenever `attention_mask` is supplied to adhere to PyTorch SDPA constraints.
   - Mathematically verified gradient isolation: $\max_{j > i} |\frac{\partial L_i}{\partial \text{emb}_j}| = \mathbf{0.00e+00}$.
2. **Explicit EOS Construction**:
   - Resolved tokenizer omission of `</s>` by explicitly appending `tokenizer.eos_token_id` (3) to all target responses in `DialogueSFTDataset`. Verified 100/100 label sequences terminate with EOS.
3. **Autoregressive Generation Engine**:
   - Enforced BOS and PAD token logit suppression (`-inf`), applied token-level repetition penalty (1.2), and ensured natural halt upon encountering `eos_token_id`.

---

## 2. Full Clean Response SFT Training Metrics

The full 3-epoch Clean Response SFT was trained from the multi-task baseline `multitask_best.pt` with classification and CSAT heads frozen to protect routing representations:

| Checkpoint / Metric | Loss | Perplexity (PPL) | Verdict |
| :--- | :---: | :---: | :--- |
| Old Broken M10.2 SFT | 0.0758 | 1.08 | ❌ **Invalid (Artifact of bidirectional leakage)** |
| **M10.2 Clean SFT (Val)** | **2.3227** | **10.20** | ✅ **Realistic, genuinely causal** |
| **M10.2 Clean SFT (Test)** | **2.6942** | **14.79** | ✅ **Realistic test perplexity** |

---

## 3. Paired Behavioral Quality Audit (80 Inquiries)

We evaluated the exact same 80 customer complaints (30 canonical benchmark inquiries + 50 stratified unseen inquiries from `clean_dialogue_test.parquet`) head-to-head on both M9.3 and M10.2.

| Behavioral Dimension | M9.3 Baseline | M10.2 Clean SFT | Absolute Delta | Relative Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Contamination Rate** | **87.5%** | **0.0%** | **−87.5 pp** | **−100.0%** |
| - Twitter `@Handles` Rate | 77.5% | 0.0% | −77.5 pp | −100.0% |
| - URL / `t.co` Link Rate | 72.5% | 0.0% | −72.5 pp | −100.0% |
| - Agent `^Signatures` Rate | 62.5% | 0.0% | −62.5 pp | −100.0% |
| - DM Deflection Phrases | 22.5% | 0.0% | −22.5 pp | −100.0% |
| **EOS Termination Rate** | **0.0%** | **100.0%** | **+100.0 pp** | — |
| **Fluency Score** | 70.5% | **93.5%** | **+23.0 pp** | +32.6% |
| **Boundary/Completeness Proxy** | 8.8% | **67.5%** | **+58.7 pp** | +667.0% |
| **Degenerate Repetition Rate** | 22.5% | **0.0%** | **−22.5 pp** | −100.0% |
| **Average Response Tokens** | 101.0 | 25.4 | −75.6 tok | −74.9% |
| **Intent/Domain Relevance Proxy** | 31.7% | 18.3% | −13.4 pp | −42.3% |
| **Appropriate Empathy Rate** | 60.0% | 42.5% | −17.5 pp | −29.2% |

---

## 4. Visualizations Generated

The following charts were generated in `sentriformer/results/clean_sft/` and copied to artifacts:

1. **Artifact Contamination Comparison**:
   ![Contamination Comparison](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/01_contamination_reduction_comparison.png)
   *Demonstrates the complete elimination of raw Twitter platform artifacts (87.5% down to 0.0%).*

2. **Behavioral Quality Dimensions**:
   ![Behavioral Quality](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/02_behavioral_quality_pillars.png)
   *Shows dramatic gains in EOS termination (100%), fluency (93.5%), and boundary closure (67.5%), along with the trade-off in response brevity.*

3. **Category Confidence & CSAT Forecasts**:
   ![Category and CSAT](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/03_category_confidence_and_csat.png)
   *Highlights the concentration of routing confidence below the 25% threshold (mean 10.2%), demonstrating the need for a triage fallback rule.*

---

## 5. Decision Gate & Next Steps

- **Verdict**: **M10.2 PASS, with behavioral limitations documented**.
- **DPO Status**: **Deferred**. Training DPO preference tuning on short deflection responses would reinforce generic responses rather than addressing domain depth.
- **Next Logical Step**: Clean-response dataset refinement (replacing URL blanks like `"at "` with enterprise portal references) and establishing the production `<25%` routing fallback.
