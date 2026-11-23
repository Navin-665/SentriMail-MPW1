# UPDATE 2026-11-09 00:40 — T5a deployed (supersedes the T4d update below)
Overnight, user-approved: SentriWSP P1 (neuron dampening) and P2 (LayerNorm-bias steering) REJECTED: no specific effect beyond matched
controls. T5a (data filter, v5, NOT human-validated) failed its first test by 0.2pp on action words, then PASSED a pre-registered replication
on fresh rows: generic phrases about 26% → 17%, broken-link replies 3–6% → under 1%, both significant vs T4d and vs the matched control.
T5c (MMI reranking) REJECTED: plain sampling did better than reranking. Default: wsp_runs/FINAL_T5a (rollback FINAL_T4d). See FINAL_T5a/MODEL_CARD.md,
wsp_runs/t5/*_DECISION.md, wsp_runs/sentriwsp01/P1_DECISION.md, P2_DECISION.md. Human blind A/B pending (LOCAL-ONLY sheet).

# UPDATE 2026-11-08 15:30 — T4d deployed (supersedes the T3 update below)
Default model: `wsp_runs/FINAL_T4d/sentriformer_t4d.pt` (see its MODEL_CARD.md). Path: T3 -> T4a (role-verified multi-turn
dialogue, removed 52k customer-written "agent" targets) -> T4d (removed 65k portal-rewrite / DM-redirect / portal-loop targets).
Sealed test, run once: portal-loop replies 31.3% -> 0.7%; category 72.4% -> 73.7%; sentiment F1 0.742 -> 0.758; CSAT F1
0.361 -> 0.352 (slightly worse); urgency partly validated only; escalation untrained. Decisions: wsp_runs/t4/T4a_DECISION.md,
T4d_DECISION.md. A/B: ChatGPT-rated (NOT human, NOT blind): T4d 60 / T3 28 / tie 12 (wsp_runs/ab_test/). Human blind A/B pending.
Test set is now used: future selection needs a fresh held-out set. Next proposed: T5 privacy + broken-sentence target filter.

# SentriFormer — Report (Phase 0 WSP recovery + Phase T multi-task training)

## UPDATE 2026-11-08 08:30 — Phase T run T3 completed and accepted (supersedes §0 "deployed model" and §6)

**Deployed model is now `wsp_runs/FINAL_T3/sentriformer_t3.pt`** (card: `FINAL_T3/MODEL_CARD.md`). Phase-0 release kept.

Runs: T1 stopped (diverged) → T2 gate-failed at step 2,000 → root cause found offline: `F.cross_entropy(weight=…,
label_smoothing=…)` applies class weights to the smoothing term too, pushing every sample toward rare classes (this was my
design error and also caused T1's flat loss) → **T3** = T2 without category label smoothing: 18,000 steps, 70 min, 2.2 GB VRAM.

Frozen acceptance (`train/TRAINING_CRITERIA.md`, full val, `eval_full/T3_val/acceptance.json`): **11/12 frozen checks pass;
deployable = yes**. Failed: urgency MAE vs mean baseline (frozen) and urgency Spearman CI > 0 (supplementary).

| Test (once) | Phase-0 | **T3** |
|---|---|---|
| Category acc / macro F1 / ECE | 63.7% / 0.205 / 0.069 | **72.4% / 0.353 / 0.058** |
| CSAT exact / ±1 / macro F1 | 73.9% / 82.3% / 0.311 | 73.3% / 82.2% / **0.361** |
| Sentiment macro F1 (human subset) | untrained | **0.742 (0.531)** |
| Urgency | untrained | not validated (val failed; test Spearman 0.28, n=202) |
| Dialogue loss / ROUGE-L | 2.98 / 0.130 | **2.78 / 0.135** |

Corrections found while validating T3:
- The project's orphaned-phrase detector misses "note **at so** we can…". Re-measured with a self-tested token rule
  (`eval_full/orphan_recount.txt`): val dangling-phrase rate Phase-0 **21.7%** -> T3 **6.3%** (the detector reported 15% -> 0%).
- Behaviour suite (`FINAL_T3/behavior_suite/`): routing 20/21 (M10.3: 10/21), triage flags 3/24; replies are on-topic
  acknowledgements but still generic ("reach out through our Customer Support Portal"); CSAT predicts 5★ for several angry CFPB texts.

Status against the five-capability objective: classification **learned**, sentiment **learned**, CSAT **learned (weak on 2-3★)**,
urgency **not learned (data: 932 rows)**, responses **cleaner but still generic — not solved**. The project is not complete.
Next phase (T4, proposal only, awaiting approval): multi-turn context, speaker roles/MSP, filtered NTP — see chat proposal.

---

# Original overnight report (2026-11-07/08)

Date: 2026-11-07/08. Hardware: RTX 3060 Laptop 6 GB, 16 GB RAM. All numbers below are measured; nothing is estimated
unless labelled. Frozen criteria: `wsp_runs/SELECTION_CRITERIA.md`, `wsp_runs/train/TRAINING_CRITERIA.md`.

## 0. Bottom line

| Objective capability | Status this morning | Evidence |
|---|---|---|
| 1. Classification (33 classes) | **Repaired, not finished.** Test acc 39.6% -> **63.7%** (+24.2 pp, CI [+23.9, +24.4]). Still 10 pp below M9.5 (74.0%) and 15 pp below the specialist (79.2%). Macro F1 0.205; e-commerce classes weak. | §3, `e1/final_test/` |
| 2. Sentiment | **Not working.** Head untrained: 46% acc vs 79.5% majority baseline. Leakage-filtered training data built. | §5, `eval_full/phase0_val/` |
| 3. Urgency | **Not working.** Head untrained: Spearman -0.003 (no signal). 1,334 labelled rows built into splits. | §5 |
| 4. CSAT | **Repaired.** Test exact 66.9% -> **73.9%**, within-1 73.7% -> **82.3%**; equal to M9.5. Still never predicts 2-3 stars. | §3 |
| 5. Response generation | **Unchanged and still weak.** Generic deflection; 15% (val) / 54% (hand-written suite) dangling "reach out here:" phrases; ROUGE-L 0.130. | §5, `FINAL/behavior_suite/` |
| Inference reliability | **Working.** Crash on >512-token inputs found and fixed; 6/6 regression tests pass. | §4 |
| Phase-T multi-task training | **Started, not completed.** T1 diverged and was stopped by me; T2 was killed during data loading by the development tool's low-memory guard and must not be restarted without your go-ahead. | §6 |

**Deployed model:** `wsp_runs/FINAL/sentriformer_final.pt` = Phase-0 model (M10.3 backbone, unchanged decoder; E1-realigned
category + CSAT heads). Originals untouched (SHA256 verified before/after; `FINAL/checkpoint_manifest.json`).

## 1. E0 — baseline (inference only)

Test set, E0 harness (`e0/E0_REPORT.md` has val, per-class F1, confusion matrices, reliability tables, hashes).

| Checkpoint | Cat acc @512 | @128 | Macro F1 @512 | ECE | Mean conf | CSAT exact | CSAT ±1 |
|---|---|---|---|---|---|---|---|
| classifier_best (M9.2) | **79.15%** | 76.90% | 0.404 | 0.051 | 0.842 | head untrained | — |
| multitask_best (M9.5) | 73.99% | 73.29% | 0.317 | 0.048 | 0.788 | **73.56%** | 81.72% |
| m10_3_sft_best (M10.3) | **39.59%** | 42.27% | 0.078 | 0.165 | 0.231 | 66.90% | 73.68% |

- Reproduced previously reported numbers exactly (79.15%, 73.28%/73.29%, 73.56%) — the harness is faithful.
- **W1 confirmed:** M10.3 lost 34.4 pp routing vs M9.5 (CI [-34.7, -34.1]) with a byte-identical head. Root cause in code:
  `train_clean_response_sft.py` froze the three heads but **not the encoder/embeddings**, so generation-only training moved
  the encoder (embeddings 66% relative drift). The M10.2 report's statement that "the shared encoder was frozen" is incorrect.
- Integrity: 0 overlap between train/val/test within and across tasks and against the pretraining corpus (exact-text hash;
  near-duplicates not checked). 16 files hashed before/after E0: 0 changed.

## 2. Discrepancies found in the project
1. `checkpoints/config.json` says `num_categories: 21`; the label map and all checkpoints use 33.
2. `csat_best.pt` contains an untrained 21-class category head; `classifier_best.pt` contains an untrained CSAT head.
3. Sentiment/urgency head never trained in any checkpoint (no script passes `sentiment_labels`).
4. M9.5 dialogue PPL 1.04 is invalid (pre-fix causal-mask leakage, as the walkthrough already notes).
5. M10.2 report claims the encoder was frozen during clean SFT; code shows only the heads were frozen.
6. Audit generation uses sampling (T=0.7), so the 80-case audit numbers are stochastic.
7. Inference path never truncates inputs: >512 tokens crashes (positional table has 512 rows); training used 128 context tokens.
8. The SentriMail web app (`Sentrimail/sentrimail/app/ai_engine.py`) does **not** use SentriFormer (DistilBERT/flan-t5/TF-IDF).
9. 33-class taxonomy contains CFPB-era duplicates (3x credit reporting, 3x payday loan, 2x credit card); 13 classes F1 = 0.

## 3. E1 / E2 / E3 — Weight-Space Programming (no gradients)

**E1 (accepted).** Closed-form ridge map A (256x256) from M10.3 pooled features to M9.5 pooled features, fitted on a fixed
200k `classification_train` subset (+ `csat_train` for CSAT), folded as `dense.weight <- dense.weight @ A`.
- Drift measurements: identity R² (M10.3 -> M9.5) = **-0.10** (category inputs), **-0.79** (CSAT inputs); mean cosine 0.36 / 0.63.
  Linear map R² = 0.674 -> the drift is only partly linear, which bounds the recovery.
- Orientation check: head(W·A, X) vs head(W, X·Aᵀ) max |Δlogit| = 7.6e-6.
- λ sweep {0, 1e-4, 1e-3, 1e-2, 1e-1}: val acc 63.94 / 63.92 / 63.88 / 63.43 / 61.58% -> λ = 0 selected.

| Val @512 | Unedited | **E1** | Control: random same-norm | Control: shuffled pairs |
|---|---|---|---|---|
| Category acc | 39.67% | **63.94%** | 21.91% | 27.38% |

- CSAT: drift precondition met (R² < 0.90); csat_train-fitted map selected; val 66.31% -> 72.20% (CI excl. 0).
- Edit scope: 2 tensors changed (`category_head.dense.weight`, `csat_head.dense.weight`), 121 byte-identical.

**E2 (rejected, both heads).** Gradient-free prior matching on train features.
- Category: val acc +1.6 pp, macro F1 +0.020 (CI [0.018, 0.023]), ECE -0.028 — but **8 important classes lost > 2 pp**
  precision or recall (incl. Mortgage, Student loan, Credit card) -> rejected by the safeguard.
- CSAT: no measurable change -> rejected.

**Test (once, Phase-0 selected model):** category 63.74% @512 (63.35% @128), macro F1 0.205, weighted F1 0.618, ECE 0.069
(was 0.165), confidence < 0.25 on 2.2% of cases (was 63%), `Returns` predicted 0.5%. CSAT 73.91% exact, 82.30% ±1, MAE 0.717,
macro F1 0.311 (vs M9.5: +0.34 pp, CI [-1.7, +2.3] — statistically equal).

## 4. Inference pipeline and regression tests
- `wsp/inference.py` reuses the project's decoder (`generate_multitask`) and quality checks; adds truncation (512 routing,
  128 generation), a <25% confidence triage flag, and honest reporting that sentiment/urgency are unavailable.
- `FINAL/regression_tests.json`: T1 strict load, T2 edit scope (heads only), T3 generation token-identical 50/50,
  T4 781-token/empty/unicode inputs, T5 originals' SHA256 unchanged, T6 reproduces recorded val predictions 1996/2000 — **ALL PASS**.
- Behaviour suite (24 hand-written complaints, greedy; qualitative only): routing 10/21 -> 14/21, triage flags 24 -> 8,
  mean confidence 0.15 -> 0.49. Strong on CFPB topics (debt collection 0.99, student loan 0.92, long mortgage 1.00);
  **worse on e-commerce** (refund/delivery complaints routed to CFPB classes). Replies identical before/after: 13/24 contain
  orphaned channel phrases.

## 5. Five-capability baseline of the deployed model (val, `eval_full/phase0_val/metrics.json`)
| Capability | Metric | Value | Baseline |
|---|---|---|---|
| Category | acc / macro F1 / ECE | 63.9% / 0.209 / 0.066 | — |
| CSAT | exact / macro F1 | 72.2% / 0.302 | — |
| Sentiment (3-class) | acc / macro F1 | 46.4% / 0.265 | majority 79.5% -> **untrained** |
| Urgency (4-level) | macro F1 / MAE / Spearman | 0.206 / 0.80 / -0.003 | mean-prediction MAE 0.67 -> **untrained** |
| Dialogue (m10_4 val) | loss / ROUGE-L / words / distinct-2 / orphaned | 2.98 / 0.130 / 18.6 / 0.214 / 15% | — |

Sentiment/urgency data (`wsp_runs/data/aux_data_report.json`): sentiment = 341k train rows (DistilBERT-SST2 **pseudo-labels**
on complaints + 25.8k **human** GoEmotions rows), val/test 20.7k/20.5k after removing anything seen in SentriCorpus training;
urgency = 932/200/202 rows from `Ataur77/ecommerce-customer-support` (annotation provenance unknown; small).

## 6. Phase T — multi-task training (gradient-based)
Design (`wsp/train_v2.py`): every step updates the shared encoder with all five losses (category CE with inverse-sqrt class
weights + label smoothing; CSAT class-weighted CE; sentiment class-weighted CE; urgency MSE; dialogue CE on M10.4), which
removes the W1 failure mode by construction. Separate backbone/head LRs, fp16, clip 1.0, cosine, hard step/minute caps,
resumable state, frozen selection score, stop gate.

- Benchmark: 0.26-0.32 s/step frozen, ~0.43 s/step joint, **peak VRAM 2.1 GB**.
- **T1 (stopped by me at step 1,500):** category train loss flat ~8.6, val-subset acc 64% -> 36%. Diagnosis: the λ=0 E1 map is
  ill-conditioned; fine in eval mode, but dropout noise is amplified into confident wrong logits, and AdamW cannot undo the
  ~10x-scaled weights fast enough. *A closed-form edit that is good at inference can be a poor initialisation for training.*
  Positive signal: dialogue val-subset loss already 2.90 (Phase-0 full val 2.98) and sentiment macro F1 0.69 after 1,500 steps.
- **T2 (fixed: category head re-initialised + 2,000-step frozen-backbone warm-up + stop gate):** killed during data loading by
  the development tool's background low-memory guard (system RAM was near the 16 GB limit with browser/IDE open). Per that guard it was
  **not restarted**. The data loader was then rewritten to stream parquet row groups (verified identical subset, far lower
  peak RAM). Restart command: `FINAL/HOW_TO_LOAD.md` §5 (est. ~2.5 h).

## 7. What remains unsolved, and the recommended path
1. **Run T2** (bounded, ~2.5 h, 2.1 GB VRAM; close browser/IDE first). Then `eval_full.py` on val, apply TRAINING_CRITERIA,
   test once. This is the step that can make sentiment and urgency real and improve the decoder on M10.4 data.
2. **Label clean-up:** merge CFPB-era duplicate classes (33 -> ~20) or report grouped metrics; it is the main macro-F1 ceiling.
3. **Urgency data:** 932 training rows is too few for a reliable head; label 2-5k real complaints or derive a weak-label set.
4. **CSAT 2-3 stars:** 16-17 test examples per class; more mid-range data is needed, not architecture changes.
5. **Responses:** the decoder reproduces deflection templates from its training mix. Decoder training on M10.4 (which passed
   the orphaned-phrase gate) is the direct lever; reply relevance should be evaluated with ROUGE-L plus human review.
6. **Capacity:** 11.65M params (4+4 layers, d=256) is adequate for routing/CSAT/sentiment/urgency, but it limits free-form
   reply quality. Within 6 GB, d_model 384-512 / 6+6 layers (~30-50M params) is trainable with fp16 + batch 16-32; only
   worth it after T2 shows whether data or capacity is the bottleneck.

## 8. Files
`e0/` baseline · `e1/` fits, controls, candidates, val/test · `e2/` calibration (rejected) · `data/` sentiment/urgency splits ·
`train/` T1 log (failed), T2 config, plans, criteria · `eval_full/phase0_val/` five-capability baseline ·
`FINAL/` model, manifest, regression tests, behaviour suite, HOW_TO_LOAD.md · `MODEL_CARD.md` · code in `../wsp/`.
