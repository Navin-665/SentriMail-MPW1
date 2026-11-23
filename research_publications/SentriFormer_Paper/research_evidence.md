# SentriFormer: Verified Research Evidence Report

Prepared 2026-11-23 for the IEEE conference paper. This was a read-only audit: nothing in the project was modified, retrained or re-evaluated.

**Conventions**
- All paths are relative to the project root `SENTRIMAIL-MPW1/` unless stated otherwise. `WR/` = `sentriformer/wsp_runs/`.
- **VERIFIED** means the number was read from a machine-written JSON or log, or recomputed from code or checkpoint tensors during this audit.
- **REPORT-ONLY** means the number appears only in a hand- or LLM-written markdown report, with no underlying JSON found.
- **UNVERIFIED** means it could not be confirmed, or there is evidence against it.
- **DERIVED (this audit)** means a trivial aggregate computed here from label counts. It is not a model evaluation.
- No dataset text (tweets, complaints, replies, examples) is reproduced anywhere in this report.

---

## 0. Headline facts (read first)

| Fact | Value | source: |
|---|---|---|
| Deployed checkpoint | `WR/FINAL_T5a/sentriformer_t5a.pt` (= `WR/train/T5a/model_best.pt`), SHA256 `4205c569b1d175ed3378a4867c5406954ffe5f408edf696b7d0aad9016ea38f5` | `sentriformer/wsp/inference.py` L25 (`DEFAULT_CKPT`); `WR/t5/t5a_retest.json` |
| Deployed checkpoint's training step | **step 2,000** of a 10,000-step run (checkpoint metadata `step: 2000`). The T5a model card's wording "T4d + 10,000 steps" describes the run length, not the selected checkpoint. | tensor metadata of `sentriformer_t5a.pt`; `WR/train/T5a/eval_history.json` |
| Unique parameters | **11,654,698** | counted from the checkpoint state dict (tied weights counted once) |
| **Sealed TEST metrics of the deployed model (T5a)** | **DO NOT EXIST.** T5a was selected and promoted on validation rows only. The last sealed-test evaluation was **T4d** (T5a's parent). | `WR/pipeline/run_final_test.sh`; `WR/eval_full/` has no `T5a_test`; `WR/FINAL_T5a/MODEL_CARD.md` |
| Test-set uses | E0 (baseline checkpoints), Phase-0 selected model (E1), T3, T4d. Each test run happened once, after selection. | `WR/WSP_FINAL_REPORT.md`; `WR/e0/test/`, `WR/e1/final_test/`, `WR/eval_full/T3_test/`, `WR/eval_full/T4d_test/` |
| Urgency | **Not learned and not validated**. T7u refit rejected; MAE worse than predicting the mean. | `WR/t7u/T7U_DECISION.md`, `WR/t7u/t7u_results.json` |
| Human evaluation | **None done.** The only A/B tallies are ChatGPT (unblinded) and LLM model labels (key still sealed). | `WR/ab_test/AB_RESULT_chatgpt_unblinded.json`; `WR/t5/AB_T5a_ratings_LLM_LABELS_LOCAL_ONLY.json` |

**Recommendation for the paper:** report T4d sealed-test numbers as "final test-set results". Report T5a as a validation-selected reply-quality refinement (pre-registered replication on fresh validation rows). Alternatively, run one sealed T5a test before submission, which requires user approval and a GPU run.

---

## 1. Architecture (from code and the deployed checkpoint)

### 1.1 Backbone

| Property | Value | source: |
|---|---|---|
| Type | Encoder-decoder Transformer (T5/BART-style seq2seq), shared encoder feeding multitask heads | `sentriformer/model/sentriformer.py`, `multi_task_heads.py` |
| Encoder / decoder layers | 4 / 4 | `sentriformer/checkpoints/config.json` (`encoder_layers`, `decoder_layers`) |
| d_model | 256 | same |
| Attention heads / head_dim | 8 / 32 | same |
| FFN hidden dim | 1024 (4 x d_model), GELU, dropout after activation and after fc2 | `config.json`; `model/feed_forward.py` |
| Normalization | **Pre-LN** (LayerNorm before each sub-layer, residual after) plus a final LayerNorm in both encoder and decoder. eps = 1e-6 | `model/encoder.py` (`SentriEncoderBlock`), `model/decoder.py` (`SentriDecoderBlock`) |
| Decoder block | causal self-attn -> cross-attn -> FFN, each Pre-LN with residual | `model/decoder.py` L14-80 |
| Causal mask | explicit lower-triangular mask AND key-padding mask, passed to SDPA with `is_causal=False` (fixed in M10.2) | `model/decoder.py` L138-155; `walkthrough.md` §1 |
| Attention kernel | `torch.nn.functional.scaled_dot_product_attention`. Bias-free Q/K/V/O projections | `model/attention.py` |
| Linear biases | `use_bias=false` for all backbone Linear layers. Head output projections (`out_proj`, `sentiment_proj`, `urgency_proj`) keep their bias | `config.json`; checkpoint tensor list |
| Positional encoding | Fixed sinusoidal, max 512 positions (non-persistent buffer). Token embeddings are scaled by sqrt(d_model) | `model/encoder.py` (`SinusoidalPositionalEmbedding`) |
| Embeddings / weight tying | One shared 16,000 x 256 embedding for encoder, decoder and LM head (`tie_word_embeddings=true`) | `model/sentriformer.py`; `multi_task_heads.py` L101-103 |
| Dropout | 0.1 (hidden and attention) | `config.json` |
| Init | Normal(0, 0.02); LayerNorm weight 1, bias 0 | `model/sentriformer.py` `_init_weights` |
| Max sequence length | 512 (position table). Training lengths: category 256, CSAT/sentiment/urgency 128, dialogue context 128 (T3) or 256 left-truncated (T4a and later), response 128 | `WR/train/*/config.json`; `wsp/train_v2.py` L180-200 |
| Inference lengths | routing 128 tokens (checkpoint has no `category_seq_len` metadata, so `CAT_LEN=128` is used); generation context 256 for multi-turn runs; max 100 new tokens | `wsp/inference.py` L27, L44, L69, L72 |

**Note for the paper:** headline category metrics are computed at 512 tokens (`category_len512`), but deployed inference routes at 128 tokens. The 128-token numbers are about 1.3-1.5 pp lower (see §4).

### 1.2 Parameter count (VERIFIED, analytic and checkpoint)

| Component | Parameters | source: |
|---|---|---|
| Shared embedding (tied with LM head) | 4,096,000 | 16000 x 256 |
| Encoder, 4 layers (787,456 each) + final LN | 3,150,336 | analytic from `model/*.py`; matches checkpoint shapes |
| Decoder, 4 layers (1,050,112 each) + final LN | 4,200,960 | same |
| **Seq2seq core** | **11,447,296** | README §4.1 (matches) |
| Category head (256x256 dense + 33x256+33 out) | 74,017 | checkpoint shapes |
| Sentiment/urgency head (256x256 dense + 3-way + 1-way) | 66,564 | checkpoint shapes |
| CSAT head (256x256 dense + 5x256+5) | 66,821 | checkpoint shapes |
| **Total (deployed, 33 classes)** | **11,654,698** | unique-tensor count of `WR/FINAL_T5a/sentriformer_t5a.pt` |

The README value of 11,651,614 for the multitask model corresponds to a 21-class category head (3,084 fewer parameters). It is outdated (`README.md` L168). `checkpoints/config.json` still says `num_categories: 21`; code overrides it to 33 from the label map (`wsp/train_v2.py` L164, `wsp/inference.py` L39).

### 1.3 Task heads

All heads read the **masked mean-pool of the final encoder states** (`pool_encoder_states`, `multi_task_heads.py` L12-28).

| Head | Structure | Output | source: |
|---|---|---|---|
| Category (routing) | dropout -> Linear 256->256 (no bias) -> GELU -> dropout -> Linear 256->33 | 33 logits | `multi_task_heads.py` `CategoryClassificationHead` |
| Sentiment + urgency (shared trunk) | dropout -> Linear 256->256 -> GELU -> dropout -> {Linear 256->3 sentiment; Linear 256->1 -> sigmoid urgency} | 3 logits, urgency in [0,1] (x3 = level 0..3) | `SentimentUrgencyHead`; `inference.py` L61 |
| CSAT | same trunk as category -> Linear 256->5 | 5 logits (1-5 stars) | `CSATPredictionHead` |
| Response generation | decoder + tied LM head over 16k vocabulary | tokens | `multi_task_heads.py` L135-160 |

### 1.4 Tokenizer

| Property | Value | source: |
|---|---|---|
| Type | **Byte-level BPE** (HF `tokenizers`), not SentencePiece | `sentriformer/tokenizer/train_tokenizer.py` L2, L71-87; `tokenizer/sentritokenizer_v1/tokenizer.json` (`model.type = BPE`) |
| Vocabulary | 16,000, including 106 special tokens: `<pad>`=0, `<unk>`=1, `<s>`=2, `</s>`=3, `<mask>`, `<sep>`=5, and 100 T5 sentinels `<extra_id_0..99>` | `tokenizer.json` `added_tokens`; `config.json` |
| Normalizer / pre-tokenizer | NFKC; ByteLevel (`add_prefix_space=False`); min_frequency 2 | `train_tokenizer.py` L80-111 |
| Vocab ablation | 16k / 24k / 32k trained and benchmarked. 16k: 4.73 chars/token overall, 0% UNK, 96.73% of texts fit in 512 tokens | `tokenizer/tokenizer_benchmark.json` |

### 1.5 Loss functions and weights (Phase T, used for T3 to T6)

Each optimizer step sums one mini-batch per task (`wsp/train_v2.py` L229-250):

| Task | Loss | Weight | Batch / frequency | source: |
|---|---|---|---|---|
| Category (33) | CE with inverse-sqrt-frequency class weights (normalized to mean 1); label smoothing **0** from T3 onward | 1.0 | 48 every step, 256 tokens | `train_v2.py` L84-86, L233; `WR/train/T3/config.json` |
| CSAT (5) | class-weighted CE (inverse sqrt) | 0.3 | 16, every 3rd step | L238; config |
| Sentiment (3) | class-weighted CE (inverse sqrt) | 0.3 | 32 every step | L240 |
| Urgency | MSE of sigmoid output vs label/3 | 0.5 | 8, every 4th step | L244 |
| Dialogue | token CE (decoder, EOS appended, pad ignored) | 1.0 | 32 every step | L246; `multi_task_heads.py` L150-160 |

Earlier M-series losses:
- M9.5 used `1.0 L_cat + 0.1 L_csat + 0.5 L_dial` (`sentriformer/training/train_multitask_joint.py` L8, L146).
- Pretraining used T5 span corruption (15% noise density, mean span length 3) (`training/train_pretrain.py` L43-44).

---

## 2. Datasets

### 2.1 Raw sources

| Source | License | Raw size | Role | source: |
|---|---|---|---|---|
| CFPB Consumer Complaint Database (`rawdata/complaints.csv`) | US public domain | 9.18 GB, 17,281,869 rows; 3,841,422 with narrative | pretraining text + product labels (routing) | `rawdata/DATA_SOURCES.md` §1 |
| TWCS, Customer Support on Twitter (Kaggle) (`rawdata/twcs.csv`) | **CC BY-NC-SA 4.0** (non-commercial) | 516 MB, 2,811,774 tweets | pretraining + dialogue pairs | `DATA_SOURCES.md` §2 |
| Bitext customer-support LLM dataset (HF) | Apache 2.0 | 26,872 rows | pretraining + dialogue pairs | `DATA_SOURCES.md` §3; `sentriformer/corpus/extract_bitext.py` |
| Kaggle "Customer Support Data" (e-commerce, `rawdata/archive (1)/Customer_support_data.csv`) | MIT (as stated) | 85,907 rows; 28,742 non-null remarks | e-commerce category labels + **CSAT labels** | `DATA_SOURCES.md` §4 |
| GoEmotions (HF, human-annotated) | not documented in project (UNVERIFIED) | 25,757 train rows used | human sentiment labels | `WR/data/aux_data_report.json` |
| `distilbert-base-uncased-finetuned-sst-2-english` | model, used as teacher | — | **pseudo-labels** for sentiment | `aux_data_report.json` `label_provenance` |
| HF `Ataur77/ecommerce-customer-support` | not documented (UNVERIFIED) | 1,334 rows used | urgency labels (annotation provenance unknown) | `aux_data_report.json`; `WR/t7u/T7U_DECISION.md` |

### 2.2 Preprocessing and cleaning (SentriCorpus)

- **Cleaning** (`sentriformer/corpus/cleaner.py`): HTML unescape, NFKC, control-character strip, whitespace collapse, at most 2 consecutive newlines. Casing and punctuation are preserved.
- **Validity filters**: CFPB at least 15 chars and 3 words (`extract_cfpb.py` L51); others at least 4 chars and 1 word.
- **Global dedup**: SHA-256 of normalized text (lowercased, whitespace-collapsed, CFPB `XX` masks removed) (`cleaner.py` `normalize_for_dedup`).
- **Group-aware deterministic split**: MD5(group_id) mod 10000 gives 90/5/5. Groups are TWCS conversations, CFPB normalized-hash clusters and e-commerce order IDs (`corpus/splitter.py`).
- **Corpus result**: 3,538,711 canonical records (6 malformed skipped). Train 3,184,956 / val 176,586 / test 177,169. Sources: CFPB 2,448,422, TWCS 1,049,079, Bitext 24,272, customer_support 16,936. "Leakage audit PASSED" (`sentriformer/data/corpus_stats.json`).
- **Train split composition**: CFPB 2,203,951; TWCS 943,965; Bitext 21,823; customer_support 15,217. Example types: complaint 2,219,168 / conversation 943,965 / support_response 21,823 (VERIFIED from parquet column counts, `sentriformer/data/train.parquet`).
- **Cross-task integrity**: supervised splits nest inside the same pretraining split. For example, all 123,229 classification-test rows are in pretraining test and none are in pretraining train (`WR/e0/pretrain_overlap.json`). Exact-text hash only; near-duplicates were not checked (`WR/WSP_FINAL_REPORT.md` §1).

### 2.3 Supervised splits (row counts VERIFIED from parquet metadata)

| Dataset | Train | Val | Test | Label construction | source: |
|---|---|---|---|---|---|
| Category routing | 2,219,168 | 122,961 | 123,229 | CFPB `Product` field plus e-commerce `category` field, unioned into **33 classes** | `sentriformer/data/supervised/classification_*.parquet`; `corpus/finalize_corpus.py` L95-111; `classification_labels.json` |
| CSAT (1-5) | 15,217 | 849 | 870 | e-commerce `CSAT Score` (survey score after handling). Input is the customer remark only (post-hoc fields excluded) | `csat_*.parquet`; `finalize_corpus.py` L121-130; `DATA_SOURCES.md` §4 |
| Sentiment (neg/neu/pos) | 341,451 | 20,721 | 20,453 | DistilBERT-SST2 pseudo-labels on CFPB/TWCS/Bitext text plus GoEmotions human labels (joy->pos; anger/fear/sadness/disgust->neg; neutral->neu; surprise dropped). Leakage-filtered against all SentriCorpus splits | `WR/data/sentiment_*.parquet`; `sentriformer/wsp/build_aux_labels.py` L1-12, L43-51; `aux_data_report.json` |
| Urgency (0..3) | 932 | 200 | 202 | `Urgency` column (Low/Medium/High/Immediate -> 0..3), stratified 70/15/15 | `WR/data/urgency_*.parquet`; `build_aux_labels.py` L9-10 |
| Dialogue M10.4 (single-turn) | 531,888 | 31,470 | 31,912 | context -> agent reply; TWCS 514,611 + Bitext 15,747 + 1,500 script-generated "resolution core" + 30 templates (train) | `data/supervised/m10_4_dialogue_*.parquet`; `sentriformer/data/build_m10_4_resolution_dataset.py` |
| Dialogue T4 multi-turn (`multi_*`) | 479,445 | 28,454 | 28,907 | M10.4 with role-verified targets (52,424 customer-authored "agent" targets removed); history prefixed "Customer:/Agent:", 250-token budget; 0 group overlap across splits | `WR/t4/data/build_report.json` |
| T4d train (portal filter) | 414,247 | (multi_val) | (multi_test) | multi_train minus 65,198 rows (portal rewrite 41,835, DM redirect 3,413, portal-loop target 56,974; overlapping flags) | `WR/t4/data/t4d_build_report.json` |
| **T5a train (deployed model's dialogue data)** | **384,751** | (multi_val) | (multi_test) | T4d train minus 29,483 regex-flagged replies (broken link 24,610, whole-reply generic 3,475, bulk personal-data request 1,427, secret request 22) and 13 train/val/test identical pairs. **No human precision audit** | `WR/t4/data/t5a_build_report.json` |
| T6 train (rejected) | 320,599 | | | T5a train minus 64,152 rows with PMI < 0 | `WR/t4/data/t6_build_report.json` |

### 2.4 Label distributions (VERIFIED, aggregate counts)

**CSAT** (`data/supervised/csat_*.parquet`):

| Stars | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train | 4,032 | 283 | 484 | 1,339 | 9,079 |
| val | 222 | 19 | 32 | 73 | 503 |
| test | 217 | 16 | 17 | 88 | 532 |

**Sentiment** (`WR/data/aux_data_report.json`):

| Split | neg | neu | pos | Origin |
|---|---|---|---|---|
| train | 299,161 | 10,234 | 32,056 | distilled_cfpb 190,660; distilled_twitter 107,670; goemotions_human 25,757; distilled_bitext 17,364 |
| test | 16,288 | 1,212 | 2,953 | 3,012 human (GoEmotions) rows |

**Urgency** (same source):
- train: 0=48, 1=251, 2=394, 3=239
- test: 0=11, 1=54, 2=85, 3=52

**Category (33 classes)**: extremely imbalanced. Largest class (CFPB "Credit reporting or other personal consumer reports"): 609,061 train / 33,547 test. Six classes have 10-25 train rows. The taxonomy contains CFPB-era duplicates: 3 credit-reporting variants, 3 payday-loan variants, and 2 credit-card variants (`WR/WSP_FINAL_REPORT.md` §2 item 9; class counts VERIFIED from `classification_*.parquet`). Majority-class test accuracy is **27.2%** (33,547 / 123,229), DERIVED (this audit).

### 2.5 Synthetic data

- **M10.4 resolution core**: 1,500 rows, 300 per archetype across 5 archetypes. Generated programmatically by `data/build_m10_4_resolution_dataset.py`, which is template/random-choice based (18 `random.choice/sample` calls). It is included in M10.4 train. The quality gates G1-G6 all passed (`sentriformer/results/m10_4_audit/M10_4_GATE_REPORT.md`, REPORT-ONLY).
- **`sentriformer/synthetic_factory/` ("SentriForge")**: a pilot in which an LLM generates and DeepSeek/ChatGPT verify. Per `forge.py`, "nothing enters training without human approval". No reference to it was found in any training data build (`wsp/*.py`, `WR/t4/data/*.json`). Treat it as **not used in training** (VERIFIED by grep; absence of evidence).

---

## 3. Development history

Legend: Accepted = became the default or base; Rejected = failed frozen criteria.

### 3.1 M-series (Sep 2026, original pipeline)

| Stage | What changed | Key result (split) | Outcome | source: |
|---|---|---|---|---|
| M5-M7 | SentriCorpus, 16k BPE tokenizer, architecture | 3.54M records; 11.45M seq2seq params | done | `multitask_report.md` §4 |
| **M8 pretraining** | T5 span corruption, 3 epochs, 149,200 steps, 14.79 h | val loss 2.4248 (best, step 140k); train 2.0898; test loss 2.3288, **test PPL 10.27** (untrained 16,445.75) | Accepted | `results/pretraining/pretraining_report.md`; `results/m9_audit/m9_1_audit_summary.json` |
| M9.2 classifier | backbone + category head, CE | **test acc 79.15%, macro-F1 0.4035, weighted-F1 0.7887** (reproduced exactly by the E0 harness) | specialist reference | `classification_results_report.md`; `WR/e0/E0_REPORT.md` |
| M9.3 dialogue SFT | seq2seq on raw TWCS/Bitext pairs | test PPL 8.94; 80-case audit: 87.5% contaminated (handles/URLs/signatures), 0% EOS | Rejected behaviourally | `dialogue_audit_report.md`; `results/m10_3_audit/M10_3_BEHAVIORAL_AUDIT.md` (REPORT-ONLY) |
| M9.4 CSAT | CSAT head | test exact 72.07%, ±1 81.49%, MAE 0.731, macro-F1 0.327 | specialist | `csat_report.md` (REPORT-ONLY) |
| M9.5 joint multitask | 1.0 cat + 0.1 csat + 0.5 dial | test cat 73.99% @512 / 73.29% @128, macro-F1 0.317; CSAT 73.56% / ±1 81.72%. **Dialogue PPL 1.04 invalid** (causal-mask leakage); greedy decoding collapsed into repetition | Accepted for heads; decoder broken | `WR/e0/E0_REPORT.md` (VERIFIED); `multitask_report.md` |
| M10.2 clean SFT | causal mask fix, explicit EOS, 615k cleaned pairs (190,475 DM deflections dropped) | val loss 2.3227 / PPL 10.20; test 2.6942 / PPL 14.79; 80-case contamination 1.2%, EOS 100% | Accepted for decoder | `walkthrough.md`; `results/m10_clean_dataset_audit.json` |
| M10.3 refined SFT | 530,388 pairs: dedup, orphan repair (30,933), templates (4,624), off-domain drop | 80-case: 0% contamination; orphaned channel phrase 15% | Accepted at the time. **Later found to break routing** | `results/m10_3_dataset_report.json`; `M10_3_BEHAVIORAL_AUDIT.md` |
| **W1 failure (found in E0)** | `train_clean_response_sft.py` froze the heads but **not the encoder/embeddings**. Generation-only training moved the shared encoder (66% relative embedding drift) | M10.3 test category acc **39.59%** (-34.4 pp vs M9.5, CI [-34.7, -34.1]); CSAT exact 66.90% | Root cause documented | `WR/WSP_FINAL_REPORT.md` §1; `WR/e0/E0_REPORT.md` |
| M10.4 data | 1,500 resolution core + gates, rebuilt dialogue set (531,888 train) | gates pass (data-level) | Used as Phase-T dialogue data | `results/m10_4_audit/M10_4_GATE_REPORT.md` |

The M10.3 PPL of 10.08 shown in `research_comparative_evaluation.md` §2 and in fig1/fig3 has no stated split and no JSON source (UNVERIFIED). Its "M10.4 pending training" row is outdated: M10.4 data was used in Phase T.

### 3.2 Weight-Space Programming, Phase 0 (E0-E3, 2026-11-07/08, no gradients)

Frozen criteria: `WR/SELECTION_CRITERIA.md`, written before fitting.

| Exp | What | Result | Outcome | source: |
|---|---|---|---|---|
| E0 | inference-only baseline of M9.2 / M9.5 / M10.3, val and test, paired bootstrap (B=1000) + McNemar | see §3.1; integrity 0 overlap; 16 files hash-unchanged | Baseline | `WR/e0/E0_REPORT.md`, `WR/e0/test/e0_metrics.json` |
| E1 | closed-form ridge map A (256x256) from M10.3 pooled features to M9.5 features (200k train subset), folded into `category_head.dense.weight` and `csat_head.dense.weight` | val cat **39.67% -> 63.94%** (λ=0 chosen); CSAT val 66.31% -> 72.20%; linear-map R² 0.674 | **Accepted** | `WR/WSP_FINAL_REPORT.md` §3; `WR/e1/acceptance.json` |
| E2 | gradient-free prior matching (bias) | cat +1.6 pp acc but 8 important classes lost >2 pp | **Rejected** (safeguard) | `WR/e2/e2_report.json`; report §3 |
| E3 | controls: random same-norm edit 21.91%; shuffled-pair fit 27.38% (val) | E1 beats both | E1 supported | report §3 |
| Phase-0 test (once) | E1 model | cat 63.74% @512, macro-F1 0.205, ECE 0.069; CSAT 73.91% / ±1 82.30% / MAE 0.717 / macro-F1 0.311 | Deployed then (`WR/FINAL/`) | `WR/e1/final_test/e1_metrics.json`; `WR/MODEL_CARD.md` |

### 3.3 Phase T: gradient multitask training (T1-T3)

Frozen criteria: `WR/train/TRAINING_CRITERIA.md`.

| Run | Change | Result | Outcome | source: |
|---|---|---|---|---|
| T1 | init Phase-0; cat CE with class weights **+ label smoothing 0.1** | cat train loss flat ~8.6; val-subset acc 64% -> **36%** at step 1,500 | **Stopped (diverged)** | `WR/train/RUN_PLAN_T1.md`; `WR/train/T1/train_log.txt` |
| T2 | category head re-initialized, 2k-step frozen-backbone warm-up, lr_heads 1e-3, stop gate | gate FAIL at step 2,000 (val-subset cat acc 0.321 < 0.55) | **Failed gate** | `WR/train/T2/train_log.txt` |
| Root cause | `F.cross_entropy(weight=w, label_smoothing=ε)` applies the class weights to the smoothing term as well, pushing every sample toward heavily weighted rare classes. Offline probe: plain CE 65.3%; weighted+smoothing 36%; weighted without smoothing 63.8% / macro-F1 0.265 | — | design error, fixed | `RUN_PLAN_T1.md` §T2 outcome; `wsp/train_v2.py` L146 |
| **T3** | T2 without category label smoothing; 18,000 steps, 70.3 min, peak 2.23 GB VRAM; best step 16,000 | val: 11/12 frozen checks pass (failed: urgency MAE vs mean, urgency Spearman CI). TEST (once): see §4.2 | **Accepted** (`WR/FINAL_T3`) | `WR/eval_full/T3_val/acceptance.json`; `WR/train/T3/eval_history.json`; `WR/FINAL_T3/MODEL_CARD.md` |

### 3.4 T4: dialogue data fixes (validation only; test sealed)

Frozen criteria: `WR/t4/T4_CRITERIA.md`.

| Run | Change | Key val numbers (300 multi-turn val rows / full val) | Outcome | source: |
|---|---|---|---|---|
| T4a-ctrl | role-verified M10.4, single message input | portal-loop 0.280 | control | `WR/t4/T4a_DECISION.md` |
| **T4a** | + role-labelled multi-turn history (≤256 tokens) | portal-loop 0.347 (T3) -> **0.237** (CI of T4a-T3 [-0.167, -0.053]); ROUGE-L +0.0114 vs ctrl (CI [0.0012, 0.0215]); history-row loss 2.809 -> 2.720; cat 73.84% | **Accepted** (base for T4d). T4a vs ctrl portal-loop CI includes 0 | `T4a_DECISION.md` |
| T4d-ctrl | T4a + 10k steps on unfiltered data | portal-loop 0.213 | control | `WR/t4/T4d_DECISION.md` |
| **T4d** | T4a + 10k steps on T4d-filtered data; best step 8,000 | portal-loop **0.003** (vs ctrl -0.210, CI [-0.257, -0.163]); redirect 0.010; cat 74.03% / macro-F1 0.3858; CSAT macro-F1 0.3361; sentiment 0.7630 | **Accepted; sealed test run once** | `T4d_DECISION.md`; `WR/eval_full/T4d_val/metrics.json` |
| Decoding tune | greedy rp 1.0 / 1.2 / 1.4 vs sampling (T 0.7 k50; T 0.5 k20) | default **greedy, repetition penalty 1.2** kept | — | `WR/t4/decode/T4d.json` |

### 3.5 Weight-space steering probes on T4d (P1, P2)

| Exp | What | Result (250 Stage-A val rows) | Outcome | source: |
|---|---|---|---|---|
| P1 | scale 4 decoder layer-3 FFN neurons {502, 619, 509, 743} by α ∈ {0.75, 0.5, 0.25} | generic rate 0.236 -> 0.244 / 0.236 / 0.232; no CI excludes 0; within the spread of 5 matched random controls | **Rejected** (correlational, not causal) | `WR/sentriwsp01/P1_CRITERIA.md`, `P1_DECISION.md` |
| P2 | add a "generic direction" to a LayerNorm bias: arm F = `decoder.final_norm.bias`, arm N = `layers.3.norm3.bias` | F α=1: -12 pp generic but matched/beaten by random directions (up to -14.8 pp); action words 0.384 -> 0.304; dialogue loss +0.28. N α=1: -6 pp, CI upper bound +0.004 | **Rejected** | `WR/sentriwsp01/P2_DECISION.md`, `WSP_P2_DESIGN.md` |

### 3.6 T5-T7 (validation rows only; test sealed)

| Exp | What | Result | Outcome | source: |
|---|---|---|---|---|
| **T5a** (test 1, 500 fresh val rows, rs=79) | T4d + 10k steps on the T5a-filtered train set; best step **2,000** | broken-link 3.0% -> **0.8%** (vs ctrl -2.2 pp, CI [-3.8, -0.8]); generic phrase 26.8% -> 17.2%; whole-reply generic 16.8% -> 7.8%; **action-word rate -2.2 pp (limit -2.0 pp; CI [-5.2, +0.6])** | **Rejected by a single point-threshold gate** | `WR/t5/T5_DECISION.md`, `WR/t5/t5_compare.json` |
| **T5a replication** (pre-registered after the failure; 500 new rows, rs=113, same thresholds) | same checkpoints | broken-link **5.8% -> 0.4%** (vs ctrl CI [-6.2, -2.6] pp); generic phrase 25.8% -> **16.6%** (CI [-12.6, -5.8] pp); action 39.6% -> 40.2%; all G2/G3 gates pass | **PASS -> deployed** | `WR/t5/T5A_RETEST_CRITERIA.md`, `WR/t5/t5a_retest.json` |
| T5c | MMI reranking of 8 candidates, λ ∈ {0.15, 0.3, 0.5} | a random sampled pick beat MMI (generic 22.4% greedy -> 8.4% random); MMI λ=0.5 raised broken-link | **Rejected** (Stage A); no `decoding.json` | `WR/t5/T5C_DECISION.md` |
| T6 | PMI filter (drop replies with PMI < 0 vs a same-company conversation) | judge-PMI vs ctrl +0.003 (CI [-0.012, +0.019]); action 39.8% -> 35.4%; whole-reply generic +3.2 pp vs ctrl | **Rejected** | `WR/t6/T6_DECISION.md`, `WR/t6/t6_compare.json` |
| T7u | refit only `urgency_proj` on T5a features | MAE 0.752 vs constant-mean 0.671 (worse; CI [-0.153, -0.014]); macro-F1 0.313 vs T5a 0.317; logistic probe macro-F1 0.347 | **Rejected**: urgency not readable from the representation | `WR/t7u/T7U_DECISION.md`, `WR/t7u/t7u_results.json` |

---

## 4. Metrics

### 4.1 Validation, full `eval_full` harness (VERIFIED)

Columns: categories evaluated at 512 tokens (n=122,961); CSAT n=849; sentiment n=20,721; urgency n=200; dialogue loss over 31,470 M10.4 val pairs; generation over 300 fixed contexts with greedy decoding and repetition penalty 1.2, single-turn, 128-token context. source: `WR/eval_full/<run>/metrics.json`

| Run | Cat acc | Cat macro-F1 | Cat ECE | Cat acc @128 | CSAT exact | CSAT ±1 | CSAT macro-F1 | Sent macro-F1 | Sent macro-F1 (human GoEmo) | Urg macro-F1 | Urg MAE (mean-baseline 0.671) | Urg Spearman | Dial loss | ROUGE-L | distinct-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase-0 | 63.94% | 0.209 | 0.066 | 63.45% | 72.20% | 80.21% | 0.302 | 0.265 (untrained) | 0.181 | 0.206 | 0.804 | -0.003 | 2.984 | 0.130 | 0.214 |
| T3 | 72.71% | 0.371 | 0.054 | 71.11% | 70.91% | 80.80% | 0.318 | 0.743 | 0.539 | 0.261 | 0.735 | 0.152 | 2.782 | 0.141 | 0.199 |
| T4a | 73.84% | 0.385 | 0.055 | 72.20% | 70.79% | 80.45% | 0.328 | 0.758 | 0.557 | 0.307 | 0.753 | 0.187 | 2.765 | 0.137 | 0.203 |
| T4d | 74.03% | 0.386 | 0.051 | 72.46% | 70.44% | 80.21% | 0.336 | 0.763 | 0.562 | 0.304 | 0.766 | 0.195 | 2.770 | 0.131 | 0.240 |
| T5a-ctrl | 74.22% | 0.387 | 0.051 | 72.71% | 69.61% | 79.62% | 0.333 | 0.764 | 0.563 | 0.313 | 0.767 | 0.206 | 2.764 | 0.135 | 0.259 |
| **T5a (deployed)** | **73.37%** | **0.377** | 0.055 | **71.79%** | 70.20% | 80.45% | 0.330 | 0.756 | 0.572 | 0.317 | 0.771 | 0.209 | 2.780 | 0.136 | 0.250 |
| T6 (rejected) | 74.02% | 0.385 | 0.055 | 72.32% | 69.73% | 79.03% | 0.327 | 0.761 | 0.553 | 0.315 | 0.750 | 0.233 | 2.778 | 0.134 | 0.259 |

The T5a val per-class F1 on CSAT is: 1★ 0.701, 2★ 0.000, 3★ 0.038, 4★ 0.095, 5★ 0.813. The model predicted 2★ once (`WR/eval_full/T5a_val/metrics.json`). 12 of 33 categories have F1 = 0 (same file).

### 4.2 Sealed TEST results (each run once) (VERIFIED)

source: `WR/eval_full/T3_test/metrics.json`, `WR/eval_full/T4d_test/metrics.json`, `WR/e0/test/e0_metrics.json`, `WR/e1/final_test/e1_metrics.json`, `WR/t4/eval/{T3,T4d}_test/metrics.json`.

| Metric (test) | M9.2 specialist | M9.5 | M10.3 | Phase-0 (E1) | T3 | **T4d (last tested)** | Baseline |
|---|---|---|---|---|---|---|---|
| Category acc @512 (n=123,229) | 79.15% | 73.99% | 39.59% | 63.74% | 72.42% | **73.71%** | majority 27.2% (DERIVED) |
| Category acc @128 | 76.90% | 73.29% | 42.27% | 63.35% | 70.98% | 72.23% | — |
| Category macro-F1 / weighted-F1 | 0.404 / 0.789 | 0.317 / 0.734 | 0.078 / 0.388 | 0.205 / 0.618 | 0.353 / 0.725 | **0.369 / 0.737** | — |
| Category ECE (15-bin) / NLL | 0.051 / 0.605 | 0.048 / 0.740 | 0.165 / 2.020 | 0.069 / — | 0.058 / 0.791 | 0.055 / 0.745 | — |
| Category conf < 0.25 (triage flag) | 0.16% | 0.37% | 63.4% | 2.2% | 0.71% | 0.50% | — |
| CSAT exact / ±1 / MAE (n=870) | untrained head | 73.56% / 81.72% / 0.732 | 66.90% / 73.68% / 1.030 | 73.91% / 82.30% / 0.717 | 73.33% / 82.18% / 0.714 | 72.41% / 81.95% / 0.708 | — |
| CSAT macro-F1 | — | 0.317 | 0.278 | 0.311 | 0.361 | 0.352 | — |
| Sentiment acc / macro-F1 (n=20,453) | untrained | untrained | untrained | untrained | 87.2% / 0.742 | **88.1% / 0.758** | majority acc 79.6% |
| Sentiment, human GoEmotions subset (n=3,012) acc / macro-F1 | — | — | — | — | 61.6% / 0.531 | 63.7% / 0.552 | — |
| Sentiment, distilled subset (n=17,441) acc / macro-F1 | — | — | — | — | 91.6% / 0.538 | 92.3% / 0.548 | — |
| Urgency macro-F1 / MAE / Spearman (n=202) | — | — | — | — | 0.367 / 0.649 / 0.279 | 0.376 / 0.691 / 0.245 | majority macro-F1 0.148; mean-pred MAE 0.675 |
| Dialogue loss / PPL (31,912 M10.4 test pairs) | — | invalid (1.04 PPL, leak) | — | 2.98 (REPORT-ONLY) | 2.782 / 16.14 | 2.769 / 15.94 | — |
| Single-turn generation (300 rows): ROUGE-L / distinct-1 / distinct-2 / mean words | — | — | — | — | 0.135 / 0.093 / 0.209 / 18.5 | 0.132 / 0.103 / 0.235 / 17.8 | — |
| EOS / repetition / placeholder / invented URL or contact | — | — | — | — | 1.0 / 0 / 1.7% / 0 | 1.0 / 0 / 1.0% / 0 | — |

**Multi-turn test (300 rows with history, 28,907 multi_test rows)** (source: `WR/t4/eval/T3_test/metrics.json`, `WR/t4/eval/T4d_test/metrics.json`):

| Metric | T3 (raw input) | T4d (multi-turn input) |
|---|---|---|
| Portal-loop reply rate | 31.3% | **0.7%** (CI of diff [-36.0, -25.7] pp: REPORT-ONLY in `WR/FINAL_T4d/MODEL_CARD.md`; no JSON found) |
| Any redirect / orphan | 32.3% / 3.0% | 1.0% / 0.3% |
| ROUGE-L / distinct-2 / mean words | 0.116 / 0.149 / 17.6 | 0.121 / 0.213 / 16.0 |
| Loss on rows with history (7,246) | 2.880 | 2.746 |
| Re-ask rate / acknowledgement proxy (n=203) | 0.0% / 5.4% | 0.0% / 24.6% |
| Reference portal-loop rate (human agents) | 15% | 15% |

The T4d test per-class report and the 33x33 confusion matrix are stored under `category_len512.per_class` and `confusion_matrix` in `WR/eval_full/T4d_test/metrics.json`, together with reliability bins. Examples of per-class F1: Mortgage 0.912, Student loan 0.844, Checking/savings 0.803, Debt collection 0.781; 12 classes have F1 = 0. CSAT per-class F1 (T4d test): 1★ 0.720, 2★ 0.000, 3★ 0.057, 4★ 0.163, 5★ 0.822. Sentiment per-class F1: neg 0.940, neu 0.630, pos 0.705. The E0 report has per-class tables and confusion matrices for M9.2/M9.5/M10.3 (`WR/e0/E0_REPORT.md`).

Only E0 comparisons carry test confidence intervals (`E0_REPORT.md`, paired bootstrap B=1000):
- M9.5 - M9.2: -5.15 pp, CI [-5.37, -4.96]
- M10.3 - M9.5: -34.41 pp, CI [-34.71, -34.10]
- Phase-0 vs M9.5 CSAT: +0.34 pp, CI [-1.7, +2.3] (`WSP_FINAL_REPORT.md` §3)

**No CI was computed for T4d-vs-T3 test category, CSAT or sentiment** (UNVERIFIED as significant).

### 4.3 Deployed-model (T5a) reply-quality evidence (validation, regex detectors)

Detectors: `WR/t5/t5_audit_v5.py`, regex-based and not human-validated (`t5a_build_report.json`: `human_precision_audit NOT done`).

| Metric | T4d (test 1 / repl.) | T5a-ctrl (test 1 / repl.) | **T5a (test 1 / repl.)** | source: |
|---|---|---|---|---|
| Broken-link reply | 3.0% / 5.8% | 3.0% / 4.8% | **0.8% / 0.4%** | `WR/t5/t5_compare.json`; `WR/t5/t5a_retest.json` |
| Generic phrase anywhere | 26.8% / 25.8% | 28.4% / 29.8% | **17.2% / 16.6%** | same |
| Whole-reply generic | 16.8% / 14.8% | 16.2% / 17.0% | **7.8% / 7.4%** | same |
| Action-word rate | 41.4% / 39.6% | 41.0% / 38.8% | 39.2% / 40.2% | same |
| ROUGE-L (non-portal refs) | 0.128 / 0.131 | 0.129 / 0.138 | 0.126 / 0.135 | same |
| Distinct replies | 0.770 / 0.778 | 0.742 / 0.762 | 0.772 / 0.780 | same |
| Portal loop (repl.) | 1.8% | 1.4% | 1.0% | `t5a_retest.json` |
| Dialogue loss (repl.) | 2.692 | 2.682 | 2.703 | same |

Behaviour suite (24 hand-written complaints, qualitative): T5a portal 2, broken 2, generic 7, routing ok 19 (T4d: 2/3/8/20). source: `WR/FINAL_T5a/MODEL_CARD.md`; `WR/FINAL_T5a/behavior_suite/behavior_suite.json`. Regression tests all pass, including strict load (123 tensors), 20/20 deterministic generation, and robustness to 781-token/empty/unicode input (`WR/FINAL_T5a/regression_tests.json`).

### 4.4 Baselines that actually exist

| Baseline | Exists? | Value | source: |
|---|---|---|---|
| Majority class: sentiment | yes | test acc 79.6% | `WR/eval_full/T4d_test/metrics.json` `sentiment.majority_baseline` |
| Majority / mean: urgency | yes | macro-F1 0.148; mean-prediction MAE 0.675 (test), 0.671 (val) | same; `WR/t7u/t7u_results.json` |
| Majority class: category | not in files | 27.2% (DERIVED from class counts) | `classification_test.parquet` |
| Single-task specialists M9.2 / M9.4 | yes | see §4.2 / §3.1 | `WR/e0/E0_REPORT.md`; `csat_report.md` |
| Untrained heads (Phase-0) | yes | §4.1 row Phase-0 | `WR/eval_full/phase0_val/metrics.json` |
| Matched controls (ctrl runs, random edits) | yes | §3 | decision files |
| Human reference rate (portal-loop in agent replies) | yes | 15% | `WR/t4/eval/T4d_test/metrics.json` |
| **TF-IDF + LR, DistilBERT/BERT/T5-small fine-tuned classifiers** | **NO** | — | none found |
| BLEU, BERTScore, METEOR | **NO** | only ROUGE-L (own word-level LCS F1, `wsp/eval_full.py` L33-47) and distinct-n | — |
| Efficiency table vs T5-Small/DistilBERT/BERT/BART/LLaMA-3-8B (latency, VRAM, throughput) | **UNVERIFIED, likely not measured**: values are hard-coded literals in the plotting script, with no benchmark output file. The "2,811 samples/s" figure originates from a CSAT-training throughput note | — | `sentriformer/generate_research_progression_plots.py` L279-284; `research_comparative_evaluation.md` §6; `dialogue_audit_report.md` §4 |
| Department-level routing table (e.g. "Mortgage 46,210 test samples") | **UNVERIFIED and inconsistent**: actual Mortgage test support is 7,320, and the department supports sum to more than 123,229 | — | `generate_research_progression_plots.py` L332-333 vs `classification_test.parquet` |

---

## 5. Hyperparameters and compute

| Stage | Optimizer / LR | Schedule | Batch | Steps / epochs | Precision / clip | Time | source: |
|---|---|---|---|---|---|---|---|
| M8 pretrain | AdamW 5e-4, β (0.9, 0.98), eps 1e-8, wd 0.01 | cosine, 2,000 warm-up | 64 | 149,200 (3 epochs) | fp16, clip 1.0 | 14.79 h | `sentriformer/training/run_pretrain.py` L45-55; `pretraining_report.md` |
| M9.2 classifier | lr 2e-4 (default) | cosine | 64 | 3 epochs | — | — | `training/train_classification.py` L118-120 (defaults; actual CLI args UNVERIFIED) |
| M9.3 dialogue | lr 1e-4 | cosine | 32 | 3 epochs | — | — | `train_dialogue_sft.py` L153-155 |
| M9.4 CSAT | lr 2e-4 | cosine | 64 | — | — | — | `train_csat.py` L171-172 |
| M9.5 joint | lr 5e-5 | cosine | — | 2 epochs | — | — | `train_multitask_joint.py` L179-180 |
| M10.2/M10.3 SFT | lr 5e-5, heads frozen (encoder NOT frozen) | cosine | 32 | 3 epochs | — | — | `train_clean_response_sft.py` L106-108, L169-171 |
| **Phase T (T3)** | AdamW, wd 0.01, β (0.9, 0.98); backbone 3e-5, heads 1e-3 | cosine, warm-up min(500, steps/10); 2,000-step head warm-up with backbone frozen | cat 48 / sent 32 / dial 32 / csat 16 (every 3rd step) / urg 8 (every 4th) | 18,000; category subset 500k (fixed seed 0) | fp16 autocast + GradScaler, clip 1.0 | 70.3 min | `wsp/train_v2.py` L208-250; `WR/train/T3/config.json`, `eval_history.json` |
| T4a / T4d / T5a / T6 (+ctrl) | as T3, but heads 3e-4, no head warm-up | same | same; dialogue context 256 | 18k (T4a) / 10k (T4d, T5a, T6) | same | 27.6-61.6 min per run (T4a value may reflect a resume) | `WR/train/*/config.json`, `eval_history.json` |
| Model selection | `score = cat_acc + csat_mF1 + sent_mF1 + urg_mF1 - 0.1 x dial_val_loss` on val subsets, every 1,000 steps | | | best steps: T3 16k, T4a 18k, T4d 8k, **T5a 2k** | | | `WR/train/TRAINING_CRITERIA.md`; `eval_history.json` |
| Hardware | NVIDIA RTX 3060 Laptop 6 GB, 16 GB RAM; Phase-T peak VRAM 2.14-2.46 GB; about 0.19-0.23 s/step | | | | | | `WR/WSP_FINAL_REPORT.md`; `WR/train/*/train_log.txt` |
| Seeds | seed 0 for all Phase-T runs; **single seed only** (no seed variance reported) | | | | | | configs |

---

## 6. Inference pipeline and deployment

- **`sentriformer/wsp/inference.py`** (`SentriFormerPipeline.analyze`) loads the checkpoint strictly.
- **Routing**: encodes the message, truncated to 128 tokens, and returns top-1 and top-3 categories with confidence. `needs_human_triage` is set when top-1 < 0.25 (L27-28, L97-99).
- **Other heads**: CSAT stars and probabilities; sentiment label and probabilities. Urgency is returned only as `urgency_experimental` with a do-not-use warning (L57-63).
- **Reply generation**: input is `"Customer: … Agent: … Customer: <msg>"` with history passed as `history=[(role, text)]`, keeping the last 256 tokens (L66-70).
- **Decoding**: greedy (temperature 0), **repetition penalty 1.2**, BOS/PAD logits suppressed, max 100 new tokens, stop at EOS (`evaluation/audit_m10_3_threeway.py` `generate_multitask` L333-370). MMI is used only if a `decoding.json` exists. **None exists for T5a** because T5c was rejected (`WR/FINAL_T5a/` listing; `WR/t5/T5C_DECISION.md`).
- **Quality flags**: EOS, repetition, template placeholder, orphaned channel phrase, invented URL, invented contact, unsupported name (L103-106).
- **Application integration**: the SentriMail FastAPI app (`Sentrimail/sentrimail/app/ai_engine.py` L243-256, L474-560) imports `wsp/inference.py`.
  - When SentriFormer is available, it uses SentriFormer for the **sentiment** label.
  - It exposes routing, CSAT and a draft reply **to admins only, never auto-sent**.
  - The urgency head is not used.
  - It falls back to DistilBERT-SST2 / flan-t5-small / rule-based logic otherwise.
  - Item 8 of `WR/WSP_FINAL_REPORT.md` §2 ("app does not use SentriFormer") is therefore **outdated**.
- **Rollback**: `WR/FINAL_T4d/sentriformer_t4d.pt`.

---

## 7. Existing figures

No PNG or SVG exists under `WR/`. **None of the existing figures reflects the Phase-T / T4d / T5a results**, so all paper figures for the final model must be newly generated from `WR/eval_full/*/metrics.json`, `WR/t4/eval/*`, `WR/t5/*.json` and `WR/train/*/eval_history.json`.

| File(s) | Shows | Status |
|---|---|---|
| `sentriformer/results/pretraining/01-08_*.png` | M8 loss curves (train, val, train-vs-val), LR schedule, gradient norm, per-epoch, dashboard | Valid for pretraining (from `checkpoints/training_log.jsonl`) |
| `sentriformer/results/classification/01-05_*.png` | M9.2 per-class F1 ranking, support vs F1, precision/recall, domain routing, dashboard | Outdated (M9.2 specialist, not deployed); domain grouping UNVERIFIED |
| `sentriformer/results/csat/01-03_*.png` | M9.4 CSAT distribution, per-class F1, dashboard | Outdated (M9.4) |
| `sentriformer/results/dialogue/01-04_*.png` | M9.3 perplexity progression, contamination breakdown, category quality | Outdated (M9.3; 30-case audit) |
| `sentriformer/results/multitask/01-03_*.png` | M9.5 vs specialists, contamination, dashboard | Outdated; includes the invalid PPL 1.04 |
| `sentriformer/results/clean_sft/01-03_*.png` | M10.2 contamination reduction, behaviour pillars, category confidence/CSAT | Outdated (M10.2) |
| `sentriformer/results/research_plots/fig1_cross_model_progression_dashboard.png` | PPL by milestone, single-task vs M9.5, contamination, EOS/fluency | Hard-coded values; mixes val and test PPL; ends at M10.3. Outdated |
| `.../fig2_behavioral_evolution_radar.png` | M9.3/M10.2/M10.3 80-case audit radar | Hard-coded; outdated |
| `.../fig3_tradeoff_perplexity_vs_contamination.png` | PPL vs cleanliness | Hard-coded; includes the invalid 1.04 |
| `.../fig4_efficiency_pareto_frontier.png` | params vs latency against T5-small/BERT/LLaMA | **UNVERIFIED (not measured); do not use** |
| `.../fig5_department_routing_matrix.png` | department F1 / support | **Inconsistent with the data; do not use** |
| `rawdata/data_details/*.png` and `processing/*.png` (10 each, duplicated) | raw dataset audits (CFPB product/length, TWCS quality, CSAT vs length, composition, pipeline diagram) | Pre-corpus EDA (2026-09-01). Check visually for embedded text samples before publishing (not inspected) |
| `Sentrimail/mermaid-diagram.svg`, `Sentrimail/sentrimail/working_images/*.png` | web-app architecture and UI screenshots | App, not model; screenshots may contain complaint text (not inspected) |

---

## 8. Open gaps (not measured or not established)

1. **No sealed-test evaluation of the deployed T5a** (§0).
2. **No human evaluation.**
   - `WR/ab_test/AB_SHEET.csv` (100 items, T3 vs T4d) was rated only by ChatGPT, with the key visible (not blind): T4d 60 / T3 28 / tie 12 (`AB_RESULT_chatgpt_unblinded.json`).
   - The T5a vs T4d sheet has 50 LLM model labels (A 6 / B 12 / tie 32). The key is sealed (`WR/sentriwsp01/_sealed_keys/AB_T5a_KEY.json`) and was not unblinded here. These are not human ratings.
   - The annotation kit exists (`WR/annotation_kit/`) but no agreement results were found.
3. **Urgency is not learned.** It has 932 training labels of unknown provenance and failed val (T3) and T7u. **Escalation is untrained.**
4. **Reply helpfulness and factual grounding are not measured.** The only measures are ROUGE-L (about 0.13), distinct-n, and regex detectors (generic, broken-link, portal-loop, action, privacy) whose precision was never human-audited. There is no BLEU, BERTScore or LLM-judge with agreement data.
5. **No CI or significance test** for the T4d-vs-T3 test differences in category, CSAT and sentiment. There is also a **single seed**.
6. **No classical or pretrained-transformer baselines** were trained on the same splits (TF-IDF+LR, DistilBERT, T5-small). The efficiency comparison is unmeasured.
7. **Label-quality limits.**
   - Sentiment labels are about 92% teacher pseudo-labels (the "distilled" subset), with neutral coming mostly from GoEmotions. The human-subset macro-F1 is only about 0.55.
   - CSAT is trained on e-commerce remarks but applied to CFPB complaints; 2★ is effectively never predicted, and 3★ F1 is below 0.06.
   - The 33-class taxonomy has CFPB-era duplicates; 12 classes have F1 = 0 on test.
8. **Near-duplicate leakage** was not checked (exact-hash only).
9. **Routing-length mismatch**: deployment uses 128 tokens; headline metrics are at 512.
10. Deduplication across splits relies on group hashing. TWCS non-commercial license (CC BY-NC-SA) restricts commercial use.
11. Inconsistent historical numbers that should **not** be cited:
    - M9.5 PPL 1.04 (leak)
    - README 11,651,614 parameters (21-class head)
    - `research_comparative_evaluation.md` efficiency table and department table
    - M10.3 PPL 10.08 (no split, no JSON)
    - the claim in the M10.2 report that the encoder was frozen (code shows otherwise)

---

## 9. Files that must NOT be published

### 9.1 Contain corpus text (TWCS/CFPB/e-commerce/Bitext/GoEmotions) or contexts derived from it

- `rawdata/` (entire folder: `complaints.csv` 9.18 GB, `twcs.csv` 516 MB, `archive (1)/` and its zip)
- `model_training/data/` (raw CSVs, `processed/*.parquet`, `emotion_supplement.parquet`, `validation_ecommerce.parquet`)
- `sentriformer/data/` (`train/val/test.jsonl` and `.parquet`, `supervised/*.parquet`)
- `sentriformer/wsp_runs/data/*.parquet` (sentiment, urgency)
- `sentriformer/wsp_runs/t4/data/*.parquet`; `sentriformer/wsp_runs/t6/t6_pmi_scores.parquet`, `t6_pmi_SMOKE.parquet`
- Every file matching `*LOCAL_ONLY*`:
  - `WR/t5/AB_T5a_vs_T4d_LOCAL_ONLY.csv`
  - `WR/t5/AB_T5a_ratings_LLM_LABELS_LOCAL_ONLY.json`
  - `WR/t5/t5_compare_replies_LOCAL_ONLY.json`
  - `WR/t5/t5_precision_sample_v5_LOCAL_ONLY.csv`
  - `WR/t6/t6_compare_replies_LOCAL_ONLY.json`
- `WR/t5/t5_precision_sample_v2.csv`, `_v3.csv`, `_v4.csv`, `t5_precision_labels_v4_chatgpt.csv`, `t5_dryrun_audit*.json`, `t5_audit_v5.json`, `t5c_results/`
- `WR/ab_test/AB_SHEET.csv`, `AB_RESULT_chatgpt_unblinded.json` (decisions may quote text), `AB_KEY.json`; `WR/sentriwsp01/_sealed_keys/`
- `WR/eval_full/*/generations.json`; `WR/t4/eval/*/generations.json`, `*/context_correctness.json`; `WR/t4/decode/T4d.json` (`samples`); `WR/eval_full/orphan_recount.txt` (model outputs)
- `WR/sentriwsp01/phase1_results.json`, `phase1b_results.json`, `p1_results/`, `p2_results/`, `p2_selftest.json`, `PHASE1_REPORT.md`, `WSP_P2_DESIGN.md`; `WR/t5/T5_DESIGN.md` (contain quoted rows; review first)
- `WR/t7u/T7U_DECISION.md` (quotes two urgency dataset rows; redact before publishing)
- `sentriformer/results/m10_3_audit/*.txt`, `*.json`; `results/m10_4_audit/M10_4_MANUAL_REVIEW_100.md`; `results/*/*_audit.json`, `results/*/*AUDIT*.md`, `results/m10_3_audit/m10_3_1_classifier_attribution.json`
- `WR/e1/feats/` and `WR/e1/*.npy` (features derived from corpus; no text, but not needed)
- `sentriformer/synthetic_factory/` (synthetic, but includes externally reviewed exports and a 200k Q&A import of unknown origin; review first)
- `WR/FINAL*/behavior_suite/` (hand-written inputs plus model replies; not corpus, review first)
- `Sentrimail/sentrimail/working_images/` (UI screenshots may show complaints)

### 9.2 Checkpoints (`.pt`, sizes in bytes)

- **Pretraining** (`sentriformer/checkpoints/`): `checkpoint_best.pt` 137,676,938; `checkpoint_final.pt` 137,688,386; `checkpoint_step_{10000..140000}.pt` (14 files) about 137.5-137.7 MB each; `sentriformer_base_weights.pt` 45,838,922
- **M-series**:
  - `checkpoints/classification/classifier_best.pt` 105,275,550
  - `checkpoints/csat/csat_best.pt` 105,203,898
  - `checkpoints/dialogue/dialogue_sft_best.pt` 137,512,506
  - `checkpoints/multitask/multitask_best.pt` 139,476,835
  - `checkpoints/clean_sft/clean_sft_best.pt` 138,345,033
  - `checkpoints/m10_3_sft/m10_3_sft_best.pt` 138,345,033
- **WSP releases**:
  - `WR/FINAL/sentriformer_final.pt` 95,812,721
  - `WR/FINAL_T3/sentriformer_t3.pt` 46,670,626
  - `WR/FINAL_T4d/sentriformer_t4d.pt` 46,671,478
  - **`WR/FINAL_T5a/sentriformer_t5a.pt` 46,671,478 (deployed)**
  - `WR/e1/m10_3_sft_wsp_candidate.pt` 95,813,292; `WR/e1/m10_3_sft_wsp_e1only_candidate.pt` 95,813,797
- **Phase-T runs** (`WR/train/{T1,T2,T3,T4a,T4a_ctrl,T4d,T4d_ctrl,T5a,T5a_ctrl,T6,T6_ctrl}/`):
  - `model_best.pt` 46,671,478 each
  - `model_last.pt` 46,671,094 each (not present for T1)
  - `state_last.pt` about 140.0 MB each (T2: 48,341,870)
- **Rejected patches**:
  - `WR/sentriwsp01/patches/P1_a{0.25,0.5,0.75}.pt` about 46.67 MB each
  - `WR/sentriwsp01/patches_p2/P2{F,N}_a{0.25,0.5,1.0}.pt` about 46.67 MB each
- **Earlier T5-small pipeline**: `model_training/checkpoints/best_model.pt` 726,362,850 and `latest_checkpoint.pt` 726,366,714

Weights were trained on TWCS (CC BY-NC-SA). If any checkpoint is released, use a non-commercial license and host it outside git; these files exceed GitHub's 100 MB limit or come close to it.

**Safe to publish**: code (`sentriformer/model/`, `tokenizer/*.py`, `corpus/*.py`, `training/`, `wsp/*.py`), tokenizer JSON, `*_CRITERIA.md`, `*_FREEZE.txt`, most `*_DECISION.md` (check T7U), `metrics.json` (numbers plus label names only), `eval_history.json`, `train_log.txt`, `config.json`, and the `*_build_report.json` files (counts only).
