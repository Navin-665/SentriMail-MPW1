# T5a criteria — FROZEN 2026-11-08 before any T5 training (SHA-256 recorded in T5_FREEZE.txt)

## Status of the evidence (stated up front)
The v5 filters have NOT passed a human precision audit. The user explicitly approved running T5a anyway (2026-11-08), so any T5a result
is labelled "filters not human-validated". The LOCAL-ONLY labelling sheet `t5_precision_sample_v5_LOCAL_ONLY.csv` remains to be labelled.

## Runs (identical except for the training data)
- **T5a:** init `wsp_runs/FINAL_T4d/sentriformer_t4d.pt`; dial_train `t5a_train.parquet` (t4d_train minus 29,483 v5-flagged rows minus 13 leaked pairs).
- **T5a-ctrl:** same init and settings; dial_train `t5a_ctrl_train.parquet` (t4d_train minus the same 13 leaked pairs only).
- Settings = T4d's: 10,000 steps, max 150 min, lr_backbone 3e-5, lr_heads 3e-4, head_warmup 0, batches cat48/sent32/dial32/csat16 (every 3)/urg8 (every 4),
  dial_ctx_len 256, eval_every 1000, n_cat_train 500k, seed 0; dial_val multi_val (selection score as T3/T4).

## Evaluation rows
500 fresh `multi_val` rows with history, random_state = 79, excluding every row used before (eval_t4 gate rs=11, decode-tune rs=29,
Phase-1 400, P1 rs=53, P2 rs=67). One pre-specified comparison per model (no selection among candidates), deployed decoding.
A further fresh set (rs=97) is reserved for any decode-time follow-up (T5c), so that it never touches the rs=79 rows.

## Primary (T5a vs T5a-ctrl, paired bootstrap 95% CI on the 500 rows)
P1. Broken-link replies (v5 BROKEN regex on generated replies) lower: CI of (T5a − ctrl) entirely below 0.
Secondary, reported but not gating: generic_whole rate, GENERIC phrase rate, privacy (secret or bulk-contact) rate, eval_t4 orphan rate.

## No-regression gates (T5a vs T4d)
G1. eval_full val: category acc ≥ −1pp, macro F1 ≥ −0.01, CSAT macro F1 ≥ −0.01, sentiment macro F1 ≥ −0.01.
G2. 500 fresh rows: portal ≤ +1pp; EOS ≥ −1pp; repetition ≤ +1pp; action-word rate ≥ −2pp; specificity ≥ −2pp; distinct ≥ −0.02;
    ROUGE-L (non-portal references) ≥ −0.005; mean length within ±20%; empty ≤ +1pp; template concentration ≤ +2pp.
G3. Teacher-forced dialogue loss on the 500 rows ≤ T4d + 0.05 (filtering is expected to raise loss slightly on unfiltered references).

## Decision
PASS (P1 and all of G1–G3) → T5a becomes the deployed default (`wsp_runs/FINAL_T5a/`, inference default repointed). T4d is kept unchanged
as the rollback, and a LOCAL-ONLY blind A/B sheet (T5a vs T4d, 50 rows) is written for human rating. Otherwise → REJECT; T4d stays the default.
The test set is not used (consumed).
