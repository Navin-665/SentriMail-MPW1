# T6 criteria — FROZEN 2026-11-09 before any T6 training (SHA-256 in T6_FREEZE.txt)

## Goal and hypothesis
Replies should respond to *this* customer, not fit any conversation. Hypothesis: removing training replies that the model finds no
more likely after their own conversation than after another conversation of the SAME company makes the trained model's replies
more conversation-specific.

## Data (train split only; val/test untouched)
- Scorer: FINAL_T5a (frozen). For every t5a_train row: pmi = mean log p(reply | own context) − mean log p(reply | context of another
  conversation of the same company). Company = TWCS author of the first reply (raw twcs.csv, local); non-TWCS rows use their source.
  Script `t6_pmi.py`, output `t6_pmi_scores.parquet`.
- **T6 data:** t5a_train minus rows with pmi < 0 (rows without a same-company partner are kept). `wsp_runs/t4/data/t6_train.parquet`.
- **T6-ctrl data:** t5a_train unchanged (already leakage-audited for T5a).
- Not human-validated: smoke inspection showed low-pmi rows include language-mismatched and template replies, but also some specific
  replies the scorer does not understand. Stated as a limitation.

## Runs (identical except for the data)
Init `wsp_runs/FINAL_T5a/sentriformer_t5a.pt`; all other settings as T5a (10,000 steps, max 150 min, lr 3e-5 / heads 3e-4, head_warmup 0,
batches cat48/sent32/dial32/csat16 every 3/urg every 4, dial_ctx_len 256, eval_every 1000, n_cat_train 500k, seed 0, dial_val multi_val).

## Evaluation rows
500 fresh multi_val rows with history, random_state = 131, excluding every row used before (rs 11, 29, Phase 1, P1 53, P2 67, T5 79,
T5c 97, retest 113). One pre-specified comparison per model, deployed decoding (greedy, repetition penalty 1.2). Test set not used.

## Primary (T6 vs T6-ctrl, paired bootstrap 95% CI on the 500 rows)
P1. Judge-PMI of the generated reply is higher: CI of (T6 − ctrl) entirely above 0. Judge = FINAL_T4d (NOT the T5a data scorer, to avoid
    scoring with the selector). Judge-PMI = mean log p_T4d(reply | own context) − mean log p_T4d(reply | context of a different
    same-company multi_val conversation; partner fixed per row by seed 131, identical for all models).
Secondary, reported, not gating: lexical specificity (content word shared with the last customer turn), generic phrase, whole-reply
generic, broken link, vs both T6-ctrl and T5a.

## No-regression gates (T6 vs T5a, the deployed model) — same thresholds as T5
G1. eval_full val: category acc ≥ −1pp, category macro F1 ≥ −0.01, CSAT macro F1 ≥ −0.01, sentiment macro F1 ≥ −0.01.
G2. 500 rows: portal ≤ +1pp; EOS ≥ −1pp; repetition ≤ +1pp; action ≥ −2pp; specificity ≥ −2pp; distinct ≥ −0.02; ROUGE-L (non-portal)
    ≥ −0.005; mean length ±20%; empty ≤ +1pp; template concentration ≤ +2pp; generic phrase ≤ +2pp; broken link ≤ +1pp.
G3. Teacher-forced dialogue loss on the 500 rows ≤ T5a + 0.05.

## Decision
PASS (P1 and all G) → T6 becomes the default (`wsp_runs/FINAL_T6/`, inference repointed; rollback FINAL_T5a), and a LOCAL-ONLY blind
A/B sheet T6 vs T5a (50 rows, key sealed) is written. Otherwise REJECT → T5a stays the default. No re-runs with changed thresholds.
