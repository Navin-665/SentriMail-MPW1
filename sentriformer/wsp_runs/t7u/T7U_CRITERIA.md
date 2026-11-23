# T7u criteria — FROZEN 2026-11-09 before any fitting (SHA-256 in T7U_FREEZE.txt)

## Question
Does the frozen T5a model already carry an urgency signal that a re-fitted urgency output layer can read out?

## Change (only one)
Init FINAL_T5a. Re-fit ONLY `sentiment_head.urgency_proj` (Linear 256->1, sigmoid, target label/3, MSE + L2 on weight).
Inputs = the layer's own input features from T5a in eval mode, fp16 autocast as deployed. Every other parameter must be
bit-identical to T5a (asserted; otherwise the run is void). L2 strength from {0, 1e-4, 1e-3, 1e-2, 1e-1}, chosen by 5-fold CV MAE on
urgency_train ONLY (seed 0), then refit on all 932 train rows. Val untouched until the single evaluation below.

## Evaluation (urgency_val, 200 rows, scored through the full model forward exactly as wsp/eval_full.py does; paired bootstrap 2000, seed 0)
PASS iff both:
C1. Per-row absolute error (levels): CI95 of (mean-baseline error - T7u error) entirely above 0. Mean baseline = urgency_train mean.
C2. Macro F1 (rounded 0..3 levels): CI95 of (T7u - T5a) entirely above 0.
Reported, not gating: acc, within-1, escalation F1 (High|Immediate), Spearman, prediction distribution, vs majority baseline.
Diagnostic only (never deployed, no selection): 4-class logistic regression on the mean-pooled encoder state (C by train CV).

## Decision
PASS -> `wsp_runs/FINAL_T7u/sentriformer_t7u.pt`, wsp/inference.py repointed (rollback FINAL_T5a); urgency_test (202 rows) scored
ONCE and reported, no selection on it. Otherwise REJECT -> T5a stays; conclusion is that urgency needs more/better labels.
Limitation: urgency labels are 932 rows of unknown annotation agreement; no human re-validation in this run.
