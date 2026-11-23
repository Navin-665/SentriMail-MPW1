# Frozen selection & acceptance criteria (written 2026-11-07 ~23:45, BEFORE any E1/E2 fitting)

Baseline under repair: `m10_3_sft_best.pt` (latest lineage checkpoint, the one the application would use).
Reference ceiling: `multitask_best.pt` (the encoder the frozen heads were trained against).

## Data usage
- Fitting (A matrices, bias offsets): `classification_train` (fixed-seed 200k subset, seed 0) and `csat_train` only.
- Candidate selection: `classification_val`, `csat_val` only.
- Test (`classification_test`, `csat_test`): evaluated exactly once, for the single selected candidate per stage, after selection. Never used for fitting or selection. No test examples are inspected.

## Candidate set (fixed in advance; nothing added after seeing val results)
- E1 category: ridge lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1} (x trace(G)/256). Select max val accuracy @512.
- E1 CSAT: map source in {csat_train fit, cls_train fit, joint fit, identity} at the selected lambda. Select max csat_val exact accuracy.
- E2 category/CSAT bias: two candidates each - (a) none, (b) iterative prior matching on the train fit subset to the train label prior (gradient-free fixed point, <=50 iters). Select by val macro F1, subject to acceptance rule below.

## Acceptance rules (val, paired vs the unedited m10_3 baseline)
1. Target metric improvement with paired-bootstrap 95% CI excluding 0.
2. Category: val accuracy @512 must improve; weighted F1 must not decrease.
3. CSAT: exact accuracy must not decrease by more than 0.5 pp AND within-1 must not decrease by more than 0.5 pp; otherwise CSAT edit is rejected (identity kept).
4. E2 bias: accepted only if macro F1 improves (CI excludes 0) AND accuracy drops by no more than 0.5 pp.
5. E3: the selected E1 edit must beat both controls (random same-norm edit, shuffled-pair fit) on val accuracy by a CI-excluding-0 margin; otherwise E1 is rejected as unsupported.
6. Edit scope: only `category_head.*` / `csat_head.*` tensors may differ; all encoder/decoder/embedding/LM tensors must be byte-identical (guarantees generation is unchanged).
7. Any rejected edit -> the original tensor is retained.

## Final test
- The selected composite candidate is evaluated once on test with the E0 harness; reported regardless of outcome.

## Addendum (written ~23:52, still BEFORE any fitting; per user safeguards)
- **CSAT drift evidence (precondition for any CSAT edit):** on `csat_train` features, the unedited M10.3 features must be
  measurably incompatible with the M9.5 features the CSAT head was trained on: identity-map R² (M10.3 -> M9.5) < 0.90
  AND E0 showed a significant CSAT drop (already true: val -5.2 pp, McNemar p=6e-5; head byte-identical across checkpoints).
  If the precondition fails, the CSAT head is left untouched regardless of val gains. Rule 3 still applies if it passes.
- **Orientation check:** the fitted map satisfies Y ≈ X @ A.T; the head computes x @ W.T, so W_new = W @ A. Before use,
  assert head(W_new, X) == head(W, X @ A.T) numerically (max abs logit diff < 1e-3) on val features.
- **Controls (rule 5, operationalised):** paired bootstrap on val @512, selected E1 minus each control: CI lower bound > 0.
- **E2 (strengthened):** fitting on train features as specified; decision on val labels. Accept only if all hold:
  macro F1 CI excludes 0 (improvement); accuracy drop <= 0.5 pp; ECE increase <= 0.01;
  no "important" class (val support >= 1% of val) loses > 2.0 pp F1, precision, or recall.
  Per-class precision/recall/F1 deltas and confusion matrices are recorded either way.
