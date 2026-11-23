# WSP-P2 criteria — FROZEN (revision 4, approved 2026-11-08, before any P2 code was written or run). Do not edit; SHA-256 recorded in P2_FREEZE.txt.
Revision 2: P0c (alignment self-test), exact split-half assignment, required arm-F output-space control matching, exact length
definition and a separate empty-reply gate.
Revision 3: per-arm norm precedence (no conflicting norm rule); Stage A = screening and selection, Stage B = confirmatory; pinned
GENERIC pattern hash; `skip_special_tokens=True` everywhere.
Revision 4: decode options fully pinned and tokenizer hashed; P0b adds arm-N hidden-norm matching; heading updated.

## Hypothesis
Subtracting a fraction α of the mean generic-vs-other activation gap d, written into a LayerNorm bias of T4d's decoder, reduces
generic replies more than matched random or shuffled-label directions (matched controls under each arm's pre-registered norm-matching rule),
with no quality regression.

## Arms and edits (exact)
- **Arm F:** `sentriformer.decoder.final_norm.bias ← bias − α·d_F` (equivalent to logits += −α·E·d_F at every position).
- **Arm N:** `sentriformer.decoder.layers.3.norm3.bias ← bias − α·d_N`.
- α ∈ {0.25, 0.5, 1.0}. d is built only from the 400 Phase-1 rows (WSP_P2_DESIGN.md section 2). Only that one 256-value bias changes per patch.

## Preconditions (checked before evaluation; failure drops the arm, no evaluation)
P0. Split-half direction stability: cos(d_half1, d_half2) ≥ 0.8. Halves are fixed: `np.random.default_rng(2026).permutation(400)`
    applied to the stored Phase-1 `row_index` order; positions 0–199 = half 1, 200–399 = half 2. The edit uses d from all 400 rows.
P0b. The self-test passes: arm-F logit equivalence; classification bitwise invariance; the P1 gate tests; save/resume; length and
     empty-reply computation; arm-F control matching (every control within ±10% in output space, fallback path exercised); **arm-N
     control matching (every control | ‖c‖ − ‖d‖ | ≤ 1e-4·‖d‖, computed in float64)**; tokenizer-file and GENERIC-pattern hash checks,
     including refusal on a deliberate mismatch.
P0c. Alignment self-test passes (WSP_P2_DESIGN.md §2a): the activation at decoder position t is labelled by token b_t of `[BOS]+body`.
     Labels use the pinned `GENERIC` regex (pattern SHA-256 `8460e8b1e89a0fa0a2a053579b47004506ad59d415ee0170acf3d4b7e163686c`, asserted at
     start-up). Every decode, both the full reply and every prefix, uses exactly `skip_special_tokens=True, clean_up_tokenization_spaces=False`,
     with the tokenizer pinned by file hash (`tokenizer.json` 528dca6b…f6f7, `tokenizer_config.json` 1101dd9d…6a4e; transformers 5.16.1). The known-reply token/label table is printed and saved. The argmax check (`argmax logits[t] == b_t` for every t, repetition
     penalty 1.0) must hold. Replies that fail the offset-integrity checks are excluded from construction and counted.

## Controls (per arm, per α; scaled by the same α as the patch)
5 random directions orthogonal to d and 3 shuffled-label directions. Norm rule per arm: **arm F, output-space matching takes
precedence** (controls start at hidden norm ‖d‖, and the fallback may change the hidden norm to get ‖E·c‖ = ‖E·d‖); **arm N, exact
hidden-space match ‖c‖ = ‖d‖ is mandatory**.
- **Arm F (required):** every control satisfies | ‖E·c‖/‖E·d‖ − 1 | ≤ 0.10. Deterministic resampling: random control k tries seeds
  10_000·k + a; shuffled control k tries permutation seeds 20_000 + 10_000·k + a; a = 0…999; the first qualifying attempt is used.
  If none qualifies, the pre-registered fallback scales the control so that ‖E·c‖ = ‖E·d‖ exactly, and it is flagged. No unmatched
  control is ever used.
- **Arm N:** exact hidden-norm match ‖c‖ = ‖d‖ (random control k uses seed k; shuffled control k uses seed 20_000 + k).
- For every control, the seed, the attempt number, ‖c‖, ‖E·c‖ and the ratio are recorded.

## Rows
500 fresh `multi_val` rows with history, random_state = 67, excluding the gate (rs=11), decode-tune (rs=29), Phase-1 (400) and P1
(rs=53, 500) rows. A = first 250, B = last 250. The test set is never used for construction, tuning or selection.

## Stage A (candidate SCREENING and SELECTION over 6 configurations; not evidence of an effect)
An α of an arm is eligible only if ALL of the following hold vs T4d on rows A:
1. Generic rate lower, with the paired bootstrap **97.5%** CI of (P2 − T4d) entirely below 0. This is a conservative screening threshold;
   it is not claimed to correct for the six Stage A comparisons.
2. Its generic reduction is larger than **each of the 8 controls** at the same α (strictly).
3. No regressions (same tolerances as P1): action ≥ −2pp; specificity ≥ −2pp; orphan ≤ +1pp; repetition ≤ +1pp; EOS ≥ −1pp;
   portal ≤ +1pp; privacy ≤ +1pp; distinct ≥ −0.02; ROUGE-L (non-portal references) ≥ −0.005; dialogue loss ≤ +0.02.
4. No template swap: template concentration ≤ +2pp; share of the single most common reply ≤ +1pp.
5. Classification outputs (category, sentiment, urgency, CSAT) bitwise identical to T4d on all 500 rows.
6. **Length:** the mean number of generated non-special tokens per reply (EOS not counted, empty reply = 0) is within ±20% of T4d's
   mean on the same rows under the same decoding.
7. **Collapse:** the empty-reply rate (length 0) is ≤ T4d's + 1pp.
Choice: per arm, the mildest eligible α; if both arms qualify, the larger generic reduction on A. No eligible α in either arm means REJECT P2, with no Stage B.

## Stage B (the independent CONFIRMATORY test; run once)
The single configuration chosen in Stage A, against T4d and its 8 controls, on the 250 untouched B rows. Rules 1–7 must hold again (CI 97.5%).
This confirmation, not the CI level, is what protects against Stage A having selected the best of six configurations. Only a Stage B pass
counts as evidence for P2. Pass → CANDIDATE only. A blind
A/B on 50 B pairs (sheet in p1_results-style folder `p2_results/`, key in `_sealed_keys/`) is required. Not promoted; T4d stays the default.

## Reporting
Every patch records d's norm, ‖β‖, the split-half cosine, the number of excluded replies and (arm F) the 20 most-penalised tokens.
Every run reports mean length and the empty-reply rate. Results are reported as PASS, REJECT (with the failed rule),
REJECT-TEMPLATE-SWAP, REJECT-SPECIFICITY, REJECT-COLLAPSE or DROPPED-UNSTABLE.
