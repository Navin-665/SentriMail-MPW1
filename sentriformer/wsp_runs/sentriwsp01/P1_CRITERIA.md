# WSP-P1 frozen criteria (written 2026-11-08, BEFORE any patch exists or any P1 evaluation is run)

## Hypothesis
Layer-3 decoder FFN neurons {502, 619, 509, 743} (found in Phase 1 from observed failures, so selection bias is possible)
drive generic, formulaic replies. Scaling their output contributions down reduces generic replies without losing specificity.

## Edit (exact)
`W = sentriformer.decoder.layers.3.ffn.fc2.weight` [256×1024]; for j in S: `W[:, j] ← α·W[:, j]`, α ∈ {0.75, 0.5, 0.25}.
This is exactly equivalent to multiplying neuron j's post-GELU activation by α. It changes 1,024 values (4 columns of 256); no optimizer, no gradients.
Decoder-only, so the category, sentiment, urgency and CSAT outputs must be bitwise identical to T4d (checked, not assumed).

## Controls
5 matched random groups of 4 layer-3 neurons, seeds 0–4. Each target neuron is paired with a random L3 neuron whose mean activation
over reply tokens (measured on the Phase-1 rows, NOT the evaluation rows) is within ±25% of the target's. Control groups exclude
the 4 targets and the 6 recurring Phase-1 neurons. Every control is scaled by the same α as the patch.

## Evaluation rows (fresh; test set never used)
500 `multi_val` rows with history, random_state=53, excluding the eval_t4 gate rows (rs=11), the decode_tune rows (rs=29) and the
400 Phase-1 rows. **Stage A** = first 250 (choose α); **Stage B** = last 250 (one confirmation of the chosen α only). B results are
never used to tune anything. Decoding is the deployed setting (greedy, repetition penalty 1.2, ≤128 new tokens).

## Metrics (per reply, then rates)
generic (Phase-1 GENERIC regex) · specificity = reply shares ≥1 content word with the current customer turn · action words
(eval_t4 ACTION regex: concrete steps or specific questions) · orphan fragment (eval_t4 rule) · repetition · EOS · portal loop ·
privacy request (verb + personal-data item, not invited by the context) · distinct-reply ratio · template concentration (share
of replies whose first 5 words are one of that model's 6 most common openings) · ROUGE-L on non-portal-loop references ·
teacher-forced dialogue loss on the references.

## Stage A rule: α is eligible only if ALL hold vs T4d on the 250 A rows
1. Generic rate lower, with the paired bootstrap 95% CI of (P1 − T4d) entirely below 0.
2. Beats the controls: P1's generic reduction is larger than **every one** of the 5 controls at the same α.
3. No regressions: action-word rate ≥ T4d−2pp; specificity ≥ T4d−2pp; orphan ≤ +1pp; repetition ≤ +1pp; EOS ≥ −1pp;
   portal loop ≤ +1pp; privacy ≤ +1pp; distinct ratio ≥ −0.02; ROUGE-L (non-portal references) ≥ −0.005; dialogue loss ≤ +0.02.
4. **No template swap:** template concentration ≤ T4d+2pp, and the patched model's single most common reply covers ≤ T4d's share +1pp.
5. Classification invariance: category, sentiment, urgency and CSAT outputs bitwise equal to T4d on all 500 rows.
Choice: the **largest eligible α** (mildest edit). If none is eligible, P1 is REJECTED and Stage B is not run.

## Stage B (once)
The chosen α vs T4d and vs the 5 controls at that α, on the 250 B rows. Rules 1–5 must hold again. Pass → P1 becomes a CANDIDATE only.
It is **not promoted**. A blind A/B on 50 B replies (P1 vs T4d) is still required, and T4d stays the default whatever the result.
Outcomes are reported as PASS, REJECT (fails a rule), or REJECT-TEMPLATE-SWAP / REJECT-SPECIFICITY where that is the reason.

## Clarifications (2026-11-08, after static review, BEFORE any run; rules unchanged)
- "the 6 recurring Phase-1 neurons" = the six recurring L3 neurons in Phase 1: n502, n619, n509, n743 (the targets) + n348, n939.
  Controls exclude exactly these six (code: EXCLUDE = TARGET | {348, 939}).
- The Stage B A/B export is blinded: random A/B labels per pair (`AB_P1_SHEET.csv`), mapping in `AB_P1_KEY.json`, opened only after rating.
- Added checks: activation tensor shape asserted ([T,1024]); model asserted in eval mode; every saved patch must reload strictly and
  equal the in-memory model parameter-for-parameter with the embedding tie intact.
