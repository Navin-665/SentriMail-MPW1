# T5a replication test — criteria FROZEN 2026-11-09 ~00:20, after T5a's narrow rejection and before any replication run
(hash in T5A_RETEST_FREEZE.txt)

## Why this test exists (disclosed)
T5a failed only G2_action (−2.2pp vs the −2.0pp limit; 95% CI [−5.2, +0.6] includes 0) and passed every other gate, with large significant
gains (generic phrase −9.6pp, broken-link −2.2pp vs T4d and vs ctrl). This is a SECOND test after a failure, so it is strictly a
replication: the same models, the same metrics and **the same unchanged thresholds**, on new rows. It is not a relaxed rule.

## Rows
500 fresh `multi_val` rows with history, random_state = 113, excluding EVERY row used before (gate rs=11, tune rs=29, Phase-1, P1 rs=53,
P2 rs=67, T5 rs=79, T5c rs=97). If fewer than 500 unused rows remain, use all of them and report n.

## Rules (all must hold; identical to T5_CRITERIA.md)
P1 broken-link lower than T5a-ctrl (95% CI below 0); G2 all ten row-level gates vs T4d with the same limits (action ≥ −2pp included);
G3 dialogue loss ≤ T4d + 0.05. G1 (eval_full) already passed on validation for the same checkpoint and is not re-run.
**Both** the original test (rs=79) and this replication must hold P1, and this replication must pass all of G2 and G3.

## Outcome
PASS → T5a is deployed (`wsp_runs/FINAL_T5a`, inference default repointed; T4d kept as the rollback; LOCAL-ONLY blind A/B written).
FAIL → T5a stays rejected and T4d stays the default. Either result is final for T5a; there is no third test.
