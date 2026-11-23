# WSP-P2 decision (2026-11-08 ~20:40) — REJECTED at Stage A (screening); Stage B not run (frozen rule)
T4d unchanged (SHA bff705ec…, verified before and after). T4d remains the default. Patches in patches_p2/ are rejected artefacts.
P0 passed for both arms (split-half cos F 0.968, N 0.969). Classification bitwise identical throughout.

| 250 rows A | generic | CI97.5 of (P2−T4d) | max control drop | action | distinct | template | dial loss | failed rules |
|---|---|---|---|---|---|---|---|---|
| T4d | 0.264 | | | 0.384 | 0.868 | 0.312 | 2.715 | |
| F α=0.25 | 0.292 | +0.028 [−0.032, +0.088] | 0.028 | 0.356 | 0.844 | 0.364 | 2.733 | 1, 2, action, portal, distinct, rouge, template |
| F α=0.5 | 0.272 | +0.008 [−0.060, +0.076] | 0.076 | 0.300 | 0.840 | 0.412 | 2.785 | 1, 2, action, portal, distinct, dial, template |
| F α=1.0 | 0.144 | **−0.120 [−0.188, −0.052]** | **0.148** | 0.304 | 0.852 | 0.376 | 2.991 | 2, action, dial, template |
| N α=0.25 | 0.276 | +0.012 [−0.032, +0.056] | 0.040 | 0.380 | 0.848 | 0.300 | 2.717 | 1, 2, template |
| N α=0.5 | 0.264 | 0.000 [−0.056, +0.056] | 0.056 | 0.368 | 0.836 | 0.300 | 2.727 | 1, 2, distinct |
| N α=1.0 | 0.204 | −0.060 [−0.124, **+0.004**] | 0.048 | 0.372 | 0.792 | 0.324 | 2.774 | 1 (CI touches 0), distinct, dial, template |

## Interpretation
- **Arm F behaves like a word ban.** It suppresses "pass", "let" and "anything", but also "share", "check" and "send" (action-word rate 0.384 → 0.30).
  It raises dialogue loss (to 2.99 at α=1) and produces new templates. At α=1.0 its generic drop (−12pp) is matched or beaten by random
  directions of the same output norm (up to −14.8pp), so the effect is not specific to the generic direction. The generic metric falls
  under ANY large perturbation of this size, and this is exactly what the controls were there to detect.
- **Arm N at α=1.0 is the only specific-looking signal.** It gave −6pp generic, beat all 8 controls (max 4.8pp), and kept action words and specificity.
  But it is not significant (the CI upper bound is +0.004), it costs distinct replies (−7.6pp: more repeated replies) and dialogue loss (+0.059),
  and it shifts templates. Per the frozen rules it is NOT eligible, and it is not adopted. Any follow-up would be a NEW pre-registered
  hypothesis on fresh rows, explicitly labelled as derived from this screening.
- Combined with P1, static weight edits (4 neurons; one bias direction at the output or at the L3 FFN input) do not remove T4d's
  generic-reply prior without collateral damage.
