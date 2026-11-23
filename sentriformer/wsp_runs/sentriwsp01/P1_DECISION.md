# WSP-P1 decision (2026-11-08 ~18:30) — REJECTED at Stage A; Stage B not run (per frozen rule)
T4d unchanged (SHA bff705ec…, verified before and after). T4d remains the default. Patches kept only as rejected artefacts in patches/.

| 250 Stage-A rows | T4d | α=0.75 | α=0.5 | α=0.25 |
|---|---|---|---|---|
| Generic rate | 0.236 | 0.244 | 0.236 | 0.232 |
| P1−T4d, 95% CI | | +0.008 [0.000, 0.020] | 0.000 [−0.016, 0.016] | −0.004 [−0.024, 0.016] |
| Controls' generic drop (5 groups) | | −0.012…+0.004 | −0.012…+0.012 | −0.008…+0.016 |
| Failed gates | | 1 (CI), 2 (controls) | 1, 2 | 1, 2 |
All no-regression gates passed (specificity, action words, orphan, repetition, EOS, portal, privacy, distinct, template, ROUGE-L,
dialogue loss unchanged at 2.735); classification bitwise identical.

Interpretation: even scaling the four neurons to 25% changed the generic rate by ≤0.4pp, smaller than the spread of random
matched controls. The Phase-1 association (neurons fire 2.4–7.8× more on generic tokens) was CORRELATIONAL, not causal:
the behaviour is distributed/redundant, and where replies changed, one generic phrase was swapped for another
("pass your comments on to our Customer Experience team"). Hypothesis H(P1) not supported. Selection-bias caveat confirmed useful.
