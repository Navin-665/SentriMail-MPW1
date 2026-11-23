# T4a decision (2026-11-08 13:15) — applied T4_CRITERIA.md as frozen; validation only, test sealed

| Gate (vs T3 unless stated) | T3 | T4a-ctrl | T4a | Result |
|---|---|---|---|---|
| Category acc (len512) | 0.7271 | 0.7362 | 0.7384 | PASS (>= T3-1pp) |
| Category macro F1 | 0.3707 | 0.3825 | 0.3852 | PASS |
| CSAT macro F1 | 0.3183 | 0.3430 | 0.3278 | PASS |
| Sentiment macro F1 | 0.7431 | 0.7565 | 0.7581 | PASS |
| Portal-loop rate (300 rows w/ history) | 0.347 | 0.280 | 0.237 | PASS; T4a-T3 = -0.110, CI [-0.167, -0.053] |
| EOS / repetition / orphan | 1.0/0/0.030 | 1.0/0/0.013 | 1.0/0/0.013 | PASS |
| ROUGE-L | 0.1317 | 0.1232 | 0.1346 | PASS |
| T4a: loss on rows with history < ctrl | | 2.809 | 2.720 | PASS |
| T4a: re-ask lower OR ROUGE-L higher vs ctrl (CI excl. 0) | | | ROUGE-L +0.0114 CI [0.0012, 0.0215] | PASS (re-ask 1.7% vs 0.9%, 4 vs 2 of 229: not lower) |

Decision: T4a passes all frozen T4a gates -> base for T4d (pending user approval). Not claimed: urgency (Spearman CI
check still FAIL), escalation, reply helpfulness (human A/B required). Portal-loop T4a vs ctrl: -0.043, CI includes 0.
