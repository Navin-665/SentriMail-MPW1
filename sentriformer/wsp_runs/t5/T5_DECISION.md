# T5a decision (2026-11-09 00:15) — REJECTED by the frozen rule (one gate, by 0.2pp). T4d stays the default.
Filters (v5) were NOT human-validated (user override). 500 fresh val rows (rs=79). T4d unchanged (SHA verified).

| 500 fresh rows | T4d | T5a-ctrl | **T5a** | T5a − T4d (95% CI) |
|---|---|---|---|---|
| Broken-link replies (primary) | 3.0% | 3.0% | **0.8%** | −2.2pp [−3.6, −0.8]; vs ctrl −2.2 [−3.8, −0.8] → **P1 PASS** |
| Whole-reply generic | 16.8% | 16.2% | **7.8%** | −9.0pp [−12.4, −5.8] |
| Generic phrase anywhere | 26.8% | 28.4% | **17.2%** | −9.6pp [−13.6, −5.8] |
| Privacy request | 0.2% | 0.2% | 0.0% | −0.2 [−0.6, 0.0] |
| Portal loop | 1.0% | 1.2% | 0.6% | ✓ |
| Specificity | 20.0% | 22.6% | 21.8% | ✓ |
| Distinct replies | 0.770 | 0.742 | 0.772 | ✓ |
| ROUGE-L (non-portal) | 0.128 | 0.129 | 0.126 | ✓ (−0.002) |
| Dialogue loss | 2.632 | 2.623 | 2.640 | ✓ (+0.008) |
| **Action-word rate** | 41.4% | 41.0% | **39.2%** | −2.2pp [−5.2, +0.6] → **G2 FAIL (limit −2.0pp)** |
| eval_full G1 (cat 73.4 vs 74.0; macroF1, CSAT, sentiment) | | | | all PASS (narrowly) |

Reading: the only failed gate is action-word retention, short of the limit by 0.2pp. Its CI includes 0, so the drop is not established, but
the rule was frozen as a point threshold and is applied as written. T5a is the strongest reply-quality result so far: generic replies are
roughly halved and broken-link replies are down 73%, both significant against T4d AND against the matched control, so the effect
comes from the filter rather than extra training.

Not promoted. What would justify adoption (user decision, NOT done automatically): a human blind A/B (T5a vs T4d) on local rows, or a
pre-registered re-test of the action-word gate on fresh rows. Checkpoint kept at wsp_runs/train/T5a/model_best.pt.
