# T4d decision (2026-11-08 14:40) — T4_CRITERIA.md T4d addendum applied as frozen; validation only, test sealed

| Gate | T4a | T4d-ctrl | T4d | Result |
|---|---|---|---|---|
| 1 PRIMARY portal-loop, T4d vs T4d-ctrl | 0.237 | 0.213 | 0.003 | PASS diff -0.210 CI [-0.257, -0.163]; <= T4a |
| 2 redirect_any <= T4a | 0.270 | 0.237 | 0.010 | PASS |
| 2 re-ask <= T4a + 2pp | 0.017 | 0.013 | 0.013 | PASS |
| 3 category acc / macro F1 | 0.7384 / 0.3852 | | 0.7403 / 0.3858 | PASS |
| 3 CSAT macro F1 | 0.3278 | | 0.3361 | PASS |
| 3 sentiment macro F1 (human subset) | 0.7581 (0.5573) | | 0.7630 (0.5615) | PASS |
| 3 EOS / repetition / orphan | 1.0/0/0.013 | 1.0/0/0.017 | 1.0/0/0.007 | PASS |
| 3 ROUGE-L, non-portal-loop references (n=255) | 0.1280 | 0.1276 | 0.1249 | PASS (-0.0031 >= -0.005; CI [-0.0104, 0.0044]) |
| (info) ROUGE-L all rows | 0.1346 | 0.1347 | 0.1245 | lower: 15% of references are portal loops |

Decision: T4d passes all frozen gates -> final candidate (`wsp_runs/train/T4d/model_best.pt`, step 8000). Next: decoding
tuning (wsp/decode_tune.py, frozen rule, disjoint rows), then ONE sealed test run vs T3 test numbers, then human A/B.
Not claimed: urgency (Spearman 0.195, still fails CI gate), escalation, helpfulness (needs human A/B).

## Final candidate frozen 2026-11-08 14:50 (before any test access)
T4d model_best (SHA bff705ec…) + decoding A (greedy, repetition_penalty 1.2; decode_tune kept default, see wsp_runs/t4/decode/T4d.json).
Single test run: eval_full --split test (T4d vs existing T3_test) and eval_t4 --split test (T3 raw vs T4d multi, same 300 rows).
