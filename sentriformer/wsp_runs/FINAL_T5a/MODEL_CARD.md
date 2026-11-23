# SentriFormer T5a (2026-11-09) — deployed default of wsp/inference.py (rollback: wsp_runs/FINAL_T4d)
Checkpoint `sentriformer_t5a.pt` (SHA256 4205c569b1d175ed3378…; = wsp_runs/train/T5a/model_best.pt). Lineage: T3 → T4a → T4d → **T5a**
= T4d + 10,000 steps on T4d's training data minus 29,483 v5-flagged replies (24,610 stripped-link leftovers such as "For now, try:.",
3,475 whole-reply templates, 1,427 bulk personal-detail requests, 22 authentication-secret requests) and 13 train/val/test duplicate pairs.
Decoding: greedy, repetition penalty 1.2; multi-turn input as T4d (pass `history=[(role, text), …]`).

**Evidence status:** the v5 filters were NOT validated by a human precision audit (the user approved training anyway). Promotion came from a
pre-registered replication after a narrow first-test failure (wsp_runs/t5/T5_DECISION.md, T5A_RETEST_CRITERIA.md, t5a_retest.json).

| Fresh val rows (no test used) | T4d | T5a-ctrl | **T5a** |
|---|---|---|---|
| Generic phrase, test 1 / replication | 26.8% / 25.8% | 28.4% / 29.8% | **17.2% / 16.6%** (−9.2pp, CI [−12.6, −5.8]) |
| Whole-reply generic, test 1 / replication | 16.8% / 14.8% | 16.2% / 17.0% | **7.8% / 7.4%** |
| Broken-link replies, test 1 / replication | 3.0% / 5.8% | 3.0% / 4.8% | **0.8% / 0.4%** |
| Action words (replication) | 39.6% | 38.8% | 40.2% |
| ROUGE-L non-portal (replication) | 0.131 | 0.138 | 0.135 |
| Category acc / CSAT F1 / sentiment F1 (val) | 74.0% / 0.336 / 0.763 | | 73.4% / 0.330 / 0.756 (within gates) |
Regression tests: ALL PASS. Behaviour suite (24): portal 2, broken 2, generic 7 (T4d: 2 / 3 / 8); routing ok 19 (T4d 20).

## Known limits
Replies are still short and often unhelpful: they ask for details already given ("provide us with the details of your complaint"), use
odd requests ("send me a pic of the account"), and give generic acknowledgements. Category routing is slightly lower than T4d's.
Urgency is only partly validated; escalation is untrained. Not ChatGPT-level. A human blind A/B (LOCAL-ONLY sheet
wsp_runs/t5/AB_T5a_vs_T4d_LOCAL_ONLY.csv, key sealed) is still needed before claiming the replies are *better* overall.
