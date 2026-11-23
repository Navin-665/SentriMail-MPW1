# SentriFormer T4d (2026-11-08) — deployed default of wsp/inference.py
Checkpoint `sentriformer_t4d.pt` (SHA256 bff705ec24b99168…; = wsp_runs/train/T4d/model_best.pt, step 8000). Lineage:
T3 -> T4a (role-verified multi-turn dialogue) -> T4d (training replies that are portal rewrites / DM redirects / portal loops removed).
Decoding: greedy, repetition penalty 1.2 (decode_tune kept default). Input for replies: "Customer: … Agent: … Customer: <msg>",
oldest tokens dropped beyond 256 (pass `history=[(role, text), …]` to `analyze`). Routing/CSAT/sentiment use the message only.

## Sealed TEST set (run once, 14:49-14:58, after the candidate was frozen)
| | T3 | T4d |
|---|---|---|
| Category acc / macro F1 | 72.42% / 0.353 | **73.71% / 0.369** |
| CSAT acc / macro F1 | 73.33% / 0.361 | 72.41% / 0.352 (slightly worse) |
| Sentiment macro F1 (human-labelled subset) | 0.742 (0.531) | **0.758 (0.552)** |
| Urgency macro F1 / Spearman | 0.367 / 0.279 | 0.376 / 0.245 (partly validated only) |
| Portal-loop replies (300 multi-turn test rows) | 31.3% | **0.7%** (diff -30.7pp, CI [-36.0, -25.7]) |
| Any redirect / orphaned phrase | 32.3% / 3.0% | 1.0% / 0.3% |
| Replies acknowledging earlier details (proxy) | 5.4% | 24.6% |
| ROUGE-L / distinct-2 | 0.116 / 0.149 | 0.121 / 0.213 |
| Dialogue loss (rows with history) | 2.880 | 2.746 |
Behaviour suite (24 hand-written complaints): portal loops 19/24 (T3) -> 2/24; routing ok 20/24 both. Regression tests: ALL PASS.

## Known limits (do not over-claim)
- Replies are short and often generic ("Please let us know if you need any help…"); some ask for personal data
  (e.g. name/zip/phone for identity theft) or keep a dangling "send us a note at so". 11.6M parameters: not ChatGPT-level.
- Urgency: Spearman CI > 0 on val but MAE does not beat the mean -> do not use for decisions. Escalation: untrained.
- Reply helpfulness is NOT established until the blind human A/B (wsp_runs/ab_test/AB_SHEET.csv) is scored.
