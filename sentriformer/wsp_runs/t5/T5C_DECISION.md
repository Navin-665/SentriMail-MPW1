# T5c (MMI reranking) decision (2026-11-09 00:22) — REJECTED at Stage A; Stage B not run. Model = T4d (the default after T5).
No decoding.json written; deployed decoding is unchanged (greedy, repetition penalty 1.2).

| 250 rows (rs=97 A) | generic phrase | broken_link | action | specificity | distinct | ROUGE-L | failed rules |
|---|---|---|---|---|---|---|---|
| greedy | 0.224 | 0.060 | 0.404 | 0.204 | 0.840 | 0.127 | |
| random sampled pick (control) | **0.084** | 0.064 | 0.380 | 0.228 | 1.000 | 0.110 | |
| MMI λ=0.15 | 0.240 | 0.056 | 0.432 | 0.204 | 0.908 | 0.124 | 1, 2 |
| MMI λ=0.3 | 0.212 | 0.068 | 0.428 | 0.228 | 0.916 | 0.125 | 1, 2 |
| MMI λ=0.5 | 0.164 (CI −0.10 to −0.024) | 0.088 | 0.420 | 0.272 | 0.936 | 0.121 | 2, broken, ROUGE |

Interpretation: sampling alone removes most template phrases (22.4% → 8.4%), but costs ROUGE-L (−0.017) and action words. MMI's
reranking moves back TOWARDS the generic, high-likelihood candidates; only at λ=0.5 does it reduce generic replies significantly,
and it still loses to a random sample and raises broken-link replies. The decode-time reranking is not a targeted fix for this model.
