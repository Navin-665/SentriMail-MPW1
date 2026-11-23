# Model card — SentriFormer (Phase-0 WSP release, 2026-11-08)

**File:** `wsp_runs/FINAL/sentriformer_final.pt` · SHA256 `b756be43…71dfa` · 11.65M parameters
**Base:** `checkpoints/m10_3_sft/m10_3_sft_best.pt` (untouched; SHA256 `2b0e4034…6d6960`)
**Change:** closed-form realignment of `category_head.dense.weight` and `csat_head.dense.weight` (no gradient training).
Encoder, decoder, embeddings and LM head are byte-identical to M10.3.

## Architecture
Encoder-decoder transformer, 4+4 pre-LN layers, d_model 256, 8 heads, FFN 1024, GELU, sinusoidal positions (max 512),
16k BPE vocabulary, tied embeddings. Mean-pooled encoder output feeds the category (33), CSAT (5) and sentiment/urgency heads.

## Intended use
Research prototype for routing and triaging customer complaints (mainly US consumer-finance / CFPB style) and drafting a
first-line acknowledgement. **Not** for autonomous customer communication: replies need human review.

## Measured performance (held-out test sets, run once)
| Task | Metric | This model | Original M10.3 | Reference M9.5 |
|---|---|---|---|---|
| Category (33) | accuracy | **63.7%** | 39.6% | 74.0% |
| | macro F1 / weighted F1 | 0.205 / 0.618 | 0.078 / 0.388 | 0.317 / 0.734 |
| | ECE (15-bin) | 0.069 | 0.165 | 0.048 |
| CSAT (1-5) | exact / within ±1 / MAE | **73.9% / 82.3% / 0.72** | 66.9% / 73.7% / 1.03 | 73.6% / 81.7% / 0.73 |
| Sentiment | — | **not available** (head untrained; below majority baseline) | same | same |
| Urgency | — | **not available** (head untrained; Spearman ≈ 0) | same | same |
| Dialogue (val) | ROUGE-L vs reference / mean words / orphaned-phrase rate | 0.130 / 18.6 / 15% | identical | — |

## Known limitations
- Routing is good on high-volume CFPB products (mortgage, student loan, debt collection, checking, credit reporting) and
  weak on rare and e-commerce classes (13 of 33 classes F1 = 0; several labels are CFPB-era duplicates).
- When top-1 confidence is below 0.25 the pipeline sets `needs_human_triage` (2.2% of test cases).
- CSAT never predicts 2 or 3 stars (too few training examples); treat it as a satisfied/dissatisfied signal.
- Replies are short, generic acknowledgements that often defer to "our support team"; 15-54% contain a dangling channel phrase
  ("reach out to us here:") inherited from URL-stripped training data. 1.7% contain template placeholders such as `{{Order Number}}`.
  No invented URLs or contact details were observed (0/300 val, 0/24 suite), but this is not a guarantee.
- English only (it routes Spanish text, but this is untested and unreliable). Inputs over 512 tokens are truncated.
- Labels: CFPB product labels are reliable; sentiment training data is partly pseudo-labelled; urgency data is small (1.3k rows).

## How to use
See `FINAL/HOW_TO_LOAD.md`: `python -I wsp/inference.py "complaint text"`.

## Provenance and reproducibility
All experiments, frozen criteria, hashes and logs are in `wsp_runs/`; the code is in `wsp/`. Test sets were used once per
selected model, never for fitting or selection.
