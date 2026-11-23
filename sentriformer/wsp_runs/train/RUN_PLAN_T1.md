# Run T1 (launched ~00:37, 2026-11-08)
- Init: wsp_runs/e1/m10_3_sft_wsp_candidate.pt (Phase-0 selected model; E1 heads, original backbone)
- Benchmark (measured): 0.32 s/step frozen, ~0.43 s/step joint, peak VRAM 2.14 GB at these batch sizes.
- Steps: 18,000 (est. 129 min + ~8 min evals); hard cap --max_minutes 150. Resumable every 1,500 steps.
- Batches/step: cat 48 @256 tok, sent 32, dial 32 (ctx128/resp128), csat 16 every 3rd step, urg 8 every 4th step.
- Exposure: cat 864k samples (~1.1 ep of 800k subset), dial 576k (~1.1 ep m10_4), sent 576k (~1.7 ep),
  csat ~96k (~6.3 ep), urg ~36k (~39 ep of 932 rows -> overfit risk, monitored on val).
- LR: backbone 3e-5, heads 3e-4, cosine, warmup 500; head warm-up 1,000 steps (backbone frozen) -- added after the
  benchmark showed the E1 head loses ~12 pp accuracy under train-mode dropout noise.
- Losses: cat CE (inv-sqrt class weights, label smoothing 0.1) x1.0, csat weighted CE x0.3, sent weighted CE x0.3,
  urg MSE x0.5, dial CE x1.0. Logged csat/urg running losses include skipped (zero) steps.
- Disk: ~47 MB per model checkpoint + ~140 MB resumable state.

## T1 outcome: STOPPED at step ~1,500 (00:43) by me
- Category train loss flat ~8.6 (uniform would be ~3.5); val-subset category acc 64% -> 36%.
- Diagnosis: selected E1 map (lambda=0) is ill-conditioned; it amplifies low-variance directions of M10.3 features.
  Harmless in eval mode, but train-mode dropout noise is amplified into confident wrong logits; AdamW at lr 3e-4 cannot
  shrink the ~10x-scaled weights quickly, and noisy gradients reached the encoder after unfreezing.
- Lesson: a closed-form head that is good in eval mode can be a poor *initialisation* for dropout training.

# Run T2 (launched ~00:48)
- Same as T1 except: category_head re-initialised (std 0.02); head warm-up 2,000 steps (backbone frozen); lr_heads 1e-3;
  stop gate: if category val-subset acc < 0.55 at step >= 2,000 -> halt. CSAT keeps its E1 init (trained normally in T1).

## T2 restart (user approved, 2026-11-08 morning)
- First T2 attempt was killed during data loading by the low-memory guard (no steps run).
- Restarted with streamed parquet loading and n_cat_train 500k (was 800k) for RAM margin; free RAM at launch 6.8 GB.

## T2 outcome: GATE FAIL at step 2,000 (category val-subset acc 0.321 < 0.55) — gate worked as designed
- Offline probe on saved M10.3 features, same head architecture: plain CE -> 65.3% val; trainer loss (class weights +
  label smoothing 0.1) -> 36%, with loss *rising*. Root cause: F.cross_entropy applies class weights to the smoothing
  term over all classes, pushing every sample toward heavily weighted rare classes. Also explains T1's flat ~8.6 loss.
- Weighted CE without smoothing on the same probe: 63.8% acc, macro F1 0.265 (plain CE: 65.3% / 0.244) -> chosen.

# Run T3 = T2 with --cat_label_smoothing 0.0 (everything else identical)
