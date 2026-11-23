# Frozen Phase-T training criteria (written 2026-11-08 ~00:10, BEFORE any training run)

## Starting point
- Init = the Phase-0 (WSP) selected model if E1 was accepted, else `m10_3_sft_best.pt`. Originals are never written.

## Bounds (each run)
- Benchmark 100 steps first: record s/step and peak VRAM. Abort if peak VRAM > 5.3 GB (6 GB card) -> halve batch sizes.
- Hard limits per run: `--max_minutes` <= 150 and `--max_steps` sized from the benchmark. No unbounded jobs.
- Resumable state every eval; checkpoints written only under `wsp_runs/train/<run>/`.

## In-training selection (val subsets only; fixed seeds)
`score = category_acc + csat_macroF1 + sentiment_macroF1 + urgency_macroF1(4-level) - 0.1 * dialogue_val_loss`
evaluated every `eval_every` steps on: 10k classification_val, full csat_val, 5k sentiment_val, full urgency_val (200),
2,098 m10_4_dialogue_val pairs. Best score -> `model_best.pt`.

## Acceptance of the trained model vs the Phase-0 model (FULL val, paired where applicable)
1. Category: accuracy @ best length >= Phase-0 - 1.0 pp; macro F1 must not drop by > 0.01.
2. CSAT: macro F1 >= Phase-0 macro F1 (class-weighted training targets the 2-4 star classes); exact acc >= Phase-0 - 2 pp.
3. Sentiment: macro F1 must beat (a) the untrained head and (b) majority-class baseline, paired CI excluding 0.
   Reported separately on the human-labelled GoEmotions subset and the distilled (teacher-agreement) subset.
4. Urgency: 4-level macro F1 and MAE must beat majority-class / mean-prediction baselines on val (n=200, small; reported with CI).
5. Dialogue: val loss <= Phase-0 + 0.05, AND on 300 fixed val contexts (greedy decoding): EOS rate >= Phase-0 - 2 pp,
   repetition rate <= Phase-0 + 2 pp, orphaned-phrase rate <= Phase-0 + 2 pp, ROUGE-L vs reference >= Phase-0 - 0.005.
6. Any failed criterion is reported. If the trained model fails 1 or 5, Phase-0 remains the deployed model and the
   trained checkpoint is kept only as an experimental candidate.

## Test
- Run once, only for the finally selected model, after all selection is complete.

## Addendum (written ~06:50, BEFORE the trained candidate's full evaluation)
- Self-test (Phase-0 vs itself) showed frozen rule 4's "macro F1 beats majority" is passed by the UNTRAINED urgency head
  (degenerate majority macro F1 ~0.15). Rule 4 is kept unchanged; a stricter SUPPLEMENTARY check is added and reported
  separately: urgency Spearman rank correlation, bootstrap 95% CI lower bound > 0. Urgency is only claimed as "learned"
  if both rule 4 and the supplementary check pass.
