# Frozen T4 criteria (written 2026-11-08 ~09:00, BEFORE any T4 training run)

Baseline: T3 = `wsp_runs/FINAL_T3/sentriformer_t3.pt` (SHA256 8d01c4fc…). Test set SEALED for all T4 decisions;
one final test comparison of the final combined candidate vs the already-measured T3 test numbers.

## Experiments (one change each, all start from T3, identical hyper-parameters to T3 unless stated)
- **T4a-ctrl**: dialogue data = role-verified M10.4 (customer-authored targets removed), input "Customer: <message>".
- **T4a**: identical rows, input = role-labelled earlier turns + "Customer: <message>" (<= 250 tokens, oldest dropped first).
  Effect of multi-turn context = T4a vs T4a-ctrl. Effect of the role fix (+ prefix) = T4a-ctrl vs T3.
- Bounds: benchmark 100 steps first; `--max_minutes 150`; steps reduced (not the cap extended) if the benchmark predicts overrun.

## Gates (validation)
No-regression vs T3 (`wsp/eval_full.py` + `wsp/accept_training.py` rules 1-3; dialogue judged by eval_t4):
category acc >= T3-1pp, macro F1 >= T3-0.01; CSAT macro F1 >= T3-0.01; sentiment macro F1 >= T3-0.01.

Dialogue (`wsp/eval_t4.py`, 300 fixed val rows with history, each model in its native input format):
- portal_loop_rate (mandatory, lower is better) must not exceed T3's on the same rows;
- EOS >= T3-2pp; repetition <= T3+2pp; orphan (token rule) <= T3+2pp; ROUGE-L >= T3-0.005.
T4a-specific (vs T4a-ctrl, paired bootstrap on the 300 rows / context-correctness set):
- loss on rows with history lower than T4a-ctrl's; AND
- context-correctness re-ask rate lower than T4a-ctrl's OR ROUGE-L higher (CI excluding 0 for at least one).
Human blind A/B (100 fixed val contexts) is required before claiming replies are better; automated metrics only gate.

## Not claimed by T4a
Urgency (still unlearned), escalation/failure detection (needs labels), reply helpfulness (needs human A/B).

## Run settings (recorded before launch; identical for T4a-ctrl and T4a)
init T3; lr_backbone 3e-5; lr_heads 3e-4; head_warmup 0 (T3 heads already aligned; T3's warm-up existed only because its
category head was re-initialised); cat_label_smoothing 0; dial_ctx_len 256; batches cat48/sent32/dial32/csat16(every 3)/urg8(every 4);
n_cat_train 500k; eval_every 1000; selection score as T3. Only --dial_train/--dial_val differ (ctrl_* vs multi_*).

## T4d addendum (written 2026-11-08 ~13:30, BEFORE T4d training; T4a passed, see T4a_DECISION.md)
- **T4d**: init T4a model_best (SHA 57b8a88c…), dial_train = t4d_train.parquet (multi_train minus portal_rewrite / dm_redirect /
  empty / target portal_loop: 65,198 rows dropped, see t4d_build_report.json); dial_val = multi_val (unchanged).
- **T4d-ctrl**: identical but dial_train = multi_train (unfiltered). Isolates the filter from the extra training steps.
- Both: T4a run settings, max_steps 10000, max_minutes 150. Run sequentially.
Gates (validation, test sealed):
1. PRIMARY: portal_loop_rate(T4d) < portal_loop_rate(T4d-ctrl), paired bootstrap 95% CI of the difference excludes 0; and <= T4a's.
2. redirect_any_rate(T4d) <= T4a's (no shift to other redirect phrasing); re-ask rate <= T4a + 2pp.
3. No-regression vs T4a: category acc >= -1pp, macro F1 >= -0.01; CSAT macro F1 >= -0.01; sentiment macro F1 >= -0.01;
   EOS >= -2pp, repetition <= +2pp, orphan <= +2pp; ROUGE-L on rows whose REFERENCE is not a portal loop >= T4a - 0.005
   (all-rows ROUGE-L reported too: it can fall simply because 15% of references are portal loops).
Pass -> T4d becomes the candidate for the single final test. Human blind A/B still required before claiming better replies.
