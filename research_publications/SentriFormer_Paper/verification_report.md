# SentriFormer paper: verification report

Checked 2026-11-23. Build: `tectonic -X compile main.tex` (IEEEtran conference class, XeTeX, TeX Gyre Termes) → `paper.pdf`, 7 pages, two columns.

## 1. Compilation and layout
| Check | Result |
|---|---|
| Compiles without errors | yes |
| Overfull/underfull boxes | none (all tables wrapped in `\resizebox`) |
| Undefined references or citations | none |
| Two-column IEEE conference layout | yes. Abstract, Index Terms, Roman-numeral sections and an IEEEtran bibliography were checked on rendered pages |
| Figures readable, nothing cut off | rendered pages 1–7 inspected at 80 dpi. All 9 figures (2 TikZ, 7 matplotlib PDF) and 6 tables fit their columns |
| Fonts | body text in TeX Gyre Termes (Times clone); figure text in DejaVu Serif (matplotlib) |

## 2. Every number traced to its source
Paths are relative to `SENTRIMAIL-MPW1/sentriformer/`.

| Claim in paper | Source |
|---|---|
| 11,654,698 params; 11,447,296 core | checkpoint tensor count (evidence §1.2); README §4.1 |
| Arch (4+4, 256, 8 heads, FFN 1024, Pre-LN, tied 16k emb, BPE 16k/106 special) | `checkpoints/config.json`, `model/*.py`, `tokenizer/sentritokenizer_v1/tokenizer.json` |
| Tokenizer 4.73 chars/token, 0% UNK, 96.73% ≤512 | `tokenizer/tokenizer_benchmark.json` |
| Corpus 3,538,711 records; splits 3,184,956 / 176,586 / 177,169; source counts | `data/corpus_stats.json` |
| Supervised split sizes (Table II) | parquet metadata; `wsp_runs/t4/data/*build_report.json` |
| Loss weights, LRs, batches, steps, time, VRAM (Table III, Eq. 2) | `wsp/train_v2.py`, `wsp_runs/train/T3/config.json`, `eval_history.json`, `train_log.txt` |
| Pretraining 149,200 steps, 14.79 h, val 2.425, test PPL 10.27, init 16,445.75 | `results/pretraining/pretraining_report.md`; `results/m9_audit/m9_1_audit_summary.json` |
| Table IV, every test value | `wsp_runs/e0/test/e0_metrics.json`, `wsp_runs/e1/final_test/e1_metrics.json`, `wsp_runs/eval_full/{T3,T4d}_test/metrics.json`, `wsp_runs/t4/eval/{T3,T4d}_test/metrics.json` |
| Majority routing 27.2% | derived from the class supports in the T4d test confusion matrix (33,547 / 123,229) |
| Per-class F1, confusion matrices, reliability (Figs 5–7) | `eval_full/T4d_test/metrics.json`, plotted by `_scripts/make_figures.py` |
| Fig. 4 history values | `e0_metrics.json`, `e1_metrics.json`, T3/T4d test metrics (script-read, not typed) |
| Encoder drift −34.4 pp, CI [−34.7, −34.1]; 66% drift | `wsp_runs/e0/E0_REPORT.md`, `WSP_FINAL_REPORT.md` §1 |
| E1 39.67→63.94, R² 0.674, controls 21.91 / 27.38 | `wsp_runs/WSP_FINAL_REPORT.md` §3; `wsp_runs/e1/acceptance.json` |
| T1 64→36%; probe 65.3 / 36 / 63.8 | `wsp_runs/train/RUN_PLAN_T1.md`, `T1/train_log.txt` |
| T5a numbers (first test and replication, CIs) | `wsp_runs/t5/t5_compare.json`, `wsp_runs/t5/t5a_retest.json`, `T5_DECISION.md` |
| Table V (validation) | `wsp_runs/eval_full/*_val/metrics.json` |
| Table VI (rejected) | `*_DECISION.md` in `wsp_runs/train`, `e2`, `sentriwsp01`, `t5`, `t6`, `t7u` |
| Urgency test 0.376 / 0.148, MAE 0.691 / 0.675; probe 0.35 | `eval_full/T4d_test/metrics.json`; `wsp_runs/t7u/t7u_results.json` |
| Latency 30.4 / 59.4 / 275.2 / 586.0 ms; 16.3 words; 46.7 MB | `research_publications/_runs/latency_benchmark.json` (measured 2026-11-23); file size |

## 3. Integrity checks
- **Test vs validation labels.** Test numbers come only from sealed-test files. T5a numbers are labelled as validation. The paper states explicitly that T5a has no sealed-test run of its own.
- **Excluded unreliable values.** No training accuracy is reported as test accuracy. The leaked M9.5 perplexity of 1.04 is footnoted and excluded. Not cited: the unmeasured efficiency table in `research_comparative_evaluation.md`, its department table, and the 21-class parameter count.
- **No superiority claims.** Pre-trained and classical baselines were not run; this is stated in Limitations (iv).
- **Architectural novelty.** The paper explicitly disclaims any; the contributions are empirical and methodological.
- **No corpus text.** Neither the paper nor its figures contain TWCS, CFPB or other corpus text. The latency messages were written for the benchmark.
- **Citations.** All 41 citations resolve to `references.bib`, and every entry was verified by Agent 3 (`_lit/literature_review.md`, verification table). The tokenizer is cited as BPE [Sennrich]; SentencePiece is not cited because the project does not use it.

## 4. Unresolved items
1. Author names, affiliations and acknowledgment are placeholders.
2. T5a has no sealed-test evaluation. The one-time run was proposed but not executed in this session (it was blocked by the permission check). Command: `python -I wsp/eval_full.py --ckpt wsp_runs/FINAL_T5a/sentriformer_t5a.pt --name T5a --split test`.
3. No human evaluation, single seed, no baselines, and the regex detectors have no precision audit. All of this is disclosed in the paper.
4. No specific conference has been chosen. Page limits and anonymisation rules for the target venue must be checked before submission. The paper has not been submitted or accepted anywhere.
