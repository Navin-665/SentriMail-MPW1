# SentriFormer: training and metrics figures

Regenerated from the project's own logs and metric files by
`research_publications/_scripts/make_training_gallery.py`. Test = sealed test split (scored once per model).

| # | Figure | Stage |
|---|---|---|
| 01 | ![](01_tokenizer_vocab_ablation.png) | **Tokenizer training**: byte-level BPE, 16k vs 24k vs 32k vocabulary (16k selected: 4.73 chars/token, 0% UNK) |
| 02 | ![](02_pretraining_curves.png) | **Pre-training** (span corruption): loss, learning rate, gradient norm; test PPL 10.27 |
| 03 | ![](03_multitask_training_losses.png) | **Multi-task training**: per-head training losses (routing, sentiment, CSAT, reply) |
| 04 | ![](04_multitask_validation_curves.png) | **Multi-task training**: validation metrics every 1k steps |
| 05 | ![](05_label_smoothing_collapse.png) | Class-weighted CE + label smoothing collapse vs the fixed recipe |
| 06 | ![](06_filter_vs_control_curves.png) | **Dialogue SFT** data interventions vs matched control runs |
| 07 | ![](07_test_routing_by_stage.png) | Sealed-test routing across all training stages |
| 08 | ![](08_final_test_metrics_ci.png) | Final sealed-test metrics with 95% bootstrap CIs |
| 09 | ![](09_routing_confusion_matrix.png) | 33×33 routing confusion matrix |
| 10 | ![](10_sentiment_csat_test.png) | Sentiment and CSAT heads: confusion matrices and per-class F1 |
| 11 | ![](11_reply_quality.png) | Reply quality: portal loops 31.3% → 0.7%; reply filter vs control |
| 12 | ![](12_routing_per_class_f1.png) | Per-class routing F1 (all 33 classes) |

Earlier per-milestone plots (pre-training dashboard, classification, CSAT, dialogue SFT, multitask, clean SFT) are in
[`sentriformer/results/`](../../../sentriformer/results/); dataset analysis in [`docs/data_analysis/`](../../data_analysis/);
paper figures in [`research_publications/`](../../../research_publications/).
