# 🎯 SentriFormer M9.2 — Multi-Task Classification Benchmark Report

**Generated**: 2026-09-02 21:41:03

---

## 📊 Executive Summary Scorecard

| Metric | Empirical Result | Target Benchmark | Status |
|---|---|---|---|
| **Final Test Accuracy** | **79.00%** | > 75.0% | ✅ **SURPASSED** |
| **Final Weighted F1-Score** | **79.00%** | > 75.0% | ✅ **SURPASSED** |
| **Final Macro F1-Score** | **0.4000** | — | Evaluated across 33 classes |
| **Test Dataset Volume** | **123,229 samples** | 100% Held-Out Test Set | Zero Data Leakage |
| **Model Backbone** | **SentriFormer Base (11.45M)** | Pretrained from `checkpoint_best.pt` | Converged |

---

## 🏆 Top Category Performance Breakdown

| Category | Precision | Recall | F1-Score | Test Samples |
|---|---|---|---|---|
| **Mortgage** | 0.92 | 0.93 | **0.92** | 7,320 |
| **Student loan** | 0.88 | 0.87 | **0.87** | 3,240 |
| **Checking or savings account** | 0.82 | 0.85 | **0.83** | 9,199 |
| **Credit reporting or other personal consumer reports** | 0.83 | 0.83 | **0.83** | 33,547 |
| **Debt collection** | 0.81 | 0.82 | **0.81** | 18,488 |
| **Money transfer, virtual currency, or money service** | 0.80 | 0.81 | **0.80** | 4,841 |
| **Credit reporting, credit repair services, or other personal consumer reports** | 0.76 | 0.80 | **0.78** | 26,627 |
| **Vehicle loan or lease** | 0.76 | 0.73 | **0.74** | 2,630 |
| **Credit card** | 0.67 | 0.66 | **0.67** | 5,756 |
| **Credit card or prepaid card** | 0.67 | 0.62 | **0.65** | 5,151 |
| **Bank account or service** | 0.71 | 0.55 | **0.62** | 768 |
| **Credit reporting** | 0.70 | 0.50 | **0.58** | 1,530 |
| **Prepaid card** | 0.67 | 0.51 | **0.58** | 592 |
| **Payday loan, title loan, personal loan, or advance loan** | 0.59 | 0.51 | **0.55** | 904 |
| **Payday loan, title loan, or personal loan** | 0.57 | 0.41 | **0.48** | 838 |

---

## 📈 Generated Visualizations

| Chart | Description | File |
|---|---|---|
| **1. Category F1 Ranking** | All 33 classes ranked by F1 performance | `01_category_f1_ranking.png` |
| **2. Support Volume vs. F1** | Scatter scaling analysis | `02_support_vs_f1_scaling.png` |
| **3. Precision vs. Recall** | Granular trade-off comparison | `03_precision_vs_recall.png` |
| **4. Domain Routing Breakdown** | Aggregated by business domain | `04_domain_routing_breakdown.png` |
| **5. Executive Dashboard** | 4-Panel complete dashboard | `05_classification_dashboard.png` |

---

## 💾 Checkpoint Artifacts

| Artifact | Path | Size |
|---|---|---|
| Best Classifier Checkpoint | `sentriformer/checkpoints/classification/classifier_best.pt` | 100.4 MB |
| Test Evaluation Text Report | `sentriformer/checkpoints/classification/test_classification_report.txt` | 4.4 KB |
