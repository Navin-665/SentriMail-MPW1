# 📊 SentriFormer M9.2 Multi-Task Classification Results & Visualizations

All classification charts, reports, and model checkpoints are saved in:
`sentriformer/results/classification/`

---

## 🏆 Key Performance Metrics (100% Unseen Test Set)

| Metric | Result | Benchmark Target | Verdict |
|---|---|---|---|
| **Final Test Accuracy** | **79.15%** | > 75.0% | ⭐ **SURPASSED** |
| **Final Weighted F1-Score** | **78.87% (0.7887)** | > 75.0% | ⭐ **SURPASSED** |
| **Final Macro F1-Score** | **0.4035** | — | Evaluated across all 33 categories |
| **Total Test Volume** | **123,229 samples** | — | Evaluated with zero data leakage |
| **Model Size** | **11.65M parameters** | ~11-15M | Optimal lightweight architecture |

---

## 🖼️ Visualizations & Analysis

### 1. Executive Classification Dashboard
![Classification Dashboard](C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/05_classification_dashboard.png)

---

### 2. Category F1-Score Ranking (All 33 Classes)
![Category F1 Ranking](C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/01_category_f1_ranking.png)
> **Key Insight:** Customer support classes with adequate real-world representation achieve exceptional accuracy:
> - **Mortgages:** `92% F1` (`0.92 precision, 0.93 recall`)
> - **Student Loans:** `87% F1` (`0.88 precision, 0.87 recall`)
> - **Checking & Savings:** `83% F1` (`0.82 precision, 0.85 recall`)
> - **Credit Reporting:** `83% F1` (`0.83 precision, 0.83 recall`)
> - **Debt Collection:** `81% F1` (`0.81 precision, 0.82 recall`)
> - **Money Transfers:** `80% F1` (`0.80 precision, 0.81 recall`)

---

### 3. Business Domain Routing Performance
![Domain Routing Breakdown](C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/04_domain_routing_breakdown.png)
> **Key Insight:** When aggregated into high-level business departments for automated routing in SentriMail:
> - **Mortgages & Loans:** `84.8% Weighted F1`
> - **Credit Bureau & Dispute Services:** `80.4% Weighted F1`
> - **Banking & Deposits:** `80.1% Weighted F1`
> - **Debt Collection:** `80.4% Weighted F1`
> - **Cards & Payments:** `65.5% Weighted F1`

---

### 4. Precision, Recall & Support Scaling Dynamics
````carousel
![Support Volume vs F1 Score](C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/02_support_vs_f1_scaling.png)
<!-- slide -->
![Precision vs Recall Breakdown](C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/03_precision_vs_recall.png)
````

---

## 📁 Stored Files in `sentriformer/results/classification/`

```text
sentriformer/results/classification/
├── 01_category_f1_ranking.png          (118.4 KB) - Complete 33-class ranking
├── 02_support_vs_f1_scaling.png        (134.7 KB) - Sample size vs performance scaling
├── 03_precision_vs_recall.png          (118.9 KB) - Trade-off comparison for active classes
├── 04_domain_routing_breakdown.png     (88.3 KB)  - Business department routing breakdown
├── 05_classification_dashboard.png    (256.1 KB) - 4-Panel executive dashboard
└── classification_report.md            (3.4 KB)   - Full Markdown benchmark report
```

Checkpoint saved at:
- **`sentriformer/checkpoints/classification/classifier_best.pt`** (100.4 MB)
