# 📊 SentriFormer M9.4 — CSAT Prediction Head Fine-Tuning Report

**Date:** September 4, 2026  
**Checkpoint Evaluated:** [`csat_best.pt`](file:///c:/Users/nan33/OneDrive/Desktop/6th%20sem%20projects/SENTRIMAIL-MPW1/sentriformer/checkpoints/csat/csat_best.pt) (Epoch 3, Val MAE: 0.717)  
**Dataset:** `csat_train.parquet` (15,217 train samples, 849 val, 870 test)  

---

## 📈 1. Executive Summary

| Evaluation Metric | Test Result | Verdict & Status |
| :--- | :---: | :--- |
| **Exact 1–5 Star Accuracy** | **72.07%** | 🟢 **Good Exact Match Rate** |
| **Within $\pm 1$ Star Tolerance** | **81.49%** | 🟢 **Strong Ordinal Prediction Reliability** |
| **Mean Absolute Error (MAE)** | **0.7310** | 🟢 **Average deviation < 3/4 of a star rating** |
| **Root Mean Squared Error (RMSE)** | **1.5470** | 🟡 **Penalty from extreme tail mispredictions** |
| **Weighted F1-Score** | **0.6898** | 🟢 **Solid representation on volume** |
| **Macro F1-Score** | **0.3274** | 🔴 **Depressed by extreme class scarcity in 2⭐ & 3⭐** |

---

## 🖼️ 2. Visual Performance Artifacts

![CSAT Distribution Imbalance](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/01_csat_distribution_imbalance.png)
*Figure 1: Extreme bimodal class distribution in CSAT training (15.2k) and test sets (870).*

---

![CSAT F1 Breakdown](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/02_csat_f1_breakdown.png)
*Figure 2: Per-Class CSAT F1-score breakdown on held-out test reviews.*

---

![CSAT Performance Dashboard](file:///C:/Users/nan33/.gemini/antigravity-ide/brain/a5de84d1-05e1-42b8-ad22-fc8883726c2b/03_csat_performance_dashboard.png)
*Figure 3: Comprehensive 4-panel CSAT performance summary, error tolerance, and multi-task integration role.*

---

## 🎯 3. Role of CSAT in the Multi-Task Pipeline

1. **Risk Triage & Priority Escalation**: 1-Star predictions ($F_1 = 0.712$) directly flag severe dissatisfaction for priority routing.
2. **Coarse Satisfaction Gauge**: Within $\pm 1$ star accuracy of **81.49%** is reliable for automated ticket routing and agent dashboard previews.
3. **Multi-Task Balance**: In M9.5, CSAT loss will be weighted modestly ($\lambda_{\text{csat}} = 0.1$) to prevent gradient interference with core Category Routing ($\lambda_{\text{cat}} = 1.0$).
