# SentriFormer M8.6 — Pretraining Results Report

**Generated**: 2026-09-02 14:24:34

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | SentriFormer (11.45M params) |
| Vocab Size | 16,000 (BPE) |
| d_model | 256 |
| Encoder/Decoder Layers | 4 / 4 |
| Attention Heads | 8 (head_dim=32) |
| FFN Dim | 1,024 |
| Max Sequence Length | 512 |
| Objective | T5 Span Corruption (15% noise, mean span 3.0) |
| Optimizer | AdamW (lr=5e-4, β₁=0.9, β₂=0.98, wd=0.01) |
| Schedule | Cosine with 2000-step linear warmup |
| Precision | FP16 Mixed Precision |
| Batch Size | 64 |
| Gradient Clipping | max_norm=1.0 |
| Training Data | 3,184,956 records (train.parquet) |
| Validation Data | 176,586 records (val.parquet) |
| Total Steps | 149,200 |
| Epochs | 3 |
| Total Time | 14.79 hours |
| Hardware | NVIDIA RTX 3060 Laptop (6 GB VRAM) |

---

## Results Summary

| Metric | Value |
|---|---|
| Initial Loss | 9.3732 |
| Final Train Loss | 2.0898 |
| Final Val Loss | 2.4248 |
| Best Val Loss | 2.4248 (step 140,000) |
| Train-Val Gap | 0.3350 |
| Total Loss Reduction | 7.2834 (77.7%) |

---

## Per-Epoch Breakdown

| Epoch | Avg Train Loss | Duration | Steps |
|---|---|---|---|
| 1 | 2.7170 | 4.68h | 49,764 |
| 2 | 2.1893 | 4.66h | 49,764 |
| 3 | 2.0926 | 5.42h | 49,764 |

---

## Validation Checkpoints

| Step | Epoch | Val Loss | Δ from Previous |
|---|---|---|---|
| 10,000 | 1 | 3.2359 | — |
| 20,000 | 1 | 2.9407 | -0.2952 |
| 30,000 | 1 | 2.7975 | -0.1432 |
| 40,000 | 1 | 2.7208 | -0.0767 |
| 50,000 | 2 | 2.6711 | -0.0497 |
| 60,000 | 2 | 2.6136 | -0.0575 |
| 70,000 | 2 | 2.5615 | -0.0521 |
| 80,000 | 2 | 2.5315 | -0.0300 |
| 90,000 | 2 | 2.5131 | -0.0184 |
| 100,000 | 3 | 2.4824 | -0.0307 |
| 110,000 | 3 | 2.4667 | -0.0157 |
| 120,000 | 3 | 2.4447 | -0.0220 |
| 130,000 | 3 | 2.4294 | -0.0153 |
| 140,000 | 3 | 2.4248 | -0.0046 |

---

## Overfitting Analysis

- **Final train loss**: 2.0898
- **Final val loss**: 2.4248
- **Gap**: 0.3350
- **Verdict**: ✅ Healthy generalization (gap < 0.5)

The model shows consistent val loss improvement across all 3 epochs,
indicating the model has not saturated and could potentially benefit from
additional training epochs.

---

## Visualizations

| # | Chart | File |
|---|---|---|
| 1 | Training Loss Curve | `01_training_loss_curve.png` |
| 2 | Validation Loss Curve | `02_validation_loss_curve.png` |
| 3 | Train vs Val Loss | `03_train_vs_val_loss.png` |
| 4 | Learning Rate Schedule | `04_learning_rate_schedule.png` |
| 5 | Gradient Norm | `05_gradient_norm.png` |
| 6 | Epoch Comparison | `06_epoch_comparison.png` |
| 7 | Loss Improvement Rate | `07_loss_improvement_rate.png` |
| 8 | Training Dashboard | `08_training_dashboard.png` |

---

## Checkpoint Files

| File | Purpose |
|---|---|
| `checkpoint_best.pt` | Best validation loss (full resumable state) |
| `checkpoint_final.pt` | Final training state (full resumable state) |
| `sentriformer_base_weights.pt` | Model weights only (for downstream tasks) |
| `config.json` | Model configuration |
| `training_log.jsonl` | Complete training log |
