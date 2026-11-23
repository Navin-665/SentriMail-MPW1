"""
==========================================================================
  M8.6 Pretraining Results — Visualization & Report Generator
==========================================================================
  Reads training_log.jsonl and produces publication-quality graphs.
  Outputs saved to: sentriformer/results/pretraining/
==========================================================================
"""
import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# Use Agg backend for headless rendering (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
LOG_FILE = os.path.join(ROOT_DIR, "sentriformer", "checkpoints", "training_log.jsonl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "pretraining")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

# Color palette
COLORS = {
    "train": "#58a6ff",       # Blue
    "val": "#f97583",         # Red/Pink
    "lr": "#56d364",          # Green
    "grad": "#d2a8ff",        # Purple
    "epoch1": "#58a6ff",      # Blue
    "epoch2": "#56d364",      # Green
    "epoch3": "#f97583",      # Red
    "gap": "#e3b341",         # Yellow/Gold
    "accent": "#79c0ff",
    "highlight": "#ff7b72",
}


def load_log(path):
    """Load all entries from the JSONL training log."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def smooth(values, window=15):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_fig(fig, name):
    """Save figure to output directory."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {name}")
    return path


# ===========================================================================
# CHART 1: Training Loss Curve (Full)
# ===========================================================================
def plot_training_loss(train_steps):
    steps = [e["step"] for e in train_steps]
    losses = [e["loss"] for e in train_steps]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Raw data (faded)
    ax.plot(steps, losses, color=COLORS["train"], alpha=0.25, linewidth=0.8, label="Raw loss")

    # Smoothed curve
    if len(losses) > 30:
        smoothed = smooth(losses, window=30)
        s_steps = steps[len(steps) - len(smoothed):]
        ax.plot(s_steps, smoothed, color=COLORS["train"], linewidth=2.0, label="Smoothed (w=30)")

    # Epoch boundaries
    epochs_at = []
    for i in range(1, len(train_steps)):
        if train_steps[i]["epoch"] != train_steps[i - 1]["epoch"]:
            epochs_at.append(train_steps[i]["step"])

    for ep_step in epochs_at:
        ax.axvline(x=ep_step, color=COLORS["accent"], linestyle="--", alpha=0.5, linewidth=1)

    # Labels for epoch boundaries
    for i, ep_step in enumerate(epochs_at):
        ax.text(ep_step, ax.get_ylim()[1] * 0.95, f" Epoch {i + 2}",
                color=COLORS["accent"], fontsize=9, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("SentriFormer Pretraining — Training Loss Curve", fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    return save_fig(fig, "01_training_loss_curve.png")


# ===========================================================================
# CHART 2: Validation Loss Curve
# ===========================================================================
def plot_validation_loss(validations):
    steps = [e["step"] for e in validations]
    losses = [e["val_loss"] for e in validations]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, losses, color=COLORS["val"], linewidth=2.5, marker="o",
            markersize=8, markerfacecolor=COLORS["highlight"], markeredgecolor="white",
            markeredgewidth=1.5, label="Validation Loss", zorder=5)

    # Annotate best
    best_idx = np.argmin(losses)
    ax.annotate(
        f"Best: {losses[best_idx]:.4f}\n(step {steps[best_idx]:,})",
        xy=(steps[best_idx], losses[best_idx]),
        xytext=(steps[best_idx] - 15000, losses[best_idx] + 0.15),
        fontsize=10, color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=COLORS["accent"]),
    )

    # Annotate each point with value
    for i, (s, l) in enumerate(zip(steps, losses)):
        if i != best_idx:
            ax.annotate(f"{l:.3f}", (s, l), textcoords="offset points",
                        xytext=(0, 12), fontsize=7, color="#8b949e", ha="center")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("SentriFormer Pretraining — Validation Loss", fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    return save_fig(fig, "02_validation_loss_curve.png")


# ===========================================================================
# CHART 3: Train vs Validation Loss (Overfitting Monitor)
# ===========================================================================
def plot_train_vs_val(train_steps, validations):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Training loss (smoothed heavily for comparison)
    t_steps = [e["step"] for e in train_steps]
    t_losses = [e["loss"] for e in train_steps]

    if len(t_losses) > 50:
        smoothed = smooth(t_losses, window=50)
        s_steps = t_steps[len(t_steps) - len(smoothed):]
        ax.plot(s_steps, smoothed, color=COLORS["train"], linewidth=2.0, label="Train Loss (smoothed)")

    # Validation loss
    v_steps = [e["step"] for e in validations]
    v_losses = [e["val_loss"] for e in validations]
    ax.plot(v_steps, v_losses, color=COLORS["val"], linewidth=2.5, marker="s",
            markersize=7, label="Validation Loss")

    # Shade the gap region
    if len(v_steps) > 1 and len(s_steps) > 0:
        # Interpolate train loss at validation steps
        train_at_val = np.interp(v_steps, s_steps, smoothed)
        ax.fill_between(v_steps, train_at_val, v_losses, alpha=0.15,
                         color=COLORS["gap"], label="Generalization Gap")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("SentriFormer — Train vs Validation Loss (Overfitting Monitor)",
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    return save_fig(fig, "03_train_vs_val_loss.png")


# ===========================================================================
# CHART 4: Learning Rate Schedule
# ===========================================================================
def plot_lr_schedule(train_steps):
    steps = [e["step"] for e in train_steps]
    lrs = [e["lr"] for e in train_steps]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(steps, lrs, color=COLORS["lr"], linewidth=2.0)
    ax.fill_between(steps, 0, lrs, alpha=0.15, color=COLORS["lr"])

    # Mark warmup end
    warmup_end = 2000
    ax.axvline(x=warmup_end, color=COLORS["accent"], linestyle=":", alpha=0.6)
    ax.text(warmup_end + 500, max(lrs) * 0.9, "Warmup\nComplete",
            color=COLORS["accent"], fontsize=9)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("SentriFormer — Cosine Learning Rate Schedule (2000-step warmup)",
                 fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))

    return save_fig(fig, "04_learning_rate_schedule.png")


# ===========================================================================
# CHART 5: Gradient Norm Over Training
# ===========================================================================
def plot_grad_norms(train_steps):
    steps = [e["step"] for e in train_steps]
    grads = [e.get("grad_norm", 0) for e in train_steps]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(steps, grads, color=COLORS["grad"], alpha=0.3, linewidth=0.8, label="Raw")

    if len(grads) > 30:
        smoothed = smooth(grads, window=30)
        s_steps = steps[len(steps) - len(smoothed):]
        ax.plot(s_steps, smoothed, color=COLORS["grad"], linewidth=2.0, label="Smoothed (w=30)")

    # Clip line
    ax.axhline(y=1.0, color=COLORS["highlight"], linestyle="--", alpha=0.5, linewidth=1,
               label="Clip threshold (1.0)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("SentriFormer — Gradient Norm Over Training", fontweight="bold", pad=15)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    return save_fig(fig, "05_gradient_norm.png")


# ===========================================================================
# CHART 6: Per-Epoch Loss Comparison (Bar Chart)
# ===========================================================================
def plot_epoch_comparison(epoch_ends, validations):
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [e["epoch"] for e in epoch_ends]
    train_losses = [e["avg_train_loss"] for e in epoch_ends]

    # Get end-of-epoch validation losses
    epoch_val_losses = []
    for ep in epochs:
        # Find val entries for this epoch, take the last one
        ep_vals = [v for v in validations if v["epoch"] == ep]
        if ep_vals:
            epoch_val_losses.append(ep_vals[-1]["val_loss"])
        else:
            epoch_val_losses.append(0)

    x = np.arange(len(epochs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_losses, width, color=COLORS["train"],
                   label="Train Loss", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, epoch_val_losses, width, color=COLORS["val"],
                   label="Val Loss", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, train_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", fontsize=10, color=COLORS["train"], fontweight="bold")
    for bar, val in zip(bars2, epoch_val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", fontsize=10, color=COLORS["val"], fontweight="bold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SentriFormer — Per-Epoch Train vs Validation Loss", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    return save_fig(fig, "06_epoch_comparison.png")


# ===========================================================================
# CHART 7: Loss Improvement Rate (Epoch-over-Epoch)
# ===========================================================================
def plot_loss_improvement(epoch_ends):
    if len(epoch_ends) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = [e["epoch"] for e in epoch_ends]
    losses = [e["avg_train_loss"] for e in epoch_ends]

    improvements = []
    labels = []
    for i in range(1, len(losses)):
        imp = losses[i - 1] - losses[i]
        pct = (imp / losses[i - 1]) * 100
        improvements.append(pct)
        labels.append(f"E{epochs[i-1]}→E{epochs[i]}")

    colors = [COLORS["epoch2"], COLORS["epoch3"]][:len(improvements)]
    bars = ax.bar(range(len(improvements)), improvements, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=12, color="#c9d1d9", fontweight="bold")

    ax.set_xlabel("Epoch Transition")
    ax.set_ylabel("Loss Improvement (%)")
    ax.set_title("SentriFormer — Epoch-over-Epoch Loss Improvement",
                 fontweight="bold", pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")

    return save_fig(fig, "07_loss_improvement_rate.png")


# ===========================================================================
# CHART 8: Training Summary Dashboard (Multi-Panel)
# ===========================================================================
def plot_dashboard(train_steps, validations, epoch_ends):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("SentriFormer M8.6 Pretraining — Dashboard",
                 fontsize=18, fontweight="bold", color="#f0f6fc", y=0.98)

    t_steps = [e["step"] for e in train_steps]
    t_losses = [e["loss"] for e in train_steps]
    v_steps = [e["step"] for e in validations]
    v_losses = [e["val_loss"] for e in validations]
    lrs = [e["lr"] for e in train_steps]
    grads = [e.get("grad_norm", 0) for e in train_steps]

    # Panel 1: Train + Val Loss
    ax1 = axes[0, 0]
    if len(t_losses) > 50:
        smoothed = smooth(t_losses, 50)
        s_steps = t_steps[len(t_steps) - len(smoothed):]
        ax1.plot(s_steps, smoothed, color=COLORS["train"], linewidth=1.5, label="Train")
    ax1.plot(v_steps, v_losses, color=COLORS["val"], linewidth=2, marker="o",
             markersize=5, label="Val")
    ax1.set_title("Loss Curves")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Panel 2: Learning Rate
    ax2 = axes[0, 1]
    ax2.plot(t_steps, lrs, color=COLORS["lr"], linewidth=1.5)
    ax2.fill_between(t_steps, 0, lrs, alpha=0.1, color=COLORS["lr"])
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Panel 3: Gradient Norm
    ax3 = axes[1, 0]
    if len(grads) > 30:
        g_smooth = smooth(grads, 30)
        g_steps = t_steps[len(t_steps) - len(g_smooth):]
        ax3.plot(g_steps, g_smooth, color=COLORS["grad"], linewidth=1.5)
    ax3.axhline(y=1.0, color=COLORS["highlight"], linestyle="--", alpha=0.5, linewidth=1)
    ax3.set_title("Gradient Norm")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Panel 4: Epoch Bars
    ax4 = axes[1, 1]
    epochs = [e["epoch"] for e in epoch_ends]
    ep_losses = [e["avg_train_loss"] for e in epoch_ends]
    ep_colors = [COLORS["epoch1"], COLORS["epoch2"], COLORS["epoch3"]][:len(epochs)]
    bars = ax4.bar([f"Epoch {e}" for e in epochs], ep_losses, color=ep_colors,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ep_losses):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=10, color="#c9d1d9", fontweight="bold")
    ax4.set_title("Per-Epoch Average Train Loss")
    ax4.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "08_training_dashboard.png")


# ===========================================================================
# TEXT SUMMARY REPORT
# ===========================================================================
def write_summary_report(train_steps, validations, epoch_ends, completes):
    report_path = os.path.join(OUTPUT_DIR, "pretraining_report.md")

    # Gather stats
    total_steps = train_steps[-1]["step"] if train_steps else 0
    first_loss = train_steps[0]["loss"] if train_steps else 0
    final_train = train_steps[-1]["loss"] if train_steps else 0
    best_val = min(v["val_loss"] for v in validations) if validations else 0
    final_val = validations[-1]["val_loss"] if validations else 0
    best_step = min(validations, key=lambda v: v["val_loss"])["step"] if validations else 0
    total_time = completes[-1]["total_seconds"] if completes else 0

    lines = [
        "# SentriFormer M8.6 — Pretraining Results Report",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Training Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| Model | SentriFormer (11.45M params) |",
        "| Vocab Size | 16,000 (BPE) |",
        "| d_model | 256 |",
        "| Encoder/Decoder Layers | 4 / 4 |",
        "| Attention Heads | 8 (head_dim=32) |",
        "| FFN Dim | 1,024 |",
        "| Max Sequence Length | 512 |",
        "| Objective | T5 Span Corruption (15% noise, mean span 3.0) |",
        "| Optimizer | AdamW (lr=5e-4, β₁=0.9, β₂=0.98, wd=0.01) |",
        "| Schedule | Cosine with 2000-step linear warmup |",
        "| Precision | FP16 Mixed Precision |",
        "| Batch Size | 64 |",
        "| Gradient Clipping | max_norm=1.0 |",
        "| Training Data | 3,184,956 records (train.parquet) |",
        "| Validation Data | 176,586 records (val.parquet) |",
        f"| Total Steps | {total_steps:,} |",
        "| Epochs | 3 |",
        f"| Total Time | {total_time/3600:.2f} hours |",
        "| Hardware | NVIDIA RTX 3060 Laptop (6 GB VRAM) |",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Initial Loss | {first_loss:.4f} |",
        f"| Final Train Loss | {final_train:.4f} |",
        f"| Final Val Loss | {final_val:.4f} |",
        f"| Best Val Loss | {best_val:.4f} (step {best_step:,}) |",
        f"| Train-Val Gap | {final_val - final_train:.4f} |",
        f"| Total Loss Reduction | {first_loss - final_train:.4f} ({((first_loss - final_train)/first_loss)*100:.1f}%) |",
        "",
        "---",
        "",
        "## Per-Epoch Breakdown",
        "",
        "| Epoch | Avg Train Loss | Duration | Steps |",
        "|---|---|---|---|",
    ]

    for e in epoch_ends:
        hrs = e["elapsed_seconds"] / 3600
        lines.append(
            f"| {e['epoch']} | {e['avg_train_loss']:.4f} | {hrs:.2f}h | {e['epoch_steps']:,} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Validation Checkpoints",
        "",
        "| Step | Epoch | Val Loss | Δ from Previous |",
        "|---|---|---|---|",
    ]

    prev = None
    for v in validations:
        delta = f"{v['val_loss'] - prev:.4f}" if prev else "—"
        lines.append(f"| {v['step']:,} | {v['epoch']} | {v['val_loss']:.4f} | {delta} |")
        prev = v["val_loss"]

    lines += [
        "",
        "---",
        "",
        "## Overfitting Analysis",
        "",
        f"- **Final train loss**: {final_train:.4f}",
        f"- **Final val loss**: {final_val:.4f}",
        f"- **Gap**: {final_val - final_train:.4f}",
        f"- **Verdict**: {'✅ Healthy generalization (gap < 0.5)' if (final_val - final_train) < 0.5 else '⚠️ Moderate overfitting (gap ≥ 0.5)'}",
        "",
        "The model shows consistent val loss improvement across all 3 epochs,",
        "indicating the model has not saturated and could potentially benefit from",
        "additional training epochs.",
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "| # | Chart | File |",
        "|---|---|---|",
        "| 1 | Training Loss Curve | `01_training_loss_curve.png` |",
        "| 2 | Validation Loss Curve | `02_validation_loss_curve.png` |",
        "| 3 | Train vs Val Loss | `03_train_vs_val_loss.png` |",
        "| 4 | Learning Rate Schedule | `04_learning_rate_schedule.png` |",
        "| 5 | Gradient Norm | `05_gradient_norm.png` |",
        "| 6 | Epoch Comparison | `06_epoch_comparison.png` |",
        "| 7 | Loss Improvement Rate | `07_loss_improvement_rate.png` |",
        "| 8 | Training Dashboard | `08_training_dashboard.png` |",
        "",
        "---",
        "",
        "## Checkpoint Files",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `checkpoint_best.pt` | Best validation loss (full resumable state) |",
        "| `checkpoint_final.pt` | Final training state (full resumable state) |",
        "| `sentriformer_base_weights.pt` | Model weights only (for downstream tasks) |",
        "| `config.json` | Model configuration |",
        "| `training_log.jsonl` | Complete training log |",
        "",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✅ Saved: pretraining_report.md")
    return report_path


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 65)
    print("  SentriFormer M8.6 — Generating Pretraining Visualizations")
    print("=" * 65)
    print(f"  Log file:   {LOG_FILE}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    # Load data
    entries = load_log(LOG_FILE)
    train_steps = [e for e in entries if e.get("event") == "train_step"]
    validations = [e for e in entries if e.get("event") == "validation"]
    epoch_ends = [e for e in entries if e.get("event") == "epoch_end"]
    completes = [e for e in entries if e.get("event") == "training_complete"]

    print(f"  Loaded {len(entries)} log entries")
    print(f"    Train steps:  {len(train_steps)}")
    print(f"    Validations:  {len(validations)}")
    print(f"    Epoch ends:   {len(epoch_ends)}")
    print()

    # Generate charts
    print("Generating charts...")
    plot_training_loss(train_steps)
    plot_validation_loss(validations)
    plot_train_vs_val(train_steps, validations)
    plot_lr_schedule(train_steps)
    plot_grad_norms(train_steps)
    plot_epoch_comparison(epoch_ends, validations)
    plot_loss_improvement(epoch_ends)
    plot_dashboard(train_steps, validations, epoch_ends)

    # Generate text report
    print()
    print("Generating report...")
    write_summary_report(train_steps, validations, epoch_ends, completes)

    print()
    print("=" * 65)
    print(f"  ✅ All results saved to: {OUTPUT_DIR}")
    print("=" * 65)

    # List generated files
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:40s}  {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
