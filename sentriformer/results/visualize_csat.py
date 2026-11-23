"""
==========================================================================
  M9.4 CSAT Results — Visualization & Report Generator
==========================================================================
  Reads test_csat_report.txt and produces high-quality dark-themed charts.
  Outputs saved to: sentriformer/results/csat/
==========================================================================
"""
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "csat")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dark theme styling
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
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

COLORS = {
    "cyan": "#38bdf8",
    "green": "#56d364",
    "red": "#f85149",
    "yellow": "#e3b341",
    "purple": "#d2a8ff",
    "orange": "#f0883e",
    "blue": "#58a6ff",
    "border": "#30363d",
}


def plot_01_distribution_imbalance():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    classes = ["1 Star\n(Very Poor)", "2 Stars\n(Poor)", "3 Stars\n(Neutral)", "4 Stars\n(Good)", "5 Stars\n(Excellent)"]
    train_counts = [4032, 283, 484, 1339, 9079]
    test_counts = [217, 16, 17, 88, 532]
    
    x = np.arange(len(classes))
    width = 0.35
    
    b1 = ax.bar(x - width/2, train_counts, width, label="Train Set (15,217 samples)", color=COLORS["blue"], edgecolor=COLORS["border"], zorder=3)
    
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    b2 = ax2.bar(x + width/2, test_counts, width, label="Test Set (870 samples)", color=COLORS["purple"], edgecolor=COLORS["border"], zorder=3)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 150, f"{h:,}", ha="center", va="bottom", color=COLORS["blue"], fontweight="bold", fontsize=9.5)
        
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 10, f"{h}", ha="center", va="bottom", color=COLORS["purple"], fontweight="bold", fontsize=9.5)
        
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylabel("Train Sample Count", color=COLORS["blue"], fontweight="bold")
    ax2.set_ylabel("Test Sample Count", color=COLORS["purple"], fontweight="bold")
    ax.set_ylim(0, 10500)
    ax2.set_ylim(0, 620)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_title("CSAT Dataset Extreme Class Imbalance\n(Bimodal Distribution: Heavy Concentration at 1-Star & 5-Star Extremes)", fontweight="bold", pad=12)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_csat_distribution_imbalance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_02_csat_f1_breakdown():
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = ["1 Star\n(Very Poor)", "2 Stars\n(Poor)", "3 Stars\n(Neutral)", "4 Stars\n(Good)", "5 Stars\n(Excellent)"]
    f1_scores = [0.7115, 0.0000, 0.0000, 0.1053, 0.8205]
    colors = [COLORS["green"], COLORS["red"], COLORS["red"], COLORS["orange"], COLORS["green"]]
    
    bars = ax.bar(classes, [s * 100 for s in f1_scores], color=colors, edgecolor=COLORS["border"], width=0.5, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1-Score (%)", fontweight="bold")
    ax.set_title("Per-Class CSAT F1-Score Breakdown on Held-Out Test Set\n(Reliable on Extremes, Zero Representation in Middle Tiers 2 & 3)", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha="center", va="bottom", color="#ffffff", fontweight="bold")
        
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_csat_f1_breakdown.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_03_csat_dashboard():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # [0, 0] Accuracy & Tolerance
    metrics = ["Exact Accuracy\n(72.07%)", "Within ±1 Star\nTolerance (81.49%)", "Weighted F1\n(68.98%)", "Macro F1\n(32.74%)"]
    vals = [72.07, 81.49, 68.98, 32.74]
    m_colors = [COLORS["blue"], COLORS["green"], COLORS["cyan"], COLORS["yellow"]]
    
    b = axs[0, 0].bar(metrics, vals, color=m_colors, edgecolor=COLORS["border"], width=0.5, zorder=3)
    axs[0, 0].set_ylim(0, 100)
    axs[0, 0].set_ylabel("Percentage (%)", fontweight="bold")
    axs[0, 0].set_title("Overall CSAT Performance & Ordinal Tolerance", fontweight="bold")
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for bar in b:
        h = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.2f}%", ha="center", va="bottom", color="#ffffff", fontweight="bold")
        
    # [0, 1] Metric Matrix
    axs[0, 1].axis("off")
    table_data = [
        ["Evaluation Metric", "Score", "Practical Interpretation"],
        ["Exact Accuracy", "72.07%", "Exact 1-5 star matching on unseen test reviews"],
        ["Within ±1 Star", "81.49%", "Prediction within 1 star point of true satisfaction"],
        ["Mean Absolute Error", "0.731", "Average deviation is less than 3/4 of a star"],
        ["Root Mean Squared Error", "1.547", "Penalty from occasional extreme misclassifications"],
        ["Macro F1", "0.327", "Depressed by 0% F1 on classes with <20 test samples"],
        ["Best Checkpoint", "Epoch 3 (0.717)", "Optimal convergence before slight overfitting"]
    ]
    table = axs[0, 1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["border"])
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=COLORS["cyan"], fontweight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color="#c9d1d9")
    axs[0, 1].set_title("M9.4 CSAT Performance Summary Matrix", fontweight="bold", pad=15)
    
    # [1, 0] Per-Class F1 Profile
    classes = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    f1s = [71.15, 0.0, 0.0, 10.53, 82.05]
    b_f1 = axs[1, 0].bar(classes, f1s, color=[COLORS["green"], COLORS["red"], COLORS["red"], COLORS["orange"], COLORS["green"]], edgecolor=COLORS["border"], width=0.45, zorder=3)
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].set_ylabel("F1 (%)", fontweight="bold")
    axs[1, 0].set_title("Per-Rating F1 Score (Data Satiation Effect)", fontweight="bold")
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for bar in b_f1:
        h = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha="center", va="bottom", color="#ffffff", fontweight="bold")
        
    # [1, 1] Strategic Role in Multi-Task M9.5
    axs[1, 1].axis("off")
    role_text = (
        ">> ROLE OF CSAT IN SENTRIFORMER M9.5 UNIFIED PIPELINE\n\n"
        "1. Triage & Risk Prioritization:\n"
        "   - Low CSAT (1-Star, 0.712 F1) triggers high urgency routing\n"
        "   - Flags dissatisfied customer tickets for senior agent review\n\n"
        "2. Coarse Sentiment Validation:\n"
        "   - Acts as an ordinal sanity check alongside 3-way sentiment\n"
        "   - Within +-1 accuracy of 81.5% is sufficient for risk scoring\n\n"
        "3. Integration Strategy for M9.5:\n"
        "   - Freeze or low-rate fine-tune the CSAT head with weight 0.1\n"
        "   - Do not let CSAT gradients dominate Category Routing (weight 1.0)"
    )
    axs[1, 1].text(0.05, 0.5, role_text, fontsize=10.5, family="monospace", va="center", color="#c9d1d9",
                   bbox=dict(boxstyle="round,pad=1.0", facecolor="#161b22", edgecolor=COLORS["cyan"], linewidth=1.5))
    axs[1, 1].set_title("M9.5 Multi-Task Integration Role", fontweight="bold", pad=15)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_csat_performance_dashboard.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("=" * 80)
    print("  GENERATING M9.4 CSAT VISUALIZATION SUITE")
    print("=" * 80)
    plot_01_distribution_imbalance()
    plot_02_csat_f1_breakdown()
    plot_03_csat_dashboard()
    print("\n✅ All CSAT visualization charts generated successfully!")


if __name__ == "__main__":
    main()
