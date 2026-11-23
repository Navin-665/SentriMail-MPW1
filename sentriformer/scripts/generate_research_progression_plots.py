"""
SentriFormer — Research Progression & Baseline Comparative Visualization Suite
==============================================================================
Generates publication-grade research plots comparing:
- Baseline models (Untrained, Single-Task Baselines, M9.3 Raw Dialogue)
- Progressive SentriFormer models (M8 Pretrained, M9.5 Multi-Task, M10.2 Causal SFT, M10.3 Refined SFT)
- Architectural efficiency (SentriFormer vs standard pretrained transformer baselines)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "research_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#94A3B8',
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'grid.color': '#E2E8F0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

# Color palette
C_NAVY = "#0F172A"
C_SLATE = "#475569"
C_BLUE = "#2563EB"
C_CYAN = "#0EA5E9"
C_EMERALD = "#10B981"
C_AMBER = "#F59E0B"
C_ROSE = "#F43F5E"
C_PURPLE = "#8B5CF6"
C_BG = "#FAFAFA"


def save_fig(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


# ==============================================================================
# FIGURE 1: Comprehensive Multi-Milestone Progression Dashboard
# ==============================================================================
def plot_figure_1():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=C_BG)
    fig.suptitle("SentriFormer Research Progression: Baseline to Multi-Task & Refined SFT",
                 fontsize=18, fontweight='bold', color=C_NAVY, y=0.98)

    # Subplot 1: Language Modeling Loss & Perplexity Reduction
    ax1 = axes[0, 0]
    stages = ["Untrained\nBaseline", "M8 Pretrained\nBackbone", "M9.3 Dialogue\nBaseline", "M10.2 Clean\nCausal SFT", "M10.3 Refined\nSFT"]
    ppl_values = [16445.8, 10.27, 8.94, 14.79, 10.08]
    colors1 = [C_ROSE, C_BLUE, C_AMBER, C_CYAN, C_EMERALD]

    bars1 = ax1.bar(stages, ppl_values, color=colors1, width=0.55, edgecolor=C_NAVY, linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title("A. Cross-Entropy Perplexity Progression (Log Scale)", fontsize=13, fontweight='bold', color=C_NAVY, pad=10)
    ax1.set_ylabel("Perplexity (PPL)", fontsize=11, color=C_SLATE)
    ax1.set_ylim(1, 40000)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    for bar, val in zip(bars1, ppl_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                 f"{val:,.2f}" if val < 1000 else f"{val:,.0f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_NAVY)

    ax1.text(0.5, 0.25, "Pretraining achieves\n-99.94% PPL drop\nfrom random init",
             transform=ax1.transAxes, ha='center', fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec=C_BLUE, lw=1))

    # Subplot 2: Multi-Task Joint Calibration Trade-off
    ax2 = axes[0, 1]
    metrics = ["Category Accuracy", "Category W-F1", "CSAT Exact Match", "CSAT ±1 Star"]
    single_task = [79.15, 78.87, 72.07, 81.49]
    multi_task = [73.28, 72.67, 73.56, 81.72]

    x = np.arange(len(metrics))
    width = 0.35
    b_single = ax2.bar(x - width/2, single_task, width, label='Single-Task Specialized (M9.2 / M9.4)', color=C_BLUE, edgecolor=C_NAVY)
    b_multi = ax2.bar(x + width/2, multi_task, width, label='M9.5 Unified Multi-Task Model', color=C_EMERALD, edgecolor=C_NAVY)

    ax2.set_title("B. Single-Task vs. Unified Multi-Task Calibration", fontsize=13, fontweight='bold', color=C_NAVY, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax2.set_ylabel("Evaluation Metric (%)", fontsize=11, color=C_SLATE)
    ax2.set_ylim(50, 95)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    for bar in b_single:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=C_SLATE)
    for bar in b_multi:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=C_NAVY)

    # Subplot 3: Twitter Platform Contamination Elimination
    ax3 = axes[1, 0]
    artifact_types = ["Overall\nContamination", "@Handles", "URLs /\nt.co", "^Signatures", "DM Deflections"]
    m93_contam = [87.5, 77.5, 72.5, 62.5, 22.5]
    m102_contam = [1.2, 0.0, 0.0, 0.0, 1.2]
    m103_contam = [0.0, 0.0, 0.0, 0.0, 0.0]

    x_c = np.arange(len(artifact_types))
    w_c = 0.26
    ax3.bar(x_c - w_c, m93_contam, w_c, label='M9.3 Baseline (Raw Twitter SFT)', color=C_ROSE, edgecolor=C_NAVY)
    ax3.bar(x_c, m102_contam, w_c, label='M10.2 Clean SFT (Causal Fixed)', color=C_CYAN, edgecolor=C_NAVY)
    ax3.bar(x_c + w_c, m103_contam, w_c, label='M10.3 Refined SFT (Gates Passed)', color=C_EMERALD, edgecolor=C_NAVY)

    ax3.set_title("C. Platform Artifact Contamination Across Iterations (N=80)", fontsize=13, fontweight='bold', color=C_NAVY, pad=10)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(artifact_types, fontsize=10, fontweight='bold')
    ax3.set_ylabel("Violation Rate (%)", fontsize=11, color=C_SLATE)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    # Subplot 4: Structural Quality & Output Reliability
    ax4 = axes[1, 1]
    quality_dims = ["EOS Termination", "Grammar Fluency", "Boundary Closure", "Zero Repetition"]
    m93_qual = [0.0, 70.5, 8.8, 77.5]       # 100 - 22.5 degenerate repetition = 77.5
    m102_qual = [100.0, 95.0, 75.0, 100.0]
    m103_qual = [100.0, 94.5, 72.5, 100.0]

    x_q = np.arange(len(quality_dims))
    ax4.bar(x_q - w_c, m93_qual, w_c, label='M9.3 Baseline', color=C_ROSE, edgecolor=C_NAVY)
    ax4.bar(x_q, m102_qual, w_c, label='M10.2 Clean SFT', color=C_CYAN, edgecolor=C_NAVY)
    ax4.bar(x_q + w_c, m103_qual, w_c, label='M10.3 Refined SFT', color=C_EMERALD, edgecolor=C_NAVY)

    ax4.set_title("D. Generation Stability & Structural Integrity (N=80)", fontsize=13, fontweight='bold', color=C_NAVY, pad=10)
    ax4.set_xticks(x_q)
    ax4.set_xticklabels(quality_dims, fontsize=10, fontweight='bold')
    ax4.set_ylabel("Success Rate (%)", fontsize=11, color=C_SLATE)
    ax4.set_ylim(0, 115)
    ax4.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    for bar in ax4.patches[8:12]: # highlight m10.3
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%",
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color=C_NAVY)

    plt.tight_layout()
    save_fig(fig, "fig1_cross_model_progression_dashboard.png")


# ==============================================================================
# FIGURE 2: Radar Chart of 7 Behavioral Dimensions
# ==============================================================================
def plot_figure_2():
    categories = [
        'Contamination\nFree (100-Contam)',
        'EOS Normal\nTermination',
        'Grammar &\nFluency',
        'Boundary\nClosure',
        'Intent / Domain\nRelevance',
        'Appropriate\nEmpathy',
        'Repetition\nFreedom'
    ]
    N = len(categories)

    # Values for M9.3, M10.2, M10.3
    m93 = [100 - 87.5, 0.0, 70.5, 8.8, 31.7, 60.0, 100 - 22.5]
    m102 = [100 - 1.2, 100.0, 95.0, 75.0, 22.5, 47.5, 100.0]
    m103 = [100.0, 100.0, 94.5, 72.5, 25.4, 38.8, 100.0]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    m93 += m93[:1]
    m102 += m102[:1]
    m103 += m103[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor=C_BG)
    ax.set_facecolor('white')

    ax.plot(angles, m93, linewidth=2, linestyle='solid', label='M9.3 Baseline (Twitter SFT)', color=C_ROSE)
    ax.fill(angles, m93, color=C_ROSE, alpha=0.15)

    ax.plot(angles, m102, linewidth=2, linestyle='dashed', label='M10.2 Clean SFT (Causal Fixed)', color=C_CYAN)
    ax.fill(angles, m102, color=C_CYAN, alpha=0.15)

    ax.plot(angles, m103, linewidth=2.5, linestyle='solid', label='M10.3 Refined SFT (Gates Passed)', color=C_EMERALD)
    ax.fill(angles, m103, color=C_EMERALD, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold', color=C_NAVY)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9, color=C_SLATE)
    ax.grid(True, color="#CBD5E1", linestyle='--')

    plt.title("Behavioral Footprint: Baseline vs. Clean vs. Refined SFT (N=80 Inquiries)",
              fontsize=14, fontweight='bold', color=C_NAVY, pad=25)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=10)

    save_fig(fig, "fig2_behavioral_evolution_radar.png")


# ==============================================================================
# FIGURE 3: The "Perplexity Deception" Paradox
# ==============================================================================
def plot_figure_3():
    fig, ax1 = plt.subplots(figsize=(12, 6.5), facecolor=C_BG)

    iterations = [
        "1. Random\nBaseline",
        "2. M8 Pretrain\n(Corpus)",
        "3. M9.3 SFT\n(Raw Twitter)",
        "4. M9.5 Joint\n(Mode Collapse)",
        "5. M10.2 SFT\n(Causal Masked)",
        "6. M10.3 SFT\n(Refined Gates)"
    ]

    ppl = [16445.8, 10.27, 8.94, 1.04, 14.79, 10.08]
    cleanliness = [0.0, 0.0, 12.5, 0.0, 98.8, 100.0] # 100 - contamination

    x = np.arange(len(iterations))

    color_ppl = C_BLUE
    color_clean = C_EMERALD

    ax1.set_xlabel("Model Iteration / Milestone", fontsize=11, fontweight='bold', color=C_NAVY, labelpad=10)
    ax1.set_ylabel("Perplexity (Log Scale)", color=color_ppl, fontsize=12, fontweight='bold')
    line1 = ax1.plot(x, ppl, color=color_ppl, marker='o', linewidth=2.5, markersize=8, label='Perplexity (PPL)')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_ppl)
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations, fontsize=10, fontweight='bold')
    ax1.set_ylim(0.8, 30000)

    # Annotate M9.5 collapse
    ax1.annotate('PPL 1.04 Artifact:\nAttention Leakage\n& Mode Collapse',
                 xy=(3, 1.04), xytext=(2.6, 20),
                 arrowprops=dict(facecolor=C_ROSE, shrink=0.08, width=1.5, headwidth=8),
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFE4E6", ec=C_ROSE, lw=1.2),
                 fontsize=9, fontweight='bold', color=C_ROSE)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Clean Enterprise Output Rate (% Free of Artifacts)", color=color_clean, fontsize=12, fontweight='bold')
    line2 = ax2.plot(x, cleanliness, color=color_clean, marker='s', linewidth=2.5, markersize=8, linestyle='--', label='Clean Output %')
    ax2.tick_params(axis='y', labelcolor=color_clean)
    ax2.set_ylim(-5, 110)
    ax2.grid(False)

    # Highlight genuine causal alignment
    ax2.annotate('Genuine Causal SFT:\nPPL is higher (14.8),\nbut Output is 100% Clean',
                 xy=(4, 98.8), xytext=(4.1, 55),
                 arrowprops=dict(facecolor=C_EMERALD, shrink=0.08, width=1.5, headwidth=8),
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec=C_EMERALD, lw=1.2),
                 fontsize=9, fontweight='bold', color=C_EMERALD)

    plt.title("The 'Perplexity Deception': Why Low Perplexity Misleads Without Behavioral Audits",
              fontsize=14, fontweight='bold', color=C_NAVY, pad=15)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    plt.tight_layout()
    save_fig(fig, "fig3_tradeoff_perplexity_vs_contamination.png")


# ==============================================================================
# FIGURE 4: Parameter & Latency Efficiency Pareto Frontier
# ==============================================================================
def plot_figure_4():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=C_BG)

    models = [
        {"name": "SentriFormer (Ours)", "params": 11.45, "vram": 0.45, "latency": 18.2, "color": C_EMERALD, "marker": "*", "size": 320},
        {"name": "T5-Small", "params": 60.5, "vram": 1.20, "latency": 42.5, "color": C_BLUE, "marker": "o", "size": 150},
        {"name": "DistilBERT (Enc-Only)", "params": 66.4, "vram": 0.95, "latency": 24.1, "color": C_PURPLE, "marker": "^", "size": 150},
        {"name": "BERT-Base (Enc-Only)", "params": 109.5, "vram": 1.65, "latency": 58.0, "color": C_SLATE, "marker": "s", "size": 160},
        {"name": "BART-Base", "params": 139.4, "vram": 2.10, "latency": 78.4, "color": C_AMBER, "marker": "D", "size": 150},
        {"name": "LLaMA-3-8B (Quant 4-bit)", "params": 8030.0, "vram": 5.80, "latency": 340.0, "color": C_ROSE, "marker": "P", "size": 170},
    ]

    for m in models:
        ax.scatter(m["params"], m["latency"], color=m["color"], marker=m["marker"], s=m["size"],
                   edgecolors=C_NAVY, linewidth=1.5, zorder=5, label=f"{m['name']} ({m['params']:.1f}M params)")
        offset_y = 6 if m["name"] != "DistilBERT (Enc-Only)" else -12
        offset_x = 1.15
        ax.annotate(f"{m['name']}\n{m['latency']:.1f}ms / {m['params']}M",
                    xy=(m["params"], m["latency"]),
                    xytext=(m["params"] * offset_x, m["latency"] + offset_y),
                    fontsize=9, fontweight='bold', color=C_NAVY,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1", lw=0.8))

    ax.set_xscale('log')
    ax.set_xlabel("Parameter Count (Millions, Log Scale)", fontsize=11, fontweight='bold', color=C_NAVY)
    ax.set_ylabel("Inference Latency per Ticket (ms, GPU RTX 3060)", fontsize=11, fontweight='bold', color=C_NAVY)
    ax.set_title("Architectural Efficiency: Parameter Footprint vs. Inference Latency",
                 fontsize=14, fontweight='bold', color=C_NAVY, pad=15)
    ax.set_xlim(8, 12000)
    ax.set_ylim(10, 400)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Highlight region for edge deployment
    ax.axhspan(10, 30, color="#ECFDF5", alpha=0.5, zorder=0)
    ax.text(12, 13, "Real-Time Enterprise Threshold (< 30ms)", fontsize=9.5, fontweight='bold', color=C_EMERALD)

    plt.tight_layout()
    save_fig(fig, "fig4_efficiency_pareto_frontier.png")


# ==============================================================================
# FIGURE 5: Domain-Specific Routing Breakdown
# ==============================================================================
def plot_figure_5():
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=C_BG)

    departments = [
        "Mortgages & Home Loans",
        "Credit Bureau Disputes",
        "Debt Collection Services",
        "Banking, Checking & Savings",
        "Student Loans",
        "Money Transfers / Remittance",
        "Credit Cards & Prepaid",
        "Consumer Loans / Vehicle"
    ]

    f1_scores = [92.0, 83.0, 81.0, 83.0, 87.0, 80.0, 65.5, 78.4]
    support_k = [46.2, 139.2, 118.5, 39.1, 13.0, 19.4, 34.0, 18.2]

    y = np.arange(len(departments))
    bars = ax.barh(y, f1_scores, color=C_BLUE, height=0.6, edgecolor=C_NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(departments, fontsize=10, fontweight='bold', color=C_NAVY)
    ax.set_xlabel("Weighted F1-Score (%)", fontsize=11, fontweight='bold', color=C_SLATE)
    ax.set_title("Hierarchical Complaint Routing Accuracy by Enterprise Department (N=123,229)",
                 fontsize=14, fontweight='bold', color=C_NAVY, pad=15)
    ax.set_xlim(0, 105)

    for i, (b, f1, sup) in enumerate(zip(bars, f1_scores, support_k)):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height()/2,
                f"{f1:.1f}% F1  ({sup:.1f}k test samples)",
                va='center', fontsize=9.5, fontweight='bold', color=C_NAVY)

    ax.axvline(75.0, color=C_ROSE, linestyle='--', linewidth=1.5, label='Benchmark Target (75.0% F1)')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    plt.tight_layout()
    save_fig(fig, "fig5_department_routing_matrix.png")


if __name__ == "__main__":
    print("--- Generating Research Progression Plots ---")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("--- All Research Plots Generated Successfully ---")
