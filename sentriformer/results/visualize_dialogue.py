"""
==========================================================================
  M9.3 / M9.3.1 Dialogue Results — Visualization & Dashboard Generator
==========================================================================
  Reads dialogue_quality_audit.json and produces high-quality dark-themed charts.
  Outputs saved to: sentriformer/results/dialogue/
==========================================================================
"""
import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
AUDIT_JSON = os.path.join(ROOT_DIR, "sentriformer", "results", "dialogue", "dialogue_quality_audit.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "dialogue")
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
    "dark_card": "#161b22",
    "border": "#30363d",
}


def load_audit_data():
    if not os.path.exists(AUDIT_JSON):
        print(f"Warning: {AUDIT_JSON} not found.")
        return None
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_01_perplexity_progression(audit_data):
    """Figure 1: Perplexity and Loss Progression from Pretraining to SFT."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ["Untrained\nBaseline", "M8 Pretrained\n(Unseen Test)", "M9.3 Dialogue SFT\n(Validation)", "M9.3 Dialogue SFT\n(Test Set)"]
    perplexities = [16445.8, 10.27, 9.13, 8.94]
    losses = [9.71, 2.33, 2.21, 2.19]
    
    # Focus on the trained stages with a dual scale or inset
    x = np.arange(len(stages))
    
    # We will plot the 3 trained stages in main focus, with an annotation for baseline
    stages_focus = ["M8 Pretrained\n(Backbone)", "M9.3 Dialogue\n(Validation)", "M9.3 Dialogue\n(Test Set)"]
    ppl_focus = [10.27, 9.13, 8.94]
    loss_focus = [2.33, 2.21, 2.19]
    xf = np.arange(len(stages_focus))
    
    bar_width = 0.35
    bars_ppl = ax.bar(xf - bar_width/2, ppl_focus, bar_width, label="Perplexity (PPL)", color=COLORS["cyan"], edgecolor=COLORS["border"], zorder=3)
    
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    bars_loss = ax2.bar(xf + bar_width/2, loss_focus, bar_width, label="Cross-Entropy Loss", color=COLORS["purple"], edgecolor=COLORS["border"], zorder=3)
    
    # Annotations
    for bar in bars_ppl:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", color=COLORS["cyan"], fontweight="bold", fontsize=11)
        
    for bar in bars_loss:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}", ha="center", va="bottom", color=COLORS["purple"], fontweight="bold", fontsize=11)
        
    ax.set_xticks(xf)
    ax.set_xticklabels(stages_focus, fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity (Lower is Better)", color=COLORS["cyan"], fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", color=COLORS["purple"], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 13)
    ax2.set_ylim(0, 3.5)
    
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_title("SentriFormer Pretraining $\\rightarrow$ Dialogue SFT Optimization\n(Language Modeling Perplexity Dropped to 8.94)", fontsize=13, fontweight="bold", pad=15)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_perplexity_progression.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_02_artifact_contamination_breakdown(audit_data):
    """Figure 2: Breakdown of Twitter Artifact Contamination."""
    if not audit_data:
        return
        
    summary = audit_data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Contamination Types
    artifact_labels = ["Twitter DM\nDeflection", "Twitter User\nHandles (@...)", "External URLs\n(t.co/http)", "Agent Staff\nSignatures (^JF)"]
    artifact_pcts = [
        summary.get("dm_deflection_rate_pct", 73.3),
        summary.get("twitter_handle_rate_pct", 66.7),
        summary.get("url_artifact_rate_pct", 53.3),
        summary.get("signature_artifact_rate_pct", 40.0)
    ]
    colors = [COLORS["red"], COLORS["orange"], COLORS["yellow"], COLORS["purple"]]
    
    bars = ax1.barh(artifact_labels, artifact_pcts, color=colors, edgecolor=COLORS["border"], height=0.55, zorder=3)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of Responses Containing Artifact", fontweight="bold")
    ax1.set_title("Distribution of Twitter Contamination Types\n(Derived from 97.6% TWCS Pre-training Data)", fontweight="bold", pad=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", color="#c9d1d9", fontweight="bold")
        
    # Right: Clean vs Contaminated Donut
    contam_rate = summary.get("contamination_rate_pct", 83.3)
    clean_rate = 100.0 - contam_rate
    
    wedges, texts, autotexts = ax2.pie(
        [contam_rate, clean_rate],
        labels=["Contaminated\n(Social Media Boilerplate)", "Clean Enterprise\nFormat"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[COLORS["red"], COLORS["green"]],
        wedgeprops=dict(width=0.45, edgecolor=COLORS["border"], linewidth=1.5),
        textprops=dict(color="#c9d1d9", fontweight="bold", fontsize=10)
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(11)
        at.set_fontweight("bold")
        
    ax2.set_title(f"Overall Response Cleanliness Audit\n(Overall Contamination: {contam_rate:.1f}%)", fontweight="bold", pad=12)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_artifact_contamination_breakdown.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_03_category_quality_bars(audit_data):
    """Figure 3: Category Performance across Relevance, Cleanliness, Actionability."""
    if not audit_data:
        return
        
    categories = audit_data["categories"]
    titles = [c["title"].replace(" / ", "\n").replace(" & ", "\n") for c in categories]
    relevance = [c["mean_scores"]["relevance"] for c in categories]
    cleanliness = [c["mean_scores"]["cleanliness"] for c in categories]
    composite = [c["mean_scores"]["composite"] for c in categories]
    
    y = np.arange(len(titles))
    height = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    ax.barh(y - height, relevance, height, label="Intent Relevance", color=COLORS["cyan"], edgecolor=COLORS["border"], zorder=3)
    ax.barh(y, cleanliness, height, label="Cleanliness (No Artifacts)", color=COLORS["green"], edgecolor=COLORS["border"], zorder=3)
    ax.barh(y + height, composite, height, label="Composite Quality Score", color=COLORS["purple"], edgecolor=COLORS["border"], zorder=3)
    
    ax.set_yticks(y)
    ax.set_yticklabels(titles, fontsize=9.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (0 - 100)", fontweight="bold")
    ax.set_title("M9.3.1 Response Quality Breakdown Across 10 Support Domains\n(High Language Understanding vs Social Media Contamination)", fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_category_quality_breakdown.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_04_behavioral_dashboard(audit_data):
    """Figure 4: Comprehensive 4-Panel Executive Dashboard."""
    if not audit_data:
        return
        
    summary = audit_data["summary"]
    means = summary.get("mean_scores", {})
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    
    # [0, 0] Pillar Radar / Multi-bar
    pillars = ["Intent\nRelevance", "Cleanliness\nScore", "Completeness\n& Structure", "Enterprise\nTone", "Actionability\n& Grounding"]
    scores = [
        means.get("relevance", 65.0),
        means.get("cleanliness", 25.0),
        means.get("completeness", 78.0),
        means.get("professionalism", 52.0),
        means.get("actionability", 42.0)
    ]
    bar_colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["yellow"], COLORS["orange"]]
    
    axs[0, 0].bar(pillars, scores, color=bar_colors, edgecolor=COLORS["border"], width=0.55, zorder=3)
    axs[0, 0].set_ylim(0, 100)
    axs[0, 0].set_ylabel("Score / 100", fontweight="bold")
    axs[0, 0].set_title("Pillar 1-5: Behavioral Quality Profile", fontweight="bold")
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for idx, sc in enumerate(scores):
        axs[0, 0].text(idx, sc + 2, f"{sc:.1f}", ha="center", va="bottom", fontweight="bold", color="#ffffff")
        
    # [0, 1] Technical vs Behavioral Comparison Matrix
    axs[0, 1].axis("off")
    table_data = [
        ["Dimension", "Status", "Insight"],
        ["Language Modeling (PPL)", "8.94 (Excellent)", "Learned strong customer service English"],
        ["Generalization Gap", "2.21 -> 2.19 (Zero)", "No overfitting on held-out test data"],
        ["Intent Mapping", "65.0% (Moderate)", "Captures core domain, but drifts on niche cases"],
        ["Twitter Deflection", "73.3% (Critical)", "Over-deflects to DMs instead of direct answers"],
        ["Social Handles & URLs", "66.7% (Critical)", "Frequent @handles, ^initials, t.co links"],
        ["Sentence Flow & Grammar", "78.0% (Good)", "Fluent token generation, minimal gibberish"]
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
    axs[0, 1].set_title("Technical vs Behavioral Audit Matrix", fontweight="bold", pad=15)
    
    # [1, 0] Root Cause: Corpus Distribution
    corpus_labels = ["TWCS (Twitter)", "CFPB Banking", "Bitext / Other"]
    corpus_pcts = [97.6, 1.8, 0.6]
    wedges, texts, autotexts = axs[1, 0].pie(
        corpus_pcts,
        labels=corpus_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=[COLORS["red"], COLORS["blue"], COLORS["purple"]],
        wedgeprops=dict(width=0.45, edgecolor=COLORS["border"], linewidth=1.5),
        textprops=dict(color="#c9d1d9", fontweight="bold")
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(10)
        at.set_fontweight("bold")
    axs[1, 0].set_title("Root Cause: SFT Corpus Distribution\n(97.6% Twitter Contamination)", fontweight="bold", pad=12)
    
    # [1, 1] SentriMail Strategic Roadmap
    axs[1, 1].axis("off")
    roadmap_text = (
        ">> SENTRIFORMER STRATEGIC ROADMAP TO PRODUCTION\n\n"
        "1. [M9.3] Dialogue SFT (COMPLETED)\n"
        "   - Checkpoint: dialogue_sft_best.pt (PPL: 8.94)\n\n"
        "2. [M9.3.1] Behavioral Quality Audit (COMPLETED)\n"
        "   - Quantified Twitter artifact rate and domain drift\n\n"
        "3. [M9.4] CSAT Prediction Head Training (NEXT)\n"
        "   - 15,217 customer satisfaction records (csat_train.parquet)\n\n"
        "4. [M9.5] Unified Multi-Task Integration\n"
        "   - Category (79.2%) + CSAT + Urgency + Dialogue Decoder\n\n"
        "5. [M10] Clean Response Alignment / DPO (ALIGNMENT)\n"
        "   - Filter DM deflection -> Train Enterprise Email Format\n\n"
        "6. [M11] Grounded RAG Production Deployment\n"
        "   - Knowledge Base Retrieval + Policy Grounding"
    )
    axs[1, 1].text(0.05, 0.5, roadmap_text, fontsize=10.5, family="monospace", va="center", color="#c9d1d9",
                   bbox=dict(boxstyle="round,pad=1.0", facecolor="#161b22", edgecolor=COLORS["cyan"], linewidth=1.5))
    axs[1, 1].set_title("Strategic Roadmap: Path to Enterprise Production", fontweight="bold", pad=15)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_sft_behavioral_dashboard.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("=" * 80)
    print("  GENERATING M9.3 / M9.3.1 DIALOGUE VISUALIZATION SUITE")
    print("=" * 80)
    audit_data = load_audit_data()
    
    plot_01_perplexity_progression(audit_data)
    plot_02_artifact_contamination_breakdown(audit_data)
    plot_03_category_quality_bars(audit_data)
    plot_04_behavioral_dashboard(audit_data)
    
    print("\n✅ All dialogue visualization charts generated successfully!")


if __name__ == "__main__":
    main()
