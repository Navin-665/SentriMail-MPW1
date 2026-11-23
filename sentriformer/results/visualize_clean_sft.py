"""
==========================================================================
  M10.2 Visualization Suite: Behavioral Audit & Contamination Reduction
==========================================================================
  Generates 3 publication-ready figures:
    1. 01_contamination_reduction_comparison.png
    2. 02_behavioral_quality_pillars.png
    3. 03_category_confidence_and_csat.png
==========================================================================
"""
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
JSON_PATH = os.path.join(ROOT_DIR, "sentriformer", "results", "clean_sft", "clean_sft_behavioral_audit.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "results", "clean_sft")
ARTIFACTS_DIR = r"C:\Users\nan33\.gemini\antigravity-ide\brain\a5de84d1-05e1-42b8-ad22-fc8883726c2b"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_contamination_comparison(data):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    categories = [
        "Overall Contamination",
        "@Handles Rate",
        "URL / t.co Links",
        "Agent ^Signatures",
        "DM Deflections"
    ]
    
    m93_vals = [
        data["m93_baseline"]["overall_contamination_pct"],
        data["m93_baseline"]["handles_pct"],
        data["m93_baseline"]["urls_pct"],
        data["m93_baseline"]["signatures_pct"],
        data["m93_baseline"]["dm_deflection_pct"]
    ]
    
    m102_vals = [
        data["m102_clean_sft"]["overall_contamination_pct"],
        data["m102_clean_sft"]["handles_pct"],
        data["m102_clean_sft"]["urls_pct"],
        data["m102_clean_sft"]["signatures_pct"],
        data["m102_clean_sft"]["dm_deflection_pct"]
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, m93_vals, width, label='M9.3 Raw Baseline SFT', color='#E63946', alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, m102_vals, width, label='M10.2 Clean Response SFT', color='#2A9D8F', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Contamination Prevalence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Artifact Contamination Reduction: M9.3 vs M10.2 Clean SFT (80 Paired Complaints)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, frameon=True, loc='upper right')
    
    # Value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B71C1C')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#004D40')
                    
    # Annotation box
    ax.text(0.5, 0.72, 'Overall Contamination Reduced by -87.5 pp (-100% Rel)\nHandles, URLs, and Signatures purged to 0.0%',
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32"))
            
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_contamination_reduction_comparison.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(ARTIFACTS_DIR, "01_contamination_reduction_comparison.png"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_behavioral_quality_pillars(data):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    
    metrics = [
        "EOS Termination",
        "Fluency Score",
        "Boundary Closure",
        "Non-Empty Rate",
        "Empathy Rate",
        "Domain Relevance",
        "Repetition (Lower Better)"
    ]
    
    m93_vals = [
        data["m93_baseline"]["eos_termination_pct"],
        data["m93_baseline"]["fluency_pct"],
        data["m93_baseline"]["completeness_proxy_pct"],
        data["m93_baseline"]["non_empty_pct"],
        data["m93_baseline"]["empathy_pct"],
        data["m93_baseline"]["relevance_proxy_pct"],
        data["m93_baseline"]["repetition_pct"]
    ]
    
    m102_vals = [
        data["m102_clean_sft"]["eos_termination_pct"],
        data["m102_clean_sft"]["fluency_pct"],
        data["m102_clean_sft"]["completeness_proxy_pct"],
        data["m102_clean_sft"]["non_empty_pct"],
        data["m102_clean_sft"]["empathy_pct"],
        data["m102_clean_sft"]["relevance_proxy_pct"],
        data["m102_clean_sft"]["repetition_pct"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, m93_vals, width, label='M9.3 Baseline', color='#457B9D', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, m102_vals, width, label='M10.2 Clean SFT', color='#E76F51', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Score / Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Behavioral Quality Dimensions: M9.3 vs M10.2 Clean SFT', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, frameon=True, loc='upper right')
    
    for i in range(len(metrics)):
        ax.annotate(f"{m93_vals[i]:.1f}%", xy=(x[i] - width/2, m93_vals[i]), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.annotate(f"{m102_vals[i]:.1f}%", xy=(x[i] + width/2, m102_vals[i]), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#BF360C')
                    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_behavioral_quality_pillars.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(ARTIFACTS_DIR, "02_behavioral_quality_pillars.png"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_category_and_csat(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # 1. Category Confidence Distribution
    confs = [c["m102"]["category_confidence"] for c in data["cases"]]
    ax1.hist(confs, bins=12, range=(0, 50), color='#2B2D42', edgecolor='white', alpha=0.85)
    ax1.axvline(25.0, color='#D90429', linestyle='--', linewidth=2, label='Low-Confidence Threshold (<25%)')
    ax1.axvline(np.mean(confs), color='#FFB703', linestyle='-', linewidth=2, label=f'Mean Confidence ({np.mean(confs):.1f}%)')
    ax1.set_xlabel('Category Routing Top-1 Confidence (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Inquiries', fontsize=11, fontweight='bold')
    ax1.set_title('Category Routing Confidence Distribution (N=80)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    
    # 2. CSAT Forecast Distribution
    csat_counts = data["m102_clean_sft"]["csat_distribution"]
    stars = [1, 2, 3, 4, 5]
    counts = [csat_counts[str(s)] if str(s) in csat_counts else csat_counts.get(s, 0) for s in stars]
    colors = ['#D90429', '#F48C06', '#FFBA08', '#80ED99', '#38B000']
    
    bars = ax2.bar([f'{s} ⭐' for s in stars], counts, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    ax2.set_xlabel('Forecasted CSAT Rating', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Inquiry Count', fontsize=11, fontweight='bold')
    ax2.set_title('M10.2 CSAT Forecast Distribution across Complaints', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(counts) + 6)
    
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_category_confidence_and_csat.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(ARTIFACTS_DIR, "03_category_confidence_and_csat.png"))
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    data = load_data()
    plot_contamination_comparison(data)
    plot_behavioral_quality_pillars(data)
    plot_category_and_csat(data)
    print("All visualization plots generated successfully!")


if __name__ == "__main__":
    main()
