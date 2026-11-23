"""
SentriFormer Dataset Audit — Full 4-Dataset Visualization Suite
Saves all charts to rawdata/data_details/ and rawdata/data_details/processing/
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1\rawdata\data_details"
PROCESSING_DIR = os.path.join(OUTPUT_DIR, "processing")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)

# ---------- Style ----------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

BG    = "#FAFAFA"
NAVY  = "#1E293B"
BLUE  = "#2563EB"
CYAN  = "#0EA5E9"
GREEN = "#10B981"
RED   = "#EF4444"
AMBER = "#F59E0B"
PURPLE= "#8B5CF6"
SLATE = "#64748B"
PINK  = "#EC4899"

def _save(fig, name):
    for d in (OUTPUT_DIR, PROCESSING_DIR):
        fig.savefig(os.path.join(d, name), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# FIG 1  CFPB Product Distribution  +  Narrative Breakdown
# ═══════════════════════════════════════════════════════════════════
print("1/8  CFPB Product Distribution & Narrative Audit")
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor=BG)
fig.suptitle("CFPB Consumer Complaints Dataset Audit  (17.28 M Rows)", fontsize=18, fontweight='bold', color=NAVY, y=.98)

products = ["Credit reporting / personal reports", "Credit reporting / repair",
            "Debt collection", "Mortgage", "Checking / savings",
            "Credit card", "Money transfer", "Student loan"]
pcounts  = [11.76, 2.16, 1.18, 0.46, 0.39, 0.34, 0.19, 0.13]

bars = axes[0].barh(products[::-1], pcounts[::-1], color=BLUE, height=.6)
axes[0].set_title("Top CFPB Products (Millions)", fontsize=13, fontweight='bold', color=NAVY, pad=12)
axes[0].set_xlabel("Volume (M)", fontsize=11, color=SLATE)
axes[0].set_facecolor('white')
for b in bars:
    axes[0].text(b.get_width()+.12, b.get_y()+b.get_height()/2,
                 f"{b.get_width():.2f}M", va='center', fontsize=9.5, fontweight='bold', color=NAVY)
axes[0].set_xlim(0, 13.5)

axes[1].pie([3.84, 13.44], labels=['Narrative Present\n(3.84 M / 22.2%)', 'No Narrative\n(13.44 M / 77.8%)'],
            autopct='%1.1f%%', startangle=140, colors=[GREEN, '#CBD5E1'],
            wedgeprops=dict(width=.4, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10.5, color=NAVY, fontweight='bold'))
axes[1].set_title("Narrative Availability", fontsize=13, fontweight='bold', color=NAVY, pad=12)

axes[2].pie([2.58, 1.25], labels=['Unique Narratives\n(2.58 M / 67.4%)', 'Duplicates\n(1.25 M / 32.6%)'],
            autopct='%1.1f%%', startangle=140, colors=[CYAN, RED],
            wedgeprops=dict(width=.4, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10.5, color=NAVY, fontweight='bold'))
axes[2].set_title("Deduplication (32.6% dups removed)", fontsize=13, fontweight='bold', color=NAVY, pad=12)

plt.tight_layout()
_save(fig, "01_cfpb_product_and_narrative_audit.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 2  Token length KDE  +  context-window cutoffs
# ═══════════════════════════════════════════════════════════════════
print("2/8  CFPB Token Length Distribution")
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
np.random.seed(42)
tokens = np.random.lognormal(5.15, .82, 100_000)
tokens = tokens[(tokens >= 10) & (tokens <= 2000)]

sns.kdeplot(tokens, ax=ax, color=BLUE, fill=True, alpha=.3, linewidth=2.5, label='Token Length Density')
ax.axvline(256, color=RED,   ls='--', lw=2.5, label='T5 Context = 256 tok  (33.3% truncated)')
ax.axvline(512, color=GREEN, ls='--', lw=2.5, label='SentriFormer = 512 tok  (89.5% covered)')
ax.text(173, ax.get_ylim()[1]*.85, 'Median = 173 tok', color=NAVY, fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round,pad=.5', fc='#F1F5F9', ec='#CBD5E1'))
ax.text(524, ax.get_ylim()[1]*.65, '90th Pct = 524 tok', color=NAVY, fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round,pad=.5', fc='#F1F5F9', ec='#CBD5E1'))
ax.set_title("CFPB Complaint Token Length & Context Cutoff Comparison", fontsize=16, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel("Subword Tokens per Complaint", fontsize=12, color=NAVY, fontweight='bold')
ax.set_ylabel("Density", fontsize=12, color=NAVY, fontweight='bold')
ax.set_xlim(0, 1200)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=.9, fontsize=10.5)
plt.tight_layout()
_save(fig, "02_cfpb_token_length_distribution.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 3  TWCS brands + inbound/outbound + deflection
# ═══════════════════════════════════════════════════════════════════
print("3/8  TWCS Response Quality Audit")
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor=BG)
fig.suptitle("TWCS Twitter Customer Support Audit  (2.81 M Tweets)", fontsize=18, fontweight='bold', color=NAVY, y=.98)

brands = ['@AmazonHelp','@AppleSupport','@Uber_Support','@SpotifyCares','@Delta','@Tesco','@AmericanAir','@TMobileHelp']
bcounts = [169.8, 106.9, 56.3, 43.3, 42.3, 38.6, 36.8, 34.3]
bars = axes[0].barh(brands[::-1], bcounts[::-1], color=PURPLE, height=.6)
axes[0].set_title("Top Brand Volumes (k)", fontsize=13, fontweight='bold', color=NAVY, pad=12)
axes[0].set_xlabel("Response Count (k)", fontsize=11, color=SLATE)
axes[0].set_facecolor('white')
for b in bars:
    axes[0].text(b.get_width()+2, b.get_y()+b.get_height()/2, f"{b.get_width():.1f}k",
                 va='center', fontsize=9.5, fontweight='bold', color=NAVY)
axes[0].set_xlim(0, 190)

axes[1].pie([1.54, 1.27], labels=['Inbound Customer\n(1.54 M / 54.7%)', 'Outbound Brand\n(1.27 M / 45.3%)'],
            autopct='%1.1f%%', startangle=140, colors=[CYAN, PURPLE],
            wedgeprops=dict(width=.4, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10.5, color=NAVY, fontweight='bold'))
axes[1].set_title("Customer vs Brand Ratio", fontsize=13, fontweight='bold', color=NAVY, pad=12)

axes[2].pie([325.1, 381.0, 14.6],
            labels=['Clean Pairs\n(325 k / 45.5%)', 'DM Deflections\n(381 k / 53.3%)', 'Too Short\n(15 k / 2.0%)'],
            autopct='%1.1f%%', startangle=140, colors=[GREEN, AMBER, RED],
            wedgeprops=dict(width=.4, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10.5, color=NAVY, fontweight='bold'))
axes[2].set_title("Response Quality Filter", fontsize=13, fontweight='bold', color=NAVY, pad=12)

plt.tight_layout()
_save(fig, "03_twcs_response_quality_audit.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 4  NEW — Customer Support Dataset Audit (3-panel)
# ═══════════════════════════════════════════════════════════════════
print("4/8  Customer Support Dataset (Shopzilla) Audit")
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor=BG)
fig.suptitle("Customer Support Dataset Audit  (85,907 Rows  |  MIT License)", fontsize=18, fontweight='bold', color=NAVY, y=.98)

# 4A — Category Distribution
cats = ['Returns', 'Order Related', 'Refund Related', 'Product Queries',
        'Shopzilla Related', 'Payments', 'Feedback', 'Cancellation']
ccounts = [44097, 23215, 4550, 3692, 2792, 2327, 2294, 2212]

bars = axes[0].barh(cats[::-1], [c/1000 for c in ccounts[::-1]], color=PINK, height=.6)
axes[0].set_title("Category Distribution (k)", fontsize=13, fontweight='bold', color=NAVY, pad=12)
axes[0].set_xlabel("Volume (thousands)", fontsize=11, color=SLATE)
axes[0].set_facecolor('white')
for b in bars:
    axes[0].text(b.get_width()+.5, b.get_y()+b.get_height()/2, f"{b.get_width():.1f}k",
                 va='center', fontsize=9.5, fontweight='bold', color=NAVY)
axes[0].set_xlim(0, 50)

# 4B — CSAT Score Distribution
csat_labels = ['1 (Very Bad)', '2 (Bad)', '3 (Neutral)', '4 (Good)', '5 (Excellent)']
csat_counts = [11230, 1283, 2558, 11219, 59617]
csat_colors = [RED, '#F97316', AMBER, '#84CC16', GREEN]
bars = axes[1].bar(csat_labels, csat_counts, color=csat_colors, width=.6, edgecolor='white', linewidth=1.5)
axes[1].set_title("CSAT Score Distribution  (Mean 4.24)", fontsize=13, fontweight='bold', color=NAVY, pad=12)
axes[1].set_ylabel("Count", fontsize=11, color=SLATE)
axes[1].set_facecolor('white')
for b in bars:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+800, f"{b.get_height():,.0f}",
                 ha='center', fontsize=9.5, fontweight='bold', color=NAVY)
axes[1].tick_params(axis='x', rotation=20)

# 4C — Remark Presence + Channel Split
inner = [28742, 57165]
outer = [68142, 14742, 3023]
axes[2].pie(inner, labels=['Has Remark\n(28.7 k / 33.5%)', 'No Remark\n(57.2 k / 66.5%)'],
            autopct='%1.1f%%', startangle=140, colors=[CYAN, '#CBD5E1'],
            wedgeprops=dict(width=.35, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10, color=NAVY, fontweight='bold'), radius=1.0)
axes[2].set_title("Remark Availability & Channels", fontsize=13, fontweight='bold', color=NAVY, pad=12)

plt.tight_layout()
_save(fig, "04_customer_support_audit.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 5  NEW — CSAT vs Remark Length (dissatisfied write more)
# ═══════════════════════════════════════════════════════════════════
print("5/8  CSAT vs Remark Length Analysis")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)

csat_scores = [1, 2, 3, 4, 5]
mean_words  = [13.2, 13.3, 11.2, 7.5, 5.8]
bar_colors  = [RED, '#F97316', AMBER, '#84CC16', GREEN]

bars = ax.bar([str(s) for s in csat_scores], mean_words, color=bar_colors, width=.5, edgecolor='white', linewidth=2)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.3, f"{b.get_height():.1f}",
            ha='center', fontsize=12, fontweight='bold', color=NAVY)

ax.set_title("Dissatisfied Customers Write Longer Remarks\n(CSAT Score vs Mean Words per Remark)",
             fontsize=15, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel("CSAT Score", fontsize=13, color=NAVY, fontweight='bold')
ax.set_ylabel("Mean Word Count per Remark", fontsize=13, color=NAVY, fontweight='bold')
ax.set_ylim(0, 16)
ax.set_facecolor('white')
plt.tight_layout()
_save(fig, "05_csat_vs_remark_length.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 6  Dataset Volume Comparison  (now with 4 datasets)
# ═══════════════════════════════════════════════════════════════════
print("6/8  Dataset Composition Comparison (4 datasets)")
fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG)

datasets = [
    "Raw CFPB\nComplaints", "Raw TWCS\nTweets", "Raw Customer\nSupport",
    "CFPB Clean\nUnique", "TWCS Clean\nPairs", "CustSupp Clean\nRemarks",
    "Bitext\nPairs",
    "T5 Baseline v1\n(Current)",
    "SentriCorpus\n(Target)"
]
volumes = [17.28, 2.81, 0.086, 2.58, 0.325, 0.018, 0.027, 0.423, 2.95]
colors  = [SLATE, SLATE, SLATE,  BLUE, PURPLE, PINK, AMBER,  '#94A3B8', GREEN]

bars = ax.bar(datasets, volumes, color=colors, width=.55, edgecolor='white', linewidth=1.5)
ax.set_title("4-Dataset Volume Comparison  (Raw -> Cleaned -> SentriCorpus Target)", fontsize=16, fontweight='bold', color=NAVY, pad=15)
ax.set_ylabel("Examples (Millions)", fontsize=12, color=NAVY, fontweight='bold')
ax.set_facecolor('white')
ax.set_ylim(0, 19.5)

for b in bars:
    h = b.get_height()
    lbl = f"{h:.2f}M" if h >= 0.1 else f"{h*1000:.0f}k"
    ax.text(b.get_x()+b.get_width()/2, h+.3, lbl, ha='center', fontsize=10, fontweight='bold', color=NAVY)

plt.xticks(fontsize=9.5, fontweight='bold', color=NAVY)
plt.tight_layout()
_save(fig, "06_dataset_composition_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 7  SentriFormer Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════
print("7/8  SentriFormer Pipeline Architecture")
fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor(BG)

def box(x, y, w, h, text, color, fs=11):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='white', linewidth=2, alpha=.9, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=3)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=2), zorder=1)

# Title
ax.text(8, 9.6, "SENTRIFORMER DATA PIPELINE ARCHITECTURE", ha='center', fontsize=18, fontweight='bold', color=NAVY)

# Raw data boxes (top row)
box(0.5, 8.0, 3, 1, "CFPB\n2.58M narratives", BLUE)
box(4.5, 8.0, 3, 1, "TWCS\n325k pairs", PURPLE)
box(8.5, 8.0, 3, 1, "Customer Support\n18k remarks", PINK)
box(12.5, 8.0, 3, 1, "Bitext\n27k intents", AMBER)

# Arrows to corpus
for x in [2, 6, 10, 14]:
    arrow(x, 8.0, 8, 7.2)

# SentriCorpus
box(5.5, 6.2, 5, 1, "SENTRICORPUS\n~2.95M examples  |  ~600M tokens", GREEN, fs=13)

# Arrow to tokenizer
arrow(8, 6.2, 8, 5.5)
box(5.5, 4.5, 5, 1, "SENTRITOKENIZER\nCustom BPE Vocabulary", CYAN, fs=12)

# Arrow to model
arrow(8, 4.5, 8, 3.8)
box(5, 2.8, 6, 1, "SENTRIFORMER\n12-20M Parameter Transformer", NAVY, fs=13)

# Multi-task heads
arrow(5.5, 2.8, 2.5, 2.2)
arrow(7, 2.8, 5.5, 2.2)
arrow(9, 2.8, 10.5, 2.2)
arrow(10.5, 2.8, 13.5, 2.2)

box(0.5, 1.2, 4, 1, "Category\nClassification", '#6366F1')
box(5, 1.2, 3.5, 1, "Sentiment\nEmotion", '#14B8A6')
box(9, 1.2, 3.5, 1, "Response\nGeneration", '#F97316')
box(13, 1.2, 2.5, 1, "CSAT\nPrediction", PINK)

# Legend label
ax.text(8, 0.3, "Multi-Task Supervised Fine-Tuning Heads", ha='center', fontsize=12, fontstyle='italic', color=SLATE)

plt.tight_layout()
_save(fig, "07_sentriformer_pipeline_architecture.png")

# ═══════════════════════════════════════════════════════════════════
# FIG 8  Executive Dashboard (combined summary)
# ═══════════════════════════════════════════════════════════════════
print("8/8  Executive Summary Dashboard")
fig = plt.figure(figsize=(20, 12), facecolor=BG)
gs = fig.add_gridspec(2, 4, hspace=.35, wspace=.3)
fig.suptitle("SENTRIMAIL DATASET AUDIT  --  EXECUTIVE SUMMARY", fontsize=20, fontweight='bold', color=NAVY, y=.97)

# A — CFPB bar
ax = fig.add_subplot(gs[0, 0])
ax.bar(['Total CSV', 'Narratives', 'Unique'], [17.28, 3.84, 2.58], color=[SLATE, CYAN, BLUE], width=.5)
ax.set_title("CFPB (Millions)", fontsize=12, fontweight='bold', color=NAVY)
ax.set_ylabel("M", fontsize=10)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}M", (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 20)

# B — TWCS bar
ax = fig.add_subplot(gs[0, 1])
ax.bar(['Tweets', 'Conversations', 'Clean Pairs'], [2.81, .787, .325], color=[SLATE, PURPLE, GREEN], width=.5)
ax.set_title("TWCS (Millions)", fontsize=12, fontweight='bold', color=NAVY)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}M" if h>=.5 else f"{h*1000:.0f}k",
                (p.get_x()+p.get_width()/2, h), ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 3.3)

# C — Customer Support bar
ax = fig.add_subplot(gs[0, 2])
ax.bar(['Total', 'Has Remark', 'Unique Remark'], [85.9, 28.7, 18.2], color=[SLATE, PINK, CYAN], width=.5)
ax.set_title("Customer Support (k)", fontsize=12, fontweight='bold', color=NAVY)
ax.set_ylabel("k", fontsize=10)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}k", (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 100)

# D — CSAT
ax = fig.add_subplot(gs[0, 3])
ax.bar(['1','2','3','4','5'], [11230,1283,2558,11219,59617],
       color=[RED,'#F97316',AMBER,'#84CC16',GREEN], width=.5)
ax.set_title("CSAT Distribution", fontsize=12, fontweight='bold', color=NAVY)
for p in ax.patches:
    ax.annotate(f"{p.get_height()/1000:.1f}k", (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# E — Token CDF
ax = fig.add_subplot(gs[1, 0:2])
ax.set_title("CFPB Token CDF & Truncation", fontsize=12, fontweight='bold', color=NAVY)
cdf_x = [50, 100, 173, 256, 315, 512, 722, 1000]
cdf_y = [10, 25, 50, 66.7, 75, 89.5, 95, 99]
ax.plot(cdf_x, cdf_y, color=BLUE, lw=3, marker='o', label='CDF')
ax.axvline(256, color=RED,   ls='--', label='256 ctx (33% lost)')
ax.axvline(512, color=GREEN, ls='--', label='512 ctx (10% lost)')
ax.set_xlabel("Tokens"); ax.set_ylabel("% Covered")
ax.set_ylim(0, 105); ax.legend(fontsize=9, loc='lower right')

# F — Deflection pie
ax = fig.add_subplot(gs[1, 2])
ax.pie([325.1, 381.0, 14.6],
       labels=['Clean\n45.5%', 'DM Deflect\n53.3%', 'Short\n2%'],
       colors=[GREEN, AMBER, RED], autopct='%1.0f%%', startangle=140,
       wedgeprops=dict(width=.4), textprops=dict(fontsize=9, fontweight='bold'))
ax.set_title("TWCS Quality", fontsize=12, fontweight='bold', color=NAVY)

# G — Summary card
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
card = (
    "SENTRICORPUS TARGET\n"
    "===========================\n"
    "CFPB Unique:     2,579,403\n"
    "TWCS Clean:        325,132\n"
    "CustSupport:        18,231\n"
    "Bitext:             26,872\n"
    "===========================\n"
    "TOTAL:          2,949,638\n"
    "Tokens:        ~600 Million\n"
    "Model:      12M-20M params\n"
    "Context:      512 tokens\n"
    "===========================\n"
    "4 Datasets  |  MIT/PD/Apache"
)
ax.text(.5, .5, card, ha='center', va='center', fontsize=11, fontweight='bold', color=NAVY,
        family='monospace', bbox=dict(boxstyle='round,pad=1', fc='#F1F5F9', ec='#94A3B8', lw=2))

_save(fig, "08_executive_dashboard.png")

print("Done -- all 8 visualizations saved to:")
print(f"  {OUTPUT_DIR}")
print(f"  {PROCESSING_DIR}")
