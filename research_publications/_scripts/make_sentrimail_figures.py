"""SentriMail paper figures from measured benchmark JSON. Usage: python make_sentrimail_figures.py <runs_dir> <out_dir>"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
L = json.load(open(os.path.join(RUNS, "latency_benchmark.json")))["results"]
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
stages = [("sentriformer_heads_plus_1_token", "SentriFormer\nheads"), ("distilbert_sst2_sentiment", "DistilBERT\nsentiment"),
          ("distilroberta_emotion", "DistilRoBERTa\nemotion"), ("sentriformer_full_analyze_greedy100", "SentriFormer\nheads+reply"),
          ("flan_t5_small_reply_sampled", "flan-T5-small\nauto-reply")]
fig, ax = plt.subplots(figsize=(3.45, 1.9))
x = range(len(stages))
for i, (dev, col, lab) in enumerate([("cpu", "#1f4e79", "CPU"), ("cuda", "#e39b1f", "GPU (RTX 3060)")]):
    med = [L[dev][k]["median_ms"] for k, _ in stages]; p95 = [L[dev][k]["p95_ms"] for k, _ in stages]
    xs = [j + (i - .5) * .38 for j in x]
    ax.bar(xs, med, .38, color=col, label=lab, yerr=[[0] * len(med), [p - m for p, m in zip(p95, med)]],
           error_kw={"lw": .6, "capsize": 1.5})
    for xx, m in zip(xs, med): ax.text(xx, m * 1.12, f"{m:.0f}", ha="center", fontsize=5.5)
ax.set_yscale("log"); ax.set_ylabel("median latency (ms), whisker = p95")
ax.set_xticks(list(x)); ax.set_xticklabels([s for _, s in stages], fontsize=5.8); ax.set_ylim(8, 12000)
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
fig.savefig(os.path.join(OUT, "latency.pdf")); print("ok")
