"""Regenerate every data figure of the SentriFormer paper from the project's own metric files (read-only).
Usage: python -I make_figures.py <path-to-sentriformer-dir> <out-figures-dir>
Every number plotted is read from a JSON/JSONL file; nothing is typed in by hand except the M-series test
values, which are read from the E0/E1 metric files too.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SF, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
WR = os.path.join(SF, "wsp_runs")
J = lambda *p: json.load(open(os.path.join(WR, *p), encoding="utf-8"))
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
W1 = 3.45  # IEEE single-column width (in)
C = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#e39b1f", "#7f7f7f"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig); print("wrote", name)


# 1. Pretraining curve (span corruption)
tr, va = [], []
for line in open(os.path.join(SF, "checkpoints", "training_log.jsonl"), encoding="utf-8"):
    e = json.loads(line)
    if e.get("event") == "train_step": tr.append((e["step"], e["loss"]))
    elif "val_loss" in e: va.append((e["step"], e["val_loss"]))
tr = np.array(tr); k = 15
sm = np.convolve(tr[:, 1], np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(figsize=(W1, 1.75))
ax.plot(tr[k - 1:, 0] / 1e3, sm, color=C[0], lw=1, label="train (moving avg.)")
if va:
    va = np.array(va); ax.plot(va[:, 0] / 1e3, va[:, 1], "o-", color=C[1], ms=2.5, lw=1, label="validation")
ax.set_xlabel("optimizer step ($\\times10^3$)"); ax.set_ylabel("span-corruption loss"); ax.set_ylim(1.8, 6)
ax.legend(frameon=False); ax.grid(alpha=.25)
save(fig, "pretrain_loss")

# 2. Sealed-test category accuracy across the development history
e0 = J("e0", "test", "e0_metrics.json"); e1 = J("e1", "final_test", "e1_metrics.json")
t3 = J("eval_full", "T3_test", "metrics.json"); t4 = J("eval_full", "T4d_test", "metrics.json")


def find_acc(d, key_hint):
    """E0/E1 files nest per-checkpoint; locate the category accuracy at 512 tokens."""
    out = {}
    def walk(x, path):
        if isinstance(x, dict):
            if "accuracy" in x and "macro_f1_sklearn" in x and "category" in "/".join(path) and "512" in "/".join(path):
                out["/".join(path)] = (x["accuracy"], x["macro_f1_sklearn"])
            for kk, v in x.items(): walk(v, path + [kk])
    walk(d, []); return out


acc_e0 = find_acc(e0, ""); acc_e1 = find_acc(e1, "")
json.dump({"e0": acc_e0, "e1": acc_e1}, open(os.path.join(OUT, "_history_sources.json"), "w"), indent=1)
pick = lambda d, name: next(v for kk, v in d.items() if name in kk)
stages = [("M9.2\nspecialist", pick(acc_e0, "classifier_best")), ("M9.5\njoint", pick(acc_e0, "multitask_best")),
          ("M10.3\nSFT", pick(acc_e0, "m10_3_sft_best")), ("E1\nridge map", list(acc_e1.values())[0]),
          ("T3", (t3["category_len512"]["accuracy"], t3["category_len512"]["macro_f1_sklearn"])),
          ("T4d", (t4["category_len512"]["accuracy"], t4["category_len512"]["macro_f1_sklearn"]))]
sup = np.array(t4["category_len512"]["confusion_matrix"]).sum(1); maj = sup.max() / sup.sum()
fig, ax = plt.subplots(figsize=(W1, 1.9))
x = np.arange(len(stages)); a = [s[1][0] * 100 for s in stages]; f = [s[1][1] * 100 for s in stages]
ax.bar(x - .19, a, .38, color=C[0], label="accuracy"); ax.bar(x + .19, f, .38, color=C[4], label="macro-F1")
for i, v in enumerate(a): ax.text(i - .19, v + 1, f"{v:.1f}", ha="center", fontsize=5.8)
for i, v in enumerate(f): ax.text(i + .19, v + 1, f"{v:.1f}", ha="center", fontsize=5.8)
ax.axhline(maj * 100, ls="--", color=C[5], lw=.8); ax.text(5.45, maj * 100 + 1.2, f"majority {maj*100:.1f}%", ha="right", fontsize=5.8, color=C[5])
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in stages]); ax.set_ylabel("test score (%)"); ax.set_ylim(0, 92)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.12))
save(fig, "history_category")

# 3. Per-class F1 (T4d test), classes with support >= 500
pc = t4["category_len512"]["per_class"]
rows = sorted([(k2, v["f1"], v["support"]) for k2, v in pc.items() if v["support"] >= 500], key=lambda r: r[1])
short = lambda s: (s[:34] + "...") if len(s) > 37 else s
fig, ax = plt.subplots(figsize=(W1, 2.9))
ax.barh(range(len(rows)), [r[1] for r in rows], color=C[0])
for i, r in enumerate(rows): ax.text(r[1] + .01, i, f"{r[1]:.2f} (n={r[2]:,})", va="center", fontsize=5.3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([short(r[0]) for r in rows], fontsize=5.6)
ax.set_xlim(0, 1.18); ax.set_xlabel("F1 on sealed test (T4d)")
save(fig, "perclass_f1")

# 4. Confusion matrices: sentiment and CSAT (row-normalised)
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.7), gridspec_kw={"width_ratios": [3, 4.2]})
for ax, cm, lab, ttl in [(axs[0], t4["sentiment"]["all"]["confusion_matrix"], ["neg", "neu", "pos"], "Sentiment (n=20,453)"),
                         (axs[1], t4["csat"]["confusion_matrix"], ["1", "2", "3", "4", "5"], "CSAT stars (n=870)")]:
    cm = np.array(cm, float); rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=5.2, color="white" if rn[i, j] > .6 else "black")
    ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    ax.set_xlabel("predicted"); ax.set_title(ttl, fontsize=7); ax.spines[:].set_visible(True)
axs[0].set_ylabel("true")
save(fig, "confusion_sent_csat")

# 5. Reliability diagram for routing (T4d test)
rel = t4["category_len512"]["reliability"]
fig, ax = plt.subplots(figsize=(1.7, 1.7))
cf = [b["conf"] for b in rel]; ac = [b["acc"] for b in rel]; n = np.array([b["n"] for b in rel])
ax.plot([0, 1], [0, 1], ls="--", color=C[5], lw=.8)
ax.scatter(cf, ac, s=4 + 40 * n / n.max(), color=C[0], zorder=3); ax.plot(cf, ac, color=C[0], lw=.8)
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(.04, .9, f"ECE={t4['category_len512']['ece_15bin']:.3f}", fontsize=6)
save(fig, "reliability")

# 6. Reply-quality: portal loop (test, T3 vs T4d) and T5a detectors (validation, replication)
p3 = J("t4", "eval", "T3_test", "metrics.json"); p4 = J("t4", "eval", "T4d_test", "metrics.json")
json.dump({"T3": p3, "T4d": p4}, open(os.path.join(OUT, "_t4_test_metrics_dump.json"), "w"), indent=1, default=str)
rt = J("t5", "t5a_retest.json")["models"]
fig, ax = plt.subplots(figsize=(W1, 1.75))
mets = [("broken_link", "broken link"), ("generic_whole", "whole-reply\ngeneric"), ("generic_phrase", "generic\nphrase"),
        ("portal", "portal\nloop"), ("action", "action\nword")]
x = np.arange(len(mets))
for i, (m, lab) in enumerate([("T4d", "T4d (parent)"), ("T5a_ctrl", "T5a-ctrl (unfiltered)"), ("T5a", "T5a (deployed)")]):
    v = [rt[m][k2] * 100 for k2, _ in mets]
    ax.bar(x + (i - 1) * .26, v, .26, color=[C[5], C[4], C[0]][i], label=lab)
    for xi, vi in zip(x, v): ax.text(xi + (i - 1) * .26, vi + .8, f"{vi:.1f}", ha="center", fontsize=4.8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets]); ax.set_ylabel("rate (%) on 500 val rows"); ax.set_ylim(0, 52)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15))
save(fig, "reply_quality_t5a")

# 7. Phase-T training dynamics: val-subset category acc and dialogue val loss vs step
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.55))
for i, run in enumerate(["T3", "T4a", "T4d", "T5a"]):
    h = J("train", run, "eval_history.json")
    s = [e["step"] / 1e3 for e in h]
    axs[0].plot(s, [e["category"]["acc"] * 100 for e in h], "-o", ms=1.6, lw=.9, color=C[i], label=run)
    axs[1].plot(s, [e["dialogue"]["val_loss"] for e in h], "-o", ms=1.6, lw=.9, color=C[i], label=run)
axs[0].set_ylabel("val-subset cat. acc (%)"); axs[1].set_ylabel("dialogue val loss")
for ax in axs: ax.set_xlabel("step ($\\times10^3$)"); ax.grid(alpha=.25)
axs[0].legend(frameon=False, ncol=2, fontsize=5.5)
fig.tight_layout(w_pad=.8)
save(fig, "phaseT_curves")

# 8. Tokenizer vocabulary ablation
tb = json.load(open(os.path.join(SF, "tokenizer", "tokenizer_benchmark.json"), encoding="utf-8"))
json.dump(tb, open(os.path.join(OUT, "_tokenizer_benchmark_copy.json"), "w"), indent=1)
print("done")
