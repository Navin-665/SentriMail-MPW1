"""Training & metrics figure gallery (PNG) for the repo, regenerated from the project's own logs/metric files.
Usage: python -I make_training_gallery.py <sentriformer_dir> <out_dir> <test_stats.json>
"""
import json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SF, OUT, TS = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
WR = os.path.join(SF, "wsp_runs")
J = lambda *p: json.load(open(os.path.join(*p), encoding="utf-8"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25})
C = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#e39b1f", "#7f7f7f", "#2a9d8f"]
NAMES = {"T1": "weighted CE + label smoothing", "T2": "head re-init + gate", "T3": "SF-MT (joint multi-task)",
         "T4a": "+ role-verified context", "T4a_ctrl": "control (single message)", "T4d": "SF-Ctx (+ portal filter)",
         "T4d_ctrl": "control (no portal filter)", "T5a": "SentriFormer final (+ reply filter)", "T5a_ctrl": "control (no reply filter)",
         "T6": "PMI filter (rejected)", "T6_ctrl": "control (PMI)"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png")); plt.close(fig); print("wrote", name)


# 1. Tokenizer training: vocabulary-size comparison
tb = J(SF, "tokenizer", "tokenizer_benchmark.json")
vs = [k for k in ["16k", "24k", "32k"] if k in tb]; srcs = list(tb[vs[0]]["per_source"])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.arange(len(srcs))
for i, v in enumerate(vs):
    axs[0].bar(x + (i - 1) * .27, [tb[v]["per_source"][s]["chars_per_token"] for s in srcs], .27, color=C[i], label=v)
    axs[1].bar(x + (i - 1) * .27, [tb[v]["per_source"][s]["avg_tokens_per_text"] for s in srcs], .27, color=C[i], label=v)
axs[0].set_xticks(x, srcs, rotation=15); axs[0].set_title("Compression (characters per token, higher = better)"); axs[0].legend()
axs[1].set_xticks(x, srcs, rotation=15); axs[1].set_title("Average tokens per text"); axs[1].legend()
ov = [tb[v]["overall"] for v in vs]
axs[2].bar(vs, [o["chars_per_token"] for o in ov], color=C[:len(vs)])
for i, o in enumerate(ov): axs[2].text(i, o["chars_per_token"] + .03, f"{o['chars_per_token']:.2f} c/t\nUNK {o['unk_rate_pct']}%", ha="center")
axs[2].set_ylim(0, max(o["chars_per_token"] for o in ov) * 1.25); axs[2].set_title("Overall (16k selected)")
fig.suptitle("Byte-level BPE tokenizer training: vocabulary size ablation", fontweight="bold")
save(fig, "01_tokenizer_vocab_ablation")

# 2. Pre-training (span corruption): loss, LR, grad norm
tr, va = [], []
for line in open(os.path.join(SF, "checkpoints", "training_log.jsonl"), encoding="utf-8"):
    e = json.loads(line)
    if e.get("event") == "train_step": tr.append((e["step"], e["loss"], e.get("lr", np.nan), e.get("grad_norm", np.nan)))
    elif "val_loss" in e: va.append((e["step"], e["val_loss"]))
tr = np.array(tr, float); va = np.array(va, float); k = 25
sm = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(tr[k - 1:, 0] / 1e3, sm(tr[:, 1]), color=C[0], label="train (smoothed)")
if len(va): axs[0].plot(va[:, 0] / 1e3, va[:, 1], "o-", color=C[1], ms=3, label="validation")
axs[0].set_ylim(1.8, 5); axs[0].set_title("Span-corruption loss"); axs[0].set_xlabel("step (k)"); axs[0].legend()
axs[1].plot(tr[:, 0] / 1e3, tr[:, 2], color=C[2]); axs[1].set_title("Learning rate (cosine, 2k warm-up)"); axs[1].set_xlabel("step (k)")
axs[2].plot(tr[k - 1:, 0] / 1e3, sm(tr[:, 3]), color=C[3]); axs[2].set_title("Gradient norm (smoothed)"); axs[2].set_xlabel("step (k)")
fig.suptitle("Stage 1: pre-training from scratch (149,200 steps, 14.8 h, test PPL 10.27)", fontweight="bold")
save(fig, "02_pretraining_curves")

# 3. Multi-task fine-tuning: per-task training losses (from train_log.txt)
pat = re.compile(r"step (\d+) cat=([\d.]+) csat=([\d.]+) sent=([\d.]+) urg=([\d.]+) dial=([\d.]+)")
fig, axs = plt.subplots(1, 4, figsize=(18, 4))
for i, run in enumerate(["T3", "T4a", "T4d", "T5a"]):
    rows = [tuple(map(float, m.groups())) for m in pat.finditer(open(os.path.join(WR, "train", run, "train_log.txt"), encoding="utf-8").read())]
    if not rows: continue
    r = np.array(rows); kk = 5; s = lambda a: np.convolve(a, np.ones(kk) / kk, mode="valid")
    for j, (col, ttl) in enumerate([(1, "routing CE"), (3, "sentiment CE"), (2, "CSAT CE"), (5, "reply (dialogue) CE")]):
        axs[j].plot(r[kk - 1:, 0] / 1e3, s(r[:, col]), color=C[i], label=NAMES[run]); axs[j].set_title(ttl); axs[j].set_xlabel("step (k)")
axs[0].legend(fontsize=8)
fig.suptitle("Stages 2-3: multi-task training losses per task (smoothed)", fontweight="bold")
save(fig, "03_multitask_training_losses")

# 4. Multi-task validation curves (selection metrics every 1k steps)
fig, axs = plt.subplots(1, 4, figsize=(18, 4))
for i, run in enumerate(["T3", "T4a", "T4d", "T5a"]):
    h = J(WR, "train", run, "eval_history.json"); st = [e["step"] / 1e3 for e in h]
    for j, (get, ttl) in enumerate([(lambda e: e["category"]["acc"] * 100, "routing accuracy (%)"),
                                     (lambda e: e["sentiment"]["macro_f1"], "sentiment macro-F1"),
                                     (lambda e: e["csat"]["macro_f1"], "CSAT macro-F1"),
                                     (lambda e: e["dialogue"]["val_loss"], "reply val loss")]):
        axs[j].plot(st, [get(e) for e in h], "o-", ms=3, color=C[i], label=NAMES[run]); axs[j].set_title(ttl); axs[j].set_xlabel("step (k)")
axs[0].legend(fontsize=8)
fig.suptitle("Stages 2-3: validation metrics during multi-task training", fontweight="bold")
save(fig, "04_multitask_validation_curves")

# 5. Label-smoothing collapse vs fix
fig, ax = plt.subplots(figsize=(7, 4))
for i, run in enumerate(["T1", "T2", "T3"]):
    h = J(WR, "train", run, "eval_history.json")
    ax.plot([e["step"] / 1e3 for e in h], [e["category"]["acc"] * 100 for e in h], "o-", color=[C[1], C[4], C[0]][i], label=NAMES[run])
ax.set_xlabel("step (k)"); ax.set_ylabel("validation routing accuracy (%)"); ax.legend()
ax.set_title("Class-weighted CE + label smoothing collapses routing; removing smoothing fixes it", fontsize=10)
save(fig, "05_label_smoothing_collapse")

# 6. Data-filter ablations vs matched controls (validation)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for j, (a, b) in enumerate([("T4a", "T4a_ctrl"), ("T4d", "T4d_ctrl"), ("T5a", "T5a_ctrl")]):
    for i, run in enumerate([a, b]):
        h = J(WR, "train", run, "eval_history.json")
        axs[j].plot([e["step"] / 1e3 for e in h], [e["dialogue"]["val_loss"] for e in h], "o-", ms=3, color=[C[0], C[5]][i], label=NAMES[run])
    axs[j].set_title(NAMES[a]); axs[j].set_xlabel("step (k)"); axs[j].set_ylabel("reply val loss"); axs[j].legend(fontsize=8)
fig.suptitle("Dialogue-data interventions vs matched control runs", fontweight="bold")
save(fig, "06_filter_vs_control_curves")

# 7. Sealed-test metrics across stages
e0 = J(WR, "e0", "test", "e0_metrics.json"); e1 = J(WR, "e1", "final_test", "e1_metrics.json")
t3 = J(WR, "eval_full", "T3_test", "metrics.json"); t4 = J(WR, "eval_full", "T4d_test", "metrics.json")
cat = lambda d: d["category_len512"]


def find(d, key):
    out = []
    def walk(x, path):
        if isinstance(x, dict):
            if key in "/".join(path) and "accuracy" in x and "category_len512" in "/".join(path): out.append(x)
            for kk, v in x.items(): walk(v, path + [kk])
    walk(d, []); return out[0]


stages = [("Routing specialist", find(e0, "classifier_best")), ("Joint heads", find(e0, "multitask_best")),
          ("Reply fine-tune\n(drift)", find(e0, "m10_3_sft_best")), ("Closed-form\nrepair", find(e1, "")),
          ("SF-MT", cat(t3)), ("SF-Ctx", cat(t4))]
fig, ax = plt.subplots(figsize=(11, 4.5)); x = np.arange(len(stages))
a = [s[1]["accuracy"] * 100 for s in stages]; f = [s[1]["macro_f1_sklearn"] * 100 for s in stages]; w = [s[1]["weighted_f1"] * 100 for s in stages]
for off, vals, col, lab in [(-.27, a, C[0], "accuracy"), (0, w, C[2], "weighted-F1"), (.27, f, C[4], "macro-F1")]:
    ax.bar(x + off, vals, .27, color=col, label=lab)
    for xi, v in zip(x, vals): ax.text(xi + off, v + 1, f"{v:.1f}", ha="center", fontsize=8)
sup = np.array(cat(t4)["confusion_matrix"]).sum(1); ax.axhline(sup.max() / sup.sum() * 100, ls="--", color=C[5]); ax.text(5.4, 29, "majority 27.2%", ha="right", color=C[5])
ax.set_xticks(x, [s[0] for s in stages]); ax.set_ylabel("sealed test (%)"); ax.set_ylim(0, 95); ax.legend(ncol=3)
ax.set_title("Routing on the sealed test set (n = 123,229) across training stages", fontweight="bold")
save(fig, "07_test_routing_by_stage")

# 8. Final model: all task metrics with 95% CIs
ts = J(TS)
fig, ax = plt.subplots(figsize=(9, 4))
items = [("Routing acc\n(33 classes)", ts["category_len512"]["acc"], ts["category_len512"]["acc_ci95"]),
         ("Routing acc\n(24 merged)", ts["merged_taxonomy"]["acc"], ts["merged_taxonomy"]["acc_ci95"]),
         ("Sentiment acc", ts["sentiment"]["acc"], ts["sentiment"]["acc_ci95"]),
         ("Sentiment\nmacro-F1", t4["sentiment"]["all"]["macro_f1_sklearn"], ts["sentiment"]["macro_f1_ci95"]),
         ("CSAT exact", ts["csat"]["acc"], ts["csat"]["acc_ci95"]), ("CSAT within\n±1 star", ts["csat"]["within1"], None)]
for i, (lab, v, ci) in enumerate(items):
    ax.bar(i, v * 100, color=C[i % len(C)])
    if ci: ax.errorbar(i, v * 100, yerr=[[v * 100 - ci[0] * 100], [ci[1] * 100 - v * 100]], color="k", capsize=4)
    ax.text(i, v * 100 + 3, f"{v*100:.1f}", ha="center", fontweight="bold")
ax.set_xticks(range(len(items)), [i[0] for i in items]); ax.set_ylim(0, 105); ax.set_ylabel("sealed test (%)")
ax.set_title("SentriFormer (SF-Ctx) sealed-test results with 95% bootstrap CIs", fontweight="bold")
save(fig, "08_final_test_metrics_ci")

# 9. 33x33 routing confusion matrix (row-normalised)
cm = np.array(cat(t4)["confusion_matrix"], float); names = list(cat(t4)["per_class"].keys())
rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
sh = lambda s: s if len(s) <= 40 else s[:38] + "…"
ax.set_xticks(range(len(names)), [sh(n) for n in names], rotation=90, fontsize=7); ax.set_yticks(range(len(names)), [sh(n) for n in names], fontsize=7)
ax.grid(False); fig.colorbar(im, fraction=.03); ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Routing confusion matrix, sealed test (row-normalised)", fontweight="bold")
save(fig, "09_routing_confusion_matrix")

# 10. Sentiment + CSAT confusion and per-class F1
fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
for ax, cmx, lab, ttl in [(axs[0], t4["sentiment"]["all"]["confusion_matrix"], ["neg", "neu", "pos"], "Sentiment confusion"),
                          (axs[1], t4["csat"]["confusion_matrix"], ["1", "2", "3", "4", "5"], "CSAT confusion")]:
    m = np.array(cmx, float); r = m / m.sum(1, keepdims=True); ax.imshow(r, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(len(lab)):
        for j in range(len(lab)): ax.text(j, i, f"{r[i,j]:.2f}", ha="center", va="center", color="white" if r[i, j] > .6 else "black")
    ax.set_xticks(range(len(lab)), lab); ax.set_yticks(range(len(lab)), lab); ax.set_title(ttl); ax.set_xlabel("predicted"); ax.set_ylabel("true")
pcs = t4["sentiment"]["all"]["per_class"]; pcc = t4["csat"]["per_class"]
labs = [f"sent {k}" for k in pcs] + [f"CSAT {k}" for k in pcc]; vals = [v["f1"] for v in pcs.values()] + [v["f1"] for v in pcc.values()]
axs[2].barh(labs, vals, color=[C[0]] * len(pcs) + [C[4]] * len(pcc)); axs[2].set_xlim(0, 1); axs[2].set_title("Per-class F1")
for i, v in enumerate(vals): axs[2].text(v + .01, i, f"{v:.2f}", va="center")
fig.suptitle("Sentiment and satisfaction heads, sealed test", fontweight="bold")
save(fig, "10_sentiment_csat_test")

# 11. Reply quality: multi-turn portal loop (test) + reply-filter replication (validation)
rt = J(WR, "t5", "t5a_retest.json")["models"]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 2.3]})
axs[0].bar(["SF-MT", "SF-Ctx", "human agents"], [31.3, 0.7, 15], color=[C[5], C[0], C[2]])
for i, v in enumerate([31.3, 0.7, 15]): axs[0].text(i, v + .8, f"{v}%", ha="center", fontweight="bold")
axs[0].set_ylabel("portal-loop replies (%)"); axs[0].set_title("Sealed multi-turn test")
mets = [("broken_link", "broken link"), ("generic_whole", "whole-reply generic"), ("generic_phrase", "generic phrase"), ("portal", "portal loop"), ("action", "action words")]
x = np.arange(len(mets))
for i, (m, lab) in enumerate([("T4d", "SF-Ctx"), ("T5a_ctrl", "control (no filter)"), ("T5a", "SentriFormer final")]):
    v = [rt[m][k] * 100 for k, _ in mets]; axs[1].bar(x + (i - 1) * .27, v, .27, color=[C[5], C[4], C[0]][i], label=lab)
    for xi, vi in zip(x, v): axs[1].text(xi + (i - 1) * .27, vi + .5, f"{vi:.1f}", ha="center", fontsize=7)
axs[1].set_xticks(x, [l for _, l in mets]); axs[1].legend(); axs[1].set_ylabel("rate (%)"); axs[1].set_title("Reply filter, 500 fresh validation rows")
fig.suptitle("Reply quality improvements", fontweight="bold")
save(fig, "11_reply_quality")

# 12. Per-class routing F1 (all 33 classes)
pc = cat(t4)["per_class"]; rows = sorted(pc.items(), key=lambda kv: kv[1]["f1"])
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh([sh(k) for k, _ in rows], [v["f1"] for _, v in rows], color=C[0])
for i, (_, v) in enumerate(rows): ax.text(v["f1"] + .01, i, f"{v['f1']:.2f} (n={v['support']:,})", va="center", fontsize=7)
ax.set_xlim(0, 1.2); ax.set_xlabel("F1"); ax.tick_params(axis="y", labelsize=7)
ax.set_title("Per-class routing F1, sealed test (all 33 classes)", fontweight="bold")
save(fig, "12_routing_per_class_f1")
