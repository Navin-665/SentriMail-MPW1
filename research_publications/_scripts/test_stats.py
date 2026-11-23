"""Bootstrap CIs on sealed-test predictions already saved by the evaluation harness (no model is run).
Also routing accuracy after merging CFPB legacy product names into their current equivalents (taxonomy revision).
Usage: python -I test_stats.py <sentriformer_dir> <out_json> [baseline_routing_preds.npz]
"""
import json, sys, os
import numpy as np

SF, OUT = sys.argv[1], sys.argv[2]
E = os.path.join(SF, "wsp_runs", "eval_full")
lab = json.load(open(os.path.join(SF, "data", "supervised", "classification_labels.json"), encoding="utf-8"))["label2id"]
L = lambda run, t: np.load(os.path.join(E, f"{run}_test", f"preds_{t}.npz"), allow_pickle=True)
rng = np.random.default_rng(0); B = 2000


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fp + fn: f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        else: f.append(0.0)
    return float(np.mean(f))


def boot(y, p, p0=None, k=None, n_boot=B):
    n = len(y); acc, d_acc, mf = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a = np.mean(p[i] == y[i]); acc.append(a)
        if p0 is not None: d_acc.append(a - np.mean(p0[i] == y[i]))
        if k is not None and n < 50000: mf.append(macro_f1(y[i], p[i], k))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return {"acc": float(np.mean(p == y)), "acc_ci95": q(acc), "diff_vs_ref_acc": None if p0 is None else float(np.mean(p == y) - np.mean(p0 == y)),
            "diff_ci95": q(d_acc), "macro_f1_ci95": q(mf)}


R = {}
for task, k in [("category_len512", 33), ("csat", 5), ("sentiment", 3)]:
    a, b = L("T4d", task), L("T3", task)
    y, p, p0 = a["targets"].astype(int), a["preds"].astype(int), b["preds"].astype(int)
    assert (b["targets"].astype(int) == y).all()
    R[task] = boot(y, p, p0, k, n_boot=1000 if task.startswith("category") else B)
    if task == "csat":
        R[task]["within1"] = float(np.mean(np.abs(p - y) <= 1))
print(json.dumps(R, indent=1))

# taxonomy merge: CFPB renamed products in 2017; old and new names are the same product family
merge = {
    "Credit reporting": "CREDIT_REPORTING", "Credit reporting or other personal consumer reports": "CREDIT_REPORTING",
    "Credit reporting, credit repair services, or other personal consumer reports": "CREDIT_REPORTING",
    "Payday loan": "PAYDAY", "Payday loan, title loan, or personal loan": "PAYDAY",
    "Payday loan, title loan, personal loan, or advance loan": "PAYDAY",
    "Credit card": "CARD", "Credit card or prepaid card": "CARD", "Prepaid card": "CARD",
    "Money transfers": "MONEY_TRANSFER", "Money transfer, virtual currency, or money service": "MONEY_TRANSFER", "Virtual currency": "MONEY_TRANSFER",
    "Bank account or service": "BANK_ACCOUNT", "Checking or savings account": "BANK_ACCOUNT",
}
id2name = {v: k for k, v in lab.items()}
grp = {i: merge.get(n, n) for i, n in id2name.items()}
gid = {g: j for j, g in enumerate(sorted(set(grp.values())))}
to = np.vectorize(lambda i: gid[grp[int(i)]])
a = L("T4d", "category_len512"); y, p = to(a["targets"]), to(a["preds"])
R["merged_taxonomy"] = {"n_classes": len(gid), **boot(y, p, None, None, 1000), "macro_f1": macro_f1(y, p, len(gid))}
for run in ["T3"]:
    b = L(run, "category_len512"); R["merged_taxonomy"][f"{run}_acc"] = float(np.mean(to(b["preds"]) == to(b["targets"])))
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    z = np.load(sys.argv[3]); yb, pb = z["targets"].astype(int), z["preds"].astype(int)
    assert len(yb) == len(a["targets"])
    R["baseline_routing"] = boot(a["targets"].astype(int), a["preds"].astype(int), pb, None, 1000)
    R["baseline_routing"]["baseline_acc"] = float(np.mean(pb == yb))
    R["baseline_routing"]["baseline_merged_acc"] = float(np.mean(to(pb) == to(yb)))
    R["baseline_routing"]["baseline_macro_f1"] = macro_f1(yb, pb, 33)
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps(R["merged_taxonomy"], indent=1))
