"""Classical baselines on the SAME splits as SentriFormer (local only; corpus never leaves the machine).
Routing: hashed word 1-2gram TF-IDF-like features + linear model, streamed over the FULL 2.2M train set.
Sentiment / CSAT: TF-IDF 1-2gram + logistic regression. Hyper-parameter (alpha / C) chosen on VAL; TEST scored once.
Usage: python -I baselines.py <sentriformer_dir> <out_json>
"""
import json, sys, time, os
import numpy as np, pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SF, OUT = sys.argv[1], sys.argv[2]
SUP, AUX = os.path.join(SF, "data", "supervised"), os.path.join(SF, "wsp_runs", "data")
labels = json.load(open(os.path.join(SUP, "classification_labels.json"), encoding="utf-8"))
R = {}


def score(y, p):
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"), "n": int(len(y))}


def rd(path, cols):
    return pq.read_table(path, columns=cols).to_pandas()


# ---------------- routing (33 classes), full train, streamed
name2id = labels["label2id"]
t0 = time.time()
hv = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm=None, lowercase=True)
tr = rd(os.path.join(SUP, "classification_train.parquet"), ["text", "product"])
ytr = tr["product"].map(name2id).to_numpy(); Xtr_txt = tr["text"].tolist(); del tr
idf = TfidfTransformer(sublinear_tf=True)
idx = np.random.RandomState(0).permutation(len(Xtr_txt))
idf.fit(hv.transform([Xtr_txt[i] for i in idx[:300000]]))  # IDF statistics from a 300k sample
va = rd(os.path.join(SUP, "classification_val.parquet"), ["text", "product"]); Xva = idf.transform(hv.transform(va["text"].tolist())); yva = va["product"].map(name2id).to_numpy()
classes = np.arange(len(name2id)); best = None
for alpha in [1e-6, 1e-5]:
    clf = SGDClassifier(loss="modified_huber", alpha=alpha, random_state=0)
    for ep in range(1):
        for s in range(0, len(idx), 200000):
            b = idx[s:s + 200000]
            clf.partial_fit(idf.transform(hv.transform([Xtr_txt[i] for i in b])), ytr[b], classes=classes)
    v = accuracy_score(yva, clf.predict(Xva)); print("routing alpha", alpha, "val acc", v, flush=True)
    if best is None or v > best[0]: best = (v, alpha, clf)
te = rd(os.path.join(SUP, "classification_test.parquet"), ["text", "product"])
yte = te["product"].map(name2id).to_numpy(); pte = best[2].predict(idf.transform(hv.transform(te["text"].tolist())))
R["routing"] = {"model": "hashed 1-2gram TF-IDF + linear (modified Huber SGD), full train", "alpha": best[1],
                "val_accuracy": best[0], "test": score(yte, pte), "minutes": (time.time() - t0) / 60}
np.savez_compressed(OUT.replace(".json", "_routing_preds.npz"), preds=pte, targets=yte)
print("routing", R["routing"], flush=True)
del Xtr_txt


# ---------------- sentiment / CSAT
def small_task(name, tr_p, va_p, te_p, ycol, mapper=None):
    t0 = time.time()
    tr, va, te = rd(tr_p, ["text", ycol]), rd(va_p, ["text", ycol]), rd(te_p, ["text", ycol])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["text"]); Xva = vec.transform(va["text"]); Xte = vec.transform(te["text"])
    ytr, yva, yte = [d[ycol].to_numpy() for d in (tr, va, te)]
    best = None
    for C in [0.3, 1.0, 3.0]:
        for cw in [None, "balanced"]:
            m = LogisticRegression(C=C, max_iter=2000, class_weight=cw).fit(Xtr, ytr)
            v = f1_score(yva, m.predict(Xva), average="macro")
            if best is None or v > best[0]: best = (v, C, cw, m)
    p = best[3].predict(Xte)
    r = {"model": "TF-IDF 1-2gram + logistic regression", "C": best[1], "class_weight": best[2], "val_macro_f1": best[0],
         "test": score(yte, p), "minutes": (time.time() - t0) / 60}
    if name == "csat":
        r["test"]["within_1"] = float(np.mean(np.abs(p.astype(int) - yte.astype(int)) <= 1))
        r["test"]["mae"] = float(np.mean(np.abs(p.astype(int) - yte.astype(int))))
    R[name] = r; print(name, r, flush=True)


small_task("sentiment", *[os.path.join(AUX, f"sentiment_{s}.parquet") for s in ("train", "val", "test")], "label")
small_task("csat", *[os.path.join(SUP, f"csat_{s}.parquet") for s in ("train", "val", "test")], "csat_score")
json.dump(R, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT)
