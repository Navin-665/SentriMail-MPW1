"""
Build leakage-filtered sentiment + urgency datasets for SentriFormer head training. Read-only on sources;
writes wsp_runs/data/{sentiment,urgency}_{train,val,test}.parquet and aux_data_report.json.

Sentiment classes (matches config.num_sentiment_classes=3): 0=negative, 1=neutral, 2=positive.
  - model_training/data/processed/{train,val,test}.parquet: sentiment distilled from DistilBERT-SST2 (teacher, 2-class).
  - model_training/data/processed/emotion_supplement.parquet: GoEmotions HUMAN labels -> sentiment
    (joy->pos; anger/fear/sadness/disgust->neg; neutral->neutral; surprise dropped).  Split 80/10/10 by text hash.
Urgency: model_training validation_ecommerce (HF Ataur77/ecommerce-customer-support): Low/Medium/High/Immediate -> 0..3.
  Stratified 70/15/15, seed 0.
Leakage: train rows never in any SentriCorpus val/test (pretraining + supervised); eval rows never in any train set.
"""
import hashlib, json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(os.path.dirname(ROOT), "model_training", "data", "processed")
OUT = os.path.join(ROOT, "wsp_runs", "data")
h = lambda t: hashlib.blake2b(str(t).strip().encode("utf-8"), digest_size=12).digest()


def hashes(path, col):
    return {h(t) for t in pq.read_table(path, columns=[col])[col].to_pylist() if t}


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    os.makedirs(OUT)
    sup = os.path.join(ROOT, "data", "supervised")
    corpus_eval = hashes(os.path.join(ROOT, "data", "val.parquet"), "text") | hashes(os.path.join(ROOT, "data", "test.parquet"), "text")
    for f in ("classification_val", "classification_test", "csat_val", "csat_test"):
        corpus_eval |= hashes(os.path.join(sup, f"{f}.parquet"), "text")
    for f in ("m10_4_dialogue_val", "m10_4_dialogue_test", "m10_3_dialogue_val", "m10_3_dialogue_test"):
        corpus_eval |= hashes(os.path.join(sup, f"{f}.parquet"), "context")
    corpus_train = hashes(os.path.join(ROOT, "data", "train.parquet"), "text") | hashes(os.path.join(sup, "m10_4_dialogue_train.parquet"), "context")
    rep = {"label_provenance": {"sentiment_complaints": "pseudo-labels distilled from distilbert-base-uncased-finetuned-sst-2-english (2-class)",
                                "sentiment_goemotions": "human-annotated GoEmotions, mapped to 3-class sentiment",
                                "urgency": "HF Ataur77/ecommerce-customer-support 'Urgency' column (provenance of annotation unknown)"}}

    # ---- sentiment ----
    SMAP = {"negative": 0, "neutral": 1, "positive": 2}
    parts = {}
    for s in ("train", "val", "test"):
        d = pd.read_parquet(os.path.join(MT, f"{s}.parquet"), columns=["text", "sentiment", "source"]).dropna(subset=["text", "sentiment"])
        d = d.assign(label=d.sentiment.map(SMAP), origin="distilled_" + d.source)
        parts[s] = [d[["text", "label", "origin"]]]
    ge = pd.read_parquet(os.path.join(MT, "emotion_supplement.parquet"), columns=["text", "emotion"]).dropna()
    EMAP = {"joy": 2, "anger": 0, "fear": 0, "sadness": 0, "disgust": 0, "neutral": 1}
    ge = ge[ge.emotion.isin(EMAP)].assign(label=lambda x: x.emotion.map(EMAP), origin="goemotions_human")
    b = np.array([h(t)[0] for t in ge.text]) % 10
    for s, m in (("train", b < 8), ("val", b == 8), ("test", b == 9)):
        parts[s].append(ge[m][["text", "label", "origin"]])
    sent = {s: pd.concat(p, ignore_index=True) for s, p in parts.items()}
    tr_h = set()
    for s in ("train", "val", "test"):
        d = sent[s]; hh = d.text.map(h)
        before = len(d)
        if s == "train":
            d = d[~hh.isin(corpus_eval)]
            tr_h = set(d.text.map(h))
        else:
            d = d[~hh.isin(corpus_train) & ~hh.isin(tr_h)]
        d = d.drop_duplicates("text").reset_index(drop=True)
        sent[s] = d
        rep[f"sentiment_{s}"] = {"rows_before_filter": before, "rows": len(d), "label_counts": {int(k): int(v) for k, v in d.label.value_counts().items()},
                                 "origin_counts": d.origin.value_counts().to_dict()}
        d.to_parquet(os.path.join(OUT, f"sentiment_{s}.parquet"), index=False)

    # ---- urgency ----
    u = pd.read_parquet(os.path.join(MT, "validation_ecommerce.parquet"), columns=["text", "urgency"]).dropna().drop_duplicates("text")
    UMAP = {"Low": 0, "Medium": 1, "High": 2, "Immediate": 3}
    u = u[u.urgency.isin(UMAP)].assign(label=lambda x: x.urgency.map(UMAP))
    u = u[~u.text.map(h).isin(corpus_eval | corpus_train | tr_h)]
    rng = np.random.default_rng(0); split = np.empty(len(u), dtype=object)
    for lab, idx in u.groupby("label").indices.items():
        idx = rng.permutation(idx); n = len(idx); a, c = int(0.7 * n), int(0.85 * n)
        split[idx[:a]], split[idx[a:c]], split[idx[c:]] = "train", "val", "test"
    for s in ("train", "val", "test"):
        d = u[split == s][["text", "label"]].reset_index(drop=True)
        d.to_parquet(os.path.join(OUT, f"urgency_{s}.parquet"), index=False)
        rep[f"urgency_{s}"] = {"rows": len(d), "label_counts": {int(k): int(v) for k, v in d.label.value_counts().items()}}
    # sanity: no cross-split overlap
    for task in ("sentiment", "urgency"):
        S = {s: set(pd.read_parquet(os.path.join(OUT, f"{task}_{s}.parquet")).text.map(h)) for s in ("train", "val", "test")}
        rep[f"{task}_cross_split_overlap"] = {f"{a}&{b}": len(S[a] & S[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        assert sum(rep[f"{task}_cross_split_overlap"].values()) == 0
    json.dump(rep, open(os.path.join(OUT, "aux_data_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
