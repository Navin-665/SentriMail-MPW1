"""
WSP E0 - Split integrity + SHA256 manifest. Read-only on project files.
Usage: python -I wsp/integrity.py <tag>      (writes wsp_runs/e0/integrity_<tag>.json)
"""
import hashlib, json, os, sys, datetime
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../sentriformer
OUT_DIR = os.path.join(ROOT, "wsp_runs", "e0")
SUP = os.path.join(ROOT, "data", "supervised")

CKPTS = {
    "classifier_best": "checkpoints/classification/classifier_best.pt",
    "multitask_best": "checkpoints/multitask/multitask_best.pt",
    "m10_3_sft_best": "checkpoints/m10_3_sft/m10_3_sft_best.pt",
}
DATA = [f"data/supervised/{n}" for n in (
    "classification_train.parquet", "classification_val.parquet", "classification_test.parquet",
    "classification_labels.json", "csat_train.parquet", "csat_val.parquet", "csat_test.parquet",
    "dialogue_train.parquet", "clean_dialogue_train.parquet", "m10_3_dialogue_train.parquet")]
OTHER = ["checkpoints/config.json", "tokenizer/sentritokenizer_v1/tokenizer.json",
         "tokenizer/sentritokenizer_v1/tokenizer_config.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hashes(path, col):
    return {hashlib.blake2b(t.strip().encode("utf-8"), digest_size=12).digest()
            for t in pq.read_table(path, columns=[col])[col].to_pylist() if t is not None}


def ids(path):
    return set(pq.read_table(path, columns=["id"])["id"].to_pylist())


def main():
    tag = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"integrity_{tag}.json")
    assert not os.path.exists(out_path), f"refusing to overwrite {out_path}"
    rep = {"timestamp": datetime.datetime.now().isoformat(), "sha256": {}, "checks": {}}

    for rel in list(CKPTS.values()) + DATA + OTHER:
        p = os.path.join(ROOT, rel)
        rep["sha256"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        print("hashed", rel, flush=True)

    if tag == "pre":  # split checks only need to run once
        c = rep["checks"]
        labels = json.load(open(os.path.join(SUP, "classification_labels.json"), encoding="utf-8"))["label2id"]
        for task, col in (("classification", "text"), ("csat", "text")):
            sp = {s: os.path.join(SUP, f"{task}_{s}.parquet") for s in ("train", "val", "test")}
            I = {s: ids(p) for s, p in sp.items()}
            T = {s: text_hashes(p, col) for s, p in sp.items()}
            rows = {s: pq.read_metadata(p).num_rows for s, p in sp.items()}
            c[task] = {
                "rows": rows,
                "unique_ids": {s: len(v) for s, v in I.items()},
                "unique_texts": {s: len(v) for s, v in T.items()},
                "id_overlap": {f"{a}&{b}": len(I[a] & I[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
                "exact_text_overlap": {f"{a}&{b}": len(T[a] & T[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
                "split_column_values": {s: sorted(set(pq.read_table(p, columns=["split"])["split"].to_pylist())) for s, p in sp.items()},
            }
            if task == "classification":
                unk = {}
                for s, p in sp.items():
                    prods = pq.read_table(p, columns=["product"])["product"].to_pylist()
                    unk[s] = sum(1 for x in prods if x not in labels)  # dataset maps these silently to id 0
                c[task]["products_not_in_label_map"] = unk
            else:
                c[task]["score_values"] = {s: sorted(set(pq.read_table(p, columns=["csat_score"])["csat_score"].to_pylist())) for s, p in sp.items()}
            del I
            c[task]["_eval_text_hashes"] = {"val": T["val"], "test": T["test"]}

        # Cross-task leakage: eval texts appearing in any other training set used by the lineage
        cross_train = {
            "classification_train": text_hashes(os.path.join(SUP, "classification_train.parquet"), "text"),
            "csat_train": text_hashes(os.path.join(SUP, "csat_train.parquet"), "text"),
            "dialogue_train(context)": text_hashes(os.path.join(SUP, "dialogue_train.parquet"), "context"),
            "clean_dialogue_train(context)": text_hashes(os.path.join(SUP, "clean_dialogue_train.parquet"), "context"),
            "m10_3_dialogue_train(context)": text_hashes(os.path.join(SUP, "m10_3_dialogue_train.parquet"), "context"),
        }
        c["cross_task_overlap"] = {}
        for task in ("classification", "csat"):
            for s, H in c[task].pop("_eval_text_hashes").items():
                for name, TH in cross_train.items():
                    c["cross_task_overlap"][f"{task}_{s} & {name}"] = len(H & TH)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
