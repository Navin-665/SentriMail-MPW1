"""WSP E0 - Do classification/CSAT eval texts appear in the pretraining corpus (data/train.parquet)?
Read-only. Writes wsp_runs/e0/pretrain_overlap.json."""
import hashlib, json, os
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "wsp_runs", "e0", "pretrain_overlap.json")
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
h = lambda t: hashlib.blake2b(t.strip().encode("utf-8"), digest_size=12).digest()

def hashes(path, col="text"):
    return {h(t) for t in pq.read_table(path, columns=[col])[col].to_pylist() if t}

pre = {s: hashes(os.path.join(ROOT, "data", f"{s}.parquet")) for s in ("train", "val", "test")}
res = {}
for task in ("classification", "csat"):
    for s in ("train", "val", "test"):
        E = hashes(os.path.join(ROOT, "data", "supervised", f"{task}_{s}.parquet"))
        res[f"{task}_{s}"] = {"n": len(E), **{f"in_pretrain_{p}": len(E & H) for p, H in pre.items()}}
        print(task, s, res[f"{task}_{s}"], flush=True)
json.dump(res, open(OUT, "w"), indent=2)
