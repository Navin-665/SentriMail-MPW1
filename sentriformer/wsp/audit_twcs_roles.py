"""
Read-only audit: how many TWCS "agent responses" were actually written by a customer?
corpus/extract_twcs.py takes response_tweet_id[0] without checking inbound==False. TWCS anonymises customers as numeric
author_ids and keeps brand handles as names, so metadata.brand numeric => the "agent" target is a customer tweet.
Also measures the M10.4 dialogue data (used by T3) by joining on (group_id, cleaned context text).
Writes wsp_runs/t4/twcs_role_audit.json.
"""
import json, os, collections
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "wsp_runs", "t4", "twcs_role_audit.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True); assert not os.path.exists(OUT)
norm = lambda t: " ".join(str(t).lower().split())
rep, key_role = {}, {}
for split in ("train", "val", "test"):
    c = collections.Counter()
    for b in pq.ParquetFile(os.path.join(ROOT, "data", f"{split}.parquet")).iter_batches(batch_size=100_000, columns=["source", "group_id", "text", "response", "metadata"]):
        d = b.to_pandas(); d = d[d.source.str.contains("twcs", na=False)]
        for g, t, r, m in zip(d.group_id, d.text, d.response, d.metadata):
            md = json.loads(m) if m else {}
            brand = str(md.get("brand", "")); cust = str(md.get("customer_author_id", ""))
            role = "self_reply" if brand == cust else ("other_customer" if brand.isdigit() else "agent")
            c[role] += 1; c["deflection_flag"] += bool(md.get("has_deflection")); c["response_null"] += r is None
            key_role[(g, norm(t))] = role
    rep[f"corpus_{split}"] = dict(c)
for split in ("train", "val", "test"):
    c = collections.Counter()
    t = pq.read_table(os.path.join(ROOT, "data", "supervised", f"m10_4_dialogue_{split}.parquet"), columns=["source", "group_id", "context"]).to_pandas()
    t = t[t.source == "twcs"]
    for g, ctx in zip(t.group_id, t.context):
        c[key_role.get((g, norm(ctx)), "unmatched")] += 1
    c["twcs_rows"] = len(t); rep[f"m10_4_{split}"] = dict(c)
json.dump(rep, open(OUT, "w"), indent=2); print(json.dumps(rep, indent=1))
