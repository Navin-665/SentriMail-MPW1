"""T5a: drop TRAIN rows flagged by the v5 detectors (wsp_runs/t5/t5_audit_v5.py: broken_link, generic_whole, privacy_secret,
privacy_bulk_contact) from T4d's training set. Val/test untouched. Leakage audit: kept ids are a subset of t4d_train, and no
group_id is shared with multi_val/multi_test. Writes wsp_runs/t4/data/t5a_train.parquet + t5a_build_report.json (refuses overwrite).
NOTE: v5 filters have NOT passed a human precision audit; the user approved training anyway (2026-11-08)."""
import hashlib, json, os, sys
import pyarrow as pa, pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, "wsp_runs", "t4", "data")
sys.path.insert(0, os.path.join(ROOT, "wsp_runs", "t5"))
import t5_audit_v5 as V

out, rep = os.path.join(D, "t5a_train.parquet"), os.path.join(D, "t5a_build_report.json")
assert not os.path.exists(out) and not os.path.exists(rep), "refusing to overwrite"
V._selftest()
t = pq.read_table(os.path.join(D, "t4d_train.parquet")); df = t.to_pandas(); assert set(df.split) == {"train"}
F = [V.flags(c, r) for c, r in zip(df.context, df.response)]
drop = [any(f[k] for k in V.DROP) for f in F]
keep = df[[not d for d in drop]]
assert len(keep) + sum(drop) == len(df) and set(keep.id) <= set(df.id)
n_after_filter = len(keep)
LEAK = {}
for split in ("val", "test"):
    other = pq.read_table(os.path.join(D, f"multi_{split}.parquet"), columns=["group_id", "context", "response"]).to_pandas()
    assert not set(keep.group_id) & set(other.group_id), f"group leak into {split}"
    # ids are numbered per split (m10_3_bitext_1002 exists in train AND val as different rows), so compare CONTENT, not ids
    pair = lambda d: {hashlib.sha1((c + "\x00" + r).encode()).hexdigest() for c, r in zip(d.context, d.response)}
    shared = pair(keep) & pair(other); LEAK[split] = len(shared)
    if shared:   # same policy as build_t4_dialogue: identical pairs are dropped from TRAIN (val/test never change)
        h = [hashlib.sha1((c + "\x00" + r).encode()).hexdigest() in shared for c, r in zip(keep.context, keep.response)]
        keep = keep[[not x for x in h]]
    assert not pair(keep) & pair(other)
pq.write_table(pa.Table.from_pandas(keep, schema=t.schema, preserve_index=False), out)
R = {"rows_in": len(df), "rows_kept": len(keep), "dropped": sum(drop), "by_filter": {k: sum(f[k] for f in F) for k in V.DROP},
     "detector": "t5_audit_v5.py", "detector_sha256": hashlib.sha256(open(V.__file__, "rb").read()).hexdigest(),
     "leakage_audit": {"kept_subset_of_t4d_train": True, "group_id_overlap": 0, "identical_pairs_found_and_dropped_from_train": LEAK,
                       "rows_dropped_for_leakage": n_after_filter - len(keep),
                       "note": "row ids are per-split counters, so id overlap is not leakage; content and group checks pass"},
     "human_precision_audit": "NOT done (user override 2026-11-08)", "kept_with_history": int((keep.n_history_segments > 0).sum())}
json.dump(R, open(rep, "w"), indent=1); print(json.dumps(R, indent=1))
