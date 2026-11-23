"""T4d: drop dialogue TRAIN rows whose target is a portal rewrite / DM redirect / empty, or is itself a portal loop
(eval_t4.portal_loop). Val/test untouched (identical evaluation rows as T4a). Writes wsp_runs/t4/data/t4d_train.parquet."""
import json, os, sys, collections
import pyarrow.parquet as pq, pyarrow as pa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from eval_t4 import portal_loop
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wsp_runs", "t4", "data")
out = os.path.join(D, "t4d_train.parquet"); assert not os.path.exists(out), "refusing to overwrite"
t = pq.read_table(os.path.join(D, "multi_train.parquet")); df = t.to_pandas()
pl = df.response.map(portal_loop)
reasons = {"portal_rewrite": df.portal_rewrite, "dm_redirect": df.dm_redirect, "empty_target": df.empty_target, "target_portal_loop": pl}
drop = df.portal_rewrite | df.dm_redirect | df.empty_target | pl
keep = df[~drop]
assert len(keep) + int(drop.sum()) == len(df) and keep.response.map(portal_loop).sum() == 0
pq.write_table(pa.Table.from_pandas(keep, schema=t.schema, preserve_index=False), out)
by_cat = {c: {"rows": int(n), "dropped": int(drop[df.intent_category == c].sum()), "frac": round(float(drop[df.intent_category == c].mean()), 3)}
          for c, n in df.intent_category.value_counts().items()}
rep = {"rows_in": len(df), "rows_kept": len(keep), "dropped": int(drop.sum()), "by_reason": {k: int(v.sum()) for k, v in reasons.items()},
       "by_intent_category": by_cat, "kept_with_history": int((keep.n_history_segments > 0).sum())}
json.dump(rep, open(os.path.join(D, "t4d_build_report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
