"""Export a stratified sample of low-quality (customer context -> agent reply) training pairs for external review.
Rows = pairs REMOVED from the dialogue training data by SentriFormer's filters:
  portal/DM redirect (portal filter), broken link + whole-reply generic (reply filter, same v5 detectors as training).
Usage: python -I export_bad_replies.py <sentriformer_dir> <out_dir> [n_total=1000]
Writes <out_dir>/sentriformer_bad_replies_<n>.csv and .jsonl (corpus text: keep out of git).
"""
import csv, json, os, sys
import pandas as pd

SF, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
D = os.path.join(SF, "wsp_runs", "t4", "data")
sys.path.insert(0, os.path.join(SF, "wsp_runs", "t5"))
import t5_audit_v5 as V  # noqa: E402  (training-time detectors)

cols = ["id", "source", "context", "response", "intent_category", "portal_rewrite", "dm_redirect"]
multi = pd.read_parquet(os.path.join(D, "multi_train.parquet"), columns=cols)
t4d_ids = set(pd.read_parquet(os.path.join(D, "t4d_train.parquet"), columns=["id"]).id)
t5a_ids = set(pd.read_parquet(os.path.join(D, "t5a_train.parquet"), columns=["id"]).id)

portal = multi[~multi.id.isin(t4d_ids) & (multi.portal_rewrite | multi.dm_redirect)]
removed5 = multi[multi.id.isin(t4d_ids) & ~multi.id.isin(t5a_ids)].copy()
fl = [V.flags(c, r) for c, r in zip(removed5.context, removed5.response)]
removed5["broken"] = [f["broken_link"] for f in fl]; removed5["generic"] = [f["generic_whole"] for f in fl]
broken = removed5[removed5.broken]; generic = removed5[removed5.generic & ~removed5.broken]

quota = {"portal_or_dm_redirect": int(N * .3), "broken_link": int(N * .35)}
quota["generic_reply"] = N - sum(quota.values())
parts = []
for name, df in [("portal_or_dm_redirect", portal), ("broken_link", broken), ("generic_reply", generic)]:
    s = df.sample(n=min(quota[name], len(df)), random_state=2026).copy(); s["problem"] = name; parts.append(s)
    print(name, "pool", len(df), "sampled", len(s))
out = pd.concat(parts).sample(frac=1, random_state=7)
out = out.rename(columns={"context": "customer_context", "response": "agent_reply"})
out = out[["id", "source", "intent_category", "customer_context", "agent_reply", "problem"]]
os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, f"sentriformer_bad_replies_{len(out)}")
out.to_csv(base + ".csv", index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
out.to_json(base + ".jsonl", orient="records", lines=True, force_ascii=False)
print("wrote", base + ".csv", len(out), "rows;", out.problem.value_counts().to_dict(), out.source.value_counts().to_dict())
