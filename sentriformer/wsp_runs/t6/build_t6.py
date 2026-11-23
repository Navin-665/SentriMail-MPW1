"""T6 data: t5a_train minus rows with same-company pmi < 0 (t6_pmi_scores.parquet). Rows without a partner (pmi NaN) are kept.
Subset of the leakage-audited t5a_train, so no new leakage is possible (asserted). Refuses overwrite."""
import hashlib, json, os
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(HERE), "t4", "data")
out, rep = os.path.join(D, "t6_train.parquet"), os.path.join(D, "t6_build_report.json")
assert not os.path.exists(out) and not os.path.exists(rep), "refusing to overwrite"
t = pq.read_table(os.path.join(D, "t5a_train.parquet")); df = t.to_pandas()
s = pd.read_parquet(os.path.join(HERE, "t6_pmi_scores.parquet")); assert list(s.id) == list(df.id), "scores must align row-for-row"
drop = (s.pmi < 0).to_numpy()   # NaN < 0 is False -> kept
keep = df[~drop]; assert set(keep.id) <= set(df.id) and len(keep) + drop.sum() == len(df)
pq.write_table(pa.Table.from_pandas(keep, schema=t.schema, preserve_index=False), out)
R = {"rows_in": len(df), "rows_kept": len(keep), "dropped_pmi_lt_0": int(drop.sum()), "no_partner_kept": int(s.pmi.isna().sum()),
     "scores_sha256": hashlib.sha256(open(os.path.join(HERE, "t6_pmi_scores.parquet"), "rb").read()).hexdigest(),
     "dropped_by_company_top10": s[drop].company.value_counts().head(10).to_dict(), "leakage": "subset of t5a_train (audited)",
     "human_precision_audit": "NOT done"}
json.dump(R, open(rep, "w"), indent=1); print(json.dumps(R, indent=1))
