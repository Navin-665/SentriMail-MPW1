"""
Sample REAL complaints for urgency/escalation annotation (local only). Never samples any test split.
Stratified: CFPB complaints (classification_train), TWCS multi-turn conversations (T4 multi_train), e-commerce urgency rows.
PII redaction: emails, phones, URLs, long digit runs, card-like numbers, @handles; CFPB 'XXXX' masks are kept.
Writes pilot_200.csv (double-labelled) and main_800.csv with blank label columns.
"""
import csv, os, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RED = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "{{Email}}"), (re.compile(r"https?://\S+|www\.\S+"), "{{URL}}"),
       (re.compile(r"@\w+"), "{{Handle}}"), (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "{{Phone/Number}}"),
       (re.compile(r"\b\d{5,}\b"), "{{Number}}")]


def redact(t):
    for p, r in RED: t = p.sub(r, t)
    return t


def main():
    assert not os.path.exists(os.path.join(HERE, "pilot_200.csv"))
    rng = np.random.default_rng(2026)
    pf = pq.ParquetFile(os.path.join(ROOT, "data", "supervised", "classification_train.parquet"))
    keep = set(rng.choice(pf.metadata.num_rows, 450, replace=False).tolist()); cfpb, off = [], 0
    for b in pf.iter_batches(batch_size=65536, columns=["id", "text"]):
        d = b.to_pandas(); sel = [i - off for i in range(off, off + len(d)) if i in keep]; cfpb += d.iloc[sel].to_dict("records"); off += len(d)
    cf = pd.DataFrame(cfpb).assign(source="cfpb", text=lambda x: x.text.str[:1500])
    tw = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_train.parquet"), columns=["group_id", "context", "source", "n_history_segments"]).to_pandas()
    tw = tw[(tw.source == "twcs") & (tw.n_history_segments > 0)].sample(400, random_state=2026)
    tw = tw.rename(columns={"group_id": "id", "context": "text"})[["id", "text", "source"]]
    ec = pq.read_table(os.path.join(ROOT, "wsp_runs", "data", "urgency_train.parquet")).to_pandas().sample(150, random_state=2026)
    ec = ec.assign(id=[f"ecom_{i}" for i in ec.index], source="ecommerce")[["id", "text", "source"]]
    allx = pd.concat([cf[["id", "text", "source"]], tw, ec]).sample(frac=1, random_state=2026).reset_index(drop=True)
    allx["text"] = allx.text.map(redact)
    cols = ["item_id", "source", "source_id", "text", "urgency_0_3", "escalate_yes_no", "borderline", "pii_fixed", "notes"]
    for name, part, prefix in (("pilot_200.csv", allx.iloc[:200], "P"), ("main_800.csv", allx.iloc[200:1000], "M")):
        with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols)
            for i, r in enumerate(part.itertuples(), 1):
                w.writerow([f"{prefix}{i:03d}", r.source, r.id, r.text, "", "", "", "", ""])
    print("wrote pilot_200.csv (copy to pilot_200_annotatorA.csv / _annotatorB.csv) and main_800.csv")
    print(allx.iloc[:200].source.value_counts().to_dict())


if __name__ == "__main__":
    main()
