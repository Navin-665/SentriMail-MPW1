"""
T4 dialogue data builder (derived files only; frozen corpus and m10_4 files are read-only).

Base = m10_4_dialogue_{split} (exactly T3's dialogue data). For each TWCS row, find its turn in the corpus conversation
(same group_id; exact normalised text, else token Jaccard >= 0.5), then:
  - target_role: who actually wrote the target reply (corpus metadata.brand; numeric author id = customer)
      agent | customer_self (customer replied to themselves) | customer_other | unmatched
  - history: earlier turns of the same conversation in turn_index order, with role labels from the AUTHOR, not position
Outputs two variants with identical rows (only context differs):
  ctrl  : "Customer: <message>"
  multi : "<history turns> Customer: <message>", left-truncated to keep the most recent turns (<= CTX_TOK tokens)
Rows whose target was written by a customer are excluded from both variants' training/val (role fix).
Flags kept for T4d (not used by T4a): portal_rewrite, dm_redirect, empty_target.
Writes wsp_runs/t4/data/{ctrl,multi}_{train,val,test}.parquet + build_report.json. Test files are written but sealed.
"""
import json, os, re, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "wsp_runs", "t4", "data")
CTX_TOK = 250
norm = lambda t: " ".join(re.sub(r"@\w+|https?://\S+|\^\w+", " ", str(t).lower()).split())
clean = lambda t: " ".join(re.sub(r"@\w+|https?://\S+|t\.co/\S+|\^\w+\s*$", " ", str(t)).split())
DM = re.compile(r"\b(dm|direct message|private message|send us a note|send us a message|message us)\b", re.I)


def role_of(md):
    b, c = str(md.get("brand", "")), str(md.get("customer_author_id", ""))
    return "customer_self" if b == c else ("customer_other" if b.isdigit() else "agent")


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    os.makedirs(OUT)
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
    ntok = lambda texts: [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
    rep = {"ctx_tok_budget": CTX_TOK}
    for split in ("train", "val", "test"):
        m = pq.read_table(os.path.join(ROOT, "data", "supervised", f"m10_4_dialogue_{split}.parquet")).to_pandas()
        tw_groups = set(m.group_id[m.source == "twcs"])
        conv = collections.defaultdict(list)   # group -> [(turn, norm_text, clean_text, clean_resp, role)]
        corpus_split = {}
        for b in pq.ParquetFile(os.path.join(ROOT, "data", f"{split}.parquet")).iter_batches(batch_size=100_000, columns=["source", "group_id", "text", "response", "metadata", "split"]):
            d = b.to_pandas(); d = d[d.source.str.contains("twcs", na=False) & d.group_id.isin(tw_groups)]
            for g, t, r, md, sp in zip(d.group_id, d.text, d.response, d.metadata, d.split):
                md = json.loads(md); conv[g].append((int(md["turn_index"]), norm(t), clean(t), clean(r) if isinstance(r, str) else "", role_of(md)))
                corpus_split[g] = sp
        for g in conv: conv[g].sort()
        roles, hist_n, ctrl_ctx, multi_ctx, match_kinds = [], [], [], [], []
        stats = collections.Counter()
        for src, g, c in zip(m.source, m.group_id, m.context):
            if src != "twcs":
                match_kinds.append("none"); roles.append("non_twcs"); hist_n.append(0); ctrl_ctx.append(f"Customer: {c}"); multi_ctx.append(f"Customer: {c}"); continue
            turns = conv.get(g, []); nc = norm(c); k = None
            for ti, tn, *_ in turns:
                if tn == nc: k = ti; stats["match_exact"] += 1; kind = "exact"; break
            if k is None:
                s = set(nc.split()); best = (0.0, None)
                for ti, tn, *_ in turns:
                    u = s | set(tn.split()); j = len(s & set(tn.split())) / max(len(u), 1)
                    if j > best[0]: best = (j, ti)
                if best[0] >= 0.5: k = best[1]; stats["match_jaccard"] += 1; kind = "jaccard"
            if k is None:
                match_kinds.append("none"); stats["unmatched"] += 1; roles.append("unmatched"); hist_n.append(0)
                ctrl_ctx.append(f"Customer: {c}"); multi_ctx.append(f"Customer: {c}"); continue
            assert corpus_split.get(g) == split, f"split mismatch for {g}"
            match_kinds.append(kind); this = next(x for x in turns if x[0] == k); roles.append(this[4])
            seg = []
            for ti, _, tc, rc, rl in turns:
                if ti >= k: break                      # only strictly earlier turns: never the target turn
                seg.append(f"Customer: {tc}")
                if rc: seg.append(("Agent: " if rl == "agent" else "Customer: ") + rc)
            hist_n.append(len(seg)); ctrl_ctx.append(f"Customer: {c}"); multi_ctx.append(" ".join(seg + [f"Customer: {c}"]))
        # left-truncate history to the token budget (drop oldest turns first; the current message is always kept)
        lens = ntok(multi_ctx)
        for i in np.where(np.array(lens) > CTX_TOK)[0]:
            parts = re.split(r"(?=(?:Customer|Agent): )", multi_ctx[i]); parts = [p for p in parts if p.strip()]
            while len(parts) > 1 and len(tok.encode(" ".join(parts))) > CTX_TOK: parts.pop(0)
            multi_ctx[i] = " ".join(parts); stats["left_truncated"] += 1
        m = m.assign(target_role=roles, n_history_segments=hist_n,
                     portal_rewrite=(m.source == "twcs") & m.response.str.contains("support portal", case=False, na=False),
                     dm_redirect=m.response.str.contains(DM, na=False),
                     empty_target=m.response.fillna("").str.split().str.len() < 3)
        # integrity: target must never be inside its own context
        leak = np.array([isinstance(r, str) and len(r) > 20 and r.strip() in cx for r, cx in zip(m.response, multi_ctx)])
        match_kind = np.array(match_kinds)
        stats["leak_dropped_exact_match(template repeated earlier)"] = int((leak & (match_kind == "exact")).sum())
        stats["leak_dropped_fuzzy_match(possible wrong turn)"] = int((leak & (match_kind == "jaccard")).sum())
        keep = ~m.target_role.isin(["customer_self", "customer_other"]) & ~leak
        assert not any(isinstance(r, str) and len(r) > 20 and r.strip() in cx for r, cx, k in zip(m.response, multi_ctx, keep) if k)
        for name, ctx in (("ctrl", ctrl_ctx), ("multi", multi_ctx)):
            m.assign(context=ctx)[keep].reset_index(drop=True).to_parquet(os.path.join(OUT, f"{name}_{split}.parquet"), index=False)
        rep[split] = {"rows_m10_4": len(m), "rows_kept": int(keep.sum()), "dropped_customer_authored_target": int(m.target_role.isin(["customer_self", "customer_other"]).sum()),
                      "target_role": m.target_role.value_counts().to_dict(), **dict(stats),
                      "kept_with_history": int(((np.array(hist_n) > 0) & keep.values).sum()),
                      "history_segments_mean(kept, twcs w/ history)": float(np.mean([h for h, k in zip(hist_n, keep) if k and h > 0]) or 0),
                      "multi_ctx_tokens_p50_p95": [int(np.percentile(ntok(list(np.array(multi_ctx)[keep.values][:20000])), q)) for q in (50, 95)],
                      "t4d_flags_in_kept": {f: int(m[f][keep].sum()) for f in ("portal_rewrite", "dm_redirect", "empty_target")}}
        print(split, json.dumps(rep[split]), flush=True)
    # conversation-level split disjointness across derived files
    G = {s: set(pq.read_table(os.path.join(OUT, f"multi_{s}.parquet"), columns=["group_id"]).to_pandas().group_id) for s in ("train", "val", "test")}
    rep["group_overlap"] = {f"{a}&{b}": len(G[a] & G[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert sum(rep["group_overlap"].values()) == 0
    json.dump(rep, open(os.path.join(OUT, "build_report.json"), "w"), indent=2)
    print("group_overlap", rep["group_overlap"])


if __name__ == "__main__":
    main()
