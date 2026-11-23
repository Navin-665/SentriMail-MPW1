"""T6 step 1: score every T5a TRAIN reply for context dependence with a fixed scorer (FINAL_T5a, eval mode, no grad):
  pmi = mean log p(reply | its own context) - mean log p(reply | a random other conversation's context)
Low pmi = the reply fits any conversation (generic). Teacher-forced exactly like p1_patch_eval.dial_loss (same tokenisation,
256-token left-truncated context, reply + EOS). Train rows only; val/test are never read. Writes t6_pmi_scores.parquet (refuses overwrite).
  python -I wsp_runs/t6/t6_pmi.py [--n N]   (--n: smoke test on N rows, writes t6_pmi_SMOKE.parquet)"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp_runs", "sentriwsp01"))
import p1_patch_eval as P1
from sentriformer.training.train_dialogue_sft import DialogueSFTCollator

SCORER = os.path.join(ROOT, "wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")


def companies(df):
    """TWCS company = author of the first outbound reply to the conversation's opening tweet (raw TWCS, read-only, local).
    Non-TWCS rows use their source name."""
    raw = pd.read_csv(os.path.join(os.path.dirname(ROOT), "rawdata", "twcs.csv"), usecols=["author_id", "inbound", "in_response_to_tweet_id"],
                      dtype={"author_id": str, "in_response_to_tweet_id": str})
    out = raw[(~raw.inbound) & raw.in_response_to_tweet_id.notna()].drop_duplicates("in_response_to_tweet_id")
    cmap = dict(zip(out.in_response_to_tweet_id, out.author_id))

    comp = [cmap.get(gid[len("twcs_conv_"):], "twcs_unknown") if gid.startswith("twcs_conv_") else f"src_{s}" for gid, s in zip(df.group_id, df.source)]
    return comp


@torch.no_grad()
def mean_lp(m, cfg, items, bs=128):
    """Per-row mean log-prob of the reply tokens (labels != -100/pad), batched by length."""
    out = np.zeros(len(items)); order = np.argsort([len(x["input_ids"]) + len(x["labels"]) for x in items])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; b = DialogueSFTCollator(0)([items[i] for i in idx]); lab = b["labels"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            o = m(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda(), generation_labels=lab)
        lp = torch.log_softmax(o["generation_logits"].float(), -1); mask = (lab != -100) & (lab != cfg.pad_token_id)
        tok = lp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        out[idx] = (tok.sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
    return out


def main():
    smoke = "--n" in sys.argv
    out = os.path.join(HERE, "t6_pmi_SMOKE.parquet" if smoke else "t6_pmi_scores.parquet")
    if not smoke: assert not os.path.exists(out), "refusing to overwrite"
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "t5a_train.parquet"), columns=["id", "group_id", "source", "context", "response", "split"]).to_pandas()
    assert set(df.split) == {"train"}
    if smoke: df = df.sample(int(sys.argv[sys.argv.index("--n") + 1]), random_state=0).reset_index(drop=True)
    pipe = P1.SentriFormerPipeline(SCORER, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg; assert not m.training
    df["company"] = companies(df)
    rng = np.random.default_rng(6); g = df.group_id.to_numpy(); perm = np.arange(len(df))
    for _, idx in df.groupby("company").indices.items():   # partner = another conversation of the SAME company (same style/language)
        p = rng.permutation(idx)
        for _ in range(20):
            bad = np.where(g[p] == g[idx])[0]
            if not len(bad) or len(np.unique(g[idx])) < 2: break
            p[bad] = rng.choice(idx, len(bad))
        perm[idx] = p
    ok = g[perm] != g   # rows of single-conversation companies keep no valid partner -> NaN, never dropped by the filter
    print({"companies": df.company.nunique(), "rows_without_partner": int((~ok).sum())})
    ctx = [tok.encode(c, add_special_tokens=True)[-256:] or [1] for c in df.context]
    rid = []
    for r in df.response:
        x = tok.encode(r, truncation=True, max_length=127, add_special_tokens=True)
        rid.append(x if x and x[-1] == cfg.eos_token_id else x + [cfg.eos_token_id])
    own = mean_lp(m, cfg, [{"input_ids": c, "labels": r} for c, r in zip(ctx, rid)])
    other = mean_lp(m, cfg, [{"input_ids": ctx[j], "labels": r} for j, r in zip(perm, rid)])
    res = pd.DataFrame({"id": df.id, "group_id": df.group_id, "company": df.company, "lp_own": own, "lp_other": other,
                        "pmi": np.where(ok, own - other, np.nan)})
    assert res.pmi.mean() > 0, "sanity: real pairs must be more likely than shuffled ones"
    res.to_parquet(out, index=False)
    print({"rows": len(res), "pmi_mean": round(res.pmi.mean(), 4), "quantiles": res.pmi.quantile([.05, .1, .2, .3, .5, .9]).round(3).to_dict(),
           "frac_pmi_le_0": round(float((res.pmi <= 0).mean()), 4)})


if __name__ == "__main__":
    main()
