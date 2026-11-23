"""Decoding tuning (inference only, no weight change). Tuning rows = 300 val rows WITH history, disjoint from eval_t4's
fixed 300 gate rows (random_state=11). FROZEN RULE (written before running): config A (current: greedy, rep 1.2) is the
default; eligible = portal_loop <= A, repetition <= A+0.02, orphan <= A+0.02, eos >= A-0.02; pick the eligible config with
the highest ROUGE-L, but keep A unless it beats A by > 0.003. Result is then confirmed once on the fixed gate rows.
Usage: python -I wsp/decode_tune.py --ckpt PATH --name NAME   -> wsp_runs/t4/decode/<name>.json"""
import argparse, json, os, sys
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_t4 import ROOT, DATA, REDIRECT, portal_loop, orphan, fmt_ctx, SentriFormerPipeline, rouge_l, generate_multitask, detect_repetition
CONFIGS = {"A_greedy_rp1.2": dict(temperature=0.0, repetition_penalty=1.2), "B_greedy_rp1.0": dict(temperature=0.0, repetition_penalty=1.0),
           "C_greedy_rp1.4": dict(temperature=0.0, repetition_penalty=1.4), "D_t0.7_k50_rp1.2": dict(temperature=0.7, top_k=50, repetition_penalty=1.2),
           "E_t0.5_k20_rp1.2": dict(temperature=0.5, top_k=20, repetition_penalty=1.2)}

def select(res):
    A = res["A_greedy_rp1.2"]
    ok = {k: v for k, v in res.items() if v["portal_loop"] <= A["portal_loop"] and v["repetition"] <= A["repetition"] + .02
          and v["orphan"] <= A["orphan"] + .02 and v["eos"] >= A["eos"] - .02}
    best = max(ok, key=lambda k: ok[k]["rougeL"])
    return best if ok[best]["rougeL"] > A["rougeL"] + 0.003 else "A_greedy_rp1.2"

def _selftest():
    base = {"portal_loop": .2, "repetition": 0, "orphan": .01, "eos": 1, "rougeL": .13}
    r = {"A_greedy_rp1.2": base, "B": {**base, "rougeL": .132}, "C": {**base, "rougeL": .14, "portal_loop": .3}, "D": {**base, "rougeL": .136}}
    assert select(r) == "D"; r["D"]["rougeL"] = .1325; assert select(r) == "A_greedy_rp1.2"
_selftest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
    od = os.path.join(ROOT, "wsp_runs", "t4", "decode"); os.makedirs(od, exist_ok=True); outp = os.path.join(od, f"{a.name}.json")
    assert not os.path.exists(outp), outp
    pipe = SentriFormerPipeline(a.ckpt, "cuda"); model, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    df = pq.read_table(os.path.join(DATA, "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); rows = hist.drop(index=list(gate)).sample(300, random_state=29)
    assert not gate & set(rows.index)
    res, samples = {}, {}
    for name, kw in CONFIGS.items():
        F = {"portal_loop": 0, "redirect_any": 0, "orphan": 0, "eos": 0, "repetition": 0}; rl, bi, tb, ex = [], set(), 0, []
        for j, (_, r) in enumerate(rows.iterrows()):
            torch.manual_seed(j)
            ids = tok.encode(fmt_ctx(r, "multi"), add_special_tokens=True)[-256:] or [1]; t = torch.tensor([ids], device="cuda")
            with torch.inference_mode(): g = generate_multitask(model, tok, cfg, t, torch.ones_like(t), max_new_tokens=128, **kw)[0].tolist()
            eos = cfg.eos_token_id in g; txt = tok.decode(g[1:g.index(cfg.eos_token_id)] if eos else g[1:], skip_special_tokens=True).strip()
            F["portal_loop"] += portal_loop(txt); F["redirect_any"] += bool(REDIRECT.search(txt)); F["orphan"] += orphan(txt)
            F["eos"] += eos; F["repetition"] += detect_repetition(txt); rl.append(rouge_l(txt, r.response))
            w = txt.lower().split(); bi.update(zip(w, w[1:])); tb += max(len(w) - 1, 0)
            if j < 8: ex.append({"input": r.context[-300:], "reference": r.response, "generated": txt})
        res[name] = {**{k: v / 300 for k, v in F.items()}, "rougeL": float(np.mean(rl)), "distinct_2": len(bi) / max(tb, 1)}
        samples[name] = ex; print(name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)
    choice = select(res); print("SELECTED", choice)
    json.dump({"ckpt": a.ckpt, "results": res, "selected": choice, "config": CONFIGS[choice], "samples": samples}, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if __name__ == "__main__": main()
