"""SentriWSP-01 Phase 1 (READ ONLY): inventory, hashes, failure mining on VAL rows (never test), and mechanistic
attribution with forward hooks. No weights are written; grad disabled, eval mode.
Rows: multi_val rows WITH history, excluding eval_t4's 300 gate rows (rs=11) and decode_tune's 300 rows (rs=29),
so both stay clean for later evaluation. Output: phase1_results.json (refuses to overwrite)."""
import hashlib, json, os, re, sys, collections
import numpy as np, torch, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp")); sys.path.insert(0, os.path.dirname(ROOT))
from inference import SentriFormerPipeline, DEFAULT_CKPT
from eval_t4 import portal_loop, orphan, REASK, IDENT
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask

OUT = os.path.join(HERE, "phase1_results.json")

GENERIC = re.compile(r"(let (us|me) know if (you need|there'?s|you have) (any(thing)?|more|further|other)|anything else|"
                     r"pass (this|that|it|your feedback|along)|sorry to hear (that|this)[.!]?\s*$)", re.I)
PRIVACY = re.compile(r"\b(full name|zip( code)?|postal code|date of birth|dob|phone( number| #)?|home address|"
                     r"(e-?mail|mailing) address|social security|ssn|card number|account number|password|pin|one[- ]time code)\b", re.I)
FRAGMENT = re.compile(r"\b(at|here|to|via)\s+(so|and|we|to)\b", re.I)
STOP = set("i me my we our you your the a an and or to of in on for is are was were be been it this that with at by from have has "
           "had do does did not no can will would could please hi hello thanks thank sorry us let know help".split())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def content(t): return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP}


def split_cur(ctx):
    cur = re.split(r"(?=Customer: )", ctx)[-1]
    return ctx[: len(ctx) - len(cur)], cur


def main():
    assert not os.path.exists(OUT), OUT
    torch.set_grad_enabled(False)
    R = {"files": {}}
    for d in ("wsp_runs/FINAL_T4d", "wsp_runs/FINAL_T3"):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            p = os.path.join(ROOT, d, f)
            if os.path.isfile(p): R["files"][f"{d}/{f}"] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
    raw = torch.load(os.path.join(ROOT, DEFAULT_CKPT), map_location="cpu", weights_only=False)
    R["checkpoint_format"] = {"container": "PyTorch pickle (torch.save dict), not safetensors", "keys": list(raw.keys()),
                              "train_run": raw.get("train_run"), "step": raw.get("step")}
    sd = raw["model_state_dict"]; del raw
    R["inventory"] = {k: {"shape": list(v.shape), "numel": v.numel(), "dtype": str(v.dtype), "fro_norm": round(float(v.float().norm()), 3)}
                      for k, v in sd.items()}
    groups = collections.Counter()
    for k, v in sd.items():
        g = ("decoder" if ".decoder." in k else "encoder" if ".encoder." in k else "shared_embeddings" if "shared" in k
             else "lm_head(tied)" if "lm_head" in k else k.split(".")[0])
        groups[g] += v.numel()
    R["param_groups"] = dict(groups)

    pipe = SentriFormerPipeline(DEFAULT_CKPT, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    R["tied"] = {"lm_head_is_shared_embeddings": m.lm_head.weight.data_ptr() == m.sentriformer.shared_embeddings.weight.data_ptr(),
                 "unique_params": sum(p.numel() for p in m.parameters())}

    # ---------------- failure mining on clean VAL rows ----------------
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas()
    hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index)
    tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    rows = hist.drop(index=list(gate | tune)).sample(400, random_state=41)
    assert not (gate | tune) & set(rows.index)
    R["rows"] = {"source": "wsp_runs/t4/data/multi_val.parquet (validation)", "n": 400, "random_state": 41,
                 "excluded": "eval_t4 gate rows (rs=11) and decode_tune rows (rs=29)", "row_index": [int(i) for i in rows.index]}
    gens = []
    for _, r in rows.iterrows():
        ids = tok.encode(r.context, add_special_tokens=True)[-256:] or [1]
        t = torch.tensor([ids], device="cuda")
        g = generate_multitask(m, tok, cfg, t, torch.ones_like(t), max_new_tokens=128, temperature=0.0, repetition_penalty=1.2)[0].tolist()
        body = g[1:g.index(cfg.eos_token_id)] if cfg.eos_token_id in g else g[1:]
        txt = tok.decode(body, skip_special_tokens=True).strip()
        earlier, cur = split_cur(r.context)
        f = {"generic": bool(GENERIC.search(txt)),
             "privacy_request": bool(PRIVACY.search(txt)) and not PRIVACY.search(r.context),
             "broken_fragment": bool(orphan(txt)) or bool(FRAGMENT.search(txt)),
             "portal_loop": bool(portal_loop(txt)),
             "no_content_overlap_with_current_turn": not (content(txt) & content(cur)),
             "reask_given_id": bool(IDENT.search(earlier)) and bool(REASK.search(txt))}
        gens.append({"idx": int(r.name), "context": r.context, "reference": r.response, "generated": txt, "ids": ids, "body": body, "flags": f})
    R["failure_rates_T4d"] = {k: round(float(np.mean([g["flags"][k] for g in gens])), 4) for k in gens[0]["flags"]}
    R["reference_rates"] = {
        "generic": round(float(np.mean([bool(GENERIC.search(g["reference"])) for g in gens])), 4),
        "privacy_request": round(float(np.mean([bool(PRIVACY.search(g["reference"])) and not PRIVACY.search(g["context"]) for g in gens])), 4),
        "broken_fragment": round(float(np.mean([bool(orphan(g["reference"])) or bool(FRAGMENT.search(g["reference"])) for g in gens])), 4)}
    R["most_repeated_replies"] = [{"count": c, "reply": t} for t, c in collections.Counter(g["generated"] for g in gens).most_common(8)]
    R["distinct_reply_ratio"] = round(len({g["generated"] for g in gens}) / len(gens), 4)
    R["opening_5_words_top"] = collections.Counter(" ".join(g["generated"].split()[:5]) for g in gens).most_common(6)

    # ---------------- hooks ----------------
    dec = m.sentriformer.decoder; W_U = m.lm_head.weight; gamma = dec.final_norm.weight
    cache, hooks = {}, []

    def keep(name, idx=None):
        def f(mod, i, o): cache[name] = o[0] if idx == 0 else o
        return f
    for li, L in enumerate(dec.layers):
        hooks += [L.self_attn.register_forward_hook(keep(f"L{li}.self_attn", 0)),
                  L.cross_attn.register_forward_hook(keep(f"L{li}.cross_attn", 0)),
                  L.ffn.fc1.register_forward_hook(keep(f"L{li}.ffn_pre")),
                  L.ffn.register_forward_hook(keep(f"L{li}.ffn")),
                  L.cross_attn.q_proj.register_forward_hook(keep(f"q{li}")),
                  L.cross_attn.k_proj.register_forward_hook(keep(f"k{li}"))]

    def trace(g):
        cache.clear()
        e = torch.tensor([g["ids"]], device="cuda"); am = torch.ones_like(e)
        enc = m.sentriformer.encoder(input_ids=e, attention_mask=am)
        d = torch.tensor([[cfg.bos_token_id] + g["body"]], device="cuda")
        dec(input_ids=d, encoder_hidden_states=enc, encoder_attention_mask=am)
        return d

    def hist_len(g):  # encoder positions belonging to earlier turns (before the last "Customer: ")
        earlier, cur = split_cur(tok.decode(g["ids"], skip_special_tokens=True))
        return max(len(g["ids"]) - len(tok.encode(cur, add_special_tokens=True)), 0)

    def xattn_share(li, pos_slice, H):
        q = cache[f"q{li}"][0, pos_slice].reshape(-1, 8, 32); k = cache[f"k{li}"][0].reshape(-1, 8, 32)
        a = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / 32 ** 0.5, -1)
        return float(a[..., :H].sum(-1).mean())

    def token_pos(g, rx):
        mt = rx.search(g["generated"])
        if not mt: return None
        for k in range(1, len(g["body"]) + 1):
            if len(tok.decode(g["body"][:k], skip_special_tokens=True).strip()) > mt.start(): return k - 1
        return None

    def attribute(g, pos):
        """Linearised direct-logit attribution (through final LayerNorm scale) of each sublayer to the token body[pos],
        which is predicted at decoder input position pos (input = [BOS] + body)."""
        d = trace(g); t = g["body"][pos]
        comps = {k: cache[k][0, pos] for k in cache if re.fullmatch(r"L\d\.(self_attn|cross_attn|ffn)", k)}
        x0 = dec.embed_tokens(d)[0, pos] * cfg.d_model ** 0.5 + dec.pos_emb(d.shape[1], d.device)[0, pos]
        resid = x0 + sum(comps.values()); sigma = (resid.var(unbiased=False) + cfg.layer_norm_eps).sqrt()
        u = (W_U[t] - W_U.mean(0)) * gamma / sigma
        dla = {k: float((v - v.mean()) @ u) for k, v in comps.items()}; dla["embed+pos"] = float((x0 - x0.mean()) @ u)
        lens, h = [], x0.clone()
        for li in range(4):
            h = h + comps[f"L{li}.self_attn"] + comps[f"L{li}.cross_attn"] + comps[f"L{li}.ffn"]
            lg = W_U @ dec.final_norm(h); lens.append(int((lg > lg[t]).sum()))
        fl = max(range(4), key=lambda li: dla[f"L{li}.ffn"])
        act = torch.nn.functional.gelu(cache[f"L{fl}.ffn_pre"][0, pos])
        W2 = dec.layers[fl].ffn.fc2.weight                                   # [256, 1024]
        contrib = act * ((W2 - W2.mean(0, keepdim=True)).T @ u)
        top = torch.topk(contrib, 8)
        H = hist_len(g)
        return {"token": tok.decode([t]), "pos": pos, "logit_lens_rank_after_layer": lens,
                "dla_top": sorted(((k, round(v, 3)) for k, v in dla.items()), key=lambda kv: -kv[1])[:6],
                "top_ffn_layer": fl, "top_neurons": [[int(i), round(float(v), 3)] for v, i in zip(top.values, top.indices)],
                "cross_attn_history_share_by_layer": [round(xattn_share(li, slice(pos, pos + 1), H), 3) for li in range(4)],
                "history_token_fraction": round(H / len(g["ids"]), 3)}

    R["cases"], counts = [], collections.defaultdict(collections.Counter)
    for kind, rx in (("generic", GENERIC), ("privacy_request", PRIVACY), ("broken_fragment", FRAGMENT)):
        picked = [g for g in gens if g["flags"][kind] and token_pos(g, rx) is not None][:12]
        for j, g in enumerate(picked):
            a = attribute(g, token_pos(g, rx))
            for i, _ in a["top_neurons"][:4]: counts[kind][(a["top_ffn_layer"], i)] += 1
            if j < 4: R["cases"].append({"kind": kind, "val_row": g["idx"], "context": g["context"][-600:], "generated": g["generated"],
                                         "reference": g["reference"], **a})
        R.setdefault("n_attributed", {})[kind] = len(picked)
    R["recurring_neurons_top4_per_case"] = {k: [[f"L{l}.ffn.n{i}", c] for (l, i), c in v.most_common(6)] for k, v in counts.items()}

    # history use: cross-attention mass on earlier-turn encoder positions, averaged over all reply positions
    earlier_words = lambda g: content(split_cur(g["context"])[0])
    uses = [g for g in gens if not g["flags"]["generic"] and (content(g["generated"]) & earlier_words(g))][:60]
    generic = [g for g in gens if g["flags"]["generic"]][:60]
    R["history_attention"] = {}
    for name, grp in (("reply_uses_earlier_turn_words", uses), ("generic_reply", generic)):
        s = []
        for g in grp:
            d = trace(g); H = hist_len(g)
            s.append(([xattn_share(li, slice(0, d.shape[1]), H) for li in range(4)], H / len(g["ids"])))
        R["history_attention"][name] = ({"n": len(s), "mean_history_share_by_layer": [round(float(np.mean([x[0][l] for x in s])), 3) for l in range(4)],
                                         "mean_history_token_fraction": round(float(np.mean([x[1] for x in s])), 3)} if s else {"n": 0})
    for h in hooks: h.remove()
    R["sample_generations"] = [{"val_row": g["idx"], "current_turn": split_cur(g["context"])[1][:200], "generated": g["generated"], "flags": g["flags"]} for g in gens[:25]]
    json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("failure_rates_T4d", "reference_rates", "distinct_reply_ratio", "opening_5_words_top", "param_groups", "tied",
                                         "n_attributed", "recurring_neurons_top4_per_case", "history_attention")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
