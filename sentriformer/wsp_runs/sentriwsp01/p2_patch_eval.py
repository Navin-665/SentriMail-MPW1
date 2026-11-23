"""WSP-P2: steering-direction edits written into decoder LayerNorm biases of T4d. Frozen rules: P2_CRITERIA.md (hash in P2_FREEZE.txt);
method: WSP_P2_DESIGN.md (revision 4).
  python -I wsp_runs/sentriwsp01/p2_patch_eval.py selftest   # checks only; Phase-1 discovery rows only; temp files in $WSP_SCRATCH, deleted
  python -I wsp_runs/sentriwsp01/p2_patch_eval.py stageA     # SCREENING: 2 arms x 3 alphas x (1 + 8 controls) on rows A
  python -I wsp_runs/sentriwsp01/p2_patch_eval.py stageB     # CONFIRMATORY, once: selected arm/alpha on rows B
Reuses p1_patch_eval helpers unchanged. T4d is only read; its SHA-256 is verified before and after every stage."""
import hashlib, json, os, re, sys, csv, random, collections
import numpy as np, pandas as pd, torch, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import p1_patch_eval as P1                                            # importing runs only its constants/defs (main is guarded)
from p1_patch_eval import sha, check_t4d, dump_atomic, generate, heads, dial_loss, reload_matches, T4D_SHA, KEYS
from eval_t4 import portal_loop, orphan, ACTION
from eval_full import rouge_l
from phase1_investigate import GENERIC, content, split_cur
from phase1b_controls import PRIV_REQ
from sentriformer.evaluation.audit_m10_3_threeway import detect_repetition
import transformers

GENERIC_SHA = "8460e8b1e89a0fa0a2a053579b47004506ad59d415ee0170acf3d4b7e163686c"
TOK_DIR = os.path.join(ROOT, "tokenizer", "sentritokenizer_v1")
TOK_SHA = {"tokenizer.json": "528dca6bc169c41313c994d17cc45032be50cd477e845c6e7973e2d22247f6f7",
           "tokenizer_config.json": "1101dd9d914d617d6b6503678fe380e0b21752b5f587def068d176df47354a6e"}
DEC = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
ALPHAS, ARMS = [0.25, 0.5, 1.0], ("F", "N")
SITE = {"F": "sentriformer.decoder.final_norm.bias", "N": "sentriformer.decoder.layers.3.norm3.bias"}
PATCH_DIR, RES = os.path.join(HERE, "patches_p2"), os.path.join(HERE, "p2_results")
KNOWN = "We're sorry to hear that. Please let us know if you need anything else."


# ---------------- pinning ----------------
def require(actual, expected, what):
    if actual != expected: raise SystemExit(f"REFUSING: {what} hash {actual} != pinned {expected}")


def check_pins():
    require(hashlib.sha256(GENERIC.pattern.encode()).hexdigest(), GENERIC_SHA, "GENERIC pattern")
    for f, h in TOK_SHA.items(): require(sha(os.path.join(TOK_DIR, f)), h, f)
    check_t4d()


def site(m, arm):
    d = m.sentriformer.decoder
    return d.final_norm.bias if arm == "F" else d.layers[3].norm3.bias


def apply2(m, base, arm=None, vec=None, alpha=0.0):
    """Reset BOTH bias sites to T4d, then (optionally) bias_arm -= alpha*vec. Never inherits a previous edit."""
    with torch.no_grad():
        for a in ARMS: site(m, a).copy_(base[a])
        if arm is not None: site(m, arm).sub_(alpha * vec.to(site(m, arm).device, site(m, arm).dtype))


# ---------------- alignment (WSP_P2_DESIGN.md 2a) ----------------
def align(tok, body):
    """labels[k] in {1 generic, 0 other, None excluded} for token b_k (= activation at decoder position k). ok=False -> exclude reply."""
    text = tok.decode(body, **DEC); c = [len(tok.decode(body[:k], **DEC)) for k in range(len(body) + 1)]
    if any(c[k + 1] < c[k] for k in range(len(body))) or any(not text.startswith(tok.decode(body[:k], **DEC)) for k in range(len(body) + 1)):
        return None, text, c, False
    spans = [(mt.start(), mt.end()) for mt in GENERIC.finditer(text)]
    special = set(tok.all_special_ids); lab = []
    for k, t in enumerate(body):
        if t in special: lab.append(None); continue
        hit = any(not text[i].isspace() for s, e in spans for i in range(max(c[k], s), min(c[k + 1], e)))
        lab.append(1 if hit else 0)
    return lab, text, c, True


def reply_len(tok, body): return sum(1 for t in body if t not in set(tok.all_special_ids))


# ---------------- direction construction (Phase-1 discovery rows only) ----------------
def phase1_rows():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas()
    idx = json.load(open(os.path.join(HERE, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"]
    return df.loc[idx], idx


def collect(m, tok, cfg, rows):
    dec, cap = m.sentriformer.decoder, {}
    hk = [dec.final_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("F", o[0])),
          dec.layers[3].norm3.register_forward_hook(lambda mod, i, o: cap.__setitem__("N", o[0]))]
    H, y, rid, excluded = {"F": [], "N": []}, [], [], 0
    for j, (_, r) in enumerate(rows.iterrows()):
        (_, _), body, ids = generate(m, tok, cfg, r.context)
        lab, _, _, ok = align(tok, body)
        if not ok: excluded += 1; continue
        e = torch.tensor([ids], device="cuda"); am = torch.ones_like(e); enc = m.sentriformer.encoder(input_ids=e, attention_mask=am)
        dec(input_ids=torch.tensor([[cfg.bos_token_id] + body], device="cuda"), encoder_hidden_states=enc, encoder_attention_mask=am)
        for a in ARMS: assert cap[a].shape == (len(body) + 1, 256), cap[a].shape
        for k, l in enumerate(lab):                                  # position k predicts b_k; position len(body) (EOS) excluded
            if l is None: continue
            for a in ARMS: H[a].append(cap[a][k].double().cpu())
            y.append(l); rid.append(j)
    for h in hk: h.remove()
    return {a: torch.stack(v) for a, v in H.items()}, np.array(y), np.array(rid), excluded


def gap(H, y): return H[y == 1].mean(0) - H[y == 0].mean(0)


def build(m, tok, cfg):
    rows, idx = phase1_rows(); H, y, rid, excluded = collect(m, tok, cfg, rows)
    perm = np.random.default_rng(2026).permutation(400); half1 = set(perm[:200].tolist())     # positions in the stored row order
    in1 = np.array([r in half1 for r in rid])
    out = {"excluded_replies": excluded, "n_generic": int(y.sum()), "n_other": int((y == 0).sum())}
    D = {}
    for a in ARMS:
        D[a] = gap(H[a], y)
        c = torch.nn.functional.cosine_similarity(gap(H[a][in1], y[in1]), gap(H[a][~in1], y[~in1]), dim=0)
        out[a] = {"norm_d": float(D[a].norm()), "split_half_cos": float(c), "P0_stable": bool(c >= 0.8)}
    return D, H, y, out


# ---------------- controls ----------------
def orth_scaled(r, d):
    u = d / d.norm(); r = r - (r @ u) * u; return r * (d.norm() / r.norm())


def controls(D, H, y, E):
    """8 controls per arm. F: output-space match |‖Ec‖/‖Ed‖-1|<=0.10 via deterministic resampling, fallback = exact output scaling.
    N: exact hidden-norm match (seed k / 20000+k)."""
    C = {}
    for a in ARMS:
        d = D[a]; Ed = float((E @ d).norm()); out = []
        def shuffled(seed):
            yp = np.random.default_rng(seed).permutation(y); s = gap(H[a], yp); return s * (d.norm() / s.norm())
        def gauss(seed): return orth_scaled(torch.from_numpy(np.random.default_rng(seed).standard_normal(256)), d)
        for kind, n_k in (("random", 5), ("shuffled", 3)):
            for k in range(n_k):
                if a == "N":
                    c = gauss(k) if kind == "random" else shuffled(20_000 + k)
                    out.append({"kind": kind, "k": k, "seed": k if kind == "random" else 20_000 + k, "attempt": 0, "fallback": False, "vec": c})
                    continue
                first, hit = None, None
                for att in range(1000):
                    seed = 10_000 * k + att if kind == "random" else 20_000 + 10_000 * k + att
                    c = gauss(seed) if kind == "random" else shuffled(seed)
                    first = first or (seed, c)
                    if abs(float((E @ c).norm()) / Ed - 1) <= 0.10: hit = (seed, att, c); break
                if hit: out.append({"kind": kind, "k": k, "seed": hit[0], "attempt": hit[1], "fallback": False, "vec": hit[2]})
                else:
                    seed, c = first; c = c * (Ed / float((E @ c).norm()))
                    out.append({"kind": kind, "k": k, "seed": seed, "attempt": None, "fallback": True, "vec": c})
        for o in out:
            o["norm"] = float(o["vec"].norm()); o["Ec_norm"] = float((E @ o["vec"]).norm()); o["Ec_ratio"] = o["Ec_norm"] / Ed
        C[a] = out
    return C


# ---------------- evaluation ----------------
def eval_rows2():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(HERE, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    A1, B1 = P1.eval_rows(); used = gate | tune | p1 | set(A1.index) | set(B1.index)
    rows = hist.drop(index=list(used)).sample(500, random_state=67); assert not used & set(rows.index) and len(rows) == 500
    return rows.iloc[:250], rows.iloc[250:]


def run2(m, tok, cfg, rows):
    per, replies = [], []
    for _, r in rows.iterrows():
        (txt, eos), body, _ = generate(m, tok, cfg, r.context); cur = split_cur(r.context)[1]
        per.append({"generic": bool(GENERIC.search(txt)), "specific": bool(content(txt) & content(cur)), "action": bool(ACTION.search(txt)),
                    "orphan": bool(orphan(txt)), "repetition": bool(detect_repetition(txt)), "eos": bool(eos), "portal": bool(portal_loop(txt)),
                    "privacy": bool(PRIV_REQ.search(txt)) and not PRIV_REQ.search(r.context), "length": reply_len(tok, body),
                    "rouge_clean": None if portal_loop(r.response) else rouge_l(txt, r.response)})
        replies.append(txt)
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k not in ("rouge_clean", "length")}
    agg["mean_length"] = float(np.mean([p["length"] for p in per])); agg["empty"] = float(np.mean([p["length"] == 0 for p in per]))
    agg["rougeL_clean"] = float(np.mean([p["rouge_clean"] for p in per if p["rouge_clean"] is not None]))
    agg["distinct"] = len(set(replies)) / len(replies)
    op = collections.Counter(" ".join(x.split()[:5]) for x in replies)
    agg["template_concentration"] = sum(c for _, c in op.most_common(6)) / len(replies)
    agg["top_reply_share"] = collections.Counter(replies).most_common(1)[0][1] / len(replies)
    agg["dial_loss"] = dial_loss(m, tok, cfg, rows)
    return agg, np.array([p["generic"] for p in per], float), replies


def ci975(d, n=10000):
    b = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 1.25)), float(np.percentile(b, 98.75))]


def gates2(base, cand, diff_ci, cand_drop, ctrl_drops, inv_ok):
    g, _ = P1.gates(base, cand, diff_ci, cand_drop, ctrl_drops, inv_ok)           # P1 rules 1-5 unchanged (rule 1 gets the 97.5% CI)
    g["6_length_within_20pct"] = abs(cand["mean_length"] / base["mean_length"] - 1) <= 0.20
    g["7_empty_le_plus_1pp"] = cand["empty"] <= base["empty"] + .01
    return g, all(g.values())


def patch_meta(arm, a, vec):
    return {"id": f"P2{arm}_a{a}", "tensor": SITE[arm], "alpha": a, "d_sha256": hashlib.sha256(vec.double().numpy().tobytes()).hexdigest(),
            "base_sha256": T4D_SHA}


def save_or_validate2(m, base, arm, a, vec, p):
    apply2(m, base, arm, vec, a)
    if os.path.exists(p):
        meta = torch.load(p, map_location="cpu", weights_only=False).get("wsp_patch")
        if meta != patch_meta(arm, a, vec): raise SystemExit(f"{p} exists with different metadata {meta}; refusing")
        if not reload_matches(m, p): raise SystemExit(f"{p} exists but does not reload to the expected model; refusing")
        return "reused (validated)"
    torch.save({"model_state_dict": m.state_dict(), "train_run": "T4d", "wsp_patch": patch_meta(arm, a, vec)}, p + ".tmp")
    os.replace(p + ".tmp", p); assert reload_matches(m, p), p
    return "saved"


def penalised_tokens(tok, E, d, n=20): return [tok.decode([i]) for i in torch.topk(E @ d, n).indices.tolist()]


# ---------------- main ----------------
def main():
    stage = sys.argv[1]; torch.set_grad_enabled(False); check_pins()
    pipe = P1.SentriFormerPipeline(P1.DEFAULT_CKPT, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    P1.PIPE_CFG = cfg; assert not m.training
    base = {a: site(m, a).detach().clone() for a in ARMS}; E = m.lm_head.weight.detach().double().cpu()
    log = {"transformers": transformers.__version__, "tokenizer_sha": TOK_SHA, "generic_sha": GENERIC_SHA}
    if stage == "selftest":
        sd = os.environ.get("WSP_SCRATCH", HERE); T = log["tests"] = {}
        # 1. pins + refusal on mismatch
        try: require("x", "y", "deliberate"); raise AssertionError("mismatch not refused")
        except SystemExit: T["pin_refusal"] = "ok"
        # 2. alignment on the known reply
        body = tok.encode(KNOWN, add_special_tokens=False); lab, text, c, ok = align(tok, body); assert ok
        mt = [x for x in GENERIC.finditer(text)]; assert mt, "known reply must contain a GENERIC match"
        table = [{"k": k, "id": t, "piece": text[c[k]:c[k + 1]], "chars": [c[k], c[k + 1]], "label": lab[k]} for k, t in enumerate(body)]
        flagged = "".join(r["piece"] for r in table if r["label"] == 1)
        for x in mt: assert x.group(0).strip() in flagged, (x.group(0), flagged)
        for r in table:
            inside = any(not text[i].isspace() for x in mt for i in range(max(r["chars"][0], x.start()), min(r["chars"][1], x.end())))
            assert (r["label"] == 1) == inside, r
        T["alignment_known_reply"] = {"text": text, "matches": [x.group(0) for x in mt], "table": table}
        assert reply_len(tok, []) == 0 and reply_len(tok, [tok.all_special_ids[0], body[0]]) == 1; T["length_empty"] = "ok"
        # 3. off-by-one: pure argmax generation, teacher-forced argmax must reproduce every token
        rows, _ = phase1_rows(); dec = m.sentriformer.decoder; nchk = 0
        from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask
        for _, r in rows.iloc[:5].iterrows():
            ids = tok.encode(r.context, add_special_tokens=True)[-256:]; t = torch.tensor([ids], device="cuda")
            g = generate_multitask(m, tok, cfg, t, torch.ones_like(t), max_new_tokens=60, temperature=0.0, repetition_penalty=1.0)[0].tolist()
            b = g[1:g.index(cfg.eos_token_id)] if cfg.eos_token_id in g else g[1:]
            enc = m.sentriformer.encoder(input_ids=t, attention_mask=torch.ones_like(t))
            hs, _ = dec(input_ids=torch.tensor([[cfg.bos_token_id] + b], device="cuda"), encoder_hidden_states=enc, encoder_attention_mask=torch.ones_like(t))
            pred = m.lm_head(hs)[0].argmax(-1).tolist(); assert pred[:len(b)] == b, "position t must predict b_t"; nchk += len(b)
        T["off_by_one_tokens_checked"] = nchk
        # 4. arm-F equivalence: logits shift by exactly E.delta
        r0 = rows.iloc[0]; ids = tok.encode(r0.context, add_special_tokens=True)[-256:]; t = torch.tensor([ids], device="cuda")
        enc = m.sentriformer.encoder(input_ids=t, attention_mask=torch.ones_like(t)); x = torch.tensor([[cfg.bos_token_id] + b], device="cuda")
        l0 = m.lm_head(dec(input_ids=x, encoder_hidden_states=enc, encoder_attention_mask=torch.ones_like(t))[0]).double()
        delta = torch.from_numpy(np.random.default_rng(5).standard_normal(256)) * 0.05; apply2(m, base, "F", -delta, 1.0)   # bias += delta
        l1 = m.lm_head(dec(input_ids=x, encoder_hidden_states=enc, encoder_attention_mask=torch.ones_like(t))[0]).double()
        err = float(((l1 - l0) - (E.cuda() @ delta.cuda())).abs().max()); assert err < 1e-3, err; T["armF_logit_equivalence_max_err"] = err
        # 5. classification invariance under both arms (Phase-1 rows only)
        apply2(m, base); h0 = heads(m, tok, rows.iloc[:20])
        for a in ARMS:
            apply2(m, base, a, torch.ones(256, dtype=torch.float64) * 0.1, 1.0); assert torch.equal(h0, heads(m, tok, rows.iloc[:20])), a
        apply2(m, base); T["classification_bitwise_both_arms"] = "ok"
        # 6. gate tests (P1 tests are re-run through gates2, plus length/empty boundaries)
        b0 = {"generic": .3, "action": .4, "specific": .3, "orphan": 0, "repetition": 0, "eos": 1, "portal": 0, "privacy": 0, "distinct": .8,
              "rougeL_clean": .12, "dial_loss": 2.7, "template_concentration": .3, "top_reply_share": .03, "mean_length": 20.0, "empty": 0.0}
        good, okc = {**b0, "generic": .2}, [-.1, -.15, -.05]; G = lambda cand, ci_=okc, drops=(0,), inv=True: gates2(b0, cand, ci_, .1, list(drops), inv)[1]
        assert G(good) and not G({**good, "repetition": .05}) and not G(good, drops=(.12,)) and not G(good, drops=(.1,))
        assert not G(good, ci_=[-.1, -.15, 0.0]) and G(good, ci_=[-.1, -.15, -1e-9]) and not G(good, inv=False)
        assert G({**good, "mean_length": 24.0}) and not G({**good, "mean_length": 24.01}) and G({**good, "mean_length": 16.0}) and not G({**good, "mean_length": 15.99})
        assert G({**good, "empty": .01 * (1 - 1e-9)}) and not G({**good, "empty": .010001})
        assert ci975(np.full(100, -.1))[2] < 0                         # constant improvement -> CI entirely below 0
        z = np.random.default_rng(1).choice([-1.0, 0.0, 1.0], 250); w975, w95 = ci975(z), P1.ci(z)
        assert w975[1] <= w95[1] and w975[2] >= w95[2], "97.5% CI must be at least as wide as the 95% CI"
        T["gates2"] = "13 gate cases + 2 CI checks ok"
        # 7. directions, P0 stability, controls (arm F output-space, arm N hidden-space)
        D, H, y, info = build(m, tok, cfg); T["construction"] = info
        C = controls(D, H, y, E)
        for o in C["F"]: assert abs(o["Ec_ratio"] - 1) <= 0.10 + 1e-12, o
        for o in C["N"]: assert abs(o["norm"] - float(D["N"].norm())) <= 1e-4 * float(D["N"].norm()), o
        T["controls"] = {a: [{k: v for k, v in o.items() if k != "vec"} for o in C[a]] for a in ARMS}
        # fallback path on a synthetic anisotropic E where +-10% is impossible for orthogonal random directions
        Es = torch.diag(torch.tensor([100.0] + [1.0] * 255, dtype=torch.float64)); ds = torch.zeros(256, dtype=torch.float64); ds[0] = 1
        Cs = controls({"F": ds, "N": ds}, {"F": torch.randn(50, 256, dtype=torch.float64), "N": torch.randn(50, 256, dtype=torch.float64)}, np.array([1] * 25 + [0] * 25), Es)
        assert any(o["fallback"] for o in Cs["F"] if o["kind"] == "random")
        for o in Cs["F"]: assert abs(o["Ec_ratio"] - 1) <= 0.10 + 1e-9, o
        T["fallback_path"] = "exercised: random controls fell back to exact output scaling"
        T["armF_top_penalised_tokens"] = penalised_tokens(tok, E, D["F"])
        # 8. save/validate/refuse in scratch (not patches_p2/), with a control edit deliberately left in memory
        tp = os.path.join(sd, "_selftest_P2.pt"); assert not os.path.exists(tp)
        apply2(m, base, "N", C["N"][0]["vec"], 1.0); assert save_or_validate2(m, base, "F", 0.5, D["F"], tp) == "saved"
        apply2(m, base, "N", C["N"][0]["vec"], 1.0); assert save_or_validate2(m, base, "F", 0.5, D["F"], tp) == "reused (validated)"
        bad = torch.load(tp, map_location="cpu", weights_only=False); bad["wsp_patch"]["alpha"] = 1.0; torch.save(bad, tp)
        try: save_or_validate2(m, base, "F", 0.5, D["F"], tp); raise AssertionError("bad metadata accepted")
        except SystemExit: pass
        os.remove(tp); assert not os.path.exists(tp + ".tmp"); T["save_validate_refuse"] = "ok"
        apply2(m, base); assert all(torch.equal(site(m, a), base[a]) for a in ARMS); check_pins(); T["reset_restores_T4d"] = "ok"
        out = os.path.join(HERE, "p2_selftest.json"); dump_atomic(log, out)
        print("SELFTEST PASS", json.dumps({k: v for k, v in T.items() if k not in ("alignment_known_reply", "controls")}, ensure_ascii=False))
        return
    os.makedirs(RES, exist_ok=True); A, B = eval_rows2()
    if stage == "stageA":
        out = os.path.join(RES, "stageA.json"); assert not os.path.exists(out)
        D, H, y, info = build(m, tok, cfg); C = controls(D, H, y, E); arms = [a for a in ARMS if info[a]["P0_stable"]]
        R = {**log, "construction": info, "arms_run": arms, "dropped_unstable": [a for a in ARMS if a not in arms],
             "d": {a: D[a].tolist() for a in ARMS}, "controls": {a: [{**{k: v for k, v in o.items() if k != "vec"}, "vec": o["vec"].tolist()} for o in C[a]] for a in ARMS},
             "armF_top_penalised_tokens": penalised_tokens(tok, E, D["F"]), "variants": {}}
        apply2(m, base); h0 = heads(m, tok, pd.concat([A, B])); bm, bg, brep = run2(m, tok, cfg, A); R["T4d"] = bm
        os.makedirs(PATCH_DIR, exist_ok=True)
        for arm in arms:
            for a in ALPHAS:
                p = os.path.join(PATCH_DIR, f"P2{arm}_a{a}.pt"); print(p, save_or_validate2(m, base, arm, a, D[arm], p), flush=True)
                inv = torch.equal(h0, heads(m, tok, pd.concat([A, B]))); cand, g, rep = run2(m, tok, cfg, A); drops = []
                for o in C[arm]:
                    apply2(m, base, arm, o["vec"], a); cm, _, _ = run2(m, tok, cfg, A); drops.append(bm["generic"] - cm["generic"])
                    R["variants"][f"C{arm}_{o['kind']}{o['k']}_a{a}"] = cm
                d_ = g - bg; gt, ok = gates2(bm, cand, ci975(d_), bm["generic"] - cand["generic"], drops, inv)
                R["variants"][f"P2{arm}_a{a}"] = {**cand, "generic_diff_CI975": ci975(d_), "control_generic_drops": drops, "gates": gt, "eligible": ok,
                                                  "checkpoint": p, "checkpoint_sha256": sha(p), "examples": list(zip(brep[:12], rep[:12]))}
                print(arm, a, "eligible" if ok else "not eligible", [k for k, v in gt.items() if not v], flush=True)
        best = {}
        for arm in arms:
            el = [a for a in ALPHAS if R["variants"][f"P2{arm}_a{a}"]["eligible"]]
            if el: best[arm] = min(el)                                # mildest = smallest alpha
        sel = max(best, key=lambda arm: bm["generic"] - R["variants"][f"P2{arm}_a{best[arm]}"]["generic"]) if best else None
        R["selected"] = {"arm": sel, "alpha": best[sel]} if sel else None
        R["decision"] = "proceed to stage B (confirmatory)" if sel else "REJECT P2 (no eligible configuration)"
        dump_atomic(R, out); print(R["decision"], R["selected"])
    elif stage == "stageB":
        out = os.path.join(RES, "stageB.json"); assert not os.path.exists(out), "stage B runs once"
        SA = json.load(open(os.path.join(RES, "stageA.json"), encoding="utf-8")); s = SA["selected"]; assert s
        arm, a = s["arm"], s["alpha"]; d = torch.tensor(SA["d"][arm], dtype=torch.float64)
        apply2(m, base); h0 = heads(m, tok, B); bm, bg, brep = run2(m, tok, cfg, B)
        apply2(m, base, arm, d, a); inv = torch.equal(h0, heads(m, tok, B)); cand, g, rep = run2(m, tok, cfg, B); drops = []
        for o in SA["controls"][arm]:
            apply2(m, base, arm, torch.tensor(o["vec"], dtype=torch.float64), a); cm, _, _ = run2(m, tok, cfg, B); drops.append(bm["generic"] - cm["generic"])
        gt, ok = gates2(bm, cand, ci975(g - bg), bm["generic"] - cand["generic"], drops, inv)
        R = {"rows": "B (250), confirmatory, run once", "arm": arm, "alpha": a, "T4d": bm, "P2": cand, "generic_diff_CI975": ci975(g - bg),
             "control_generic_drops": drops, "gates": gt, "decision": "CANDIDATE (not promoted; blind A/B required)" if ok else "REJECT"}
        if ok:
            rng = random.Random(11); key = {}; os.makedirs(KEYS, exist_ok=True)
            with open(os.path.join(RES, "AB_P2_SHEET.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
                for k, (i, x, yv) in enumerate(zip(B.index[:50], brep[:50], rep[:50])):
                    flip = rng.random() < .5; key[f"P2AB{k + 1:03d}"] = {"A": "P2" if flip else "T4d", "B": "T4d" if flip else "P2", "val_row": int(i)}
                    w.writerow([f"P2AB{k + 1:03d}", B.loc[i].context, yv if flip else x, x if flip else yv, "", ""])
            dump_atomic(key, os.path.join(KEYS, "AB_P2_KEY.json"))
        dump_atomic(R, out); print(R["decision"], [k for k, v in gt.items() if not v])
    apply2(m, base); check_pins()


if __name__ == "__main__":
    main()
