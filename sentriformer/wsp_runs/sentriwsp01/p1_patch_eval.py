"""WSP-P1: graded dampening of L3 FFN neurons {502,619,509,743} vs 5 matched random controls. Criteria: P1_CRITERIA.md (frozen).
  python -I wsp_runs/sentriwsp01/p1_patch_eval.py selftest   # checks only; one temp file in $WSP_SCRATCH, deleted; no patches/
  python -I wsp_runs/sentriwsp01/p1_patch_eval.py stageA     # saves P1 checkpoints, evaluates T4d / P1 x3 / controls x15 on rows A
  python -I wsp_runs/sentriwsp01/p1_patch_eval.py stageB     # ONCE: chosen alpha vs T4d + its controls on rows B
T4d (wsp_runs/FINAL_T4d/sentriformer_t4d.pt) is only ever read; its SHA-256 is verified before and after every stage."""
import hashlib, json, os, re, sys, collections, csv, random
import pandas as pd
import numpy as np, torch, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp")); sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, HERE)
from inference import SentriFormerPipeline, DEFAULT_CKPT
from eval_t4 import portal_loop, orphan, ACTION
from eval_full import rouge_l
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask, detect_repetition
from sentriformer.training.train_dialogue_sft import DialogueSFTCollator
from phase1_investigate import GENERIC, content, split_cur
from phase1b_controls import PRIV_REQ

T4D, T4D_SHA = os.path.join(ROOT, DEFAULT_CKPT), "bff705ec24b9916880c9df4b16eda284f961814f05828c09e61208b384af1efe"
TARGET, LAYER, ALPHAS = [502, 619, 509, 743], 3, [0.75, 0.5, 0.25]
EXCLUDE = set(TARGET) | {348, 939}                                  # recurring Phase-1 L3 neurons are not valid controls
PATCH_DIR, RES, KEYS = os.path.join(HERE, "patches"), os.path.join(HERE, "p1_results"), os.path.join(HERE, "_sealed_keys")  # keys never shared with sheets


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def check_t4d(): assert sha(T4D) == T4D_SHA, "T4d checkpoint changed!"


def dump_atomic(obj, path):
    """A killed write leaves only <path>.tmp, never a partial <path> that would block the next run."""
    with open(path + ".tmp", "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def fc2(m): return m.sentriformer.decoder.layers[LAYER].ffn.fc2.weight


def apply(m, base_w, neurons, alpha):
    """Reset fc2 to T4d's weights, then scale the given columns. Exact: W[:,j]*a == scaling neuron j's activation by a."""
    with torch.no_grad():
        w = fc2(m); w.copy_(base_w)
        if neurons: w[:, neurons] *= alpha


def eval_rows():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas()
    hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(HERE, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    rows = hist.drop(index=list(gate | tune | p1)).sample(500, random_state=53)
    assert not (gate | tune | p1) & set(rows.index) and len(rows) == 500
    return rows.iloc[:250], rows.iloc[250:]


def matched_controls(m, tok, cfg):
    """Mean post-GELU activation of every L3 neuron over reply tokens of the Phase-1 rows (never the evaluation rows)."""
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas()
    rows = df.loc[json.load(open(os.path.join(HERE, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"][:150]]
    dec, tot, n, acts = m.sentriformer.decoder, torch.zeros(1024, device="cuda"), 0, {}
    h = dec.layers[LAYER].ffn.fc1.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", torch.nn.functional.gelu(o[0])))
    for _, r in rows.iterrows():
        body, ids = generate(m, tok, cfg, r.context)[1:]
        e = torch.tensor([ids], device="cuda"); am = torch.ones_like(e); enc = m.sentriformer.encoder(input_ids=e, attention_mask=am)
        dec(input_ids=torch.tensor([[cfg.bos_token_id] + body], device="cuda"), encoder_hidden_states=enc, encoder_attention_mask=am)
        a = acts["a"]; assert a.dim() == 2 and a.shape == (len(body) + 1, 1024), a.shape   # hook stores gelu(o[0]) -> [T, 1024]
        tot += a[: len(body)].sum(0); n += len(body)                                     # positions 0..len-1 predict the reply tokens
    h.remove(); mean = (tot / n).cpu().numpy(); groups = []
    for seed in range(5):
        rng, g = np.random.default_rng(seed), []
        for t in TARGET:
            pool = [j for j in range(1024) if j not in EXCLUDE and j not in g and abs(mean[j] - mean[t]) <= 0.25 * abs(mean[t])]
            assert pool, f"no matched control for neuron {t}"
            g.append(int(rng.choice(pool)))
        groups.append(g)
    return groups, {int(t): round(float(mean[t]), 4) for t in TARGET}, [[round(float(mean[j]), 4) for j in g] for g in groups]


def reload_matches(m, path):
    """Saved checkpoint must reload strictly into a fresh model and equal the in-memory model parameter-for-parameter."""
    from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
    fresh = SentriFormerForMultiTask(m.config if hasattr(m, "config") else PIPE_CFG)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    a, b = dict(m.named_parameters()), dict(fresh.named_parameters())
    ok = a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k]) for k in a)
    ok = ok and fresh.lm_head.weight.data_ptr() == fresh.sentriformer.shared_embeddings.weight.data_ptr()
    return ok


def patch_meta(a):
    return {"id": f"P1_a{a}", "tensor": f"sentriformer.decoder.layers.{LAYER}.ffn.fc2.weight", "columns": TARGET, "alpha": a, "base_sha256": T4D_SHA}


def save_or_validate_patch(m, base_w, a, p):
    """Model is rebuilt from baseline T4d + this patch. New file: atomic write (tmp + os.replace). Existing file (left by a
    killed run): reused only if metadata matches exactly AND it reloads strictly to the same parameters; otherwise stop."""
    apply(m, base_w, TARGET, a)                                         # reset to T4d, then patch: never inherits control state
    if os.path.exists(p):
        meta = torch.load(p, map_location="cpu", weights_only=False).get("wsp_patch")
        if meta != patch_meta(a): raise SystemExit(f"{p} exists with different metadata {meta}; refusing to reuse or overwrite")
        if not reload_matches(m, p): raise SystemExit(f"{p} exists but does not reload to the expected patched model; refusing")
        return "reused (validated)"
    tmp = p + ".tmp"
    torch.save({"model_state_dict": m.state_dict(), "train_run": "T4d", "wsp_patch": patch_meta(a)}, tmp)
    os.replace(tmp, p)                                                  # atomic on the same volume
    assert reload_matches(m, p), f"{p} does not reload to the in-memory patched model"
    return "saved"


def generate(m, tok, cfg, ctx):
    ids = tok.encode(ctx, add_special_tokens=True)[-256:] or [1]; t = torch.tensor([ids], device="cuda")
    g = generate_multitask(m, tok, cfg, t, torch.ones_like(t), max_new_tokens=128, temperature=0.0, repetition_penalty=1.2)[0].tolist()
    eos = cfg.eos_token_id in g; body = g[1:g.index(cfg.eos_token_id)] if eos else g[1:]
    return (tok.decode(body, skip_special_tokens=True).strip(), eos), body, ids


def heads(m, tok, rows):
    """Classification outputs on the message only (as inference.py does); must be bitwise identical across decoder edits."""
    out = []
    for _, r in rows.iterrows():
        msg = split_cur(r.context)[1][len("Customer: "):]
        ids = torch.tensor([tok.encode(msg, truncation=True, max_length=128, add_special_tokens=True) or [1]], device="cuda")
        o = m(input_ids=ids, attention_mask=torch.ones_like(ids))
        out.append(torch.cat([o["category_logits"].flatten(), o["csat_logits"].flatten(), o["sentiment_logits"].flatten(), o["urgency_score"].flatten()]).cpu())
    return torch.stack(out)


def dial_loss(m, tok, cfg, rows):
    items = []
    for c, r in zip(rows.context, rows.response):
        rid = tok.encode(r, truncation=True, max_length=127, add_special_tokens=True)
        if not rid or rid[-1] != cfg.eos_token_id: rid = rid + [cfg.eos_token_id]
        items.append({"input_ids": tok.encode(c, add_special_tokens=True)[-256:] or [1], "labels": rid})
    tot = n = 0
    for i in range(0, len(items), 64):
        b = DialogueSFTCollator(0)(items[i:i + 64])
        o = m(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda(), generation_labels=b["labels"].cuda())
        k = int((b["labels"] != -100).sum()); tot += o["generation_loss"].float().item() * k; n += k
    return tot / n


def run(m, tok, cfg, rows):
    per, replies = [], []
    for _, r in rows.iterrows():
        (txt, eos), _, _ = generate(m, tok, cfg, r.context); cur = split_cur(r.context)[1]
        per.append({"generic": bool(GENERIC.search(txt)), "specific": bool(content(txt) & content(cur)), "action": bool(ACTION.search(txt)),
                    "orphan": bool(orphan(txt)), "repetition": bool(detect_repetition(txt)), "eos": bool(eos), "portal": bool(portal_loop(txt)),
                    "privacy": bool(PRIV_REQ.search(txt)) and not PRIV_REQ.search(r.context),
                    "rouge_clean": None if portal_loop(r.response) else rouge_l(txt, r.response)})
        replies.append(txt)
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "rouge_clean"}
    agg["rougeL_clean"] = float(np.mean([p["rouge_clean"] for p in per if p["rouge_clean"] is not None]))
    agg["distinct"] = len(set(replies)) / len(replies)
    op = collections.Counter(" ".join(x.split()[:5]) for x in replies)
    agg["template_concentration"] = sum(c for _, c in op.most_common(6)) / len(replies)
    agg["top_reply_share"] = collections.Counter(replies).most_common(1)[0][1] / len(replies)
    agg["dial_loss"] = dial_loss(m, tok, cfg, rows)
    return agg, np.array([p["generic"] for p in per], float), replies


def ci(d, n=10000):
    b = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def gates(base, cand, diff_ci, cand_drop, ctrl_drops, inv_ok):
    g = {"1_generic_lower_CI": diff_ci[2] < 0,
         "2_beats_every_control": all(cand_drop > c for c in ctrl_drops),
         "3_action": cand["action"] >= base["action"] - .02, "3_specific": cand["specific"] >= base["specific"] - .02,
         "3_orphan": cand["orphan"] <= base["orphan"] + .01, "3_repetition": cand["repetition"] <= base["repetition"] + .01,
         "3_eos": cand["eos"] >= base["eos"] - .01, "3_portal": cand["portal"] <= base["portal"] + .01,
         "3_privacy": cand["privacy"] <= base["privacy"] + .01, "3_distinct": cand["distinct"] >= base["distinct"] - .02,
         "3_rouge": cand["rougeL_clean"] >= base["rougeL_clean"] - .005, "3_dial_loss": cand["dial_loss"] <= base["dial_loss"] + .02,
         "4_no_template_swap": cand["template_concentration"] <= base["template_concentration"] + .02 and cand["top_reply_share"] <= base["top_reply_share"] + .01,
         "5_classification_bitwise": inv_ok}
    return g, all(g.values())


def main():
    stage = sys.argv[1]; torch.set_grad_enabled(False); check_t4d()
    pipe = SentriFormerPipeline(DEFAULT_CKPT, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    global PIPE_CFG; PIPE_CFG = cfg; assert not m.training, "model must be in eval mode"
    base_w = fc2(m).detach().clone(); A, B = eval_rows()
    if stage == "selftest":
        b0 = {"generic": .3, "action": .4, "specific": .3, "orphan": 0, "repetition": 0, "eos": 1, "portal": 0, "privacy": 0, "distinct": .8,
              "rougeL_clean": .12, "dial_loss": 2.7, "template_concentration": .3, "top_reply_share": .03}
        good = {**b0, "generic": .2}
        assert gates(b0, good, [-.1, -.15, -.05], .1, [.01, 0, -.02], True)[1], "good candidate must pass"
        assert not gates(b0, {**good, "repetition": .05}, [-.1, -.15, -.05], .1, [0], True)[1], "repetition regression must fail"
        assert not gates(b0, good, [-.1, -.15, -.05], .1, [.12], True)[1], "losing to a control must fail"
        assert not gates(b0, good, [-.1, -.15, .01], .1, [0], True)[1], "CI touching 0 must fail"
        assert not gates(b0, {**good, "template_concentration": .4}, [-.1, -.15, -.05], .1, [0], True)[1], "template swap must fail"
        assert not gates(b0, good, [-.1, -.15, -.05], .1, [0], False)[1], "classification change must fail"
        assert not gates(b0, good, [-.1, -.15, 0.0], .1, [0], True)[1], "CI upper bound exactly 0 must fail"
        assert gates(b0, good, [-.1, -.15, -1e-9], .1, [0], True)[1], "CI upper bound just below 0 must pass"
        assert not gates(b0, good, [-.1, -.15, -.05], .1, [.1], True)[1], "a control TYING P1 must fail (strictly larger required)"
        assert not gates(b0, {**good, "action": .37}, [-.1, -.15, -.05], .1, [0], True)[1], "action-word loss >2pp must fail"
        # boundary tests for every regression threshold: exactly at the tolerance passes, just beyond fails
        e = 1e-6
        for key, tol in [("specific", -.02), ("orphan", .01), ("repetition", .01), ("eos", -.01), ("portal", .01), ("privacy", .01),
                         ("distinct", -.02), ("rougeL_clean", -.005), ("dial_loss", .02), ("top_reply_share", .01), ("template_concentration", .02)]:
            beyond = b0[key] + tol + (e if tol > 0 else -e)
            assert gates(b0, {**good, key: b0[key] + tol * (1 - 1e-9)}, [-.1, -.15, -.05], .1, [0], True)[1], f"{key} at tolerance must pass"
            assert not gates(b0, {**good, key: beyond}, [-.1, -.15, -.05], .1, [0], True)[1], f"{key} beyond tolerance must fail"
        n_gate_tests = 10 + 2 * 11
        x = torch.randn(3, 1024, device="cuda"); a = 0.5
        ref = x.clone(); ref[:, TARGET] *= a; y0 = ref @ base_w.T
        apply(m, base_w, TARGET, a); assert torch.allclose(x @ fc2(m).T, y0, atol=1e-5), "column scaling != activation scaling"
        changed = int((fc2(m) != base_w).sum()); assert changed <= 1024, changed
        h1 = heads(m, tok, A.iloc[:20]); apply(m, base_w, [], 1.0); h0 = heads(m, tok, A.iloc[:20])
        assert torch.equal(h0, h1), "decoder edit changed classification outputs"; assert torch.equal(fc2(m), base_w)
        groups, tm, cm = matched_controls(m, tok, cfg); assert len(groups) == 5 and all(len(set(g)) == 4 and not set(g) & EXCLUDE for g in groups)
        # save / resume path, exercised in the scratch dir (not patches/): new save, validated reuse, refusal on bad metadata
        sd_dir = os.environ.get("WSP_SCRATCH", HERE); tp = os.path.join(sd_dir, "_selftest_P1.pt")
        assert not os.path.exists(tp)
        apply(m, base_w, groups[0], .9)                                  # leave a CONTROL edit in memory on purpose
        assert save_or_validate_patch(m, base_w, a, tp) == "saved"
        apply(m, base_w, groups[0], .9)
        assert save_or_validate_patch(m, base_w, a, tp) == "reused (validated)", "resume must rebuild from baseline, not inherit"
        bad = torch.load(tp, map_location="cpu", weights_only=False); bad["wsp_patch"]["alpha"] = .3; torch.save(bad, tp)
        try: save_or_validate_patch(m, base_w, a, tp); raise AssertionError("bad metadata must be refused")
        except SystemExit: pass
        os.remove(tp); assert not os.path.exists(tp + ".tmp")
        apply(m, base_w, [], 1.0)
        assert not os.path.exists(os.path.join(sd_dir, "_dump.json"))
        dump_atomic({"x": 1}, os.path.join(sd_dir, "_dump.json")); assert json.load(open(os.path.join(sd_dir, "_dump.json")))["x"] == 1
        assert not os.path.exists(os.path.join(sd_dir, "_dump.json.tmp")); os.remove(os.path.join(sd_dir, "_dump.json"))
        print(f"selftest ok: {n_gate_tests} gate tests, atomic json, exact equivalence, {changed} values changed, heads bitwise identical, activation shape ok, "
              f"controls {groups} (target mean act {tm}, control {cm}), atomic save + validated resume + bad-metadata refusal, "
              f"reset restores T4d"); check_t4d(); return
    os.makedirs(RES, exist_ok=True)
    if stage == "stageA":
        out = os.path.join(RES, "stageA.json"); assert not os.path.exists(out)
        groups, tgt_mean, ctrl_mean = matched_controls(m, tok, cfg)
        rows_all = pd.concat([A, B]); apply(m, base_w, [], 1.0); h0 = heads(m, tok, rows_all)
        base, base_g, base_rep = run(m, tok, cfg, A); R = {"rows": "A (250)", "controls": groups, "target_mean_act": tgt_mean,
                                                           "control_mean_act": ctrl_mean, "T4d": base, "variants": {}}
        os.makedirs(PATCH_DIR, exist_ok=True)
        for a in ALPHAS:
            p = os.path.join(PATCH_DIR, f"P1_a{a}.pt")
            print(p, save_or_validate_patch(m, base_w, a, p), flush=True)   # baseline reset + patch; atomic save or strict validated reuse
            inv = torch.equal(h0, heads(m, tok, rows_all)); cand, g, rep = run(m, tok, cfg, A)
            drops = []
            for gi, grp in enumerate(groups):
                apply(m, base_w, grp, a); c, cg, _ = run(m, tok, cfg, A); drops.append(base["generic"] - c["generic"])
                R["variants"][f"C{gi}_a{a}"] = c
            d = g - base_g; gt, ok = gates(base, cand, ci(d), base["generic"] - cand["generic"], drops, inv)
            R["variants"][f"P1_a{a}"] = {**cand, "generic_diff_CI": ci(d), "control_generic_drops": drops, "gates": gt, "eligible": ok,
                                          "checkpoint": p, "checkpoint_sha256": sha(p), "examples": list(zip(base_rep[:12], rep[:12]))}
            print(a, "eligible" if ok else "not eligible", {k: v for k, v in gt.items() if not v}, flush=True)
        el = [a for a in ALPHAS if R["variants"][f"P1_a{a}"]["eligible"]]
        R["selected_alpha"] = max(el) if el else None; R["decision"] = "proceed to stage B" if el else "REJECT P1 (no eligible alpha)"
        dump_atomic(R, out); print(R["decision"], R["selected_alpha"])
    elif stage == "stageB":
        out = os.path.join(RES, "stageB.json"); assert not os.path.exists(out), "stage B runs once"
        SA = json.load(open(os.path.join(RES, "stageA.json"), encoding="utf-8")); a = SA["selected_alpha"]; assert a is not None
        apply(m, base_w, [], 1.0); h0 = heads(m, tok, B); base, base_g, base_rep = run(m, tok, cfg, B)
        apply(m, base_w, TARGET, a); inv = torch.equal(h0, heads(m, tok, B)); cand, g, rep = run(m, tok, cfg, B)
        drops = []
        for grp in SA["controls"]:
            apply(m, base_w, grp, a); c, _, _ = run(m, tok, cfg, B); drops.append(base["generic"] - c["generic"])
        gt, ok = gates(base, cand, ci(g - base_g), base["generic"] - cand["generic"], drops, inv)
        R = {"rows": "B (250), run once", "alpha": a, "T4d": base, "P1": cand, "generic_diff_CI": ci(g - base_g), "control_generic_drops": drops,
             "gates": gt, "decision": "CANDIDATE (not promoted; blind A/B required)" if ok else "REJECT",
             "ab_export": "p1_results/AB_P1_SHEET.csv (key in _sealed_keys/, never share it)" if ok else "none: P1 rejected, no A/B"}
        if ok:   # frozen criteria: the blind A/B exists only for a Stage-B CANDIDATE
            rng = random.Random(7); key = {}; os.makedirs(KEYS, exist_ok=True)
            with open(os.path.join(RES, "AB_P1_SHEET.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
                for k, (i, x, y) in enumerate(zip(B.index[:50], base_rep[:50], rep[:50])):
                    flip = rng.random() < .5; key[f"P1AB{k + 1:03d}"] = {"A": "P1" if flip else "T4d", "B": "T4d" if flip else "P1", "val_row": int(i)}
                    w.writerow([f"P1AB{k + 1:03d}", B.loc[i].context, y if flip else x, x if flip else y, "", ""])
            dump_atomic(key, os.path.join(KEYS, "AB_P1_KEY.json"))
        dump_atomic(R, out); print(R["decision"], {k: v for k, v in gt.items() if not v})
    apply(m, base_w, [], 1.0); check_t4d()


if __name__ == "__main__":
    main()
