"""T5c: MMI reranking (decode-time, no weight change) per frozen T5C_CRITERIA.md. Model = default after the T5 decision.
  python -I wsp_runs/t5/t5c_mmi.py selftest | stageA | stageB"""
import csv, json, os, random, sys
import numpy as np, pyarrow.parquet as pq, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); S1 = os.path.join(ROOT, "wsp_runs", "sentriwsp01")
sys.path.insert(0, S1); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "wsp"))
import p1_patch_eval as P1, p2_patch_eval as P2, t5_compare as T5, t5_audit_v5 as V
import mmi as M
from eval_t4 import portal_loop, orphan, ACTION
from eval_full import rouge_l
from phase1_investigate import GENERIC, content, split_cur
from sentriformer.evaluation.audit_m10_3_threeway import detect_repetition
import collections

LAMS = [0.15, 0.3, 0.5]; RES = os.path.join(HERE, "t5c_results")


def model_path():
    d = json.load(open(os.path.join(HERE, "t5_compare.json")))
    return ("wsp_runs/FINAL_T5a/sentriformer_t5a.pt" if d["decision"].startswith("PASS") else "wsp_runs/FINAL_T4d/sentriformer_t4d.pt"), d["decision"]


def rows_rs97():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(S1, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    a1, b1 = P1.eval_rows(); a2, b2 = P2.eval_rows2(); r79 = T5.rows_rs79()
    used = gate | tune | p1 | set(a1.index) | set(b1.index) | set(a2.index) | set(b2.index) | set(r79.index)
    rows = hist.drop(index=list(used)).sample(500, random_state=97); assert not used & set(rows.index)
    return rows.iloc[:250], rows.iloc[250:]


def metrics(items, tok):
    """items: list of (row, body). Same metric definitions as t5_compare / P2."""
    per, reps = [], []
    for r, body in items:
        txt = tok.decode(body, skip_special_tokens=True).strip(); cur = split_cur(r.context)[1]
        per.append({"generic_phrase": bool(GENERIC.search(txt)), "broken_link": bool(V.BROKEN.search(txt)), "orphan": bool(orphan(txt)),
                    "privacy": V.secret_request(txt) or V.bulk_contact(r.context, txt), "portal": bool(portal_loop(txt)),
                    "repetition": bool(detect_repetition(txt)), "eos": len(body) < 128, "action": bool(ACTION.search(txt)),
                    "specific": bool(content(txt) & content(cur)), "length": P2.reply_len(tok, body),
                    "rouge_clean": None if portal_loop(r.response) else rouge_l(txt, r.response)})
        reps.append(txt)
    keys = [k for k in per[0] if k not in ("length", "rouge_clean")]
    agg = {k: float(np.mean([p[k] for p in per])) for k in keys}
    agg.update(mean_length=float(np.mean([p["length"] for p in per])), empty=float(np.mean([p["length"] == 0 for p in per])),
               rougeL_clean=float(np.mean([p["rouge_clean"] for p in per if p["rouge_clean"] is not None])), distinct=len(set(reps)) / len(reps),
               template_concentration=sum(c for _, c in collections.Counter(" ".join(x.split()[:5]) for x in reps).most_common(6)) / len(reps))
    return agg, np.array([p["generic_phrase"] for p in per], float), reps


def gates(b, c, diff_ci, drop, ctrl_drop):
    g = {"1_generic_lower_CI": diff_ci[2] < 0, "2_beats_random_pick": drop > ctrl_drop,
         "3_action": c["action"] >= b["action"] - .02, "3_specific": c["specific"] >= b["specific"] - .02, "3_broken": c["broken_link"] <= b["broken_link"] + .01,
         "3_orphan": c["orphan"] <= b["orphan"] + .01, "3_privacy": c["privacy"] <= b["privacy"] + .01, "3_portal": c["portal"] <= b["portal"] + .01,
         "3_repetition": c["repetition"] <= b["repetition"] + .01, "3_eos": c["eos"] >= b["eos"] - .01, "3_distinct": c["distinct"] >= b["distinct"] - .02,
         "3_rouge": c["rougeL_clean"] >= b["rougeL_clean"] - .005, "3_length": abs(c["mean_length"] / b["mean_length"] - 1) <= .2,
         "3_empty": c["empty"] <= b["empty"] + .01, "3_template": c["template_concentration"] <= b["template_concentration"] + .02}
    return g, all(g.values())


def pools(m, tok, cfg, rows, offset):
    out = []
    for j, (_, r) in enumerate(rows.iterrows()):
        ids = torch.tensor([tok.encode(r.context, add_special_tokens=True)[-256:] or [1]], device="cuda")
        c = M.candidates(m, tok, cfg, ids, n=8, seed_base=1000 * (offset + j)); out.append((r, c, M.score_all(m, tok, cfg, ids, c), offset + j))
    return out


def evaluate(P, tok, lam_list):
    base, bg, brep = metrics([(r, c[0]) for r, c, s, _ in P], tok)                                    # greedy = candidate 0
    rnd, rg, _ = metrics([(r, c[1 + random.Random(7 * pos).randrange(len(c) - 1)]) for r, c, s, pos in P], tok)   # random sampled pick
    res = {"greedy": base, "random_pick": rnd}
    for lam in lam_list:
        cm, cg, crep = metrics([(r, c[M.pick(s, lam)]) for r, c, s, _ in P], tok)
        d = cg - bg; gt, ok = gates(base, cm, P1.ci(d), base["generic_phrase"] - cm["generic_phrase"], base["generic_phrase"] - rnd["generic_phrase"])
        res[f"mmi_{lam}"] = {**cm, "generic_diff_CI95": P1.ci(d), "gates": gt, "eligible": ok, "examples": list(zip(brep[:10], crep[:10]))}
        print(lam, "eligible" if ok else "not eligible", [k for k, v in gt.items() if not v], flush=True)
    return res, brep


def main():
    stage = sys.argv[1]; torch.set_grad_enabled(False); P1.check_t4d()
    if stage == "selftest":
        pipe = P1.SentriFormerPipeline(P1.DEFAULT_CKPT, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
        A, B = rows_rs97(); r = A.iloc[0]; ids = torch.tensor([tok.encode(r.context, add_special_tokens=True)[-256:]], device="cuda")
        c = M.candidates(m, tok, cfg, ids, n=3, seed_base=5)
        g = P1.generate(m, tok, cfg, r.context)[1]; assert c[0] == g, "candidate 0 must equal the deployed greedy reply"
        s = M.score_all(m, tok, cfg, ids, c); assert all(np.isfinite(x) for p in s for x in p)
        assert M.pick(s, 0.0) == max(range(3), key=lambda i: s[i][0])        # lam=0 -> plain likelihood
        b0 = {"generic_phrase": .3, "broken_link": 0, "orphan": 0, "privacy": 0, "portal": 0, "repetition": 0, "eos": 1, "action": .4, "specific": .3,
              "mean_length": 20, "empty": 0, "rougeL_clean": .12, "distinct": .8, "template_concentration": .3}
        good = {**b0, "generic_phrase": .2}
        assert gates(b0, good, [-.1, -.15, -.05], .1, .02)[1] and not gates(b0, good, [-.1, -.15, -.05], .1, .1)[1]
        assert not gates(b0, {**good, "action": .37}, [-.1, -.15, -.05], .1, 0)[1] and not gates(b0, good, [-.1, -.15, 0], .1, 0)[1]
        print("T5c selftest ok: greedy candidate == deployed reply; scores finite; lam=0 = likelihood; gate tests ok"); return
    os.makedirs(RES, exist_ok=True); ckpt, t5dec = model_path()
    pipe = P1.SentriFormerPipeline(ckpt, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg; A, B = rows_rs97()
    if stage == "stageA":
        out = os.path.join(RES, "stageA.json"); assert not os.path.exists(out)
        res, _ = evaluate(pools(m, tok, cfg, A, 0), tok, LAMS)
        el = [l for l in LAMS if res[f"mmi_{l}"]["eligible"]]
        R = {"model": ckpt, "t5_decision": t5dec, "results": res, "selected_lambda": min(el) if el else None,
             "decision": "proceed to stage B" if el else "REJECT T5c (no eligible lambda)"}
        P1.dump_atomic(R, out); print(R["decision"], R["selected_lambda"])
    elif stage == "stageB":
        out = os.path.join(RES, "stageB.json"); assert not os.path.exists(out)
        SA = json.load(open(os.path.join(RES, "stageA.json"))); lam = SA["selected_lambda"]; assert lam is not None and SA["model"] == ckpt
        P = pools(m, tok, cfg, B, 250); res, brep = evaluate(P, tok, [lam]); ok = res[f"mmi_{lam}"]["eligible"]
        R = {"model": ckpt, "lambda": lam, "results": res, "decision": "PASS -> MMI enabled by default for this checkpoint" if ok else "REJECT"}
        if ok:
            dj = os.path.join(ROOT, os.path.dirname(ckpt), "decoding.json"); assert not os.path.exists(dj)
            P1.dump_atomic({"mmi_lambda": lam, "n_candidates": 8, "source": "T5c stage B pass", "criteria": "wsp_runs/t5/T5C_CRITERIA.md"}, dj)
            mrep = [tok.decode(c[M.pick(s, lam)], skip_special_tokens=True).strip() for r, c, s, _ in P]
            rng = random.Random(31); key = {}; os.makedirs(os.path.join(S1, "_sealed_keys"), exist_ok=True)
            with open(os.path.join(HERE, "AB_MMI_vs_greedy_LOCAL_ONLY.csv"), "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh); w.writerow(["LOCAL ONLY - contains TWCS text - do not upload"] + [""] * 5)
                w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
                for k, ((r, *_), x, y) in enumerate(list(zip(P, brep, mrep))[:50]):
                    flip = rng.random() < .5; key[f"MMI{k + 1:03d}"] = {"A": "MMI" if flip else "greedy", "B": "greedy" if flip else "MMI"}
                    w.writerow([f"MMI{k + 1:03d}", r.context, y if flip else x, x if flip else y, "", ""])
            P1.dump_atomic(key, os.path.join(S1, "_sealed_keys", "AB_MMI_KEY.json"))
        P1.dump_atomic(R, out); print(R["decision"])
    P1.check_t4d()


if __name__ == "__main__":
    main()
