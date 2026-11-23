"""T5a decision per frozen T5_CRITERIA.md: T4d vs T5a vs T5a-ctrl on 500 fresh val rows (rs=79), plus eval_full gates.
On PASS: package wsp_runs/FINAL_T5a, repoint wsp/inference.py DEFAULT_CKPT, write LOCAL-ONLY blind A/B sheet (key in _sealed_keys).
  python -I wsp_runs/t5/t5_compare.py [--rows N]   (--rows only for a smoke test; writes to t5_compare_SMOKE.json)"""
import csv, hashlib, json, os, random, re, shutil, sys, collections
import numpy as np, pyarrow.parquet as pq, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); S1 = os.path.join(ROOT, "wsp_runs", "sentriwsp01")
sys.path.insert(0, S1); sys.path.insert(0, HERE)
import p1_patch_eval as P1, p2_patch_eval as P2
import t5_audit_v5 as V
from eval_t4 import portal_loop, orphan, ACTION
from eval_full import rouge_l
from phase1_investigate import GENERIC, content, split_cur
from sentriformer.evaluation.audit_m10_3_threeway import detect_repetition

MODELS = {"T4d": "wsp_runs/FINAL_T4d/sentriformer_t4d.pt", "T5a": "wsp_runs/train/T5a/model_best.pt", "T5a_ctrl": "wsp_runs/train/T5a_ctrl/model_best.pt"}


def rows_rs79():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(S1, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    a1, b1 = P1.eval_rows(); a2, b2 = P2.eval_rows2()
    used = gate | tune | p1 | set(a1.index) | set(b1.index) | set(a2.index) | set(b2.index)
    rows = hist.drop(index=list(used)).sample(500, random_state=79); assert not used & set(rows.index)
    return rows


def evaluate(name, ckpt, rows):
    pipe = P1.SentriFormerPipeline(ckpt, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg; assert not m.training
    per, replies = [], []
    with torch.no_grad():
        for _, r in rows.iterrows():
            (txt, eos), body, _ = P1.generate(m, tok, cfg, r.context); cur = split_cur(r.context)[1]
            per.append({"broken_link": bool(V.BROKEN.search(txt)), "generic_whole": V.generic_whole(txt), "generic_phrase": bool(GENERIC.search(txt)),
                        "privacy": V.secret_request(txt) or V.bulk_contact(r.context, txt), "orphan": bool(orphan(txt)), "portal": bool(portal_loop(txt)),
                        "eos": bool(eos), "repetition": bool(detect_repetition(txt)), "action": bool(ACTION.search(txt)),
                        "specific": bool(content(txt) & content(cur)), "length": P2.reply_len(tok, body),
                        "rouge_clean": None if portal_loop(r.response) else rouge_l(txt, r.response)})
            replies.append(txt)
        loss = P1.dial_loss(m, tok, cfg, rows)
    del m, pipe; torch.cuda.empty_cache()
    keys = [k for k in per[0] if k not in ("length", "rouge_clean")]
    agg = {k: float(np.mean([p[k] for p in per])) for k in keys}
    agg.update(mean_length=float(np.mean([p["length"] for p in per])), empty=float(np.mean([p["length"] == 0 for p in per])),
               rougeL_clean=float(np.mean([p["rouge_clean"] for p in per if p["rouge_clean"] is not None])), distinct=len(set(replies)) / len(replies),
               template_concentration=sum(c for _, c in collections.Counter(" ".join(x.split()[:5]) for x in replies).most_common(6)) / len(replies),
               dial_loss=loss)
    arr = {k: np.array([float(p[k]) for p in per]) for k in keys}
    print(name, {k: round(v, 4) for k, v in agg.items()}, flush=True)
    return agg, arr, replies


def main():
    smoke = "--rows" in sys.argv; n = int(sys.argv[sys.argv.index("--rows") + 1]) if smoke else 500
    out = os.path.join(HERE, "t5_compare_SMOKE.json" if smoke else "t5_compare.json")
    if not smoke: assert not os.path.exists(out), "decision already made"
    P1.check_t4d(); rows = rows_rs79().iloc[:n]
    names = ["T4d"] if smoke else list(MODELS)
    R = {"rows": f"{n} fresh multi_val rows with history, rs=79", "filters_human_validated": False, "models": {}}
    A, REP = {}, {}
    for k in names:
        R["models"][k], A[k], REP[k] = evaluate(k, os.path.join(ROOT, MODELS[k]), rows)
    if smoke:
        json.dump(R, open(out, "w"), indent=1); print("smoke ok"); P1.check_t4d(); return
    b, c, t = R["models"]["T4d"], R["models"]["T5a_ctrl"], R["models"]["T5a"]
    ci = lambda d: P1.ci(d)
    R["primary_broken_link_T5a_minus_ctrl_CI95"] = ci(A["T5a"]["broken_link"] - A["T5a_ctrl"]["broken_link"])
    R["secondary_CI95_T5a_minus_ctrl"] = {k: ci(A["T5a"][k] - A["T5a_ctrl"][k]) for k in ("generic_whole", "generic_phrase", "privacy", "orphan")}
    R["secondary_CI95_T5a_minus_T4d"] = {k: ci(A["T5a"][k] - A["T4d"][k]) for k in ("broken_link", "generic_whole", "generic_phrase", "privacy", "orphan")}
    F = lambda nm: json.load(open(os.path.join(ROOT, "wsp_runs", "eval_full", f"{nm}_val", "metrics.json")))
    f4, f5 = F("T4d"), F("T5a")
    g = {"P1_broken_link_lower_than_ctrl": R["primary_broken_link_T5a_minus_ctrl_CI95"][2] < 0,
         "G1_cat_acc": f5["category_len512"]["accuracy"] >= f4["category_len512"]["accuracy"] - .01,
         "G1_cat_macroF1": f5["category_len512"]["macro_f1_sklearn"] >= f4["category_len512"]["macro_f1_sklearn"] - .01,
         "G1_csat_macroF1": f5["csat"]["macro_f1_sklearn"] >= f4["csat"]["macro_f1_sklearn"] - .01,
         "G1_sent_macroF1": f5["sentiment"]["all"]["macro_f1_sklearn"] >= f4["sentiment"]["all"]["macro_f1_sklearn"] - .01,
         "G2_portal": t["portal"] <= b["portal"] + .01, "G2_eos": t["eos"] >= b["eos"] - .01, "G2_repetition": t["repetition"] <= b["repetition"] + .01,
         "G2_action": t["action"] >= b["action"] - .02, "G2_specific": t["specific"] >= b["specific"] - .02, "G2_distinct": t["distinct"] >= b["distinct"] - .02,
         "G2_rouge": t["rougeL_clean"] >= b["rougeL_clean"] - .005, "G2_length": abs(t["mean_length"] / b["mean_length"] - 1) <= .2,
         "G2_empty": t["empty"] <= b["empty"] + .01, "G2_template": t["template_concentration"] <= b["template_concentration"] + .02,
         "G3_dial_loss": t["dial_loss"] <= b["dial_loss"] + .05}
    R["gates"], ok = g, all(g.values()); R["decision"] = "PASS -> T5a deployed as default" if ok else "REJECT -> T4d stays default"
    R["examples"] = [{"row": int(i), "T4d": x, "T5a": y} for i, x, y in list(zip(rows.index, REP["T4d"], REP["T5a"]))[:0]]  # replies kept local below
    if ok:
        fd = os.path.join(ROOT, "wsp_runs", "FINAL_T5a"); os.makedirs(fd, exist_ok=True); dst = os.path.join(fd, "sentriformer_t5a.pt")
        assert not os.path.exists(dst); shutil.copy2(os.path.join(ROOT, MODELS["T5a"]), dst)
        R["final_checkpoint"] = {"path": "wsp_runs/FINAL_T5a/sentriformer_t5a.pt", "sha256": P1.sha(dst)}
        inf = os.path.join(ROOT, "wsp", "inference.py"); s = open(inf, encoding="utf-8").read()
        old = 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T4d", "sentriformer_t4d.pt")'
        assert s.count(old) == 1; open(inf, "w", encoding="utf-8").write(s.replace(old, 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")  # T5a (rollback: FINAL_T4d)', 1))
        rng = random.Random(23); key = {}; os.makedirs(os.path.join(S1, "_sealed_keys"), exist_ok=True)
        with open(os.path.join(HERE, "AB_T5a_vs_T4d_LOCAL_ONLY.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(["LOCAL ONLY - contains TWCS text - do not upload"] + [""] * 5)
            w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
            for k, (i, x, y) in enumerate(list(zip(rows.index, REP["T4d"], REP["T5a"]))[:50]):
                flip = rng.random() < .5; key[f"T5AB{k + 1:03d}"] = {"A": "T5a" if flip else "T4d", "B": "T4d" if flip else "T5a", "val_row": int(i)}
                w.writerow([f"T5AB{k + 1:03d}", rows.loc[i].context, y if flip else x, x if flip else y, "", ""])
        P1.dump_atomic(key, os.path.join(S1, "_sealed_keys", "AB_T5a_KEY.json"))
    with open(os.path.join(HERE, "t5_compare_replies_LOCAL_ONLY.json"), "w", encoding="utf-8") as fh:
        json.dump({"row_index": [int(i) for i in rows.index], "replies": REP}, fh, ensure_ascii=False)
    P1.dump_atomic(R, out); P1.check_t4d()
    print(R["decision"], [k for k, v in g.items() if not v])


if __name__ == "__main__":
    main()
