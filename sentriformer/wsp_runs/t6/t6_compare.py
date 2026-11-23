"""T6 decision per frozen T6_CRITERIA.md: T5a vs T6 vs T6-ctrl on 500 fresh val rows (rs=131) + eval_full gates.
Reuses t5_compare.evaluate (same generation and metrics) and t6_pmi.mean_lp (same teacher forcing) with judge FINAL_T4d.
On PASS: package wsp_runs/FINAL_T6, repoint wsp/inference.py, write LOCAL-ONLY blind A/B sheet T6 vs T5a (key sealed).
  python -I wsp_runs/t6/t6_compare.py [--smoke]   (--smoke: 20 rows, T5a only, writes t6_compare_SMOKE.json)"""
import csv, json, os, random, shutil, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp_runs", "t5")); sys.path.insert(0, HERE)
import t5_compare as T5, t5a_retest as RT, t6_pmi as PM
P1 = T5.P1; S1 = T5.S1
MODELS = {"T5a": "wsp_runs/FINAL_T5a/sentriformer_t5a.pt", "T6": "wsp_runs/train/T6/model_best.pt", "T6_ctrl": "wsp_runs/train/T6_ctrl/model_best.pt"}
JUDGE = "wsp_runs/FINAL_T4d/sentriformer_t4d.pt"
PINS = {JUDGE: "bff705ec24b9916880c9df4b16eda284f961814f05828c09e61208b384af1efe", MODELS["T5a"]: "4205c569b1d175ed3378a4867c5406954ffe5f408edf696b7d0aad9016ea38f5"}


def check_pins():   # originals must never change (P1.check_t4d hashes inference.DEFAULT_CKPT, which is now T5a)
    for p, h in PINS.items(): assert P1.sha(os.path.join(ROOT, p)) == h, f"{p} changed!"


def rows_rs131():
    """Same exclusions as the replication (rs 11/29/Phase1/53/67/79/97) plus the replication's own rs=113 rows."""
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(S1, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    a1, b1 = P1.eval_rows(); a2, b2 = T5.P2.eval_rows2(); a97, b97 = RT.C.rows_rs97()
    used = gate | tune | p1 | set(a1.index) | set(b1.index) | set(a2.index) | set(b2.index) | set(T5.rows_rs79().index) \
        | set(a97.index) | set(b97.index) | set(RT.rows_rs113().index)
    rows = hist.drop(index=list(used)).sample(500, random_state=131); assert not used & set(rows.index)
    return df, rows


def partners(df, rows):
    """Fixed per row: context of a different multi_val conversation of the same company (seed 131)."""
    comp = pd.Series(PM.companies(df), index=df.index); rng = np.random.default_rng(131); out = {}
    for i in rows.index:
        pool = df.index[(comp == comp[i]).to_numpy() & (df.group_id != df.group_id[i]).to_numpy()]
        out[i] = df.context[rng.choice(pool)] if len(pool) else None
    return out


def judge_pmi(replies, rows, part):
    pipe = P1.SentriFormerPipeline(os.path.join(ROOT, JUDGE), "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    enc = lambda c: tok.encode(c, add_special_tokens=True)[-256:] or [1]
    res = {}
    for k, reps in replies.items():
        rid = []
        for r in reps:
            x = tok.encode(r, truncation=True, max_length=127, add_special_tokens=True)
            rid.append(x if x and x[-1] == cfg.eos_token_id else x + [cfg.eos_token_id])
        own = PM.mean_lp(m, cfg, [{"input_ids": enc(c), "labels": r} for c, r in zip(rows.context, rid)])
        oth = PM.mean_lp(m, cfg, [{"input_ids": enc(part[i] or "Customer:"), "labels": r} for i, r in zip(rows.index, rid)])
        res[k] = np.where([part[i] is not None for i in rows.index], own - oth, 0.0)
    del m, pipe; torch.cuda.empty_cache(); return res


def main():
    smoke = "--smoke" in sys.argv; out = os.path.join(HERE, "t6_compare_SMOKE.json" if smoke else "t6_compare.json")
    if not smoke: assert not os.path.exists(out), "decision already made"
    check_pins(); df, rows = rows_rs131(); rows = rows.iloc[:20] if smoke else rows; part = partners(df, rows)
    R = {"rows": f"{len(rows)} fresh multi_val rows with history, rs=131", "filters_human_validated": False, "models": {},
         "rows_without_partner": sum(v is None for v in part.values())}
    A, REP = {}, {}
    for k in (["T5a"] if smoke else MODELS):
        R["models"][k], A[k], REP[k] = T5.evaluate(k, os.path.join(ROOT, MODELS[k]), rows)
    J = judge_pmi(REP, rows, part)
    for k in J: R["models"][k]["judge_pmi"] = float(J[k].mean())
    if smoke:
        P1.dump_atomic(R, out); print("smoke ok", R); return
    t, b = R["models"]["T6"], R["models"]["T5a"]
    R["primary_judge_pmi_T6_minus_ctrl_CI95"] = P1.ci(J["T6"] - J["T6_ctrl"])
    R["judge_pmi_T6_minus_T5a_CI95"] = P1.ci(J["T6"] - J["T5a"])
    R["secondary_CI95"] = {f"{k}_T6_minus_{o}": P1.ci(A["T6"][k] - A[o][k]) for k in ("specific", "generic_phrase", "generic_whole", "broken_link") for o in ("T6_ctrl", "T5a")}
    F = lambda nm: json.load(open(os.path.join(ROOT, "wsp_runs", "eval_full", f"{nm}_val", "metrics.json")))
    f0, f6 = F("T5a"), F("T6")
    g = {"P1_judge_pmi_above_ctrl": R["primary_judge_pmi_T6_minus_ctrl_CI95"][1] > 0,
         "G1_cat_acc": f6["category_len512"]["accuracy"] >= f0["category_len512"]["accuracy"] - .01,
         "G1_cat_macroF1": f6["category_len512"]["macro_f1_sklearn"] >= f0["category_len512"]["macro_f1_sklearn"] - .01,
         "G1_csat_macroF1": f6["csat"]["macro_f1_sklearn"] >= f0["csat"]["macro_f1_sklearn"] - .01,
         "G1_sent_macroF1": f6["sentiment"]["all"]["macro_f1_sklearn"] >= f0["sentiment"]["all"]["macro_f1_sklearn"] - .01,
         "G2_portal": t["portal"] <= b["portal"] + .01, "G2_eos": t["eos"] >= b["eos"] - .01, "G2_repetition": t["repetition"] <= b["repetition"] + .01,
         "G2_action": t["action"] >= b["action"] - .02, "G2_specific": t["specific"] >= b["specific"] - .02, "G2_distinct": t["distinct"] >= b["distinct"] - .02,
         "G2_rouge": t["rougeL_clean"] >= b["rougeL_clean"] - .005, "G2_length": abs(t["mean_length"] / b["mean_length"] - 1) <= .2,
         "G2_empty": t["empty"] <= b["empty"] + .01, "G2_template": t["template_concentration"] <= b["template_concentration"] + .02,
         "G2_generic": t["generic_phrase"] <= b["generic_phrase"] + .02, "G2_broken": t["broken_link"] <= b["broken_link"] + .01,
         "G3_dial_loss": t["dial_loss"] <= b["dial_loss"] + .05}
    R["gates"], ok = g, all(g.values()); R["decision"] = "PASS -> T6 deployed as default" if ok else "REJECT -> T5a stays default"
    if ok:
        fd = os.path.join(ROOT, "wsp_runs", "FINAL_T6"); os.makedirs(fd, exist_ok=True); dst = os.path.join(fd, "sentriformer_t6.pt")
        assert not os.path.exists(dst); shutil.copy2(os.path.join(ROOT, MODELS["T6"]), dst); R["final_checkpoint_sha256"] = P1.sha(dst)
        inf = os.path.join(ROOT, "wsp", "inference.py"); s = open(inf, encoding="utf-8").read()
        old = 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")  # T5a (rollback: FINAL_T4d)'; assert s.count(old) == 1
        open(inf, "w", encoding="utf-8").write(s.replace(old, 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T6", "sentriformer_t6.pt")  # T6 (rollback: FINAL_T5a)', 1))
        rng = random.Random(29); key = {}
        with open(os.path.join(HERE, "AB_T6_vs_T5a_LOCAL_ONLY.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(["LOCAL ONLY - contains TWCS text - do not upload"] + [""] * 5)
            w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
            for k, (i, x, y) in enumerate(list(zip(rows.index, REP["T5a"], REP["T6"]))[:50]):
                flip = rng.random() < .5; key[f"T6AB{k + 1:03d}"] = {"A": "T6" if flip else "T5a", "B": "T5a" if flip else "T6", "val_row": int(i)}
                w.writerow([f"T6AB{k + 1:03d}", rows.loc[i].context, y if flip else x, x if flip else y, "", ""])
        P1.dump_atomic(key, os.path.join(S1, "_sealed_keys", "AB_T6_KEY.json"))
    with open(os.path.join(HERE, "t6_compare_replies_LOCAL_ONLY.json"), "w", encoding="utf-8") as fh:
        json.dump({"row_index": [int(i) for i in rows.index], "replies": REP}, fh, ensure_ascii=False)
    P1.dump_atomic(R, out); check_pins(); print(R["decision"], [k for k, v in g.items() if not v])


if __name__ == "__main__":
    main()
