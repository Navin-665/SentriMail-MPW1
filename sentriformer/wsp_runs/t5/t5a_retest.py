"""T5a replication test per frozen T5A_RETEST_CRITERIA.md. Reuses t5_compare.evaluate and its gate definitions unchanged."""
import csv, json, os, random, shutil, sys
import numpy as np, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t5_compare as T5, t5c_mmi as C
P1 = T5.P1; ROOT = T5.ROOT


def rows_rs113():
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas(); hist = df[df.n_history_segments > 0]
    gate = set(hist.sample(300, random_state=11).index); tune = set(hist.drop(index=list(gate)).sample(300, random_state=29).index)
    p1 = set(json.load(open(os.path.join(T5.S1, "phase1_results.json"), encoding="utf-8"))["rows"]["row_index"])
    a1, b1 = P1.eval_rows(); a2, b2 = T5.P2.eval_rows2(); r79 = T5.rows_rs79(); a97, b97 = C.rows_rs97()
    used = gate | tune | p1 | set(a1.index) | set(b1.index) | set(a2.index) | set(b2.index) | set(r79.index) | set(a97.index) | set(b97.index)
    rest = hist.drop(index=list(used)); n = min(500, len(rest))
    rows = rest.sample(n, random_state=113); assert not used & set(rows.index)
    return rows


def main():
    out = os.path.join(HERE, "t5a_retest.json"); assert not os.path.exists(out), "the replication runs once"
    P1.check_t4d(); rows = rows_rs113(); R = {"rows": len(rows), "random_state": 113, "models": {}}
    A, REP = {}, {}
    for k in ("T4d", "T5a", "T5a_ctrl"):
        R["models"][k], A[k], REP[k] = T5.evaluate(k, os.path.join(ROOT, T5.MODELS[k]), rows)
    b, t = R["models"]["T4d"], R["models"]["T5a"]
    R["P1_broken_link_T5a_minus_ctrl_CI95"] = P1.ci(A["T5a"]["broken_link"] - A["T5a_ctrl"]["broken_link"])
    R["generic_phrase_T5a_minus_T4d_CI95"] = P1.ci(A["T5a"]["generic_phrase"] - A["T4d"]["generic_phrase"])
    R["action_T5a_minus_T4d_CI95"] = P1.ci(A["T5a"]["action"] - A["T4d"]["action"])
    orig = json.load(open(os.path.join(HERE, "t5_compare.json")))
    g = {"P1_replication": R["P1_broken_link_T5a_minus_ctrl_CI95"][2] < 0, "P1_original": orig["gates"]["P1_broken_link_lower_than_ctrl"],
         "G2_portal": t["portal"] <= b["portal"] + .01, "G2_eos": t["eos"] >= b["eos"] - .01, "G2_repetition": t["repetition"] <= b["repetition"] + .01,
         "G2_action": t["action"] >= b["action"] - .02, "G2_specific": t["specific"] >= b["specific"] - .02, "G2_distinct": t["distinct"] >= b["distinct"] - .02,
         "G2_rouge": t["rougeL_clean"] >= b["rougeL_clean"] - .005, "G2_length": abs(t["mean_length"] / b["mean_length"] - 1) <= .2,
         "G2_empty": t["empty"] <= b["empty"] + .01, "G2_template": t["template_concentration"] <= b["template_concentration"] + .02,
         "G3_dial_loss": t["dial_loss"] <= b["dial_loss"] + .05, "G1_eval_full_val": all(v for k, v in orig["gates"].items() if k.startswith("G1"))}
    R["gates"], ok = g, all(g.values()); R["decision"] = "PASS -> T5a deployed as default" if ok else "FAIL -> T5a stays rejected; T4d default"
    if ok:
        fd = os.path.join(ROOT, "wsp_runs", "FINAL_T5a"); os.makedirs(fd, exist_ok=True); dst = os.path.join(fd, "sentriformer_t5a.pt")
        assert not os.path.exists(dst); shutil.copy2(os.path.join(ROOT, T5.MODELS["T5a"]), dst); R["final_checkpoint_sha256"] = P1.sha(dst)
        inf = os.path.join(ROOT, "wsp", "inference.py"); s = open(inf, encoding="utf-8").read()
        old = 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T4d", "sentriformer_t4d.pt")'; assert s.count(old) == 1
        open(inf, "w", encoding="utf-8").write(s.replace(old, 'DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")  # T5a (rollback: FINAL_T4d)', 1))
        rng = random.Random(23); key = {}; os.makedirs(os.path.join(T5.S1, "_sealed_keys"), exist_ok=True)
        with open(os.path.join(HERE, "AB_T5a_vs_T4d_LOCAL_ONLY.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(["LOCAL ONLY - contains TWCS text - do not upload"] + [""] * 5)
            w.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
            for k, (i, x, y) in enumerate(list(zip(rows.index, REP["T4d"], REP["T5a"]))[:50]):
                flip = rng.random() < .5; key[f"T5AB{k + 1:03d}"] = {"A": "T5a" if flip else "T4d", "B": "T4d" if flip else "T5a", "val_row": int(i)}
                w.writerow([f"T5AB{k + 1:03d}", rows.loc[i].context, y if flip else x, x if flip else y, "", ""])
        P1.dump_atomic(key, os.path.join(T5.S1, "_sealed_keys", "AB_T5a_KEY.json"))
    P1.dump_atomic(R, out); P1.check_t4d(); print(R["decision"], [k for k, v in g.items() if not v])


if __name__ == "__main__":
    main()
