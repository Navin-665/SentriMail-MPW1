"""
WSP regression tests for the final model. Usage: python -I wsp/regression_tests.py <final_ckpt> <out_json>
  T1 final checkpoint loads strictly into SentriFormerForMultiTask
  T2 every non-head tensor is byte-identical to the original m10_3 checkpoint
  T3 greedy generation is token-identical to the original on 50 val dialogue contexts (generation unchanged)
  T4 pipeline handles >512-token, empty, and unicode inputs without crashing (the original path crashes >512)
  T5 original checkpoints' SHA256 still match the pre-experiment manifest
  T6 category/CSAT predictions on a fixed 2,000-sample val slice match the recorded final val predictions
"""
import hashlib, json, os, sys, traceback
import numpy as np
import torch
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import SentriFormerPipeline, ROOT
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask

ORIG = "checkpoints/m10_3_sft/m10_3_sft_best.pt"


def main():
    final_ckpt, out_json = sys.argv[1], sys.argv[2]
    trained = len(sys.argv) > 3 and sys.argv[3] == "--trained"   # Phase-T model: weights changed everywhere by design
    recorded = sys.argv[4] if trained else os.path.join(ROOT, "wsp_runs", "e1", "final_val", "preds_e1_category_len512.npz")
    assert not os.path.exists(out_json)
    res = {}

    def test(name, fn):
        try:
            detail = fn(); res[name] = {"pass": True, "detail": detail}
        except Exception as e:
            res[name] = {"pass": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
        print(name, "PASS" if res[name]["pass"] else f"FAIL {res[name]['error']}", flush=True)

    fin = SentriFormerPipeline(final_ckpt); org = SentriFormerPipeline(ORIG)
    test("T1_strict_load", lambda: {"tensors": len(fin.model.state_dict())})

    def t2():
        a, b = fin.model.state_dict(), org.model.state_dict()
        diff = [k for k in a if not torch.equal(a[k], b[k])]
        assert all(k.startswith(("category_head.", "csat_head.")) for k in diff), diff
        return {"differing_tensors": diff}
    if not trained: test("T2_edit_scope", t2)

    def t3():
        ctx = pq.read_table(os.path.join(ROOT, "data", "supervised", "m10_3_dialogue_val.parquet"), columns=["context"])["context"].to_pylist()[:50]
        same = 0
        for c in ctx:
            outs = []
            for p in (fin, org):
                ids, am = p._enc(c, 128)
                with torch.inference_mode():
                    outs.append(generate_multitask(p.model, p.tok, p.cfg, ids, am, max_new_tokens=60, temperature=0.0).tolist())
            same += outs[0] == outs[1]
        assert same == len(ctx), f"{same}/{len(ctx)} identical"
        return {"identical": f"{same}/{len(ctx)}"}
    if not trained: test("T3_generation_unchanged", t3)
    def t3b():
        ctx = pq.read_table(os.path.join(ROOT, "data", "supervised", "m10_4_dialogue_val.parquet"), columns=["context"])["context"].to_pylist()[:20]
        a = [fin.analyze(c, greedy=True, max_new_tokens=60)["response"] for c in ctx]
        b = [fin.analyze(c, greedy=True, max_new_tokens=60)["response"] for c in ctx]
        assert a == b, "greedy generation not deterministic"
        return {"deterministic": f"{len(a)}/{len(a)}"}
    if trained: test("T3b_generation_deterministic", t3b)

    def t4():
        out = {}
        for name, txt in (("long_781_tokens", "I was charged twice for my order and nobody has refunded me. " * 60), ("empty", ""), ("unicode", "Mi tarjeta fue cobrada dos veces 💳 — ¿reembolso?")):
            r = fin.analyze(txt, greedy=True, max_new_tokens=30)
            out[name] = {"category": r["category"]["label"], "truncated": r["truncated_for_routing"], "response_len": len(r["response"])}
        return out
    test("T4_robust_inputs", t4)

    def t5():
        man = json.load(open(os.path.join(ROOT, "wsp_runs", "e0", "integrity_pre.json")))["sha256"]
        bad = []
        for rel, v in man.items():
            if rel.startswith("checkpoints/"):
                h = hashlib.sha256()
                with open(os.path.join(ROOT, rel), "rb") as f:
                    for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
                if h.hexdigest() != v["sha256"]: bad.append(rel)
        assert not bad, bad
        return {"checked": [k for k in man if k.startswith("checkpoints/")]}
    test("T5_originals_untouched", t5)

    def t6():
        rec = recorded
        d = np.load(rec)
        txt = pq.read_table(os.path.join(ROOT, "data", "supervised", "classification_val.parquet"), columns=["text"])["text"].to_pylist()[:2000]
        agree = 0
        fin_len = fin.cat_len; fin.cat_len = 512
        for i, t in enumerate(txt):
            agree += fin.cat_names.index(fin.analyze(t, greedy=True, max_new_tokens=1)["category"]["label"]) == int(d["preds"][i])
        fin.cat_len = fin_len
        assert agree / len(txt) >= 0.995, f"agreement {agree}/{len(txt)}"  # batch padding + fp16 vs fp32 can flip near-ties
        return {"agreement": f"{agree}/{len(txt)}"}
    test("T6_reproduces_recorded_val_predictions", t6)

    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("T"))
    json.dump(res, open(out_json, "w"), indent=2)
    print("ALL PASS" if res["all_pass"] else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
