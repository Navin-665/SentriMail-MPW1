"""SentriWSP-01 Phase 1b (READ ONLY): corrected detectors + neuron controls on the same 400 VAL rows as phase1.
1) privacy = a REQUEST for personal data (verb + data item), not the word "phone" in "reach us via phone".
2) broken fragment = eval_t4.orphan token rule only ("here to help" is not broken).
3) For recurring L3 FFN neurons: activation at generic-phrase positions vs all other reply positions (control), and
   which output tokens each neuron's fc2 column promotes through the tied unembedding.
Output: phase1b_results.json (refuses to overwrite)."""
import json, os, re, sys, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp")); sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, HERE)
from inference import SentriFormerPipeline, DEFAULT_CKPT
from eval_t4 import orphan
import pyarrow.parquet as pq
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask
from phase1_investigate import GENERIC
OUT = os.path.join(HERE, "phase1b_results.json")
PRIV_REQ = re.compile(r"\b(send|share|provide|confirm|give|dm|include|tell)\b[^.?!]{0,40}\b(full name|name|zip( code)?|postal code|date of birth|"
                      r"phone number|phone #|address|e-?mail( address)?|account number|card number|ssn|social security|password|pin)\b", re.I)
NEURONS = [("L3", 502), ("L3", 619), ("L3", 509), ("L3", 743), ("L3", 348), ("L2", 219)]


def main():
    assert not os.path.exists(OUT); torch.set_grad_enabled(False)
    P1 = json.load(open(os.path.join(HERE, "phase1_results.json"), encoding="utf-8"))
    pipe = SentriFormerPipeline(DEFAULT_CKPT, "cuda"); m, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    df = pq.read_table(os.path.join(ROOT, "wsp_runs", "t4", "data", "multi_val.parquet")).to_pandas()
    rows = df.loc[P1["rows"]["row_index"]]
    dec = m.sentriformer.decoder; W_U = m.lm_head.weight; gamma = dec.final_norm.weight
    acts = {}
    hk = [dec.layers[l].ffn.fc1.register_forward_hook(lambda mod, i, o, l=l: acts.__setitem__(l, torch.nn.functional.gelu(o[0]))) for l in (2, 3)]
    rate = collections.Counter(); at_generic = collections.defaultdict(list); elsewhere = collections.defaultdict(list); n = 0
    ex_priv = []
    for _, r in rows.iterrows():
        ids = tok.encode(r.context, add_special_tokens=True)[-256:] or [1]; t = torch.tensor([ids], device="cuda")
        g = generate_multitask(m, tok, cfg, t, torch.ones_like(t), max_new_tokens=128, temperature=0.0, repetition_penalty=1.2)[0].tolist()
        body = g[1:g.index(cfg.eos_token_id)] if cfg.eos_token_id in g else g[1:]
        txt = tok.decode(body, skip_special_tokens=True).strip(); n += 1
        pr = bool(PRIV_REQ.search(txt)) and not PRIV_REQ.search(r.context)
        rate["privacy_request"] += pr; rate["broken_fragment_orphan_rule"] += bool(orphan(txt)); rate["generic"] += bool(GENERIC.search(txt))
        if pr and len(ex_priv) < 6: ex_priv.append({"current_turn": re.split(r"(?=Customer: )", r.context)[-1][:200], "generated": txt})
        # neuron activations over the model's own reply (teacher forced)
        e = torch.tensor([ids], device="cuda"); am = torch.ones_like(e); enc = m.sentriformer.encoder(input_ids=e, attention_mask=am)
        d = torch.tensor([[cfg.bos_token_id] + body], device="cuda"); acts.clear()
        dec(input_ids=d, encoder_hidden_states=enc, encoder_attention_mask=am)
        mt = GENERIC.search(txt); gpos = set()
        if mt:
            for k in range(1, len(body) + 1):
                L = len(tok.decode(body[:k], skip_special_tokens=True).strip())
                if mt.start() < L <= mt.end() + 1: gpos.add(k - 1)
        for (ln, j) in NEURONS:
            a = acts[int(ln[1])][:, j].float().cpu().numpy()[: len(body)]
            for p in range(len(body)): (at_generic if p in gpos else elsewhere)[f"{ln}.n{j}"].append(float(a[p]))
    for h in hk: h.remove()
    R = {"n": n, "rates_corrected": {k: round(v / n, 4) for k, v in rate.items()}, "privacy_request_examples": ex_priv, "neurons": {}}
    for (ln, j) in NEURONS:
        key = f"{ln}.n{j}"; a, b = np.array(at_generic[key]), np.array(elsewhere[key])
        col = dec.layers[int(ln[1])].ffn.fc2.weight[:, j]                    # what this neuron writes into the residual
        logits = W_U @ (col * gamma); top = torch.topk(logits, 12).indices.tolist(); bot = torch.topk(-logits, 8).indices.tolist()
        pooled = np.concatenate([a, b]); rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):  # permutation test on mean difference
            rng.shuffle(pooled); diffs.append(pooled[: len(a)].mean() - pooled[len(a):].mean())
        obs = a.mean() - b.mean() if len(a) else float("nan")
        R["neurons"][key] = {"mean_act_at_generic_tokens": round(float(a.mean()), 4) if len(a) else None, "n_generic_positions": len(a),
                             "mean_act_elsewhere": round(float(b.mean()), 4), "n_other_positions": len(b),
                             "perm_p_one_sided": round(float(np.mean(np.array(diffs) >= obs)), 4) if len(a) else None,
                             "promotes_tokens": [tok.decode([i]) for i in top], "suppresses_tokens": [tok.decode([i]) for i in bot],
                             "fc2_col_norm": round(float(col.norm()), 4),
                             "layer_fc2_col_norm_median": round(float(dec.layers[int(ln[1])].ffn.fc2.weight.norm(dim=0).median()), 4)}
    json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False); print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
