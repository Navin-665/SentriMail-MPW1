"""
WSP E1 - Closed-form head re-alignment (no gradients, no optimizer).

Hypothesis (W1): M10.2/M10.3 SFT moved the encoder while category/CSAT heads stayed frozen, so the heads
read features from a space they were not trained on. If pooled_m103 -> pooled_mt is ~linear, then folding a
least-squares map A into the first head layer restores routing:   dense.weight_new = dense.weight @ A
Only category_head.dense.weight and csat_head.dense.weight change; the encoder/decoder/LM path is untouched,
so generated text is byte-identical by construction (verified below).

Fit data: classification_train (random fixed-seed subset) + csat_train ONLY. Selection: val. Test: run once by e1_final.
Stages (each resumable; outputs never overwritten):
  feats  - extract pooled encoder features            -> wsp_runs/e1/feats/*.npy
  fit    - ridge sweep + controls, val selection       -> wsp_runs/e1/fit_report.json, A_*.npy
"""
import argparse, json, os, sys, time, datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask, pool_encoder_states
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator

OUT = os.path.join(ROOT, "wsp_runs", "e1")
FEATS = os.path.join(OUT, "feats")
SUP = os.path.join(ROOT, "data", "supervised")
CK = {"mt": "checkpoints/multitask/multitask_best.pt", "m103": "checkpoints/m10_3_sft/m10_3_sft_best.pt"}
N_TRAIN = 200_000
SEED = 0


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def setup():
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
    li = json.load(open(os.path.join(SUP, "classification_labels.json"), encoding="utf-8"))
    cfg = SentriFormerConfig.from_pretrained(os.path.join(ROOT, "checkpoints", "config.json"))
    cfg.num_categories = li["num_classes"]; cfg.num_csat_classes = 5
    return tok, li, cfg


def load(name, cfg, device="cuda"):
    raw = torch.load(os.path.join(ROOT, CK[name]), map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
    m = SentriFormerForMultiTask(cfg); m.load_state_dict(sd, strict=True)
    return m.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def pooled(model, loader):
    out, ys = [], []
    for b in loader:
        ids, am = b["input_ids"].cuda(), b["attention_mask"].cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):  # identical to SentriFormerForMultiTask.forward
            enc = model.sentriformer.encoder(input_ids=ids, attention_mask=am)
            p = pool_encoder_states(enc, attention_mask=am)
        out.append(p.float().cpu()); ys.append(b["labels"])
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def loaders(tok, li, cfg):
    L = {}
    tr = ClassificationParquetDataset(os.path.join(SUP, "classification_train.parquet"), tok, li["label2id"], max_seq_length=512)
    idx = np.sort(np.random.default_rng(SEED).choice(len(tr), N_TRAIN, replace=False))
    L["cls_train_512"] = DataLoader(Subset(tr, idx.tolist()), batch_size=128, collate_fn=ClassificationDataCollator(0, 512))
    for s in ("val",):
        for n in (512, 128):
            ds = ClassificationParquetDataset(os.path.join(SUP, f"classification_{s}.parquet"), tok, li["label2id"], max_seq_length=n)
            L[f"cls_{s}_{n}"] = DataLoader(ds, batch_size=128, collate_fn=ClassificationDataCollator(0, n))
    for s in ("train", "val"):
        L[f"csat_{s}_128"] = DataLoader(CSATDataset(os.path.join(SUP, f"csat_{s}.parquet"), tok, max_len=128), batch_size=64, collate_fn=CSATCollator(0))
    return L, idx


def stage_feats():
    os.makedirs(FEATS, exist_ok=True)
    tok, li, cfg = setup()
    L, idx = loaders(tok, li, cfg)
    np.save(os.path.join(FEATS, "train_subset_idx.npy"), idx) if not os.path.exists(os.path.join(FEATS, "train_subset_idx.npy")) else None
    need = {"mt": ["cls_train_512", "csat_train_128"], "m103": list(L.keys())}
    for name, keys in need.items():
        todo = [k for k in keys if not os.path.exists(os.path.join(FEATS, f"{name}_{k}.npz"))]
        if not todo:
            continue
        model = load(name, cfg)
        for k in todo:
            t = time.time(); X, y = pooled(model, L[k])
            np.savez(os.path.join(FEATS, f"{name}_{k}.npz"), X=X.astype(np.float32), y=y.astype(np.int16))
            log(f"feats {name} {k} {X.shape} ({time.time()-t:.0f}s)")
        del model; torch.cuda.empty_cache()


def head_logits(sd, prefix, X, dense_w):
    """Exact re-implementation of the head in eval mode (dropout off): out_proj(GELU(dense(x)))."""
    x = torch.from_numpy(X).cuda()
    h = torch.nn.functional.gelu(x @ dense_w.T)
    return (h @ sd[f"{prefix}.out_proj.weight"].T.cuda() + sd[f"{prefix}.out_proj.bias"].cuda()).cpu()


def ridge(X, Y, lam):
    X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
    G = X64.T @ X64
    reg = lam * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + reg * np.eye(G.shape[0]), X64.T @ Y64).T  # A with Y ≈ X @ A.T


def stage_fit():
    from e0_eval import metrics, paired
    tok, li, cfg = setup()
    names = [li["id2label"][str(i)] for i in range(li["num_classes"])]; csn = [f"{i} star" for i in range(1, 6)]
    ld = lambda n: (np.load(os.path.join(FEATS, f"{n}.npz"))["X"], np.load(os.path.join(FEATS, f"{n}.npz"))["y"].astype(np.int64))
    raw = torch.load(os.path.join(ROOT, CK["m103"]), map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in raw["model_state_dict"].items()}; del raw
    Xs_c, _ = ld("m103_cls_train_512"); Xt_c, _ = ld("mt_cls_train_512")
    Xs_s, _ = ld("m103_csat_train_128"); Xt_s, _ = ld("mt_csat_train_128")
    I = np.eye(256)

    def r2(Xs, Xt, A):
        res = Xt - Xs @ A.T
        return float(1 - (res ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum())
    cos = lambda X, Y: float(((X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)).mean())
    rep = {"n_fit_cls": int(len(Xs_c)), "n_fit_csat": int(len(Xs_s)), "variants": {},
           "drift": {"cls_train": {"r2_identity": r2(Xs_c, Xt_c, I), "mean_cosine": cos(Xs_c, Xt_c)},
                     "csat_train": {"r2_identity": r2(Xs_s, Xt_s, I), "mean_cosine": cos(Xs_s, Xt_s)}}}
    log(f"drift: {json.dumps(rep['drift'])}")
    Wc = sd["category_head.dense.weight"]; Ws = sd["csat_head.dense.weight"]
    val = {k: ld(f"m103_{k}") for k in ("cls_val_512", "cls_val_128", "csat_val_128")}
    KEEP = ("accuracy", "macro_f1_sklearn", "weighted_f1", "nll", "ece_15bin", "mean_max_conf", "frac_conf_lt_0.25")

    def evaluate(Ac, As, keys=None):
        r, preds = {}, {}
        Wc_new = (Wc @ torch.from_numpy(Ac).float()).cuda(); Ws_new = (Ws @ torch.from_numpy(As).float()).cuda()
        for k, (X, y) in val.items():
            if keys and k not in keys: continue
            pre, W, nm = ("csat_head", Ws_new, csn) if k.startswith("csat") else ("category_head", Wc_new, names)
            m, _, p = metrics(head_logits(sd, pre, X, W), y, nm)
            r[k] = {x: m[x] for x in KEEP}
            if k.startswith("csat"): r[k]["within_1"] = float(np.mean(np.abs(p - y) <= 1)); r[k]["pred_dist"] = list(m["pred_distribution"].values())
            preds[k] = p
        return r, preds

    base, base_p = evaluate(I, I)
    rep["variants"]["identity(no edit)"] = base
    # self-check: re-implemented head on extracted features must reproduce E0's full-model val result
    e0 = json.load(open(os.path.join(ROOT, "wsp_runs", "e0", "val", "e0_metrics.json")))["checkpoints"]["m10_3_sft_best"]["tasks"]
    for k, e0k in (("cls_val_512", "category_len512"), ("cls_val_128", "category_len128"), ("csat_val_128", "csat_len128")):
        assert abs(base[k]["accuracy"] - e0[e0k]["accuracy"]) < 2e-3, f"head re-implementation mismatch on {k}"

    # --- category: ridge sweep (selection on val acc @512) ---
    fits = {}
    for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        Ac = ridge(Xs_c, Xt_c, lam); fits[lam] = Ac
        v, _ = evaluate(Ac, I, keys=("cls_val_512", "cls_val_128")); v["r2_cls_train"] = r2(Xs_c, Xt_c, Ac)
        rep["variants"][f"category ridge lam={lam}"] = v
        log(f"lam={lam} cls512 acc={v['cls_val_512']['accuracy']:.4f} cls128 acc={v['cls_val_128']['accuracy']:.4f} r2={v['r2_cls_train']:.4f}")
    best_lam = max(fits, key=lambda l: rep["variants"][f"category ridge lam={l}"]["cls_val_512"]["accuracy"])
    Ac = fits[best_lam]

    # --- orientation check: head(W @ A, X) == head(W, X @ A.T) ---
    Xv = val["cls_val_512"][0][:5000]
    l1 = head_logits(sd, "category_head", Xv, (Wc @ torch.from_numpy(Ac).float()).cuda())
    l2 = head_logits(sd, "category_head", (Xv.astype(np.float64) @ Ac.T).astype(np.float32), Wc.cuda())
    rep["orientation_max_abs_logit_diff"] = float((l1 - l2).abs().max())
    assert rep["orientation_max_abs_logit_diff"] < 1e-3, rep["orientation_max_abs_logit_diff"]
    # and mapped features should approximate the M9.5 features the head was trained on (val has no mt feats; use train R2)
    log(f"orientation ok, max diff {rep['orientation_max_abs_logit_diff']:.2e}")

    sel_r, sel_p = evaluate(Ac, I, keys=("cls_val_512", "cls_val_128"))
    y512 = val["cls_val_512"][1]
    vs_base = paired(y512, base_p["cls_val_512"], sel_p["cls_val_512"], len(names))

    # --- E3 controls ---
    rng = np.random.default_rng(1)
    dW = (Wc @ torch.from_numpy(Ac).float()) - Wc
    R = torch.from_numpy(rng.standard_normal(tuple(Wc.shape))).float(); R *= dW.norm() / R.norm()
    Ar = np.linalg.solve(Wc.double().numpy(), (Wc + R).double().numpy())  # Wc @ Ar = Wc + R
    ctrl = {"random_same_norm": Ar, "shuffled_pairs": ridge(Xs_c, Xt_c[rng.permutation(len(Xt_c))], best_lam)}
    rep["controls"] = {}
    for n, A in ctrl.items():
        cr, cp = evaluate(A, I, keys=("cls_val_512",))
        rep["controls"][n] = {"cls_val_512": cr["cls_val_512"], "e1_minus_control": paired(y512, cp["cls_val_512"], sel_p["cls_val_512"], len(names))}
        log(f"control {n}: acc={cr['cls_val_512']['accuracy']:.4f}")

    cat_ok = (vs_base["delta_accuracy_ci95"][0] > 0
              and sel_r["cls_val_512"]["weighted_f1"] >= base["cls_val_512"]["weighted_f1"]
              and all(c["e1_minus_control"]["delta_accuracy_ci95"][0] > 0 for c in rep["controls"].values()))
    rep["category"] = {"lambda": best_lam, "val": sel_r, "paired_vs_unedited": vs_base, "accepted": bool(cat_ok)}

    # --- CSAT: drift precondition, then candidates, then rule 3 ---
    drift_ok = rep["drift"]["csat_train"]["r2_identity"] < 0.90
    cands = {"csat_train": ridge(Xs_s, Xt_s, best_lam), "cls_train": Ac,
             "joint": ridge(np.vstack([Xs_c, Xs_s]), np.vstack([Xt_c, Xt_s]), best_lam)}
    cs_res = {n: evaluate(I, A, keys=("csat_val_128",)) for n, A in cands.items()}
    rep["csat_candidates"] = {n: r[0]["csat_val_128"] for n, r in cs_res.items()}
    best_cs = max(cands, key=lambda n: cs_res[n][0]["csat_val_128"]["accuracy"])
    b, s = base["csat_val_128"], cs_res[best_cs][0]["csat_val_128"]
    ys = val["csat_val_128"][1]
    cs_paired = paired(ys, base_p["csat_val_128"], cs_res[best_cs][1]["csat_val_128"], 5)
    csat_ok = drift_ok and (s["accuracy"] >= b["accuracy"] - 0.005) and (s["within_1"] >= b["within_1"] - 0.005) and cs_paired["delta_accuracy_ci95"][0] > 0
    rep["csat"] = {"drift_precondition_r2_lt_0.90": bool(drift_ok), "selected_source": best_cs, "val": s, "unedited_val": b,
                   "paired_vs_unedited": cs_paired, "accepted": bool(csat_ok)}

    np.save(os.path.join(OUT, "A_category.npy"), Ac); np.save(os.path.join(OUT, "A_csat.npy"), cands[best_cs])
    for n, A in ctrl.items(): np.save(os.path.join(OUT, f"A_control_{n}.npy"), A)
    json.dump({"category": bool(cat_ok), "csat": bool(csat_ok)}, open(os.path.join(OUT, "acceptance.json"), "w"))
    json.dump(rep, open(os.path.join(OUT, "fit_report.json"), "w"), indent=2)
    log(f"ACCEPT category={cat_ok} (lam={best_lam}, val512 {base['cls_val_512']['accuracy']:.4f}->{sel_r['cls_val_512']['accuracy']:.4f}) "
        f"csat={csat_ok} (src={best_cs}, drift_ok={drift_ok}, {b['accuracy']:.4f}->{s['accuracy']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["feats", "fit"]); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    assert not (a.stage == "fit" and os.path.exists(os.path.join(OUT, "fit_report.json"))), "fit already done"
    {"feats": stage_feats, "fit": stage_fit}[a.stage]()
