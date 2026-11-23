"""
WSP E0 - Baseline evaluation (inference only). Reuses the project's datasets, collators,
model class and fp16 autocast exactly as training/train_classification.py and train_csat.py do;
adds probability-based metrics (NLL, ECE, Brier, confidence). Writes only to wsp_runs/e0/<split>/.

Usage: python -I wsp/e0_eval.py --split val|test
"""
import argparse, hashlib, json, os, sys, time, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../sentriformer
sys.path.insert(0, os.path.dirname(ROOT))

from transformers import AutoTokenizer
from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator

CKPTS = {
    "classifier_best": "checkpoints/classification/classifier_best.pt",
    "multitask_best": "checkpoints/multitask/multitask_best.pt",
    "m10_3_sft_best": "checkpoints/m10_3_sft/m10_3_sft_best.pt",
}
CSAT_SUPPORTED = {"multitask_best", "m10_3_sft_best"}  # classifier_best's CSAT head was never trained (M9.2 = category loss only)
CLS_LENGTHS = (512, 128)   # 512 = M9.2 protocol, 128 = M9.5 multitask protocol
CSAT_LEN = 128             # both CSAT pipelines used 128
RETURNS_ID = 28


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def state_fingerprint(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def run(model, loader, device, key):
    logits_all, tgts = [], []
    for b in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
        logits_all.append(out[key].float().cpu()); tgts.append(b["labels"])
    return torch.cat(logits_all), torch.cat(tgts).numpy()


def confusion(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def prf_from_cm(cm):
    tp = np.diag(cm).astype(float); pred = cm.sum(0); sup = cm.sum(1)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, sup


def macro_f1_over_present(y, p, k):
    # sklearn f1_score(average="macro") averages over labels present in y OR p; replicate that.
    cm = confusion(y, p, k); _, _, f1, sup = prf_from_cm(cm)
    present = (sup > 0) | (cm.sum(0) > 0)
    return f1[present].mean()


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            table.append({"lo": round(lo, 4), "hi": round(hi, 4), "n": int(m.sum()), "acc": float(correct[m].mean()), "conf": float(conf[m].mean())})
    return float(e), table


def metrics(logits, y, names):
    from sklearn.metrics import f1_score, accuracy_score
    k = len(names)
    probs = F.softmax(logits, -1).numpy(); p = probs.argmax(1)
    conf = probs.max(1); correct = (p == y).astype(float)
    cm = confusion(y, p, k); prec, rec, f1, sup = prf_from_cm(cm)
    e, rel = ece(conf, correct)
    onehot = np.eye(k)[y]
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1_sklearn": float(f1_score(y, p, average="macro", zero_division=0)),
        "macro_f1_all_classes": float(f1.mean()),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "nll": float(F.cross_entropy(logits, torch.from_numpy(y)).item()),
        "ece_15bin": e,
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "mean_max_conf": float(conf.mean()),
        "frac_conf_lt_0.25": float((conf < 0.25).mean()),
        "mean_entropy_nats": float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1).mean()),
        "pred_distribution": {names[i]: int(c) for i, c in enumerate(np.bincount(p, minlength=k))},
        "per_class": {names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])} for i in range(k)},
        "confusion_matrix": cm.tolist(),
        "reliability": rel,
    }, probs, p


def csat_extra(p, y):
    ps, ts = p + 1, y + 1
    return {"within_1": float(np.mean(np.abs(ps - ts) <= 1)), "mae": float(np.mean(np.abs(ps - ts))),
            "rmse": float(np.sqrt(np.mean((ps - ts) ** 2)))}


def paired(y, pa, pb, k, n_boot=1000, seed=0):
    """Paired bootstrap of (B - A) for accuracy and macro F1, plus exact McNemar on accuracy."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed); n = len(y)
    ca, cb = pa == y, pb == y
    d_acc, d_f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d_acc.append(cb[i].mean() - ca[i].mean())
        d_f1.append(macro_f1_over_present(y[i], pb[i], k) - macro_f1_over_present(y[i], pa[i], k))
    b01, b10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"delta_accuracy": float(cb.mean() - ca.mean()), "delta_accuracy_ci95": q(d_acc),
            "delta_macro_f1": float(macro_f1_over_present(y, pb, k) - macro_f1_over_present(y, pa, k)), "delta_macro_f1_ci95": q(d_f1),
            "mcnemar": {"A_wrong_B_right": b01, "A_right_B_wrong": b10, "p_value": float(pv)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "wsp_runs", "e0", args.split)
    assert not os.path.exists(out_dir), f"refusing to overwrite {out_dir}"
    os.makedirs(out_dir)
    assert torch.cuda.is_available(), "existing eval pipeline uses CUDA fp16 autocast"
    device = torch.device("cuda"); torch.manual_seed(0)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
    li = json.load(open(os.path.join(ROOT, "data", "supervised", "classification_labels.json"), encoding="utf-8"))
    label2id = li["label2id"]; cat_names = [li["id2label"][str(i)] for i in range(li["num_classes"])]
    csat_names = [f"{i} star" for i in range(1, 6)]
    config = SentriFormerConfig.from_pretrained(os.path.join(ROOT, "checkpoints", "config.json"))
    config.num_categories = li["num_classes"]; config.num_csat_classes = 5  # same override as the training scripts

    cls_path = os.path.join(ROOT, "data", "supervised", f"classification_{args.split}.parquet")
    loaders = {}
    for L in CLS_LENGTHS:
        ds = ClassificationParquetDataset(cls_path, tok, label2id, max_seq_length=L)
        loaders[("category", L)] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ClassificationDataCollator(config.pad_token_id, L))
    csat_ds = CSATDataset(os.path.join(ROOT, "data", "supervised", f"csat_{args.split}.parquet"), tok, max_len=CSAT_LEN)
    loaders[("csat", CSAT_LEN)] = DataLoader(csat_ds, batch_size=64, shuffle=False, collate_fn=CSATCollator(config.pad_token_id))
    log(f"split={args.split} category n={len(loaders[('category', 512)].dataset)} csat n={len(csat_ds)}")

    summary = {"split": args.split, "protocol": {"category_lengths": CLS_LENGTHS, "csat_len": CSAT_LEN, "autocast": "fp16",
               "mode": "model.eval() + torch.inference_mode()", "batch_size": args.batch_size}, "checkpoints": {}}
    preds = {}
    for name, rel in CKPTS.items():
        raw = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
        sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
        meta = {k: v for k, v in raw.items() if k in ("epoch", "step", "global_step", "best_val_f1", "best_val_loss", "best_val_score")}
        del raw
        model = SentriFormerForMultiTask(config)
        model.load_state_dict(sd, strict=True)
        model.to(device).eval().requires_grad_(False)
        fp_before = state_fingerprint(model)
        rec = {"path": rel, "meta": meta, "tasks": {}}
        for (task, L), loader in loaders.items():
            if task == "csat" and name not in CSAT_SUPPORTED:
                rec["tasks"]["csat"] = "SKIPPED: CSAT head untrained in this checkpoint"; continue
            t = time.time()
            logits, y = run(model, loader, device, "category_logits" if task == "category" else "csat_logits")
            m, probs, p = metrics(logits, y, cat_names if task == "category" else csat_names)
            if task == "csat": m.update(csat_extra(p, y))
            if task == "category": m["returns_pred_rate"] = float((p == RETURNS_ID).mean())
            key = f"{task}_len{L}"
            rec["tasks"][key] = m
            preds[(name, key)] = (y, p)
            np.savez_compressed(os.path.join(out_dir, f"preds_{name}_{key}.npz"), targets=y.astype(np.int16), preds=p.astype(np.int16), probs=probs.astype(np.float16))
            log(f"{name:16s} {key:16s} acc={m['accuracy']:.4f} macroF1={m['macro_f1_sklearn']:.4f} wF1={m['weighted_f1']:.4f} ece={m['ece_15bin']:.4f} conf={m['mean_max_conf']:.3f} ({time.time()-t:.0f}s)")
        fp_after = state_fingerprint(model)
        assert fp_before == fp_after, f"{name}: weights changed during evaluation"
        rec["weights_fingerprint_sha256"] = fp_after
        summary["checkpoints"][name] = rec
        del model, sd; torch.cuda.empty_cache()

    # Paired comparisons on identical samples
    comps = {}
    for key, k in [(f"category_len{L}", len(cat_names)) for L in CLS_LENGTHS] + [(f"csat_len{CSAT_LEN}", 5)]:
        for a, b in (("classifier_best", "multitask_best"), ("multitask_best", "m10_3_sft_best"), ("classifier_best", "m10_3_sft_best")):
            if (a, key) in preds and (b, key) in preds:
                y, pa = preds[(a, key)]; _, pb = preds[(b, key)]
                comps[f"{key}: {b} - {a}"] = paired(y, pa, pb, k)
    summary["paired_comparisons"] = comps
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "e0_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"done in {summary['elapsed_seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
