"""
WSP E1 final - build edited checkpoint from selected A matrices, verify edit scope, evaluate with the
identical E0 harness on val then test (test once), paired stats vs m10_3 (original) and multitask_best.
Writes only wsp_runs/e1/. Original checkpoints are read-only inputs.
Usage: python -I wsp/e1_final.py --split val|test
"""
import argparse, json, os, sys, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_eval import run, metrics, csat_extra, paired, state_fingerprint, CLS_LENGTHS, CSAT_LEN, RETURNS_ID
from e1_realign import setup, SUP, CK
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator

OUT = os.path.join(ROOT, "wsp_runs", "e1")
E2 = os.path.join(ROOT, "wsp_runs", "e2")
EDITED = os.path.join(OUT, "m10_3_sft_wsp_candidate.pt")


def build():
    """Applies ONLY accepted edits: E1 maps per wsp_runs/e1/acceptance.json, E2 bias deltas (zeros if rejected)."""
    acc = json.load(open(os.path.join(OUT, "acceptance.json")))
    raw = torch.load(os.path.join(ROOT, CK["m103"]), map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"]
    new = {k: v.clone() for k, v in sd.items()}
    for head, f in (("category", "A_category.npy"), ("csat", "A_csat.npy")):
        if acc[head]:
            key = f"{head}_head.dense.weight"
            new[key] = (sd[key].float() @ torch.from_numpy(np.load(os.path.join(OUT, f))).float()).to(sd[key].dtype)
    e1_only = os.path.join(OUT, "m10_3_sft_wsp_e1only_candidate.pt")
    if not os.path.exists(e1_only):
        torch.save({"model_state_dict": new, "wsp_edit": {"base": CK["m103"], "stage": "E1 accepted maps only", "acceptance": acc}}, e1_only)
    new = {k: v.clone() for k, v in new.items()}
    for head in ("category", "csat"):
        bpath = os.path.join(E2, f"bias_delta_{head}.npy")
        if os.path.exists(bpath):
            key = f"{head}_head.out_proj.bias"
            new[key] = (sd[key].float() + torch.from_numpy(np.load(bpath)).float()).to(sd[key].dtype)
    for k in new: assert torch.isfinite(new[k]).all(), k
    changed = [k for k in sd if not torch.equal(sd[k], new[k])]
    EDIT_KEYS = changed
    assert all(k.startswith(("category_head.", "csat_head.")) for k in changed), f"edit outside heads: {changed}"
    diff = {k: {"rel_change": float((new[k] - sd[k]).norm() / sd[k].norm()),
                "orig_sha256": hashlib.sha256(sd[k].numpy().tobytes()).hexdigest(),
                "new_sha256": hashlib.sha256(new[k].numpy().tobytes()).hexdigest()} for k in EDIT_KEYS}
    if not os.path.exists(EDITED):
        torch.save({"model_state_dict": new, "wsp_edit": {"base": CK["m103"], "edited_tensors": list(EDIT_KEYS), "op": "dense.weight @ A (ridge, fit on train)", "diff": diff}}, EDITED)
        json.dump({"changed_tensors": changed, "diff": diff, "unchanged_tensors": len(sd) - len(changed)}, open(os.path.join(OUT, "edit_diff.json"), "w"), indent=2)
    return new


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["val", "test"]); a = ap.parse_args()
    od = os.path.join(OUT, f"final_{a.split}"); assert not os.path.exists(od), f"refusing to overwrite {od}"
    tok, li, cfg = setup()
    sd = build()
    cat_names = [li["id2label"][str(i)] for i in range(li["num_classes"])]; csat_names = [f"{i} star" for i in range(1, 6)]
    model = SentriFormerForMultiTask(cfg); model.load_state_dict(sd, strict=True); model.cuda().eval().requires_grad_(False)
    fp = state_fingerprint(model)
    os.makedirs(od)
    res = {"edited_checkpoint": os.path.relpath(EDITED, ROOT), "tasks": {}, "paired": {}}
    jobs = [(f"category_len{L}", DataLoader(ClassificationParquetDataset(os.path.join(SUP, f"classification_{a.split}.parquet"), tok, li["label2id"], max_seq_length=L), batch_size=128, collate_fn=ClassificationDataCollator(0, L)), "category_logits", cat_names) for L in CLS_LENGTHS]
    jobs.append((f"csat_len{CSAT_LEN}", DataLoader(CSATDataset(os.path.join(SUP, f"csat_{a.split}.parquet"), tok, max_len=CSAT_LEN), batch_size=64, collate_fn=CSATCollator(0)), "csat_logits", csat_names))
    for key, loader, out_key, names in jobs:
        logits, y = run(model, loader, "cuda", out_key)
        m, probs, p = metrics(logits, y, names)
        if key.startswith("csat"): m.update(csat_extra(p, y))
        else: m["returns_pred_rate"] = float((p == RETURNS_ID).mean())
        res["tasks"][key] = m
        np.savez_compressed(os.path.join(od, f"preds_e1_{key}.npz"), targets=y.astype(np.int16), preds=p.astype(np.int16), probs=probs.astype(np.float16))
        k = len(names)
        for ref in ("m10_3_sft_best", "multitask_best"):
            r = np.load(os.path.join(ROOT, "wsp_runs", "e0", a.split, f"preds_{ref}_{key}.npz"))
            assert np.array_equal(r["targets"], y), "sample order mismatch vs E0"
            res["paired"][f"{key}: e1 - {ref}"] = paired(y, r["preds"].astype(np.int64), p, k)
        print(key, {x: round(m[x], 4) for x in ("accuracy", "macro_f1_sklearn", "weighted_f1", "ece_15bin", "mean_max_conf")}, flush=True)
    assert state_fingerprint(model) == fp
    json.dump(res, open(os.path.join(od, "e1_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
