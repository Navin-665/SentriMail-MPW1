"""
WSP E2 - Gradient-free output-bias prior matching on top of the selected E1 heads.
Fixed-point iteration on TRAIN features: b <- b + log(train_label_prior) - log(mean predicted prob), <=50 iters.
Acceptance (frozen in wsp_runs/SELECTION_CRITERIA.md, rule 4): val macro F1 improves with CI excluding 0 AND
val accuracy drops <= 0.5 pp vs E1-only. Otherwise rejected (delta = 0). Writes wsp_runs/e2/ only.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_eval import metrics, paired
from e1_realign import setup, head_logits, CK, FEATS

OUT = os.path.join(ROOT, "wsp_runs", "e2")


def prior_match(logits, y, k, iters=50):
    prior = np.bincount(y, minlength=k) / len(y)
    b = torch.zeros(k); target = torch.log(torch.from_numpy(np.clip(prior, 1e-8, 1)).float())
    for i in range(iters):
        q = torch.softmax(logits + b, -1).mean(0)
        step = target - torch.log(q.clamp_min(1e-8))
        b += step
        if step.abs().max() < 1e-4:
            break
    b[torch.from_numpy(prior == 0)] = 0.0  # classes absent from train: leave untouched
    return b, i + 1


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    os.makedirs(OUT)
    tok, li, cfg = setup()
    cat_names = [li["id2label"][str(i)] for i in range(li["num_classes"])]; csat_names = [f"{i} star" for i in range(1, 6)]
    raw = torch.load(os.path.join(ROOT, CK["m103"]), map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in raw["model_state_dict"].items()}; del raw
    E1 = os.path.join(ROOT, "wsp_runs", "e1")
    acc = json.load(open(os.path.join(E1, "acceptance.json")))
    fit = json.load(open(os.path.join(E1, "fit_report.json")))
    A = lambda head: torch.from_numpy(np.load(os.path.join(E1, f"A_{head}.npy"))).float() if acc[head] else torch.eye(256)
    Wc = (sd["category_head.dense.weight"] @ A("category")).cuda()   # E2 applies on top of the ACCEPTED E1 state only
    Ws = (sd["csat_head.dense.weight"] @ A("csat")).cuda()
    csat_allowed = fit["csat"]["drift_precondition_r2_lt_0.90"]  # safeguard: no CSAT edit without established drift
    ld = lambda n: (np.load(os.path.join(FEATS, f"{n}.npz"))["X"], np.load(os.path.join(FEATS, f"{n}.npz"))["y"].astype(np.int64))
    rep = {}
    for task, prefix, W, names, tr, vals, sel in (
        ("category", "category_head", Wc, cat_names, "m103_cls_train_512", ("m103_cls_val_512", "m103_cls_val_128"), "m103_cls_val_512"),
        ("csat", "csat_head", Ws, csat_names, "m103_csat_train_128", ("m103_csat_val_128",), "m103_csat_val_128")):
        k = len(names)
        if task == "csat" and not csat_allowed:
            np.save(os.path.join(OUT, "bias_delta_csat.npy"), np.zeros(k, dtype=np.float32))
            rep[task] = {"accepted": False, "skipped": "CSAT drift precondition not met; CSAT head left untouched"}
            print("csat SKIPPED (drift precondition not met)", flush=True); continue
        Xt, yt = ld(tr)
        lt = head_logits(sd, prefix, Xt, W)
        delta, n_it = prior_match(lt, yt, k)
        r = {"bias_delta": delta.tolist(), "iterations": n_it, "val": {}}
        for v in vals:
            Xv, yv = ld(v)
            base = head_logits(sd, prefix, Xv, W); cal = base + delta
            mb, _, pb = metrics(base, yv, names); mc, _, pc = metrics(cal, yv, names)
            keep = ("accuracy", "macro_f1_sklearn", "weighted_f1", "nll", "ece_15bin", "mean_max_conf")
            n_val = len(yv)
            per_class = {nm: {"support": mb["per_class"][nm]["support"],
                              **{f"d_{q}": mc["per_class"][nm][q] - mb["per_class"][nm][q] for q in ("precision", "recall", "f1")}}
                         for nm in names}
            important = [nm for nm, d in per_class.items() if d["support"] >= 0.01 * n_val]
            damaged = [nm for nm in important if min(per_class[nm]["d_precision"], per_class[nm]["d_recall"], per_class[nm]["d_f1"]) < -0.02]
            r["val"][v] = {"e1_only": {x: mb[x] for x in keep}, "e1+e2": {x: mc[x] for x in keep},
                           "pred_dist_e1": list(mb["pred_distribution"].values()), "pred_dist_e1e2": list(mc["pred_distribution"].values()),
                           "per_class_delta": per_class, "important_classes": important, "damaged_important_classes": damaged,
                           "confusion_e1": mb["confusion_matrix"], "confusion_e1e2": mc["confusion_matrix"],
                           "paired_e2_minus_e1": paired(yv, pb, pc, k)}
        sv = r["val"][sel]; p = sv["paired_e2_minus_e1"]
        d_ece = sv["e1+e2"]["ece_15bin"] - sv["e1_only"]["ece_15bin"]
        checks = {"macro_f1_ci_excludes_0": p["delta_macro_f1_ci95"][0] > 0, "acc_drop_le_0.5pp": p["delta_accuracy"] >= -0.005,
                  "ece_increase_le_0.01": d_ece <= 0.01, "no_important_class_damaged": not sv["damaged_important_classes"]}
        accepted = all(checks.values())
        r["accepted"] = bool(accepted)
        r["decision_basis"] = {"selection_split": sel, "checks": checks, "delta_macro_f1": p["delta_macro_f1"], "ci": p["delta_macro_f1_ci95"],
                               "delta_acc": p["delta_accuracy"], "delta_ece": d_ece, "damaged": sv["damaged_important_classes"]}
        np.save(os.path.join(OUT, f"bias_delta_{task}.npy"), (delta if accepted else torch.zeros(k)).numpy())
        rep[task] = r
        print(task, "accepted" if accepted else "REJECTED", json.dumps(r["decision_basis"]), flush=True)
    json.dump(rep, open(os.path.join(OUT, "e2_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
