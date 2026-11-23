"""
Apply the frozen wsp_runs/train/TRAINING_CRITERIA.md to full-val eval_full results of Phase-0 vs a trained candidate.
Usage: python -I wsp/accept_training.py <candidate_name>   (expects wsp_runs/eval_full/{phase0,<name>}_val/)
Writes wsp_runs/eval_full/<name>_val/acceptance.json
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_eval import paired, macro_f1_over_present

E = os.path.join(ROOT, "wsp_runs", "eval_full")


def boot_ci(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    name = sys.argv[1]
    P = json.load(open(os.path.join(E, "phase0_val", "metrics.json"))); C = json.load(open(os.path.join(E, f"{name}_val", "metrics.json")))
    ld = lambda n, f: np.load(os.path.join(E, f"{n}_val", f), allow_pickle=True)
    out = {"candidate": name, "checks": {}, "numbers": {}}

    # 1. Category (best length per model)
    bl = lambda R: max((512, 128), key=lambda L: R[f"category_len{L}"]["accuracy"])
    lp, lc = bl(P), bl(C)
    pa, ca = P[f"category_len{lp}"], C[f"category_len{lc}"]
    out["numbers"]["category"] = {"phase0": {"len": lp, "acc": pa["accuracy"], "macro_f1": pa["macro_f1_sklearn"], "ece": pa["ece_15bin"]},
                                  "cand": {"len": lc, "acc": ca["accuracy"], "macro_f1": ca["macro_f1_sklearn"], "ece": ca["ece_15bin"]}}
    if lp == lc:
        yp = ld("phase0", f"preds_category_len{lp}.npz"); yc = ld(name, f"preds_category_len{lc}.npz")
        out["numbers"]["category"]["paired"] = paired(yp["targets"].astype(int), yp["preds"].astype(int), yc["preds"].astype(int), 33)
    out["checks"]["1_category_acc_ge_phase0_minus_1pp"] = ca["accuracy"] >= pa["accuracy"] - 0.01
    out["checks"]["1_category_macroF1_drop_le_0.01"] = ca["macro_f1_sklearn"] >= pa["macro_f1_sklearn"] - 0.01

    # 2. CSAT
    pc, cc = P["csat"], C["csat"]
    out["numbers"]["csat"] = {"phase0": {k: pc[k] for k in ("accuracy", "macro_f1_sklearn", "within_1", "mae")},
                              "cand": {k: cc[k] for k in ("accuracy", "macro_f1_sklearn", "within_1", "mae")},
                              "cand_pred_dist": cc["pred_distribution"], "cand_per_class_f1": {k: v["f1"] for k, v in cc["per_class"].items()}}
    out["checks"]["2_csat_macroF1_ge_phase0"] = cc["macro_f1_sklearn"] >= pc["macro_f1_sklearn"]
    out["checks"]["2_csat_acc_ge_phase0_minus_2pp"] = cc["accuracy"] >= pc["accuracy"] - 0.02

    # 3. Sentiment: beat untrained head (phase0) and majority baseline, paired CI excluding 0
    sp, sc = ld("phase0", "preds_sentiment.npz"), ld(name, "preds_sentiment.npz")
    y = sc["targets"].astype(int); p0 = sp["preds"].astype(int); p1 = sc["preds"].astype(int)
    maj = C["sentiment"]["majority_baseline"]["label"]; pm = np.full_like(y, maj)
    f1 = lambda yy, pp: macro_f1_over_present(yy, pp, 3)
    d_unt = boot_ci(lambda i: f1(y[i], p1[i]) - f1(y[i], p0[i]), len(y))
    d_maj = boot_ci(lambda i: f1(y[i], p1[i]) - f1(y[i], pm[i]), len(y))
    out["numbers"]["sentiment"] = {"cand_macro_f1": C["sentiment"]["all"]["macro_f1_sklearn"], "cand_acc": C["sentiment"]["all"]["accuracy"],
                                   "untrained_macro_f1": P["sentiment"]["all"]["macro_f1_sklearn"], "majority_macro_f1": float(f1(y, pm)),
                                   "human_goemotions_macro_f1": C["sentiment"]["goemotions_human"]["macro_f1_sklearn"],
                                   "human_goemotions_acc": C["sentiment"]["goemotions_human"]["accuracy"],
                                   "distilled_teacher_agreement_acc": C["sentiment"]["distilled"]["accuracy"],
                                   "distilled_macro_f1": C["sentiment"]["distilled"]["macro_f1_sklearn"],
                                   "ci_vs_untrained": d_unt, "ci_vs_majority": d_maj,
                                   "per_class": C["sentiment"]["all"]["per_class"]}
    out["checks"]["3_sentiment_beats_untrained_CI"] = d_unt[0] > 0
    out["checks"]["3_sentiment_beats_majority_CI"] = d_maj[0] > 0

    # 4. Urgency: beat majority macro F1 and mean-prediction MAE (bootstrap CI, n=200)
    uc = ld(name, "preds_urgency.npz"); yu = uc["targets"].astype(int); s = uc["scores"].astype(float)
    pcu = np.clip(np.rint(s), 0, 3).astype(int); U = C["urgency"]
    from sklearn.metrics import f1_score
    tr_maj = None
    import pyarrow.parquet as pq
    tr = pq.read_table(os.path.join(ROOT, "wsp_runs", "data", "urgency_train.parquet")).to_pandas().label
    tr_maj, tr_mean = int(tr.mode()[0]), float(tr.mean())
    d_f1 = boot_ci(lambda i: f1_score(yu[i], pcu[i], average="macro", zero_division=0) - f1_score(yu[i], np.full(len(i), tr_maj), average="macro", zero_division=0), len(yu))
    d_mae = boot_ci(lambda i: np.abs(tr_mean - yu[i]).mean() - np.abs(s[i] - yu[i]).mean(), len(yu))
    out["numbers"]["urgency"] = {k: U[k] for k in ("n", "acc", "macro_f1", "mae_levels", "within_1", "escalation_f1", "spearman", "pred_dist")}
    out["numbers"]["urgency"].update({"baseline_majority_macro_f1": U["baseline_majority"]["macro_f1"], "baseline_mean_mae": U["baseline_mean_mae"],
                                      "ci_macroF1_minus_majority": d_f1, "ci_mae_improvement_vs_mean": d_mae,
                                      "phase0_spearman": P["urgency"]["spearman"]})
    out["checks"]["4_urgency_macroF1_beats_majority_CI"] = d_f1[0] > 0
    out["checks"]["4_urgency_mae_beats_mean_CI"] = d_mae[0] > 0
    # SUPPLEMENTARY (added after self-test showed the untrained head passes the frozen macro-F1-vs-majority check):
    rank = lambda a: np.argsort(np.argsort(a, kind="stable"), kind="stable")
    sp_ci = boot_ci(lambda i: np.corrcoef(rank(s[i]), rank(yu[i]))[0, 1], len(yu))
    out["numbers"]["urgency"]["spearman_ci"] = sp_ci
    out["supplementary_checks"] = {"4s_urgency_spearman_CI_gt_0": sp_ci[0] > 0}

    # 5. Dialogue
    gp, gc = P["dialogue"]["generation"], C["dialogue"]["generation"]
    out["numbers"]["dialogue"] = {"phase0": {"loss": P["dialogue"]["teacher_forced_loss"], **gp}, "cand": {"loss": C["dialogue"]["teacher_forced_loss"], **gc}}
    out["checks"]["5_dial_loss_le_phase0_plus_0.05"] = C["dialogue"]["teacher_forced_loss"] <= P["dialogue"]["teacher_forced_loss"] + 0.05
    out["checks"]["5_eos_ge_phase0_minus_2pp"] = gc["eos_rate"] >= gp["eos_rate"] - 0.02
    out["checks"]["5_repetition_le_phase0_plus_2pp"] = gc["repetition_rate"] <= gp["repetition_rate"] + 0.02
    out["checks"]["5_orphaned_le_phase0_plus_2pp"] = gc["orphaned_rate"] <= gp["orphaned_rate"] + 0.02
    out["checks"]["5_rougeL_ge_phase0_minus_0.005"] = gc["rougeL_mean"] >= gp["rougeL_mean"] - 0.005

    c = out["checks"]
    out["deployable"] = all(v for k, v in c.items() if k.startswith(("1_", "5_")))  # rule 6
    out["all_criteria_pass"] = all(c.values())
    json.dump(out, open(os.path.join(E, f"{name}_val", "acceptance.json"), "w"), indent=2, default=float)
    for k, v in list(c.items()) + list(out["supplementary_checks"].items()): print(("PASS " if v else "FAIL ") + k)
    print("deployable (rules 1+5):", out["deployable"], "| all criteria:", out["all_criteria_pass"])


if __name__ == "__main__":
    main()
