"""WSP E0 - Build wsp_runs/e0/E0_REPORT.md from val/test e0_metrics.json + integrity files. Read-only on inputs."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E0 = os.path.join(ROOT, "wsp_runs", "e0")
OUT = os.path.join(E0, "E0_REPORT.md")
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
R = {s: json.load(open(os.path.join(E0, s, "e0_metrics.json"), encoding="utf-8")) for s in ("val", "test")}
pre = json.load(open(os.path.join(E0, "integrity_pre.json"), encoding="utf-8"))
post = json.load(open(os.path.join(E0, "integrity_post.json"), encoding="utf-8"))
pt = json.load(open(os.path.join(E0, "pretrain_overlap.json"), encoding="utf-8"))
CK = ["classifier_best", "multitask_best", "m10_3_sft_best"]
pct = lambda x: f"{100 * x:.2f}%"
L = ["# E0 Baseline Evaluation Report", "",
     f"Generated from `wsp_runs/e0/{{val,test}}/e0_metrics.json`. Protocol: {R['test']['protocol']}.", ""]

for split in ("val", "test"):
    r = R[split]["checkpoints"]
    L += [f"## Category routing - {split}", "",
          "| Checkpoint | Len | Acc | Macro F1 | Weighted F1 | NLL | ECE | Brier | Mean conf | Conf<25% | Returns rate |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CK:
        for Ln in (512, 128):
            m = r[c]["tasks"][f"category_len{Ln}"]
            L.append(f"| {c} | {Ln} | {pct(m['accuracy'])} | {m['macro_f1_sklearn']:.4f} | {m['weighted_f1']:.4f} | {m['nll']:.3f} | {m['ece_15bin']:.4f} | {m['brier']:.3f} | {m['mean_max_conf']:.3f} | {pct(m['frac_conf_lt_0.25'])} | {pct(m['returns_pred_rate'])} |")
    L += ["", f"## CSAT - {split}", "", "| Checkpoint | Exact | Within±1 | MAE | RMSE | Macro F1 | Weighted F1 | ECE | Mean conf | Pred dist (1..5) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CK:
        m = r[c]["tasks"].get("csat_len128")
        if isinstance(m, dict):
            L.append(f"| {c} | {pct(m['accuracy'])} | {pct(m['within_1'])} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['macro_f1_sklearn']:.4f} | {m['weighted_f1']:.4f} | {m['ece_15bin']:.4f} | {m['mean_max_conf']:.3f} | {list(m['pred_distribution'].values())} |")
        else:
            L.append(f"| {c} | {r[c]['tasks']['csat']} |||||||||")
    L += ["", f"## Paired comparisons - {split} (B - A; 1000x paired bootstrap 95% CI; exact McNemar)", "",
          "| Comparison | ΔAcc | 95% CI | ΔMacro F1 | 95% CI | McNemar (A✗B✓ / A✓B✗, p) |", "|---|---|---|---|---|---|"]
    for k, v in R[split]["paired_comparisons"].items():
        mc = v["mcnemar"]
        L.append(f"| {k} | {100*v['delta_accuracy']:+.2f} pp | [{100*v['delta_accuracy_ci95'][0]:+.2f}, {100*v['delta_accuracy_ci95'][1]:+.2f}] | {v['delta_macro_f1']:+.4f} | [{v['delta_macro_f1_ci95'][0]:+.4f}, {v['delta_macro_f1_ci95'][1]:+.4f}] | {mc['A_wrong_B_right']} / {mc['A_right_B_wrong']}, p={mc['p_value']:.2g} |")
    L.append("")

# Per-class F1 on test @512
r = R["test"]["checkpoints"]
names = list(r["multitask_best"]["tasks"]["category_len512"]["per_class"].keys())
L += ["## Per-class F1 - test, len 512", "", "| Class | Support | " + " | ".join(CK) + " | m10_3 pred count |", "|---|---|---|---|---|---|"]
for n in names:
    pcs = [r[c]["tasks"]["category_len512"]["per_class"][n] for c in CK]
    L.append(f"| {n} | {pcs[0]['support']} | " + " | ".join(f"{p['f1']:.3f}" for p in pcs) + f" | {r['m10_3_sft_best']['tasks']['category_len512']['pred_distribution'][n]} |")

# Integrity
changed = [k for k in pre["sha256"] if pre["sha256"][k]["sha256"] != post["sha256"].get(k, {}).get("sha256")]
L += ["", "## Integrity", "",
      f"- Files hashed before and after: {len(pre['sha256'])}; changed: **{len(changed)}** {changed if changed else ''}",
      "- In-memory weight fingerprint identical before/after evaluation for every checkpoint (asserted).",
      f"- Split checks: {json.dumps({t: {'id_overlap': pre['checks'][t]['id_overlap'], 'text_overlap': pre['checks'][t]['exact_text_overlap']} for t in ('classification', 'csat')})}",
      f"- Cross-task eval/train exact-text overlaps: all {sum(pre['checks']['cross_task_overlap'].values())} (zero means none).",
      f"- Pretraining-corpus overlap: {json.dumps(pt)}", "",
      "### SHA256", "", "| File | SHA256 | Bytes |", "|---|---|---|"]
L += [f"| {k} | `{v['sha256']}` | {v['bytes']:,} |" for k, v in pre["sha256"].items()]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT, "changed files:", changed)
