"""T7u: re-fit only sentiment_head.urgency_proj on frozen T5a features. Criteria frozen in T7U_CRITERIA.md."""
import json, os, sys, hashlib
import numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.join(ROOT, "wsp"))
from transformers import AutoTokenizer
from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from train_v2 import TextLabelDataset, FloatLabelCollator
from eval_full import urgency_scores

OUT = os.path.join(ROOT, "wsp_runs", "t7u"); AUX = os.path.join(ROOT, "wsp_runs", "data")
SRC = os.path.join(ROOT, "wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
li = json.load(open(os.path.join(ROOT, "data", "supervised", "classification_labels.json"), encoding="utf-8"))
cfg = SentriFormerConfig.from_pretrained(os.path.join(ROOT, "checkpoints", "config.json"))
cfg.num_categories, cfg.num_csat_classes = li["num_classes"], 5
raw = torch.load(SRC, map_location="cpu", weights_only=False)
model = SentriFormerForMultiTask(cfg); model.load_state_dict(raw["model_state_dict"], strict=True); model.cuda().eval()
ref = {k: v.clone() for k, v in model.state_dict().items()}


def loader(split):
    d = pd.read_parquet(os.path.join(AUX, f"urgency_{split}.parquet"))
    return DataLoader(TextLabelDataset(d.text.tolist(), d.label.astype(float).tolist(), tok, 128), batch_size=64, collate_fn=FloatLabelCollator(0, 128))


cap = {}
h1 = model.sentiment_head.urgency_proj.register_forward_hook(lambda m, i, o: cap.setdefault("x", []).append(i[0].float().cpu()))
h2 = model.sentiment_head.register_forward_hook(lambda m, i, o: cap.setdefault("pool", []).append(i[0].float().cpu()))
feats = {}
for s in ("train", "val"):
    cap.clear(); _, y = urgency_scores(model, loader(s))
    feats[s] = (torch.cat(cap["x"]), torch.cat(cap["pool"]).numpy(), y)
h1.remove(); h2.remove()
Xtr, Ptr, ytr = feats["train"]; Xva, Pva, yva = feats["val"]
w0, b0 = model.sentiment_head.urgency_proj.weight.detach().cpu().clone(), model.sentiment_head.urgency_proj.bias.detach().cpu().clone()


def fit(X, y, lam):
    lin = torch.nn.Linear(256, 1); lin.weight.data, lin.bias.data = w0.clone(), b0.clone()
    t = torch.tensor(y, dtype=torch.float32) / 3
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = ((torch.sigmoid(lin(X).squeeze(-1)) - t) ** 2).mean() + lam * (lin.weight ** 2).sum(); l.backward(); return l
    opt.step(closure); return lin


cv = {}
for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
    errs = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xtr):
        lin = fit(Xtr[tr], ytr[tr], lam)
        with torch.no_grad(): errs.append(np.abs(torch.sigmoid(lin(Xtr[te])).squeeze(-1).numpy() * 3 - ytr[te]).mean())
    cv[lam] = float(np.mean(errs))
lam = min(cv, key=cv.get); lin = fit(Xtr, ytr, lam)

# T5a scores (before) and T7u scores (after), both through the full model forward
s_old, y = urgency_scores(model, loader("val"))
with torch.no_grad():
    model.sentiment_head.urgency_proj.weight.copy_(lin.weight); model.sentiment_head.urgency_proj.bias.copy_(lin.bias)
s_new, _ = urgency_scores(model, loader("val"))
sd = model.state_dict()
changed = [k for k in sd if not torch.equal(sd[k], ref[k])]
assert set(changed) <= {"sentiment_head.urgency_proj.weight", "sentiment_head.urgency_proj.bias"}, changed

tr_mean = float(ytr.mean()); tr_maj = int(np.bincount(ytr).argmax())
rnd = lambda s: np.clip(np.rint(s), 0, 3).astype(int)
mf1 = lambda yy, pp: f1_score(yy, pp, average="macro", zero_division=0)


def summary(s):
    p = rnd(s)
    return {"acc": float((p == y).mean()), "macro_f1": float(mf1(y, p)), "mae_levels": float(np.abs(s - y).mean()),
            "within_1": float((np.abs(p - y) <= 1).mean()), "escalation_f1": float(f1_score(y >= 2, p >= 2, zero_division=0)),
            "spearman": float(pd.Series(s).corr(pd.Series(y), method="spearman")), "pred_dist": np.bincount(p, minlength=4).tolist()}


rng = np.random.default_rng(0); n = len(y); B = [rng.integers(0, n, n) for _ in range(2000)]
ci = lambda f: [float(f(np.arange(n)))] + np.percentile([f(i) for i in B], [2.5, 97.5]).tolist()
c1 = ci(lambda i: (np.abs(tr_mean - y[i]) - np.abs(s_new[i] - y[i])).mean())
c2 = ci(lambda i: mf1(y[i], rnd(s_new[i])) - mf1(y[i], rnd(s_old[i])))
mae_vs_t5a = ci(lambda i: (np.abs(s_old[i] - y[i]) - np.abs(s_new[i] - y[i])).mean())

# diagnostic: 4-class logistic on pooled encoder state
best_c = max((1e-3, 1e-2, 1e-1, 1.0, 10.0), key=lambda c: cross_val_score(LogisticRegression(C=c, max_iter=3000, class_weight="balanced"), Ptr, ytr,
                                                                    cv=KFold(5, shuffle=True, random_state=0), scoring="f1_macro").mean())
diag = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Ptr, ytr).predict(Pva)

R = {"criteria_sha256": hashlib.sha256(open(os.path.join(OUT, "T7U_CRITERIA.md"), "rb").read()).hexdigest(),
     "cv_mae_by_l2": {str(k): v for k, v in cv.items()}, "chosen_l2": lam, "changed_params": changed,
     "T5a": summary(s_old), "T7u": summary(s_new),
     "baseline_mean": {"value": tr_mean, "mae_levels": float(np.abs(tr_mean - y).mean())},
     "baseline_majority": {"label": tr_maj, "acc": float((y == tr_maj).mean()), "macro_f1": float(mf1(y, np.full_like(y, tr_maj)))},
     "C1_abs_err_meanbase_minus_T7u_CI95": c1, "C2_macroF1_T7u_minus_T5a_CI95": c2, "mae_T5a_minus_T7u_CI95": mae_vs_t5a,
     "diagnostic_logreg_pooled": {"C": best_c, "macro_f1": float(mf1(y, diag)), "acc": float((diag == y).mean())}}
R["C1"], R["C2"] = c1[1] > 0, c2[1] > 0
R["decision"] = "PASS" if R["C1"] and R["C2"] else "REJECT"

if R["decision"] == "PASS":
    fd = os.path.join(ROOT, "wsp_runs", "FINAL_T7u"); os.makedirs(fd, exist_ok=True)
    out = {"model_state_dict": {k: v.cpu() for k, v in sd.items()}, "train_run": "T7u (urgency_proj refit on T5a)", "step": 0, "quick_eval": None}
    torch.save(out, os.path.join(fd, "sentriformer_t7u.pt"))
    s_te, y_te = urgency_scores(model, loader("test"))  # once, reported only
    y, _y = y_te, y; R["test_once_T7u"] = summary(s_te); y = _y
    R["test_baseline_mean_mae"] = float(np.abs(tr_mean - y_te).mean())
json.dump(R, open(os.path.join(OUT, "t7u_results.json"), "w"), indent=1)
print(json.dumps({k: R[k] for k in ("chosen_l2", "T5a", "T7u", "baseline_mean", "C1_abs_err_meanbase_minus_T7u_CI95",
                                    "C2_macroF1_T7u_minus_T5a_CI95", "diagnostic_logreg_pooled", "decision")}, indent=1))
