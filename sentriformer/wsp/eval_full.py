"""
Full five-capability evaluation of any SentriFormerForMultiTask checkpoint (inference only).
  category  : E0 harness metrics @512 and @128 on classification_<split>
  csat      : E0 harness metrics on csat_<split>
  sentiment : 3-class metrics on wsp_runs/data/sentiment_<split>, overall + per origin (human GoEmotions vs distilled)
  urgency   : 4-level acc / macro-F1 / MAE / within-1 + escalation (High|Immediate) F1 on urgency_<split>
  dialogue  : teacher-forced loss/PPL on full m10_4_dialogue_<split>; greedy generation on 300 fixed contexts with
              EOS / repetition / orphaned-phrase / placeholder / invented-contact rates, distinct-1/2, length, ROUGE-L vs reference
Usage: python -I wsp/eval_full.py --ckpt PATH --name NAME --split val|test
Writes wsp_runs/eval_full/<name>_<split>/ (refuses to overwrite).
"""
import argparse, json, math, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_eval import run, metrics, csat_extra
from train_v2 import TextLabelDataset, FloatLabelCollator
from inference import SentriFormerPipeline
from sentriformer.training.classification_dataset import ClassificationParquetDataset, ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask, safety_check_response, detect_repetition

SUP = os.path.join(ROOT, "data", "supervised"); AUX = os.path.join(ROOT, "wsp_runs", "data")
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")


def lcs(a, b):
    if not a or not b: return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref):
    h, r = hyp.lower().split(), ref.lower().split(); l = lcs(h, r)
    if l == 0: return 0.0
    p, rc = l / len(h), l / len(r); return 2 * p * rc / (p + rc)


@torch.inference_mode()
def urgency_scores(model, loader):
    P, Y = [], []
    for b in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda())
        P.append(o["urgency_score"].float().cpu()); Y.append(b["labels"])
    return torch.cat(P).numpy() * 3, torch.cat(Y).numpy().astype(int)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"]); ap.add_argument("--n_gen", type=int, default=300)
    a = ap.parse_args()
    od = os.path.join(ROOT, "wsp_runs", "eval_full", f"{a.name}_{a.split}"); assert not os.path.exists(od), od
    os.makedirs(od); torch.manual_seed(0)
    pipe = SentriFormerPipeline(a.ckpt, "cuda"); model, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    li = json.load(open(os.path.join(SUP, "classification_labels.json"), encoding="utf-8"))
    R = {"ckpt": a.ckpt, "split": a.split}

    for L in (512, 128):
        ds = ClassificationParquetDataset(os.path.join(SUP, f"classification_{a.split}.parquet"), tok, li["label2id"], max_seq_length=L)
        lg, y = run(model, DataLoader(ds, batch_size=128, collate_fn=ClassificationDataCollator(0, L)), "cuda", "category_logits")
        m, probs, p = metrics(lg, y, pipe.cat_names); m["returns_pred_rate"] = float((p == 28).mean()); R[f"category_len{L}"] = m
        np.savez_compressed(os.path.join(od, f"preds_category_len{L}.npz"), targets=y.astype(np.int16), preds=p.astype(np.int16))
    lg, y = run(model, DataLoader(CSATDataset(os.path.join(SUP, f"csat_{a.split}.parquet"), tok, max_len=128), batch_size=64, collate_fn=CSATCollator(0)), "cuda", "csat_logits")
    m, _, p = metrics(lg, y, [f"{i} star" for i in range(1, 6)]); m.update(csat_extra(p, y)); R["csat"] = m
    np.savez_compressed(os.path.join(od, "preds_csat.npz"), targets=y.astype(np.int16), preds=p.astype(np.int16))

    sd = pq.read_table(os.path.join(AUX, f"sentiment_{a.split}.parquet")).to_pandas()
    lg, y = run(model, DataLoader(TextLabelDataset(sd.text.tolist(), sd.label.tolist(), tok, 128), batch_size=128, collate_fn=ClassificationDataCollator(0, 128)), "cuda", "sentiment_logits")
    names = ["negative", "neutral", "positive"]
    m, _, p = metrics(lg, y, names); R["sentiment"] = {"all": m}
    maj = int(pq.read_table(os.path.join(AUX, "sentiment_train.parquet"), columns=["label"]).to_pandas().label.mode()[0])  # train majority
    for org in ("goemotions_human", "distilled"):
        msk = (sd.origin == org).values if org != "distilled" else sd.origin.str.startswith("distilled").values
        mm, _, _ = metrics(lg[torch.from_numpy(msk)], y[msk], names); R["sentiment"][org] = mm
    R["sentiment"]["majority_baseline"] = {"label": int(maj), "accuracy": float((y == maj).mean())}
    np.savez_compressed(os.path.join(od, "preds_sentiment.npz"), targets=y.astype(np.int16), preds=p.astype(np.int16), origin=sd.origin.values.astype(str))

    ud = pq.read_table(os.path.join(AUX, f"urgency_{a.split}.parquet")).to_pandas()
    s, yu = urgency_scores(model, DataLoader(TextLabelDataset(ud.text.tolist(), ud.label.astype(float).tolist(), tok, 128), batch_size=64, collate_fn=FloatLabelCollator(0, 128)))
    pc = np.clip(np.rint(s), 0, 3).astype(int)
    from sklearn.metrics import f1_score
    tr_mean = pq.read_table(os.path.join(AUX, "urgency_train.parquet")).to_pandas().label.mean()
    tr_maj = int(pq.read_table(os.path.join(AUX, "urgency_train.parquet")).to_pandas().label.mode()[0])
    R["urgency"] = {"n": int(len(yu)), "acc": float((pc == yu).mean()), "macro_f1": float(f1_score(yu, pc, average="macro", zero_division=0)),
                    "mae_levels": float(np.abs(s - yu).mean()), "within_1": float((np.abs(pc - yu) <= 1).mean()),
                    "escalation_f1": float(f1_score(yu >= 2, pc >= 2, zero_division=0)),
                    "spearman": float(np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(yu)))[0, 1]),
                    "pred_dist": np.bincount(pc, minlength=4).tolist(),
                    "baseline_majority": {"acc": float((yu == tr_maj).mean()), "macro_f1": float(f1_score(yu, np.full_like(yu, tr_maj), average="macro", zero_division=0))},
                    "baseline_mean_mae": float(np.abs(tr_mean - yu).mean())}
    np.savez_compressed(os.path.join(od, "preds_urgency.npz"), targets=yu, scores=s)

    dpath = os.path.join(SUP, f"m10_4_dialogue_{a.split}.parquet")
    dds = DialogueSFTDataset(dpath, tok, max_ctx_len=128, max_resp_len=128)
    tot, n = 0.0, 0
    with torch.inference_mode():
        for b in DataLoader(dds, batch_size=64, collate_fn=DialogueSFTCollator(pad_token_id=0)):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                o = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda(), generation_labels=b["labels"].cuda())
            k = int((b["labels"] != -100).sum()); tot += o["generation_loss"].float().item() * k; n += k
    R["dialogue"] = {"teacher_forced_loss": tot / n, "ppl": math.exp(tot / n), "n_pairs": len(dds)}
    t = pq.read_table(dpath, columns=["context", "response", "intent_category"]).to_pandas()
    idx = np.random.default_rng(7).choice(len(t), a.n_gen, replace=False)  # fixed contexts, same for every model
    gens, flags = [], {"eos": 0, "repetition": 0, "orphaned": 0, "placeholder": 0, "invented_contact_or_url": 0, "empty": 0}
    rl, toks, uni, bi = [], [], set(), set(); total_w = 0; total_b = 0
    for i in idx:
        ctx, ref = t.context[i], t.response[i]
        ids, am = pipe._enc(ctx, 128)
        with torch.inference_mode():
            g = generate_multitask(model, tok, cfg, ids, am, max_new_tokens=128, temperature=0.0, repetition_penalty=1.2)[0].tolist()
        eos = cfg.eos_token_id in g; body = g[1:g.index(cfg.eos_token_id)] if eos else g[1:]
        txt = tok.decode(body, skip_special_tokens=True).strip(); sf = safety_check_response(txt, ctx)
        flags["eos"] += eos; flags["repetition"] += detect_repetition(txt); flags["orphaned"] += sf["has_orphaned_phrase"]
        flags["placeholder"] += bool(PLACEHOLDER.search(txt)) and not PLACEHOLDER.search(ctx); flags["empty"] += not txt
        flags["invented_contact_or_url"] += sf["has_invented_contact"] or sf["has_invented_url"]
        w = txt.lower().split(); toks.append(len(w)); uni.update(w); bi.update(zip(w, w[1:])); total_w += len(w); total_b += max(len(w) - 1, 0)
        rl.append(rouge_l(txt, ref)); gens.append({"intent": t.intent_category[i], "context": ctx, "reference": ref, "generated": txt, "rougeL": rl[-1]})
    N = len(idx)
    R["dialogue"]["generation"] = {"n": N, **{f"{k}_rate": v / N for k, v in flags.items()}, "mean_words": float(np.mean(toks)),
                                   "distinct_1": len(uni) / max(total_w, 1), "distinct_2": len(bi) / max(total_b, 1),
                                   "rougeL_mean": float(np.mean(rl))}
    json.dump(gens, open(os.path.join(od, "generations.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(R, open(os.path.join(od, "metrics.json"), "w"), indent=2)
    c = R["category_len512"]; g = R["dialogue"]["generation"]
    print(json.dumps({"cat512_acc": c["accuracy"], "cat512_macroF1": c["macro_f1_sklearn"], "cat_ece": c["ece_15bin"],
                      "csat_acc": R["csat"]["accuracy"], "csat_macroF1": R["csat"]["macro_f1_sklearn"],
                      "sent_macroF1": R["sentiment"]["all"]["macro_f1_sklearn"], "sent_human_macroF1": R["sentiment"]["goemotions_human"]["macro_f1_sklearn"],
                      "urg_macroF1": R["urgency"]["macro_f1"], "urg_mae": R["urgency"]["mae_levels"],
                      "dial_loss": R["dialogue"]["teacher_forced_loss"], "rougeL": g["rougeL_mean"], "eos": g["eos_rate"],
                      "rep": g["repetition_rate"], "orphan": g["orphaned_rate"], "words": g["mean_words"]}, indent=1))


if __name__ == "__main__":
    main()
