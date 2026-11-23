"""
SentriFormer v2 - bounded, staged multi-task training (gradient-based; Phase T, after WSP Phase 0).

Root cause being fixed: M10.2/M10.3 trained the shared encoder with ONLY the generation loss while the category/CSAT
heads were frozen, so the encoder drifted away from the heads (E0: -34 pp routing). Here every task's loss reaches the
shared encoder in every step, so representations stay compatible with all heads.

Tasks per optimizer step (one mini-batch each, summed weighted loss):
  category  classification_train (fixed-seed subset), 33 classes, CE + label smoothing
  csat      csat_train, CE with inverse-sqrt-frequency class weights
  sentiment wsp_runs/data/sentiment_train (distilled + GoEmotions human), 3 classes, class-weighted CE
  urgency   wsp_runs/data/urgency_train (932 rows), MSE on sigmoid(urgency_proj) vs label/3
  dialogue  m10_4_dialogue_train, causal decoder CE (EOS appended, as in DialogueSFTDataset)
Optimisation: AdamW, separate LR for backbone vs heads, cosine schedule, fp16 autocast + GradScaler, clip 1.0.
Bounded: hard --max_steps and --max_minutes; resumable (state saved every eval). Writes only wsp_runs/train/<run>/.

Usage: python -I wsp/train_v2.py --run NAME --init <ckpt> [--max_steps N] [--max_minutes M] [--benchmark 100]
"""
import argparse, itertools, json, math, os, random, sys, time, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask, pool_encoder_states
from sentriformer.training.classification_dataset import ClassificationDataCollator
from sentriformer.training.train_csat import CSATDataset, CSATCollator
from sentriformer.training.train_dialogue_sft import DialogueSFTDataset, DialogueSFTCollator

SUP = os.path.join(ROOT, "data", "supervised")
AUX = os.path.join(ROOT, "wsp_runs", "data")


def log(m, f=None):
    s = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(s, flush=True)
    if f: f.write(s + "\n"); f.flush()


class TextLabelDataset(Dataset):
    """text -> token ids + integer/float label (used for category subset, sentiment, urgency)."""
    def __init__(self, texts, labels, tok, max_len):
        self.t, self.y, self.tok, self.L = texts, labels, tok, max_len
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        return {"token_ids": self.tok.encode(self.t[i], truncation=True, max_length=self.L, add_special_tokens=True) or [1], "label_id": self.y[i]}


class FloatLabelCollator(ClassificationDataCollator):
    def __call__(self, batch):
        out = super().__call__([{"token_ids": b["token_ids"], "label_id": 0} for b in batch])
        out["labels"] = torch.tensor([b["label_id"] for b in batch], dtype=torch.float32)
        return out


def category_subset(n, seed, tok, label2id, max_len, split="train"):
    # Stream row groups and keep only sampled rows: peak RAM ~ one batch + the subset, not the whole 2.2M-row table
    # (the full read is what coincided with the low-memory kill of run T2).
    pf = pq.ParquetFile(os.path.join(SUP, f"classification_{split}.parquet"))
    N = pf.metadata.num_rows
    keep = np.sort(np.random.default_rng(seed).choice(N, n, replace=False)) if n and n < N else np.arange(N)
    texts, labels, off, j = [], [], 0, 0
    for batch in pf.iter_batches(batch_size=65536, columns=["text", "product"]):
        m = batch.num_rows; lo = j
        while j < len(keep) and keep[j] < off + m: j += 1
        if j > lo:
            sub = batch.take(keep[lo:j] - off)
            texts += sub.column(0).to_pylist(); labels += [label2id[p] for p in sub.column(1).to_pylist()]
        off += m
    assert len(texts) == len(keep)
    return TextLabelDataset(texts, labels, tok, max_len)


def aux(name, split, tok, max_len, limit=None, seed=0):
    d = pq.read_table(os.path.join(AUX, f"{name}_{split}.parquet")).to_pandas()
    if limit and len(d) > limit: d = d.sample(limit, random_state=seed)
    return TextLabelDataset(d.text.tolist(), d.label.tolist(), tok, max_len)


def class_weights(labels, k):
    c = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k).astype(float)
    w = np.where(c > 0, 1.0 / np.sqrt(np.maximum(c, 1)), 0.0); w = w / w[c > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


def forever(loader):
    while True:
        for b in loader: yield b


@torch.inference_mode()
def quick_eval(model, loaders, cfg, device):
    """Small, fixed val subsets for model selection during training (full eval is separate)."""
    from sklearn.metrics import f1_score
    model.eval(); r = {}
    for task, key in (("category", "category_logits"), ("csat", "csat_logits"), ("sentiment", "sentiment_logits")):
        P, Y = [], []
        for b in loaders[task]:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            P.append(o[key].float().argmax(-1).cpu()); Y.append(b["labels"])
        P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        r[task] = {"acc": float((P == Y).mean()), "macro_f1": float(f1_score(Y, P, average="macro", zero_division=0))}
    P, Y = [], []
    for b in loaders["urgency"]:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
        P.append(o["urgency_score"].float().cpu()); Y.append(b["labels"])
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    pc, yc = np.clip(np.rint(P * 3), 0, 3).astype(int), Y.astype(int)
    r["urgency"] = {"acc": float((pc == yc).mean()), "macro_f1": float(f1_score(yc, pc, average="macro", zero_division=0)),
                    "mae_levels": float(np.abs(P * 3 - Y).mean())}
    tot, n = 0.0, 0
    for b in loaders["dialogue"]:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), generation_labels=b["labels"].to(device))
        k = int((b["labels"] != -100).sum()); tot += o["generation_loss"].float().item() * k; n += k
    r["dialogue"] = {"val_loss": tot / n, "ppl": math.exp(tot / n)}
    # selection score frozen in wsp_runs/train/TRAINING_CRITERIA.md
    r["selection_score"] = (r["category"]["acc"] + r["csat"]["macro_f1"] + r["sentiment"]["macro_f1"] + r["urgency"]["macro_f1"]
                            - 0.1 * r["dialogue"]["val_loss"])
    model.train()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--max_steps", type=int, default=20000); ap.add_argument("--max_minutes", type=float, default=150)
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps, report s/step + VRAM, save nothing")
    ap.add_argument("--lr_backbone", type=float, default=3e-5); ap.add_argument("--lr_heads", type=float, default=3e-4)
    ap.add_argument("--bs_cat", type=int, default=24); ap.add_argument("--bs_csat", type=int, default=8)
    ap.add_argument("--bs_sent", type=int, default=16); ap.add_argument("--bs_urg", type=int, default=8)
    ap.add_argument("--bs_dial", type=int, default=16)
    ap.add_argument("--w_cat", type=float, default=1.0); ap.add_argument("--w_csat", type=float, default=0.3)
    ap.add_argument("--w_sent", type=float, default=0.3); ap.add_argument("--w_urg", type=float, default=0.5)
    ap.add_argument("--w_dial", type=float, default=1.0)
    ap.add_argument("--cat_len", type=int, default=256); ap.add_argument("--n_cat_train", type=int, default=400_000)
    ap.add_argument("--csat_every", type=int, default=1); ap.add_argument("--urg_every", type=int, default=1)
    ap.add_argument("--reinit_heads", default="", help="comma list: category,csat,sentiment")
    ap.add_argument("--gate_step", type=int, default=0); ap.add_argument("--gate_cat_acc", type=float, default=0.0)
    ap.add_argument("--cat_label_smoothing", type=float, default=0.0, help="NOTE: with class weights, PyTorch weights the smoothing term too -> collapses to rare classes")
    ap.add_argument("--dial_train", default=os.path.join(SUP, "m10_4_dialogue_train.parquet")); ap.add_argument("--dial_val", default=os.path.join(SUP, "m10_4_dialogue_val.parquet"))
    ap.add_argument("--dial_ctx_len", type=int, default=128)
    ap.add_argument("--head_warmup", type=int, default=1000, help="steps with backbone frozen");
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = os.path.join(ROOT, "wsp_runs", "train", a.run)
    resume = os.path.join(out, "state_last.pt")
    if not a.benchmark:
        os.makedirs(out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=2)
    lf = open(os.path.join(out, "train_log.txt"), "a", encoding="utf-8") if not a.benchmark else None
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
    li = json.load(open(os.path.join(SUP, "classification_labels.json"), encoding="utf-8"))
    cfg = SentriFormerConfig.from_pretrained(os.path.join(ROOT, "checkpoints", "config.json"))
    cfg.num_categories, cfg.num_csat_classes = li["num_classes"], 5

    model = SentriFormerForMultiTask(cfg)
    raw = torch.load(a.init if os.path.isabs(a.init) else os.path.join(ROOT, a.init), map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw, strict=True); del raw
    for hname in filter(None, a.reinit_heads.split(",")):
        for mod in getattr(model, f"{hname}_head").modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.normal_(mod.weight, 0.0, cfg.initializer_range)
                if mod.bias is not None: torch.nn.init.zeros_(mod.bias)
        log(f"re-initialised {hname}_head", lf)
    model.to(device).train()

    # ---- data ----
    log("loading data", lf)
    cat_tr = category_subset(a.n_cat_train if not a.benchmark else 20000, a.seed, tok, li["label2id"], a.cat_len)
    csat_tr = CSATDataset(os.path.join(SUP, "csat_train.parquet"), tok, max_len=128)
    sent_tr = aux("sentiment", "train", tok, 128, limit=None if not a.benchmark else 20000)
    urg_tr = aux("urgency", "train", tok, 128)
    dial_tr = DialogueSFTDataset(a.dial_train if os.path.isabs(a.dial_train) else os.path.join(ROOT, a.dial_train), tok, max_ctx_len=a.dial_ctx_len, max_resp_len=128,
                                 limit=None if not a.benchmark else 20000)
    w_cat = class_weights(cat_tr.y, cfg.num_categories)
    w_csat = class_weights(csat_tr.labels, 5); w_sent = class_weights(sent_tr.y, 3)
    cc = ClassificationDataCollator(0, a.cat_len)
    L = {
        "category": forever(DataLoader(cat_tr, batch_size=a.bs_cat, shuffle=True, collate_fn=cc, drop_last=True)),
        "csat": forever(DataLoader(csat_tr, batch_size=a.bs_csat, shuffle=True, collate_fn=CSATCollator(0), drop_last=True)),
        "sentiment": forever(DataLoader(sent_tr, batch_size=a.bs_sent, shuffle=True, collate_fn=ClassificationDataCollator(0, 128), drop_last=True)),
        "urgency": forever(DataLoader(urg_tr, batch_size=a.bs_urg, shuffle=True, collate_fn=FloatLabelCollator(0, 128), drop_last=True)),
        "dialogue": forever(DataLoader(dial_tr, batch_size=a.bs_dial, shuffle=True, collate_fn=DialogueSFTCollator(pad_token_id=0), drop_last=True)),
    }
    EV = {  # fixed val subsets for selection
        "category": DataLoader(category_subset(10000, 123, tok, li["label2id"], a.cat_len, split="val"), batch_size=128, collate_fn=cc),
        "csat": DataLoader(CSATDataset(os.path.join(SUP, "csat_val.parquet"), tok, max_len=128), batch_size=64, collate_fn=CSATCollator(0)),
        "sentiment": DataLoader(aux("sentiment", "val", tok, 128, limit=5000, seed=123), batch_size=128, collate_fn=ClassificationDataCollator(0, 128)),
        "urgency": DataLoader(aux("urgency", "val", tok, 128), batch_size=64, collate_fn=FloatLabelCollator(0, 128)),
        "dialogue": DataLoader(torch.utils.data.Subset(DialogueSFTDataset(a.dial_val if os.path.isabs(a.dial_val) else os.path.join(ROOT, a.dial_val), tok, a.dial_ctx_len, 128), list(range(0, 28000, 14))),
                               batch_size=64, collate_fn=DialogueSFTCollator(pad_token_id=0)),
    }
    log(f"data: cat={len(cat_tr)} csat={len(csat_tr)} sent={len(sent_tr)} urg={len(urg_tr)} dial={len(dial_tr)}", lf)

    head_params = [p for n, p in model.named_parameters() if n.startswith(("category_head", "csat_head", "sentiment_head"))]
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr_backbone}, {"params": head_params, "lr": a.lr_heads}], weight_decay=0.01, betas=(0.9, 0.98))
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(500, a.max_steps // 10), num_training_steps=a.max_steps)
    scaler = torch.amp.GradScaler("cuda")
    step, best = 0, -1e9
    hist = []
    if os.path.exists(resume) and not a.benchmark:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); scaler.load_state_dict(st["scaler"])
        step, best, hist = st["step"], st["best"], st["hist"]
        log(f"resumed at step {step}", lf)

    t0 = time.time(); step0 = step; torch.cuda.reset_peak_memory_stats()
    run_steps = a.benchmark or a.max_steps
    run_loss = {k: 0.0 for k in ("cat", "csat", "sent", "urg", "dial")}
    def set_backbone(trainable):
        for p in backbone: p.requires_grad_(trainable)
    set_backbone(step >= a.head_warmup)
    if step < a.head_warmup: log(f"head warm-up: backbone frozen until step {a.head_warmup}", lf)
    while step < run_steps:
        if step == a.head_warmup:
            set_backbone(True); log(f"step {step}: backbone unfrozen", lf)
        if not a.benchmark and (time.time() - t0) / 60 > a.max_minutes:
            log(f"time budget reached at step {step}", lf); break
        with torch.amp.autocast("cuda", dtype=torch.float16):
            b = next(L["category"]); o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            l_cat = F.cross_entropy(o["category_logits"].float(), b["labels"].to(device), weight=w_cat.to(device), label_smoothing=a.cat_label_smoothing)
            zero = torch.zeros((), device=device)
            l_csat = zero
            if step % a.csat_every == 0:
                b = next(L["csat"]); o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                l_csat = F.cross_entropy(o["csat_logits"].float(), b["labels"].to(device), weight=w_csat.to(device))
            b = next(L["sentiment"]); o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            l_sent = F.cross_entropy(o["sentiment_logits"].float(), b["labels"].to(device), weight=w_sent.to(device))
            l_urg = zero
            if step % a.urg_every == 0:
                b = next(L["urgency"]); o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                l_urg = F.mse_loss(o["urgency_score"].float(), b["labels"].to(device) / 3.0)
            b = next(L["dialogue"]); o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), generation_labels=b["labels"].to(device))
            l_dial = o["generation_loss"].float()
            loss = a.w_cat * l_cat + a.w_csat * l_csat + a.w_sent * l_sent + a.w_urg * l_urg + a.w_dial * l_dial
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(); step += 1
        for k, v in zip(run_loss, (l_cat, l_csat, l_sent, l_urg, l_dial)): run_loss[k] += v.item()
        if step % 100 == 0:
            el = time.time() - t0
            log(f"step {step} " + " ".join(f"{k}={v/100:.3f}" for k, v in run_loss.items()) + f" | {el/max(step - step0, 1):.3f}s/step | vram_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", lf)
            run_loss = {k: 0.0 for k in run_loss}
        if not a.benchmark and (step % a.eval_every == 0 or step == run_steps):
            r = quick_eval(model, EV, cfg, device); r["step"] = step; r["minutes"] = (time.time() - t0) / 60; hist.append(r)
            log(f"EVAL step {step}: cat={r['category']['acc']:.4f} csatF1={r['csat']['macro_f1']:.4f} sentF1={r['sentiment']['macro_f1']:.4f} "
                f"urgF1={r['urgency']['macro_f1']:.4f} dial_loss={r['dialogue']['val_loss']:.4f} score={r['selection_score']:.4f}", lf)
            if r["selection_score"] > best:
                best = r["selection_score"]
                torch.save({"model_state_dict": model.state_dict(), "train_run": a.run, "step": step, "quick_eval": r}, os.path.join(out, "model_best.pt"))
                log("  new best -> model_best.pt", lf)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "best": best, "hist": hist}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
            json.dump(hist, open(os.path.join(out, "eval_history.json"), "w"), indent=2)
            if a.gate_step and step >= a.gate_step and r["category"]["acc"] < a.gate_cat_acc:
                log(f"GATE FAIL at step {step}: category acc {r['category']['acc']:.4f} < {a.gate_cat_acc}; stopping", lf); break
    if a.benchmark:
        el = time.time() - t0
        print(json.dumps({"benchmark_steps": step, "sec_per_step": el / max(step - step0, 1), "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30}))
    else:
        torch.save({"model_state_dict": model.state_dict(), "train_run": a.run, "step": step}, os.path.join(out, "model_last.pt"))
        log(f"finished at step {step}; best score {best:.4f}", lf)


if __name__ == "__main__":
    main()
