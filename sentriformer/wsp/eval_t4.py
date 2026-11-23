"""
T4 dialogue evaluation (inference only). Same fixed rows for every model; each model gets its native input format:
  --fmt raw   : current customer message only (T3 / Phase-0 native)
  --fmt ctrl  : "Customer: <message>"
  --fmt multi : role-labelled history + "Customer: <message>"
Metrics: teacher-forced loss (all val rows; rows with history), greedy generation on 300 fixed rows that HAVE history,
portal-loop / deflection / orphan / EOS / repetition / ROUGE-L / distinct-2, and a context-correctness set where an earlier
customer turn contains an identifier (redacted to {{Order Number}}): re-ask rate + acknowledgement proxy.
Usage: python -I wsp/eval_t4.py --ckpt PATH --name NAME --fmt raw|ctrl|multi
Writes wsp_runs/t4/eval/<name>_val/. Validation only (test is sealed during T4).
"""
import argparse, json, math, os, re, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import SentriFormerPipeline
from eval_full import rouge_l
from sentriformer.training.train_dialogue_sft import DialogueSFTCollator
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask, detect_repetition

DATA = os.path.join(ROOT, "wsp_runs", "t4", "data")
REDIRECT = re.compile(r"(support portal|our portal|support team|customer (support|service|care)( team)?\b|contact (us|our)|reach out|raise a ticket|"
                      r"submit a (ticket|request)|send us a (note|message|dm)|\bdm us\b|direct message|private message|give us a call|call us|message us)", re.I)
ACTION = re.compile(r"(\b(share|provide|send|confirm|include) (us )?(with )?(your|the) ([a-z]+ ){0,3}(number|id|details|email|name|address|screenshot|error)\b|\b(could|can|would) you (please )?(confirm|share|provide|tell|send|let us know|check|try)\b|\bplease (confirm|share|provide|tell us|let us know (the|your|what|which|when|if))\b|"
                    r"\b(what|which|when|where|how long|is it|are you|did you|have you)\b[^.?!]*\?|"
                    r"\b(try|restart|reinstall|update|reset|log ?in|sign in|go to (settings|your account|the app|the [a-z]+ app)|tap|select|"
                    r"check (your|the) (email|inbox|statement|settings|order history|spam|account)|freeze|lock|cancel|unplug|clear (the |your )?cache|turn (it )?off)\b)", re.I)
REASK = re.compile(r"\b(order|account|ticket|reference|tracking|confirmation|case|transaction) (number|no\.?|#|id)\b|\border #", re.I)
ACK = re.compile(r"\b(your|the|this) (order|account|ticket|reference|tracking|case|transaction)\b|\{\{order number\}\}|order number you (provided|shared|sent)", re.I)
IDENT = re.compile(r"(#\s?\d{4,}|\b\d{6,}\b|\b[A-Z]{2,}\d{5,}\b)")


def portal_loop(t): return bool(REDIRECT.search(t)) and not ACTION.search(t)


def orphan(t):
    w = re.findall(r"[A-Za-z']+:?|[.!,]", t.lower())
    for a, b in zip(w, w[1:]):
        if a in ("at", "via") and b in ("so", "and", "to", "for", "if", ".", "!", ","): return True
        if a == "here:" or (a == "here" and b in ("so", "and")): return True
    return bool(w) and w[-1] in ("at", "via", "here:")


def _selftest():
    assert portal_loop("Please reach out to us through our Customer Support Portal so we can look into this for you.")
    assert portal_loop("We're sorry to hear that. Please contact our customer support team so we can look into this further.")
    assert not portal_loop("Please contact us and share your order number so we can check.")
    assert not portal_loop("Try restarting your device. If that doesn't help, contact us.")
    assert not portal_loop("Can you confirm the error message you see?")
    assert not portal_loop("I'm sorry your refund is delayed. Which payment method did you use?")
    assert REASK.search("Could you send us your order number?") and not REASK.search("Thanks for sharing that.")
    assert orphan("Please send us a note at so we can help.") and not orphan("We're here to help.")
_selftest()


def fmt_ctx(row, fmt):
    if fmt == "multi": return row["context"]
    cur = re.split(r"(?=Customer: )", row["context"])[-1]           # the current customer message (last Customer: segment)
    return cur if fmt == "ctrl" else cur[len("Customer: "):]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--fmt", required=True, choices=["raw", "ctrl", "multi"])
    ap.add_argument("--split", default="val", choices=["val", "test"]); a = ap.parse_args()  # test: ONLY for the single final run
    od = os.path.join(ROOT, "wsp_runs", "t4", "eval", f"{a.name}_{a.split}"); assert not os.path.exists(od), od; os.makedirs(od)
    pipe = SentriFormerPipeline(a.ckpt, "cuda"); model, tok, cfg = pipe.model, pipe.tok, pipe.cfg
    df = pq.read_table(os.path.join(DATA, f"multi_{a.split}.parquet")).to_pandas()  # multi file holds full context; other formats derived from it
    df["inp"] = [fmt_ctx(r, a.fmt) for _, r in df.iterrows()]
    R = {"ckpt": a.ckpt, "fmt": a.fmt}

    def tf_loss(sub):
        tot = n = 0
        items = []
        for c, r in zip(sub.inp, sub.response):
            ids = tok.encode(c, add_special_tokens=True)[-256:] or [1]   # left-truncate: keep the most recent context
            rid = tok.encode(r, truncation=True, max_length=127, add_special_tokens=True)
            if not rid or rid[-1] != cfg.eos_token_id: rid = rid + [cfg.eos_token_id]
            items.append({"input_ids": ids, "labels": rid})
        with torch.inference_mode():
            for i in range(0, len(items), 64):
                b = DialogueSFTCollator(0)(items[i:i + 64])
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda(), generation_labels=b["labels"].cuda())
                k = int((b["labels"] != -100).sum()); tot += o["generation_loss"].float().item() * k; n += k
        return tot / n
    hist = df[df.n_history_segments > 0]
    R["loss_all_val"] = tf_loss(df); R["loss_rows_with_history"] = tf_loss(hist); R["n_val"] = len(df); R["n_with_history"] = len(hist)

    def gen(c):
        ids = tok.encode(c, add_special_tokens=True)[-256:] or [1]
        t = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            g = generate_multitask(model, tok, cfg, t, torch.ones_like(t), max_new_tokens=128, temperature=0.0, repetition_penalty=1.2)[0].tolist()
        eos = cfg.eos_token_id in g; body = g[1:g.index(cfg.eos_token_id)] if eos else g[1:]
        return tok.decode(body, skip_special_tokens=True).strip(), eos

    rows = hist.sample(300, random_state=11)                         # fixed: same rows for every model
    out, F = [], {"portal_loop": 0, "redirect_any": 0, "orphan": 0, "eos": 0, "repetition": 0}
    rl, uni, bi, tw, tb = [], set(), set(), 0, 0
    for _, r in rows.iterrows():
        txt, eos = gen(r.inp); w = txt.lower().split()
        F["portal_loop"] += portal_loop(txt); F["redirect_any"] += bool(REDIRECT.search(txt)); F["orphan"] += orphan(txt)
        F["eos"] += eos; F["repetition"] += detect_repetition(txt); rl.append(rouge_l(txt, r.response))
        uni.update(w); bi.update(zip(w, w[1:])); tw += len(w); tb += max(len(w) - 1, 0)
        out.append({"group_id": r.group_id, "input": r.inp, "reference": r.response, "generated": txt})
    R["generation_300_with_history"] = {**{f"{k}_rate": v / 300 for k, v in F.items()}, "rougeL": float(np.mean(rl)),
                                        "distinct_2": len(bi) / max(tb, 1), "mean_words": tw / 300}
    # reference-side portal-loop rate on the same rows (how often the TRAINING TARGET itself is a portal loop)
    R["generation_300_with_history"]["reference_portal_loop_rate"] = float(np.mean([portal_loop(x) for x in rows.response]))

    # context-correctness: earlier customer turn has an identifier -> redact -> does the reply ask for it again?
    has_id = hist[hist.context.apply(lambda c: bool(IDENT.search(" ".join(re.split(r"(?=Customer: )", c)[:-1]))))]
    cc = has_id.sample(min(300, len(has_id)), random_state=12)
    reask = ack = 0; cc_out = []
    for _, r in cc.iterrows():
        inp = IDENT.sub("{{Order Number}}", r.inp)
        txt, _ = gen(inp); reask += bool(REASK.search(txt)) and "{{order number}}" not in txt.lower(); ack += bool(ACK.search(txt))
        cc_out.append({"input": inp, "generated": txt})
    R["context_correctness"] = {"n": len(cc), "reask_rate": reask / max(len(cc), 1), "ack_proxy_rate": ack / max(len(cc), 1),
                                "note": "identifier given in an EARLIER customer turn (redacted); ack proxy is weak - human A/B decides"}
    json.dump(out, open(os.path.join(od, "generations.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(cc_out, open(os.path.join(od, "context_correctness.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(R, open(os.path.join(od, "metrics.json"), "w"), indent=2)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
