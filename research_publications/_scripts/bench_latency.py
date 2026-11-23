"""Per-stage inference latency of the SentriMail AI pipeline (read-only; no app/DB is started).
Usage: python -I bench_latency.py <sentriformer_dir> <out_json>
Inputs: 30 benchmark complaint messages written for this evaluation (below); no corpus text.
"""
import json, sys, time, statistics, importlib.util, os
import torch

SF, OUT = sys.argv[1], sys.argv[2]
MSGS = [
    "My card was charged twice for the same order and I still have not received a refund after ten days.",
    "The mobile app logs me out every time I try to view my statement. Please fix this.",
    "I was promised a call back yesterday about my mortgage escrow, nobody called.",
    "Package shows delivered but it never arrived at my address. Order placed two weeks ago.",
    "Thank you for resolving my issue so quickly, the new router works perfectly.",
    "A debt collector keeps calling me about a loan I already paid off in full last year.",
    "Why is there a late fee on my account when I paid before the due date?",
    "I cannot reset my password, the verification code email never comes.",
    "My student loan servicer applied my payment to the wrong loan and now I am marked delinquent.",
    "The product I received is damaged and the return label link does not work.",
    "Someone opened a credit card in my name. I need this removed from my credit report immediately.",
    "Your website checkout keeps failing with an error at the payment step.",
    "I have been on hold for over an hour trying to cancel my subscription.",
    "The interest rate on my auto loan is different from what was agreed at signing.",
    "Could you tell me when my replacement card will be shipped?",
    "I transferred money to a friend and it has been pending for four days.",
    "The agent I spoke to was rude and hung up on me twice.",
    "My account was frozen without any notice and I cannot pay my rent.",
    "Great service today, the representative explained everything clearly.",
    "I keep receiving marketing emails even after unsubscribing three times.",
    "The credit bureau refuses to correct an address that is not mine despite my dispute letters.",
    "My order was cancelled automatically and I was not told why.",
    "I was charged an overdraft fee even though I opted out of overdraft coverage.",
    "The prepaid card balance disappeared after the last update to your system.",
    "I want to file a formal complaint about repeated billing errors on my statement over the past three months, "
    "including duplicate charges, missing credits for returned items, and a fee that customer service promised to "
    "waive but never did. I have called five times and each time I was told it would be fixed in the next cycle.",
    "Refund still not processed.",
    "My payday loan lender is withdrawing more than the agreed amount from my bank account every week.",
    "The delivery driver left my parcel in the rain and everything inside is ruined.",
    "Please update the phone number on my account, I cannot do it online.",
    "I received a notice of foreclosure even though I am enrolled in the loan modification program and have made "
    "every trial payment on time. Nobody at the servicer can explain this and I am worried about losing my home.",
]


def timeit(fn, reps):
    ts = []
    for m in MSGS:
        for _ in range(reps):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t = time.perf_counter(); fn(m)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"n": len(ts), "mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts),
            "p95_ms": ts[int(.95 * (len(ts) - 1))], "min_ms": ts[0], "max_ms": ts[-1]}


spec = importlib.util.spec_from_file_location("sf_inference", os.path.join(SF, "wsp", "inference.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from transformers import pipeline

R = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     "cpu_threads": torch.get_num_threads(), "n_messages": len(MSGS),
     "message_words": [min(len(m.split()) for m in MSGS), max(len(m.split()) for m in MSGS)], "results": {}}
for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
    t = time.perf_counter(); sf = mod.SentriFormerPipeline(device=dev); load = (time.perf_counter() - t) * 1e3
    sf.analyze(MSGS[0])  # warm-up
    gen_lens = []
    def full(m): gen_lens.append(len(sf.analyze(m)["response"].split()))
    r = {"sentriformer_load_ms": load,
         "sentriformer_heads_plus_1_token": timeit(lambda m: sf.analyze(m, max_new_tokens=1), 2),
         "sentriformer_full_analyze_greedy100": timeit(full, 2)}
    r["sentriformer_mean_reply_words"] = statistics.mean(gen_lens)
    di = 0 if dev == "cuda" else -1
    sst = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=di)
    emo = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=di)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tk = AutoTokenizer.from_pretrained("google/flan-t5-small")
    t5m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(dev).eval()
    def t5(prompt, **kw):  # same settings as SentriMail ai_engine (sampled, top_p 0.95, max_length 150)
        with torch.no_grad():
            return t5m.generate(**tk(prompt, return_tensors="pt", truncation=True).to(dev), **kw)
    sst(MSGS[0]); emo(MSGS[0]); t5("x", max_length=5)
    r["distilbert_sst2_sentiment"] = timeit(lambda m: sst(m[:512]), 2)
    r["distilroberta_emotion"] = timeit(lambda m: emo(m[:512]), 2)
    r["flan_t5_small_reply_sampled"] = timeit(lambda m: t5("Write a polite customer support reply to: " + m,
                                                            do_sample=True, top_p=0.95, max_length=150), 1)
    R["results"][dev] = r
    del sf, sst, emo, t5m
    if dev == "cuda": torch.cuda.empty_cache()
    print(dev, json.dumps({k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT)
