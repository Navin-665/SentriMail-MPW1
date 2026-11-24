"""Compare original vs Adaption-Labs-adapted replies with SentriFormer's own reply detectors (local only).
Usage: python -I score_adapted.py <sentriformer_dir> <original.csv> <adapted.csv|.jsonl> [<out_dir>]
Matches rows by `id` when present (else by customer_context), auto-detects the adapted completion column,
and writes a per-row CSV with flags plus a JSON summary.
"""
import json, os, re, sys
import pandas as pd

SF, ORIG, ADAPT = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(ADAPT))
sys.path.insert(0, os.path.join(SF, "wsp_runs", "t5")); sys.path.insert(0, os.path.join(SF, "wsp"))
import t5_audit_v5 as V          # noqa: E402  broken_link / generic_whole / privacy detectors used in training
from eval_t4 import ACTION       # noqa: E402

PORTAL = re.compile(r"\b(dm|direct message|private message|message us|reach out to|contact (us|our)|visit|go to|"
                    r"support (portal|page|site)|website|web ?site|online at|call (us|our)|via (dm|email|phone|chat))\b", re.I)
URL = re.compile(r"(https?://|www\.|\.com\b|\{\{|\}\}|\[[A-Z_ ]+\])", re.I)
NUM = re.compile(r"[$£€]?\d[\d,.:/]*")

o = pd.read_csv(ORIG)
a = pd.read_json(ADAPT, lines=True) if ADAPT.endswith(".jsonl") else pd.read_csv(ADAPT)
print("adapted columns:", list(a.columns))
cand = [c for c in a.columns if re.search(r"(adapt|evolv|new|improv|rewrit|output|completion|response|answer)", c, re.I)
        and c not in ("customer_context", "prompt")]
ccol = next((c for c in cand if c not in ("agent_reply",)), "completion" if "completion" in a.columns else "agent_reply")
key = "id" if "id" in a.columns and "id" in o.columns else ("customer_context" if "customer_context" in a.columns else None)
if key is None:
    pcol = next(c for c in a.columns if re.search("prompt", c, re.I)); a = a.rename(columns={pcol: "customer_context"}); key = "customer_context"
print("using adapted column:", ccol, "| join key:", key)
m = o.merge(a[[key, ccol]].rename(columns={ccol: "adapted_reply"}), on=key, how="inner")


def score(ctx, reply):
    reply = str(reply or "")
    f = V.flags(str(ctx), reply)
    nums_new = [n for n in NUM.findall(reply) if n.strip("$£€") not in str(ctx)]
    return {"broken_link": f["broken_link"], "generic_whole": f["generic_whole"],
            "privacy": f["privacy_secret"] or f["privacy_bulk_contact"], "portal_redirect": bool(PORTAL.search(reply)),
            "url_or_placeholder": bool(URL.search(reply)), "invented_number": bool(nums_new),
            "action_or_question": bool(ACTION.search(reply)) or "?" in reply, "words": len(reply.split())}


rows = []
for _, r in m.iterrows():
    so, sa = score(r.customer_context, r.agent_reply), score(r.customer_context, r.adapted_reply)
    rows.append({"id": r.get("id"), "problem": r.get("problem"), **{f"orig_{k}": v for k, v in so.items()}, **{f"new_{k}": v for k, v in sa.items()}})
d = pd.DataFrame(rows)
metrics = ["broken_link", "generic_whole", "portal_redirect", "url_or_placeholder", "invented_number", "privacy", "action_or_question"]
summary = {"rows_matched": len(d), "rows_original": len(o), "rows_adapted": len(a),
           "original": {k: round(float(d[f"orig_{k}"].mean()) * 100, 1) for k in metrics},
           "adapted": {k: round(float(d[f"new_{k}"].mean()) * 100, 1) for k in metrics},
           "mean_words": {"original": round(float(d.orig_words.mean()), 1), "adapted": round(float(d.new_words.mean()), 1)},
           "by_problem_fixed_pct": {}}
bad = ["broken_link", "generic_whole", "portal_redirect", "url_or_placeholder", "invented_number", "privacy"]
d["new_clean"] = ~d[[f"new_{k}" for k in bad]].any(axis=1) & d.new_action_or_question & (d.new_words.between(5, 60))
summary["adapted_clean_and_usable_pct"] = round(float(d.new_clean.mean()) * 100, 1)
for p, g in d.groupby("problem"):
    summary["by_problem_fixed_pct"][p] = round(float(g.new_clean.mean()) * 100, 1)
os.makedirs(OUT, exist_ok=True)
d.to_csv(os.path.join(OUT, "adapted_scores_LOCAL_ONLY.csv"), index=False)
json.dump(summary, open(os.path.join(OUT, "adapted_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
