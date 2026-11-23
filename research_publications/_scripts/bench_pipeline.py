"""Run SentriMail's integrated analyze_complaint() (read-only, no DB/app) on the benchmark messages.
Usage: python -I -B bench_pipeline.py <sentrimail_app_dir> <out_json>
Replicates main.py's post-analysis keyword escalation and the LOW -> auto-reply branch to count decisions.
"""
import ast, json, os, sys, time, statistics, importlib.util, collections

APP, OUT = sys.argv[1], sys.argv[2]
src = open(os.path.join(os.path.dirname(__file__), "bench_latency.py"), encoding="utf-8").read()
MSGS = ast.literal_eval(src.split("MSGS = ", 1)[1].split("\n]\n", 1)[0] + "\n]")
KW = ["legal", "police", "harassment", "abuse", "threat", "court", "violence", "urgent", "emergency", "lawsuit", "assault"]  # main.py L345
BLOCK = ["refund", "chargeback", "legal", "fraud", "threat", "injury", "security breach", "data leak"]  # ai_engine.py L356

spec = importlib.util.spec_from_file_location("ai_engine", os.path.join(APP, "ai_engine.py"))
ae = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae)
t = time.perf_counter(); ae.analyze_complaint("warm up message about a late delivery"); first = (time.perf_counter() - t) * 1e3
rows, ts = [], []
for m in MSGS:
    t = time.perf_counter(); a = ae.analyze_complaint(m); ts.append((time.perf_counter() - t) * 1e3)
    pr = a["priority"].upper(); esc = any(k in m.lower() for k in KW)
    final = "CRITICAL" if esc else pr
    sf = a.get("sentriformer") or {}
    rows.append({"priority_engine": pr, "final_priority": final, "keyword_escalated": esc,
                 "auto_replied": final == "LOW", "auto_resolvable": a["auto_resolvable"],
                 "has_blocker_term": any(b in m.lower() for b in BLOCK), "sentiment": a["sentiment_label"],
                 "emotion": a["emotion_label"], "sf_category": sf.get("category"), "sf_triage": sf.get("needs_human_triage"),
                 "sf_csat": sf.get("csat_stars"), "sf_flags": sf.get("reply_flags"), "sf_present": bool(sf)})
ts.sort()
C = collections.Counter
R = {"n": len(MSGS), "first_call_incl_model_load_ms": first,
     "latency_ms": {"mean": statistics.mean(ts), "median": statistics.median(ts), "p95": ts[int(.95 * (len(ts) - 1))], "max": ts[-1]},
     "sentriformer_present": sum(r["sf_present"] for r in rows),
     "final_priority": C(r["final_priority"] for r in rows), "engine_priority": C(r["priority_engine"] for r in rows),
     "keyword_escalated": sum(r["keyword_escalated"] for r in rows), "auto_replied": sum(r["auto_replied"] for r in rows),
     "auto_replied_but_not_auto_resolvable": sum(r["auto_replied"] and not r["auto_resolvable"] for r in rows),
     "auto_replied_with_blocker_term": sum(r["auto_replied"] and r["has_blocker_term"] for r in rows),
     "sentiment": C(r["sentiment"] for r in rows), "emotion": C(r["emotion"] for r in rows),
     "sf_triage_flagged": sum(bool(r["sf_triage"]) for r in rows), "sf_csat": C(r["sf_csat"] for r in rows),
     "sf_category_top": C(r["sf_category"] for r in rows).most_common(8), "rows": rows}
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k != "rows"}, default=str, indent=1))
