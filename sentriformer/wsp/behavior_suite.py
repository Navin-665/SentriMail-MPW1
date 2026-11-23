"""
WSP behaviour suite: hand-written complaints (not drawn from any dataset split) run through the inference pipeline
for the original M10.3 checkpoint and the final candidate. Deterministic (greedy) decoding so the two are comparable.
'expected' is a coarse set of acceptable labels, used only to summarise routing; it is a qualitative probe, not a benchmark.
Usage: python -I wsp/behavior_suite.py <final_ckpt> <out_dir>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import SentriFormerPipeline

CR = {"Credit reporting", "Credit reporting or other personal consumer reports", "Credit reporting, credit repair services, or other personal consumer reports"}
CARD = {"Credit card", "Credit card or prepaid card", "Prepaid card"}
BANK = {"Checking or savings account", "Bank account or service"}
LOAN = {"Payday loan", "Payday loan, title loan, or personal loan", "Payday loan, title loan, personal loan, or advance loan", "Consumer Loan"}
ECOM = {"Returns", "Order Related", "Refund Related", "Cancellation", "Product Queries", "Payments related", "Shopzilla Related", "Others", "Feedback"}
MONEY = {"Money transfer, virtual currency, or money service", "Money transfers", "Virtual currency"}

CASES = [
    ("short", "Charged twice.", CARD | BANK | ECOM),
    ("short", "Mortgage escrow wrong.", {"Mortgage"}),
    ("short", "Where is my refund?", ECOM),
    ("short", "Debt collector keeps calling me.", {"Debt collection"}),
    ("typical", "There is an account on my credit report that I never opened. I disputed it with the bureau twice and it is still there.", CR),
    ("typical", "My mortgage servicer applied my payment late even though I paid on the 1st, and now they charged me a late fee and reported me.", {"Mortgage"}),
    ("typical", "A debt collector is calling my workplace about a medical bill I already paid. They refuse to send validation of the debt.", {"Debt collection"}),
    ("typical", "My student loan servicer put my loans into forbearance without asking me and interest capitalized.", {"Student loan"}),
    ("typical", "I sent money through the app to the wrong person and the company says they cannot reverse it.", MONEY),
    ("typical", "The bank closed my checking account without notice and is holding my remaining balance.", BANK),
    ("typical", "My credit card company won't remove a fraudulent charge from a merchant I've never heard of.", CARD),
    ("typical", "The dealership financed my car at a higher rate than we agreed and the lender won't fix it.", {"Vehicle loan or lease"}),
    ("typical", "I returned the shoes three weeks ago and still haven't got my money back.", ECOM),
    ("typical", "My order says delivered but I never received the package.", ECOM),
    ("typical", "I want to cancel my order before it ships.", ECOM),
    ("ambiguous", "I'm really unhappy with how I've been treated and I want this fixed now.", None),
    ("ambiguous", "Something is wrong with my account and nobody will tell me what.", None),
    ("ambiguous", "The charge on my statement doesn't look right.", CARD | BANK | ECOM),
    ("multi-issue", "I have contacted your team four times. My payment was deducted, but my order was never delivered. Nobody is helping me.", ECOM | CARD | BANK),
    ("multi-issue", "My payday lender took two withdrawals from my checking account in one week, causing overdraft fees, and now a collector is calling.", LOAN | BANK | {"Debt collection"}),
    ("multi-issue", "My identity was stolen: someone opened a credit card in my name, it went to collections, and it is on my credit report.", CR | CARD | {"Debt collection"}),
    ("long", "I am writing about my mortgage. " + "In March the servicer transferred my loan without notice, then lost my escrow records, paid my insurance twice, and raised my monthly payment. " * 12, {"Mortgage"}),
    ("long", "Credit report dispute. " + "Equifax and TransUnion keep reporting a collection account that was deleted last year; my disputes are returned as verified without investigation. " * 40, CR | {"Debt collection"}),
    ("edge", "", None),
]


def main():
    final_ckpt, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=False)
    rows, agg = [], {}
    for tag, ck in (("m10_3_original", "checkpoints/m10_3_sft/m10_3_sft_best.pt"), ("final", final_ckpt)):
        pipe = SentriFormerPipeline(ck)
        a = agg.setdefault(tag, {"n": 0, "routing_ok": 0, "routing_scored": 0, "triage": 0, "eos": 0, "repetition": 0, "placeholder": 0, "orphaned": 0, "invented_contact_or_url": 0, "mean_conf": 0.0, "mean_tokens": 0.0})
        for i, (kind, text, expected) in enumerate(CASES):
            r = pipe.analyze(text, greedy=True)
            ok = None if expected is None else r["category"]["label"] in expected
            rows.append({"model": tag, "case": i, "kind": kind, "input_preview": text[:140], "expected": sorted(expected) if expected else None, "routing_ok": ok, **r})
            a["n"] += 1; a["mean_conf"] += r["category"]["confidence"]; a["mean_tokens"] += len(r["response"].split())
            if ok is not None: a["routing_scored"] += 1; a["routing_ok"] += int(ok)
            q = r["quality_flags"]
            a["triage"] += r["category"]["needs_human_triage"]; a["eos"] += q["eos_terminated"]; a["repetition"] += q["repetition"]
            a["placeholder"] += q["template_placeholder"]; a["orphaned"] += q["orphaned_channel_phrase"]; a["invented_contact_or_url"] += q["invented_contact"] or q["invented_url"]
        a["mean_conf"] /= a["n"]; a["mean_words"] = a.pop("mean_tokens") / a["n"]
        del pipe
    json.dump({"aggregate": agg, "cases": rows}, open(os.path.join(out_dir, "behavior_suite.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    L = ["# Behaviour suite (24 hand-written complaints, greedy decoding)", "", "| Metric | m10_3 original | final |", "|---|---|---|"]
    for k in agg["final"]:
        f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        L.append(f"| {k} | {f(agg['m10_3_original'][k])} | {f(agg['final'][k])} |")
    L += ["", "| # | Kind | Input | Expected | m10_3 category (conf) | final category (conf) | final CSAT | final response |", "|---|---|---|---|---|---|---|---|"]
    byc = {(r["model"], r["case"]): r for r in rows}
    for i, (kind, text, exp) in enumerate(CASES):
        o, f = byc[("m10_3_original", i)], byc[("final", i)]
        esc = lambda s: s.replace("|", "/").replace("\n", " ")
        L.append(f"| {i} | {kind} | {esc(text[:80])}{'…' if len(text) > 80 else ''} | {'any' if not exp else esc(', '.join(sorted(exp))[:60])} | "
                 f"{esc(o['category']['label'])} ({o['category']['confidence']:.2f}){' ✓' if o['routing_ok'] else (' ✗' if o['routing_ok'] is False else '')} | "
                 f"{esc(f['category']['label'])} ({f['category']['confidence']:.2f}){' ✓' if f['routing_ok'] else (' ✗' if f['routing_ok'] is False else '')} | "
                 f"{f['csat']['stars']}★ | {esc(f['response'][:160])} |")
    open(os.path.join(out_dir, "behavior_suite.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
