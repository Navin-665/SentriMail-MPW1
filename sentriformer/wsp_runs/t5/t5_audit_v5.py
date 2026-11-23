"""T5 detector v5 (READ ONLY audit; precision over coverage). v4 rules/outputs are kept unchanged in t5_dryrun_audit.py / *_v4.*.
Writes t5_audit_v5.json and the LOCAL-ONLY human labelling sheet t5_precision_sample_v5_LOCAL_ONLY.csv (contains TWCS text:
never upload it to any external service). No dataset is written, no model loaded, no val/test row read."""
import csv, hashlib, json, os, random, re, sys, collections
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp")); sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.join(ROOT, "wsp_runs", "sentriwsp01"))
from eval_t4 import ACTION
from phase1_investigate import GENERIC, split_cur

VERSION, SEED = "v5", 2028
OUT = os.path.join(HERE, f"t5_audit_{VERSION}.json"); SAMPLE = os.path.join(HERE, f"t5_precision_sample_{VERSION}_LOCAL_ONLY.csv")

# ---- broken: ONLY unambiguous missing-link / dangling-destination patterns (the broad orphan rule is dropped) ----
BROKEN = re.compile(
    r"(\b(at|via)\s+(for|and|so|to|if|or|they|we)\b"                        # "track online at for updates", "call us at and" (not "at our store", "at with this")
    r"|\b(at|via)\s*[.;](?!\d)"                                             # "call the Rewards Dept directly at." / "via." (not "look at, or" / "looking at?")
    r"|\b(here|link|below|following|website)\s*:\s*([.,;!?]|$|and\b|so\b|let\b|to\b|we\b|for\b)"   # "steps listed here: let us know", "here:."
    r"|:\s*[.!?]?\s*$)", re.I)                                              # reply ends with a dangling colon ("...if this helps:.")

# ---- generic: the WHOLE reply is template (every sentence is filler or a GENERIC phrase), >=1 GENERIC, no action, no question ----
FILLER = re.compile(r"^\W*(hi|hello|hey|hiya|thanks?( so much| again| for [^.!?]{0,40})?|thank you( so much| for [^.!?]{0,40})?|"
                    r"(we('| a)re|i('| a)m) (so |very |really )?(sorry|glad|happy)[^.!?]{0,40}|(so )?sorry[^.!?]{0,40}|"
                    r"(we|i) (do )?apologi[sz]e[^.!?]{0,40}|apologies[^.!?]{0,40}|(we|i) understand[^.!?]{0,30}|"
                    r"(that'?s )?(great|awesome|wonderful|good to hear)[^.!?]{0,30}|(we('| a)re|i('| a)m) here to help|"
                    r"have a (great|good|nice|lovely) (day|evening|weekend|one)|cheers|regards|take care|"
                    r"((ty|thanks|thx|cheers|best|regards|many thanks),?\s*)?-?\s*\^?[a-z]{1,12}( [a-z]{1,12})?( \d/\d)?\.?)\W*$", re.I)  # sign-offs: "Ty, Andy 3/3."

def sentences(t): return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

def generic_whole(reply):
    ss = sentences(reply)
    return (bool(ss) and any(GENERIC.search(s) for s in ss) and all(GENERIC.search(s) or FILLER.match(s) for s in ss)
            and "?" not in reply and not ACTION.search(reply))

# ---- privacy: (a) authentication secrets; (b) bulk personal-detail collection not raised by the customer ----
ASK = re.compile(r"\b(send|share|provide|confirm|give|dm|include|tell|reply with|let us (know|have)|(we|i)( will| would)? (need|require)( your| the)?)\b", re.I)
SECRET = re.compile(r"\b(password|passcode|one[- ]time (code|password|pin)|otp|verification code|(card|atm|debit|credit|bank|account|security) pin|"
                    r"cvv|cvc|(full|complete|long|16[- ]digit) (credit |debit |bank )?card number|(credit|debit) card number|social security( number)?|ssn)\b", re.I)
ITEM = re.compile(r"\b(name|zip( ?code)?|postcode|postal code|phone( number| #)?|mobile number|address|e-?mail( address)?|date of birth|dob)\b", re.I)
INVITED = re.compile(r"\b(e-?mail|address|phone|mobile|cell|name|details|profile|birth|dob|zip|postcode|postal code)\b", re.I)
EXEMPT = re.compile(r"\b(open(ing)?|creat(e|ing)|register(ing)?|sign(ing)? up|apply(ing)? for|updat(e|ing)|chang(e|ing)|subscri(be|ption)|newsletter)\b", re.I)

def clauses(t, width):
    for m in ASK.finditer(t): yield re.split(r"[.?!;]", t[m.end(): m.end() + width], maxsplit=1)[0]

def secret_request(reply): return any(SECRET.search(c) for c in clauses(reply, 40))

def bulk_contact(ctx, reply):   # >=2 distinct personal items in one request clause, none raised by the customer, not an onboarding/update flow
    if INVITED.search(ctx) or EXEMPT.search(ctx): return False
    return any(len({m.group(1).lower().split()[0] for m in ITEM.finditer(c)}) >= 2 for c in clauses(reply, 70))

def flags(ctx, reply):
    return {"broken_link": bool(BROKEN.search(reply)), "generic_whole": generic_whole(reply),
            "privacy_secret": secret_request(reply), "privacy_bulk_contact": bulk_contact(ctx, reply)}

DROP = ("broken_link", "generic_whole", "privacy_secret", "privacy_bulk_contact")


def _selftest():
    B = lambda r: flags("Customer: x", r)["broken_link"]
    assert B("Please continue to track your package online at for updates.") and B("I kindly ask that you call the Rewards Dept. directly at. They are happy to help.")
    assert B("Follow the steps listed here and let us know if this helps:.") and B("Check the steps here: let us know if it works.")
    assert not B("We're here to help. What device are you using?") and not B("Turn it off and on.") and not B("Call us at 1-800-555-0100.")
    assert not B("Is it an upgrade you want to look at, or take advantage of the BOGO offer?")
    assert not B("You can find more details by visiting our website.") and not B("What are you looking at?")
    assert not B("We'll pass this on to the team at our Culverhouse Cross store.") and not B("We understand where you're at with this.")
    assert B("For now, try:.") and B("You can check out our coverage areas at We're expanding quickly.")
    G = lambda r: flags("Customer: x", r)["generic_whole"]
    assert G("I'm sorry for the inconvenience caused. If you need anything else please let me know. Ty, Andy 3/3.")
    assert G("We're sorry to hear that. We'll pass your feedback on to our team.")
    assert not G("We're afraid that isn't possible yet when using the mobile app. Don't worry, we'll pass your feedback onto the team.")
    assert not G("In the new interface head to Keep Watching > All Watch History. We'll pass along your feedback.")
    assert not G("Hi Lauren, can you let me know if this was online or in store? I can pass this on.")
    S = lambda r: flags("Customer: x", r)["privacy_secret"]
    assert S("Please send us your password so we can check.") and S("Can you share the one-time code you received?") and S("Please provide your full card number.")
    assert not S("You can reset your password from the login page.") and not S("Can you send us your Nectar card number?")
    assert not S("Please share the PIN code of the store.") and not S("Kindly share your pincode and order ID.")
    C = lambda ctx, r: flags(ctx, r)["privacy_bulk_contact"]
    assert C("Customer: why am I charged $12 on savings?", "Pls click the link below to send us your name/zip/phone # and we will connect.")
    assert C("Customer: my order is late", "We will need your full name, address and phone number to look into it.")
    assert not C("Customer: my order is late", "Please send your order number.")
    assert not C("Customer: I want to open an account for my dad", "Please provide your full name and address.")
    assert not C("Customer: my email is wrong on file", "Please confirm your name and email address.")
    assert not C("Customer: subscribe to the newsletter", "Please share your name and email address.")


def main():
    assert not os.path.exists(OUT) and not os.path.exists(SAMPLE)
    src = os.path.join(ROOT, "wsp_runs", "t4", "data", "t4d_train.parquet")
    df = pq.read_table(src, columns=["id", "context", "response", "intent_category", "split"]).to_pandas(); assert set(df.split) == {"train"}
    tot, by_cat, flagged, combo, union = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(list), collections.Counter(), 0
    cats = collections.Counter(df.intent_category)
    for i, (ctx, rep, cat) in enumerate(zip(df.context, df.response, df.intent_category)):
        f = flags(ctx, rep); hit = tuple(k for k in DROP if f[k])
        if hit: union += 1; combo[" + ".join(hit)] += 1
        for k in hit: tot[k] += 1; by_cat[k][cat] += 1; flagged[k].append(i)
    n = len(df)
    R = {"version": VERSION, "rows": n, "source": src, "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
         "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
         "flag_counts": {k: {"rows": tot[k], "share": round(tot[k] / n, 4)} for k in DROP},
         "union_of_drop_filters": {"rows": union, "share": round(union / n, 4), "kept": n - union}, "overlap_combinations": dict(combo.most_common()),
         "removal_share_by_category": {c: round(sum(by_cat[k][c] for k in DROP) / cats[c], 3) for c, _ in cats.most_common(15)}}
    rng = random.Random(SEED)
    with open(SAMPLE, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["LOCAL ONLY - contains TWCS text - do not upload to any external service (ChatGPT, etc.)"] + [""] * 10)
        w.writerow(["sample_id", "filter", "row_id", "also_flagged_by", "customer_turn", "reply", "human_label (yes/no)", "reviewer",
                    "second_review (yes/no; REQUIRED for privacy_secret)", "second_reviewer", "adjudicated_label"])
        for k in DROP:
            pick = sorted(rng.sample(flagged[k], min(100, len(flagged[k]))))
            pre = {"broken_link": "BRK", "generic_whole": "GEN", "privacy_secret": "SEC", "privacy_bulk_contact": "CON"}[k]
            for j, i in enumerate(pick):
                f = flags(df.context.iat[i], df.response.iat[i])
                w.writerow([f"{pre}{j + 1:03d}", k, df.id.iat[i], ",".join(x for x in DROP if f[x] and x != k),
                            split_cur(df.context.iat[i])[1][-300:], df.response.iat[i], "", "", "", "", ""])
    R["precision_sample"] = {"file": os.path.basename(SAMPLE), "seed": SEED, "per_filter": {k: min(100, len(flagged[k])) for k in DROP},
                             "protocol": "human labels only; privacy_secret double-reviewed; model suggestions (if any) kept in a separate file"}
    json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("flag_counts", "union_of_drop_filters", "overlap_combinations", "precision_sample")}, indent=1))


if __name__ == "__main__":
    _selftest(); main()
