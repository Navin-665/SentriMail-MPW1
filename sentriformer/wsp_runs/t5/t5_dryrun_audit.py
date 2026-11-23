"""T5 DRY-RUN audit (READ ONLY): how many T4d TRAINING rows each proposed T5 filter would flag, by intent category, with
examples. Writes only wsp_runs/t5/t5_dryrun_audit.json. No dataset is written, no model is loaded, no val/test rows are read."""
import json, os, re, sys, collections
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp")); sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.join(ROOT, "wsp_runs", "sentriwsp01"))
from eval_t4 import orphan, ACTION
from phase1_investigate import GENERIC, content, split_cur

VERSION = "v4"   # v4: bare "need/require" is not a request; looser 'invited' check (customer mentioned the item)   # v1 first dry run; v2 review fixes A (address noun / same-clause) + B (postal PIN);
# v3 spot-check fixes: loyalty/membership card numbers are not sensitive; DOB moved to the contact tier (uninvited check);
# phone number as the telecom service identifier and name/address in account-opening flows are legitimate.
OUT = os.path.join(HERE, f"t5_dryrun_audit_{VERSION}.json"); SAMPLE = os.path.join(HERE, f"t5_precision_sample_{VERSION}.csv")
# B: authentication PINs only. "pin code"/"pincode"/"PIN code of the store" is a postal code (India) and is NOT sensitive.
LOYALTY = re.compile(r"(nectar|loyalty|reward|rewards|membership|member|club|gift|clubcard|boots|advantage|points|frequent flyer|library|student|nus)\s*$", re.I)
SENSITIVE = re.compile(r"\b(password|passcode|(account|card|atm|debit|credit|bank|security|secret) pin|your pin\b(?!\s*-?\s*code)|pin number|"
                       r"one[- ]time (code|password|pin)|otp|social security( number)?|ssn|((full|complete|long|16[- ]digit) )?((credit|debit|bank|payment) )?card number|"
                       r"cvv|cvc|security code)\b", re.I)
# A: noun forms only (address as a noun with a possessive/qualifier, never the verb "to address"); v3: DOB here (uninvited check)
CONTACT = re.compile(r"\b(your (full )?name|full name|(your|home|mailing|postal|billing|shipping|delivery|street) address|e-?mail address|your e-?mail|"
                     r"(your )?(phone|mobile|cell|contact) (number|no\.?|#)|phone #|zip( ?code)?|postcode|postal code|date of birth|dob)\b", re.I)
PHONE_ITEM = re.compile(r"\b(phone|mobile|cell|contact) (number|no\.?|#)|phone #", re.I)
TELECOM = re.compile(r"\b(sim|network|signal|recharge|prepaid|postpaid|port(ed|ing)?|airtel|idea|vodafone|jio|tariff|data pack|mobile number|"
                     r"affected number|registered (mobile )?number|calls?|sms|roaming|coverage)\b", re.I)
ONBOARD = re.compile(r"\b(open(ing)?|creat(e|ing)|set(ting)? up|register(ing)?|sign(ing)? up for|apply(ing)? for)\b[^.?!]{0,40}\baccount\b", re.I)


def sensitive_hit(clause):
    for mt in SENSITIVE.finditer(clause):
        if "card number" in mt.group(0).lower() and LOYALTY.search(clause[: mt.start()].rstrip()): continue   # "your Nectar card number"
        return True
    return False


def contact_hit(clause, ctx):
    for mt in CONTACT.finditer(clause):
        if PHONE_ITEM.fullmatch(mt.group(0).lower().replace("your ", "")) and TELECOM.search(ctx + " " + clause): continue  # number = service id
        if ONBOARD.search(ctx) and re.search(r"name|address", mt.group(0), re.I): continue                              # account opening
        return True
    return False
ASK = re.compile(r"\b(send|share|provide|confirm|give|dm|include|tell|reply with|let us (know|have)|(we|i)( will| would)? (need|require)( your| the)?)\b", re.I)
INVITED = re.compile(r"\b(e-?mail|address|phone|mobile( number)?|cell|name|details|profile|birth|dob|zip|postcode|postal code)\b", re.I)


def clauses(t):
    """A: text after each request verb, cut at 40 characters or the first sentence boundary (. ? ! ;), whichever comes first."""
    for m in ASK.finditer(t): yield re.split(r"[.?!;]", t[m.end(): m.end() + 40], maxsplit=1)[0]


def ask_for(rx, t, ctx=""):
    for cl in clauses(t):
        if rx is SENSITIVE and sensitive_hit(cl): return True
        if rx is CONTACT and contact_hit(cl, ctx): return True
        if rx not in (SENSITIVE, CONTACT) and rx.search(cl): return True
    return False


def _selftest():
    assert not ask_for(CONTACT, "We'll share this with our team to address as a training opportunity.")       # A: verb "address"
    assert ask_for(CONTACT, "Please send your name/zip/phone #/account type.")
    assert ask_for(CONTACT, "Can you DM us your email address?")
    assert not ask_for(CONTACT, "Please send us a DM. Your address is on file.")                               # A: crosses a sentence
    assert not ask_for(SENSITIVE, "Please share the PIN code of the store so we can check.")                  # B: postal PIN
    assert not ask_for(SENSITIVE, "Kindly share your pincode and order ID.")                                   # B: postal PIN
    assert ask_for(SENSITIVE, "Please confirm your card PIN to proceed.") and ask_for(SENSITIVE, "Please send your password.")
    assert ask_for(SENSITIVE, "Please provide the long card number on the front.")
    assert not ask_for(SENSITIVE, "Thanks Paul, can you send us your Nectar card number via the link below?")   # v3: loyalty card
    assert ask_for(SENSITIVE, "Can you send us your debit card number?")
    assert not ask_for(CONTACT, "Please provide the affected mobile number for checks.", "my sim has no network")  # v3: telecom id
    assert ask_for(CONTACT, "Pls send us your phone #/Zip and we will connect with you.", "Why am I charged $12 on savings?")
    assert not ask_for(CONTACT, "Please provide your full name and address.", "I want to open a savings account for my dad")
    assert not flags("Customer: the date of birth is 13.8.1941, proceed",
                     "Kindly provide the correct date of birth for guest 2.")["privacy_contact_uninvited"]           # invited by context
    assert flags("Customer: I was charged twice", "Please send us your date of birth and full name.")["privacy_contact_uninvited"]
    assert not flags("Customer: move my login from my old address to my new email",
                     "If you need to transfer your login email address, visit Security Settings.")["privacy_contact_uninvited"]  # v4
    assert flags("Customer: my order is late", "We will need your full name and phone number to look into it.")["privacy_contact_uninvited"]


def flags(ctx, reply):
    earlier, cur = split_cur(ctx); words = reply.split()
    f = {}
    # generic: formulaic phrase AND no task information (no content-word overlap with the customer's turn, no concrete action/question)
    f["generic_no_info"] = bool(GENERIC.search(reply)) and not (content(reply) & content(cur)) and not ACTION.search(reply)
    f["generic_phrase_any"] = bool(GENERIC.search(reply))
    # privacy: sensitive data requested -> always flagged; contact PII requested and NOT raised by the customer anywhere in the context
    f["privacy_sensitive"] = ask_for(SENSITIVE, reply)
    f["privacy_contact_uninvited"] = ask_for(CONTACT, reply, ctx) and not INVITED.search(ctx)
    # broken: orphan token rule, empty or under 3 words, ends with ':' or a dangling connector
    f["broken"] = (bool(orphan(reply)) or len(words) < 3 or reply.rstrip().endswith(":")
                   or bool(re.search(r"\b(at|to|via|here|and|so|the|our|your)\s*[.:]?\s*$", reply.strip(), re.I)))
    return f


DROP = ("broken", "privacy_sensitive", "privacy_contact_uninvited", "generic_no_info")


def main():
    import csv, hashlib, random
    assert not os.path.exists(OUT) and not os.path.exists(SAMPLE)
    src = os.path.join(ROOT, "wsp_runs", "t4", "data", "t4d_train.parquet")
    t = pq.read_table(src, columns=["id", "context", "response", "intent_category", "split"])
    df = t.to_pandas(); assert set(df.split) == {"train"}, set(df.split)
    tot, by_cat, flagged = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(list)
    combo, any_drop, cats = collections.Counter(), 0, collections.Counter(df.intent_category)
    for i, (ctx, rep, cat) in enumerate(zip(df.context, df.response, df.intent_category)):
        f = flags(ctx, rep); hit = tuple(k for k in DROP if f[k])
        if hit: any_drop += 1; combo[" + ".join(hit)] += 1
        for k, v in f.items():
            if v: tot[k] += 1; by_cat[k][cat] += 1; flagged[k].append(i)
    n = len(df)
    R = {"version": VERSION, "rows": n, "source": src, "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
         "flag_counts": {k: {"rows": v, "share": round(v / n, 4)} for k, v in tot.items()},
         "union_of_drop_filters": {"rows": any_drop, "share": round(any_drop / n, 4), "kept": n - any_drop},
         "overlap_combinations": dict(combo.most_common()),
         "removal_share_by_category": {c: round(sum(by_cat[k][c] for k in DROP) / cats[c], 3) for c, _ in cats.most_common(15)},
         "by_category_share": {k: {c: round(by_cat[k][c] / cats[c], 3) for c, _ in cats.most_common(12)} for k in tot}}
    # precision sample: 100 flagged rows per DROP filter (all if fewer), fixed seed, for manual labelling
    rng = random.Random(2027)
    with open(SAMPLE, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["sample_id", "filter", "row_id", "also_flagged_by", "customer_turn", "reply",
                                        "true_failure (yes/no)", "note"])
        for k in DROP:
            pick = sorted(rng.sample(flagged[k], min(100, len(flagged[k]))))
            for j, i in enumerate(pick):
                f = flags(df.context.iat[i], df.response.iat[i])
                w.writerow([f"{k[:3].upper()}{j + 1:03d}", k, df.id.iat[i], ",".join(x for x in DROP if f[x] and x != k),
                            split_cur(df.context.iat[i])[1][-300:], df.response.iat[i], "", ""])
    R["precision_sample"] = {"file": os.path.basename(SAMPLE), "seed": 2027, "per_filter": {k: min(100, len(flagged[k])) for k in DROP}}
    json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("rows", "flag_counts", "union_of_drop_filters", "overlap_combinations", "precision_sample")}, indent=1))


if __name__ == "__main__":
    _selftest()   # detector unit tests run before any data is read
    main()
