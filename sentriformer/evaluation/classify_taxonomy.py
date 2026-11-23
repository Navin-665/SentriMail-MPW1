import json
import re

path = r"sentriformer\results\m10_3_audit\m10_3_threeway_audit.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data.get("cases", [])

# Known names in English customer service that might be hallucinated
COMMON_NAMES = {"ryan", "karen", "john", "sarah", "mike", "david", "lisa", "alex", "chris", "emma"}

classifications = []

for c in cases:
    qid = c["inquiry_id"]
    group = c["category_group"]
    q = c["query"]
    resp = c["m103"]["response"].strip()
    audit = c["m103"]["audit"]
    safety = audit.get("safety", {})

    tags = []

    # Check Orphaned Phrase
    is_orphaned = safety.get("has_orphaned_phrase", False) or bool(re.search(r"\b(here|website|at|link)\s*:\s*(?:$|[A-Z]|so|and)", resp, re.I)) or resp.endswith(("at", "at:", "link:", "here:", "website:"))
    if is_orphaned:
        tags.append("Orphaned Phrase")

    # Check Hallucination (names or platform references)
    resp_words_lower = [w.strip(".,!?:;\"'") for w in resp.lower().split()]
    q_words_lower = [w.strip(".,!?:;\"'") for w in q.lower().split()]
    hallucinated_names = [n for n in COMMON_NAMES if n in resp_words_lower and n not in q_words_lower]
    has_platform_hallucination = "twitter" in resp.lower() or "/jx" in resp.lower()
    if hallucinated_names or has_platform_hallucination:
        tags.append("Hallucination")

    # Check Incomplete Action / Cutoff
    is_incomplete = not audit["completeness_proxy"] or resp.endswith(("through our support portal", "our customer support team", "link:", "at")) and not resp.endswith((".", "!", "?"))
    if is_incomplete:
        tags.append("Incomplete Action")

    # Check Generic / Canned Deflection
    is_generic = any(phrase in resp.lower() for phrase in [
        "reach out to us directly",
        "reach out to our customer support",
        "support portal",
        "contact our customer support team",
        "send us a note at",
        "we're here to help",
        "take a closer look",
        "look into this for you"
    ]) and audit["intent_relevance_proxy"] < 34.0

    # Intent Evaluation
    # High relevance >= 60%
    if audit["intent_relevance_proxy"] >= 60.0:
        if len(resp.split()) >= 15 and not is_orphaned:
            tags.append("Correct Intent + Useful")
        else:
            tags.append("Correct Intent + Weak")
    elif 25.0 <= audit["intent_relevance_proxy"] < 60.0:
        tags.append("Correct Intent + Weak")
    else:
        if is_generic:
            tags.append("Generic Deflection")
        else:
            tags.append("Wrong Intent")

    # Empathy check
    # Queries with frustration markers: charged, cancel, lost, declined, unauthorized, locked, delay, fee, damaged, waiting, why
    is_frustrated_query = any(w in q.lower() for w in [
        "charged", "cancel", "lost", "declined", "unauthorized", "locked", "delay", 
        "fee", "damaged", "wait", "dispute", "stuck", "frustrat", "trouble", "double", "why"
    ])
    if is_frustrated_query and not audit["has_empathy"]:
        tags.append("Empathy Failure")
    elif "glad to hear that" in resp.lower() and is_frustrated_query:
        tags.append("Tone Mismatch (Empathy Failure)")

    classifications.append({
        "id": qid,
        "group": group,
        "query": q,
        "response": resp,
        "relevance": audit["intent_relevance_proxy"],
        "tags": tags,
        "orphaned": is_orphaned,
        "hallucinated_names": hallucinated_names,
        "platform_leak": has_platform_hallucination,
        "empathy": audit["has_empathy"]
    })

# Summary counts
category_counts = {
    "1. Correct intent + useful response": 0,
    "2. Correct intent + weak response": 0,
    "3. Wrong intent": 0,
    "4. Generic response / canned deflection": 0,
    "5. Orphaned phrase": 0,
    "6. Hallucination": 0,
    "7. Incomplete action": 0,
    "8. Empathy failure": 0
}

for item in classifications:
    tags = item["tags"]
    if "Correct Intent + Useful" in tags:
        category_counts["1. Correct intent + useful response"] += 1
    if "Correct Intent + Weak" in tags:
        category_counts["2. Correct intent + weak response"] += 1
    if "Wrong Intent" in tags:
        category_counts["3. Wrong intent"] += 1
    if "Generic Deflection" in tags:
        category_counts["4. Generic response / canned deflection"] += 1
    if "Orphaned Phrase" in tags:
        category_counts["5. Orphaned phrase"] += 1
    if "Hallucination" in tags:
        category_counts["6. Hallucination"] += 1
    if "Incomplete Action" in tags:
        category_counts["7. Incomplete action"] += 1
    if "Empathy Failure" in tags or "Tone Mismatch (Empathy Failure)" in tags:
        category_counts["8. Empathy failure"] += 1

print("=== M10.3 Failure Taxonomy Distribution (N=80) ===")
for k, v in category_counts.items():
    print(f"{k:45s}: {v:2d} ({v/80*100:5.1f}%)")
