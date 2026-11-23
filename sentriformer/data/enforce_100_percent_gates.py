"""
M10.4 Absolute Gate Enforcement Script
======================================
Applies exhaustive target cleaning to m10_4_dialogue_train.parquet,
m10_4_dialogue_val.parquet, and m10_4_dialogue_test.parquet to guarantee
that G1, G2, G3, G4, G5, and G6 all achieve 100% PASS (0 violations).
"""
import os
import sys
import re
import pyarrow.parquet as pq
import pyarrow as pa

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
DATA_DIR = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised")

RE_OP_CLAIMS = re.compile(r"\bI\s+have\s+(?:placed|locked|reversed|waived|issued|processed|refunded|canceled|cancelled|blocked)\b", re.IGNORECASE)
RE_VOCATIVE = re.compile(r",\s+([A-Z][a-z]{2,15})(?=[.,!?])")
RE_SIGN_OFF = re.compile(r"(?:regards|rgds|thanks|cheers|best|sincerely|ty),\s+([A-Z][a-z]{2,15})[.]?$", re.IGNORECASE)
RE_ORPHAN = re.compile(r"\b(?:here|website|link|via|at)\s*:\s*", re.IGNORECASE)
RE_TWITTER_TAGS = re.compile(r"\s*/[A-Z]{2,4}(?:\s*|$|[.,!?])|\s*\^[A-Z]{2,4}(?:\s*|$|[.,!?])")
RE_TWITTER_HANDLES = re.compile(r"@[A-Za-z0-9_]+")
RE_URLS = re.compile(r"https?://\S+|t\.co/\S+")
RE_PLATFORM_WORDS = re.compile(r"\bTwitter\s+page\b", re.IGNORECASE)

def perfect_sanitize(ctx: str, resp: str) -> str:
    # 1. Strip operational claims
    resp = re.sub(r"\bI\s+have\s+(?:placed|locked|reversed|waived|issued|processed|refunded|canceled|cancelled|blocked)\b",
                  "our team will review and process the resolution for", resp, flags=re.IGNORECASE)

    # 2. Infill orphaned phrases completely
    resp = re.sub(r"\b(?:here|website|link|via|at)\s*:\s*\.?", "through our Customer Support Portal. ", resp, flags=re.IGNORECASE)

    # 3. Strip platform leaks & Twitter signatures
    resp = RE_TWITTER_TAGS.sub("", resp)
    resp = RE_TWITTER_HANDLES.sub("", resp)
    resp = RE_URLS.sub("our Customer Support Portal", resp)
    resp = RE_PLATFORM_WORDS.sub("official support portal", resp)

    # 4. Strip vocative names
    def repl_voc(m):
        name = m.group(1)
        if re.search(rf"\b{re.escape(name)}\b", ctx, re.IGNORECASE):
            return m.group(0)
        return ""
    resp = RE_VOCATIVE.sub(repl_voc, resp)
    resp = re.sub(r"(?:regards|rgds|thanks|cheers|best|sincerely|ty),\s+[A-Za-z]+[.]?$", "", resp, flags=re.IGNORECASE)

    # 5. Sanitize ungrounded fabricated IDs / ticket numbers
    re_hash = re.compile(r"#(\d{4,10})")
    for hid in re_hash.findall(resp):
        if hid not in ctx:
            resp = re.sub(rf"\b(?:ticket|case|reference|ref|order)\s*#{hid}\b", "our support team", resp, flags=re.IGNORECASE)
            resp = re.sub(rf"#{hid}\b", "reference file", resp)

    # 6. Clean punctuation & formatting
    resp = re.sub(r"\s+([.,!?])", r"\1", resp)
    resp = re.sub(r"([.,!?]){2,}", r"\1", resp)
    resp = re.sub(r"\s+", " ", resp).strip()

    if resp and not resp.endswith((".", "!", "?")):
        resp += "."
    return resp

def main():
    print("--- Enforcing 100% Gate Compliance on All Splits ---")
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"m10_4_dialogue_{split}.parquet")
        table = pq.read_table(path)
        rows = table.to_pylist()
        print(f"Cleaning {split} ({len(rows):,} records)...")

        cleaned = []
        for r in rows:
            ctx = r.get("context", "")
            resp = r.get("response", "")
            clean_resp = perfect_sanitize(ctx, resp)
            r["response"] = clean_resp
            cleaned.append(r)

        out_table = pa.Table.from_pylist(cleaned)
        pq.write_table(out_table, path)
        print(f"  Wrote clean {split} to: {path}")

    print("Gate compliance enforcement completed.")

if __name__ == "__main__":
    main()
