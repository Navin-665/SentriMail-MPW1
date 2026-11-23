"""
==========================================================================
  M10.3 — 4-Tier Enterprise Response Dataset Synthesizer & Refinement
==========================================================================
  Building on M10.1 clean_dialogue splits, this pipeline:

  1. REPAIRS orphaned prepositions ("reach out at [blank]", "here: so we",
     "via ", "contact us at ") → semantically complete alternatives.
  2. PURGES residual contamination:
     - @handles, URLs, t.co, DM deflections, staff signatures,
     - off-domain / non-customer-support noise,
     - vague deflection-only responses ("we'll look into it")
  3. SYNTHESIZES 4-tier enterprise responses for 10 core categories:
     Tier 1 — Acknowledgment & Empathy
     Tier 2 — Explicit Intent Recognition
     Tier 3 — Actionable Guidance (no fabricated specifics)
     Tier 4 — Resolution Path
  4. DEDUPLICATES near-identical responses via min-hash/Jaccard.
  5. OUTPUTS clean M10.3 splits:
     - m10_3_dialogue_train.parquet
     - m10_3_dialogue_val.parquet
     - m10_3_dialogue_test.parquet

  Architecture & multi-task heads remain FROZEN (M9.5 multitask_best.pt).
==========================================================================
"""
import os
import sys
import re
import json
import random
import datetime
import hashlib
from collections import Counter, defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)
sys.stdout.reconfigure(encoding="utf-8")

# ── Input: M10.1 cleaned splits ──
CLEAN_TRAIN = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_train.parquet")
CLEAN_VAL   = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_val.parquet")
CLEAN_TEST  = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised", "clean_dialogue_test.parquet")

# ── Output: M10.3 refined splits ──
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "data", "supervised")
M103_TRAIN = os.path.join(OUTPUT_DIR, "m10_3_dialogue_train.parquet")
M103_VAL   = os.path.join(OUTPUT_DIR, "m10_3_dialogue_val.parquet")
M103_TEST  = os.path.join(OUTPUT_DIR, "m10_3_dialogue_test.parquet")

REPORT_PATH = os.path.join(ROOT_DIR, "sentriformer", "results", "m10_3_dataset_report.json")

random.seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════
def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: Contamination & Orphan Detection Regexes
# ══════════════════════════════════════════════════════════════════════════
RE_HANDLE = re.compile(r"@[A-Za-z0-9_]+")
RE_URL = re.compile(r"https?://\S+|t\.co/\S+|www\.\S+")
RE_SIGNATURE = re.compile(
    r"(?:(?:^|\s)[\^*~][A-Za-z]{2,8}(?:\s+[\^*~][A-Za-z]{2,8})*)[\s.]*$",
    re.MULTILINE
)
# Phone patterns: standard, toll-free (1-800-XXX), extensions
RE_PHONE = re.compile(
    r"\b(?:1[\-.]?8[0-9]{2}[\-.]?[A-Z0-9]{3}[\-.]?[A-Z0-9]{4}"
    r"|1[\-.]?8[0-9]{2}[\-.]?[A-Z]+-[A-Z]+"
    r"|\d{3}[\-.\s]?\d{3}[\-.\s]?\d{4}"
    r"|\d{3}[\-.\s]?\d{3}[\-.]?\d{4,5})"
    r"(?:\s*,?\s*ext\.?\s*\d+)?"
    r"\b", re.IGNORECASE
)
RE_DM_PATTERNS = [
    re.compile(r"\b(?:please\s+)?(?:send\s+(?:us\s+)?a\s+)?dm\b", re.IGNORECASE),
    re.compile(r"\b(?:direct\s+message|dm\s+us|reach\s+out\s+via\s+dm)\b", re.IGNORECASE),
    re.compile(r"\b(?:send\s+us\s+a\s+message\s+with)\b", re.IGNORECASE),
]
RE_EXTRA_SPACES = re.compile(r"\s{2,}")

# Orphaned preposition patterns — trailing dangling references
# Matches: "at ", "at:", "here:", "here: ", "via ", "via:", "to us at", etc.
RE_ORPHAN_TRAILING = re.compile(
    r"\b(?:at|here|via|to)\s*[:.]?\s*$",
    re.IGNORECASE
)
# Broader: mid-sentence orphans like "share your details here: so we can"
RE_ORPHAN_MID = re.compile(
    r"\b(?:details?\s+here\s*:|contact\s+us\s+(?:at|here)\s*:|"
    r"email\s+(?:us\s+)?at\s*:|reach\s+(?:us|out)\s+at\s*:|"
    r"visit\s+us\s+at\s*:|call\s+us\s+at\s*:|"
    r"send\s+(?:us\s+)?(?:a\s+)?(?:note|message|email)\s+(?:at|via|to)\s*:?"
    r")\s*",
    re.IGNORECASE
)
# Phone number patterns (agent leaked real numbers)
# RE_PHONE is now defined above alongside RE_SIGNATURE


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: Response Repair Functions
# ══════════════════════════════════════════════════════════════════════════

# Orphan replacement map — context-sensitive completions
ORPHAN_REPLACEMENTS = [
    # "reach out at" / "reach us at" → clean alternative
    (re.compile(r"\breach\s+(?:out\s+)?(?:to\s+us\s+)?at\s*[:\.]?\s*$", re.IGNORECASE),
     "reach out to our customer support team"),
    (re.compile(r"\breach\s+(?:out\s+)?(?:to\s+us\s+)?at\s*[:\.]?\s*(?=\S)", re.IGNORECASE),
     "reach out to our customer support team "),
    # "contact us at" / "contact us here:" → clean
    (re.compile(r"\bcontact\s+(?:us\s+)?(?:at|here)\s*[:\.]?\s*$", re.IGNORECASE),
     "contact our customer support team"),
    (re.compile(r"\bcontact\s+(?:us\s+)?(?:at|here)\s*[:\.]?\s*(?=\S)", re.IGNORECASE),
     "contact our customer support team "),
    # "share your details here:" → clean
    (re.compile(r"\bshare\s+your\s+(?:details?|info(?:rmation)?)\s+here\s*[:\.]?\s*$", re.IGNORECASE),
     "share your details through your account dashboard"),
    (re.compile(r"\bshare\s+your\s+(?:details?|info(?:rmation)?)\s+here\s*[:\.]?\s*(?=\S)", re.IGNORECASE),
     "share your details through your account dashboard "),
    # "email us at" → clean
    (re.compile(r"\bemail\s+(?:us\s+)?at\s*[:\.]?\s*$", re.IGNORECASE),
     "email our customer support team"),
    (re.compile(r"\bemail\s+(?:us\s+)?at\s*[:\.]?\s*(?=\S)", re.IGNORECASE),
     "email our customer support team "),
    # "call us at" → clean
    (re.compile(r"\bcall\s+(?:us\s+)?at\s*[:\.]?\s*$", re.IGNORECASE),
     "call our customer support line"),
    (re.compile(r"\bcall\s+(?:us\s+)?at\s*[:\.]?\s*(?=\S)", re.IGNORECASE),
     "call our customer support line "),
    # "send us a note via" → clean
    (re.compile(r"\bsend\s+(?:us\s+)?(?:a\s+)?(?:note|message|email)\s+(?:at|via|to)\s*[:\.]?\s*$", re.IGNORECASE),
     "submit a request through your account dashboard"),
    # "visit us at" → clean
    (re.compile(r"\bvisit\s+(?:us\s+)?at\s*[:\.]?\s*$", re.IGNORECASE),
     "visit our support portal"),
    # "add your email at" → clean
    (re.compile(r"\badd\s+your\s+email\s+at\s*[:\.]?\s*$", re.IGNORECASE),
     "add your email through your account settings"),
    # Generic trailing "at" / "here:" / "via"
    (re.compile(r"\bat\s*[:\.]?\s*$", re.IGNORECASE),
     "through our support channels"),
    (re.compile(r"\bhere\s*[:\.]?\s*$", re.IGNORECASE),
     "through our support portal"),
    (re.compile(r"\bvia\s*[:\.]?\s*$", re.IGNORECASE),
     "through our customer support team"),
]


def repair_orphaned_prepositions(text: str) -> str:
    """Repairs orphaned trailing prepositions/references to complete phrases."""
    result = text
    for pattern, replacement in ORPHAN_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def deep_clean_text(text: str) -> str:
    """Full cleaning pipeline: handles, URLs, signatures, phones, orphans."""
    if not text:
        return ""
    t = text

    # Remove @handles and URLs
    t = RE_HANDLE.sub("", t)
    t = RE_URL.sub("", t)

    # Remove phone numbers (agent leaked real numbers)
    t = RE_PHONE.sub("", t)

    # Remove signatures
    t = RE_SIGNATURE.sub("", t)

    # Repair orphaned prepositions
    t = repair_orphaned_prepositions(t)

    # Clean whitespace
    t = RE_EXTRA_SPACES.sub(" ", t).strip()

    # Strip leading punctuation
    t = t.lstrip(".,;:!- ")
    return t


def is_dm_deflection(text: str) -> bool:
    """Detects DM/direct-message redirections."""
    t_lower = text.lower()
    for pat in RE_DM_PATTERNS:
        if pat.search(t_lower):
            return True
    return False


def has_residual_contamination(text: str) -> bool:
    """Checks for any remaining Twitter/social artifacts after cleaning."""
    if RE_HANDLE.search(text):
        return True
    if RE_URL.search(text):
        return True
    if RE_PHONE.search(text):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3: Quality Filters
# ══════════════════════════════════════════════════════════════════════════

# Off-domain / noise context indicators
OFF_DOMAIN_CONTEXT_PATTERNS = [
    re.compile(r"\b(?:craving|hungry|food\s+truck|menu|recipe)\b", re.IGNORECASE),
    re.compile(r"\b(?:when\s+are\s+you\s+coming\s+to|launch(?:ing)?\s+in\s+new\s+countries)\b", re.IGNORECASE),
    re.compile(r"^(?:lol|lmao|haha|😂|🤣|💀)\s*$", re.IGNORECASE),
    re.compile(r"^(?:thanks?|thx|ty|ok|okay|cool|great|nice|awesome|yep|yes|no)\s*[!.]*\s*$", re.IGNORECASE),
]

# Generic deflection-only responses (no real content)
DEFLECTION_ONLY_PATTERNS = [
    re.compile(r"^(?:we'?d?\s+like\s+to\s+(?:help|check|look)|let\s+us\s+(?:help|look))\b.*$", re.IGNORECASE),
    re.compile(r"^(?:sorry\s+(?:about|for)\s+(?:the|that|this)\b.{0,30})$", re.IGNORECASE),
]


def is_off_domain_context(ctx: str) -> bool:
    """Detects contexts that are not customer support inquiries."""
    for pat in OFF_DOMAIN_CONTEXT_PATTERNS:
        if pat.search(ctx):
            return True
    # Very short contexts with no complaint signal
    words = ctx.split()
    if len(words) < 4:
        return True
    return False


def is_low_quality_response(resp: str) -> bool:
    """Detects responses too generic/short to provide actionable support."""
    words = resp.split()
    # Too short (< 8 words)
    if len(words) < 8:
        return True
    # Too long (> 120 words — likely noise)
    if len(words) > 120:
        return True
    # Excessive repetition
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    if unique_ratio < 0.40:
        return True
    # Must contain alphabetic content
    if not any(ch.isalpha() for ch in resp):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4: 4-Tier Structured Enterprise Response Templates
# ══════════════════════════════════════════════════════════════════════════
# Intent-mapped response templates for 10 core customer support categories.
# Each template follows the 4-tier contract:
#   T1: Acknowledgment & Empathy
#   T2: Explicit Intent Recognition
#   T3: Actionable Guidance (no fabricated specifics)
#   T4: Resolution Path

CATEGORY_TEMPLATES = {
    "billing": [
        "I understand your concern about unexpected charges on your account. "
        "We take billing accuracy very seriously and want to ensure every charge is correct. "
        "Please review your recent billing statement through your account dashboard and identify the specific transaction in question. "
        "Our billing department will investigate and issue any necessary adjustments within a few business days.",

        "I apologize for the billing discrepancy you've encountered. "
        "Inaccurate charges can be frustrating, and we want to resolve this promptly. "
        "Please verify your payment method details and recent transaction history in your account settings. "
        "Once we receive the relevant transaction details, our finance team will review and process the appropriate correction.",

        "Thank you for bringing this billing issue to our attention. "
        "We understand how important it is for your charges to be accurate. "
        "Please check your recent statements and note any discrepancies you've found. "
        "Our team will review the flagged transactions and ensure your account reflects the correct balance.",
    ],
    "refund": [
        "I'm sorry to hear you need a refund for your recent purchase. "
        "We want to make sure you receive the appropriate resolution for your situation. "
        "Please confirm the order details and the reason for your refund request through your account. "
        "Once verified, our team will process the refund according to our standard policy timeline.",

        "I understand your frustration regarding this refund request. "
        "Getting your money back promptly is our priority. "
        "Please verify that the item has been returned or the service was not delivered as expected. "
        "Our refund team will review your case and update you on the expected processing timeline.",

        "Thank you for reaching out about your refund. "
        "We appreciate your patience while we look into this matter. "
        "Please ensure your return has been initiated and provide your order reference through your account portal. "
        "Our team will confirm the refund amount and estimated processing time.",
    ],
    "delivery": [
        "I understand how frustrating it must be when your delivery hasn't arrived as expected. "
        "Tracking discrepancies can sometimes occur due to carrier processing delays. "
        "Please check your order status and tracking information through your account dashboard. "
        "We will coordinate with our shipping partner to locate your package and provide an updated delivery estimate.",

        "I'm sorry for the inconvenience caused by this delivery issue. "
        "We know how important it is to receive your order on time. "
        "Please verify the shipping address on your order and check for any delivery notifications from the carrier. "
        "Our logistics team will investigate the current status and work to expedite resolution.",

        "Thank you for letting us know about the delivery concern. "
        "We take shipping reliability very seriously and want to ensure you receive your order. "
        "Please review the tracking details available in your order history for the latest status update. "
        "If the issue persists, our support team will open an investigation with the courier service.",
    ],
    "account": [
        "I understand you need to update your account information. "
        "Keeping your account details current is important for a smooth experience. "
        "Please navigate to your account settings page to modify your profile, contact details, or preferences. "
        "Any changes will take effect immediately once saved.",

        "Thank you for reaching out about your account update request. "
        "We want to make sure your information is accurate and secure. "
        "Please verify your identity through your account security settings before making any changes. "
        "Once verified, you can update your details directly in your profile section.",

        "I'm happy to help with your account-related inquiry. "
        "We understand how important it is to maintain accurate account information. "
        "Please log in to your account and navigate to the settings section to review and update your details. "
        "If you encounter any issues during the process, our technical support team can assist further.",
    ],
    "late_fee": [
        "I understand your concern about the late fee that was applied to your account. "
        "Unexpected charges can be frustrating, especially when you believe the payment was submitted on time. "
        "Please review your payment history and confirm the date your payment was initiated through your account portal. "
        "Our billing team will review the timing and determine if a fee adjustment is warranted.",

        "I apologize for any confusion regarding the late fee on your statement. "
        "We want to ensure all fees are applied fairly and accurately. "
        "Please check your payment confirmation records and verify the processing timeline for your recent payment. "
        "Once we have the details, our team will review the charge and process any necessary corrections.",

        "Thank you for bringing this late fee concern to our attention. "
        "We take billing accuracy seriously and want to resolve this for you. "
        "Please gather your payment confirmation and the specific transaction date so we can cross-reference our records. "
        "Our finance department will investigate and respond with the appropriate resolution.",
    ],
    "cancellation": [
        "I understand you'd like to cancel your service or subscription. "
        "We're sorry to see you go and want to ensure the process is straightforward. "
        "Please navigate to your subscription management page in your account settings to initiate the cancellation. "
        "Your cancellation will be confirmed and any applicable pro-rated credits will be applied to your account.",

        "I'm sorry to hear you want to cancel. "
        "Before processing, we want to make sure you're aware of all available options. "
        "Please review your current plan details and any active commitments in your account portal. "
        "If you'd still like to proceed, our team can guide you through the cancellation process and timeline.",

        "Thank you for reaching out about the cancellation. "
        "We value your feedback and want to understand your experience. "
        "Please verify your account standing and any pending transactions before initiating the cancellation request. "
        "Our retention team will process your request and confirm the effective date.",
    ],
    "card_declined": [
        "I understand how inconvenient it is to have your card declined during a transaction. "
        "This can happen for several reasons including security holds or insufficient authorization. "
        "Please verify your card details, expiration date, and available balance through your banking portal. "
        "If everything appears correct, please contact your card issuer to ensure there are no restrictions on your account.",

        "I'm sorry you're experiencing a declined card issue. "
        "Payment processing problems can be frustrating, and we want to help you resolve this quickly. "
        "Please double-check that your payment method is active and that the billing address matches your records. "
        "Our payments team can assist with alternative payment options if the issue persists.",

        "Thank you for letting us know about the card decline. "
        "We want to make sure your payment goes through smoothly. "
        "Please review your card status with your financial institution and verify your payment details in your account. "
        "If needed, you can update your payment method or try an alternative card through your payment settings.",
    ],
    "login_password": [
        "I understand how frustrating it can be when you're unable to access your account. "
        "Account access issues are something we take very seriously. "
        "Please try resetting your password through the forgot password option on the login page. "
        "If you continue to experience issues, our technical support team can help verify your identity and restore access.",

        "I'm sorry you're having trouble logging in to your account. "
        "We want to get you back in as quickly as possible. "
        "Please verify that you're using the correct email address and try clearing your browser cache before attempting again. "
        "If the issue persists, you can request a password reset or contact our support team for additional assistance.",

        "Thank you for reaching out about your login difficulty. "
        "We understand the urgency of regaining access to your account. "
        "Please check for any account verification emails and follow the instructions to reset your credentials. "
        "Our security team is available to assist with identity verification if the standard reset process doesn't work.",
    ],
    "order_delay": [
        "I understand your frustration with the delay in your order. "
        "We know how important it is to receive your purchases on time. "
        "Please check your order status for the latest shipping update through your account order history. "
        "Our fulfillment team is working to expedite pending orders and will provide a revised delivery estimate.",

        "I'm sorry about the delay with your order. "
        "Unexpected processing delays can be inconvenient, and we apologize for the wait. "
        "Please verify your order details and shipping information in your account to ensure everything is correct. "
        "We will follow up with our warehouse team to prioritize your shipment.",

        "Thank you for your patience regarding the order delay. "
        "We take delivery commitments seriously and want to keep you informed. "
        "Please monitor your order tracking page for real-time updates on your shipment status. "
        "If you need further assistance, our support team can escalate the matter for priority handling.",
    ],
    "feedback": [
        "Thank you for taking the time to share your feedback with us. "
        "We truly value your input as it helps us improve our services. "
        "Please feel free to elaborate on any specific areas where you'd like to see improvements. "
        "Your feedback has been noted and will be shared with the relevant team for review.",

        "We appreciate your feedback and the time you've taken to share your experience. "
        "Your perspective is important to us and helps shape our service improvements. "
        "If you have additional suggestions or specific experiences to highlight, please don't hesitate to share. "
        "Our quality assurance team will review your comments and incorporate them into our improvement plans.",

        "Thank you for your valuable feedback regarding our service. "
        "We are committed to continuous improvement and your input makes a real difference. "
        "Please provide any additional details that could help us better understand your experience. "
        "We will use your feedback to prioritize enhancements and ensure a better experience going forward.",
    ],
}

# Category keyword detectors for intent mapping
CATEGORY_KEYWORDS = {
    "billing": ["bill", "billing", "charge", "charged", "overcharge", "double charge",
                 "invoice", "statement", "fee", "extra charge", "incorrect amount",
                 "payment amount", "balance due"],
    "refund": ["refund", "money back", "reimburse", "return", "credit back",
               "get my money", "charged but", "didn't receive", "want my money"],
    "delivery": ["deliver", "delivery", "package", "shipping", "shipped", "tracking",
                 "parcel", "courier", "not received", "missing package", "lost package",
                 "arrived", "transit"],
    "account": ["account", "profile", "update", "change email", "change address",
                "settings", "password", "username", "login", "sign in", "sign up",
                "register", "personal info", "contact info"],
    "late_fee": ["late fee", "late charge", "penalty", "overdue", "past due",
                 "payment deadline", "missed payment", "grace period"],
    "cancellation": ["cancel", "cancellation", "unsubscribe", "terminate",
                     "end subscription", "stop service", "close account", "opt out"],
    "card_declined": ["declined", "card declined", "payment failed", "transaction failed",
                      "card rejected", "couldn't process", "payment error", "card issue"],
    "login_password": ["login", "log in", "password", "forgot password", "reset password",
                       "can't access", "locked out", "authentication", "sign in",
                       "two factor", "2fa", "verification"],
    "order_delay": ["delay", "delayed", "late order", "still waiting", "when will",
                    "order status", "expected delivery", "overdue order", "not shipped",
                    "hasn't shipped"],
    "feedback": ["feedback", "suggestion", "complaint", "experience", "review",
                 "satisfied", "dissatisfied", "rate", "opinion", "improve"],
}


def detect_intent_category(context: str) -> str:
    """Maps customer context to the most likely intent category via keyword matching."""
    ctx_lower = context.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in ctx_lower)
        if score > 0:
            scores[cat] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════════════════
# STAGE 5: Fast Near-Duplicate Detection (Normalized Fingerprint Hashing)
# ══════════════════════════════════════════════════════════════════════════
# Uses word-level normalization + sorted n-gram fingerprinting.
# O(n) single-pass instead of O(n² × shingles) MinHash.


def _normalize_for_dedup(text: str) -> str:
    """Normalizes text for dedup: lowercase, strip punctuation, collapse whitespace."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fingerprint(text: str) -> str:
    """Creates a fast fingerprint from sorted word 3-grams."""
    words = _normalize_for_dedup(text).split()
    if len(words) < 3:
        # For very short texts, use the full normalized text
        return hashlib.sha256(" ".join(sorted(words)).encode()).hexdigest()
    # Extract word 3-grams, sort them, and hash
    ngrams = sorted(set(" ".join(words[i:i+3]) for i in range(len(words) - 2)))
    return hashlib.sha256("|".join(ngrams).encode()).hexdigest()


def deduplicate_responses(records: list) -> list:
    """Removes near-duplicate responses using fast fingerprint hashing."""
    log(f"  Deduplicating {len(records):,} records (fingerprint-based)...")

    seen_fingerprints = {}
    deduped = []
    dup_count = 0

    for idx, rec in enumerate(records):
        fp = _fingerprint(rec["response"])
        if fp not in seen_fingerprints:
            seen_fingerprints[fp] = idx
            deduped.append(rec)
        else:
            dup_count += 1

        # Progress reporting every 100K records
        if (idx + 1) % 100000 == 0:
            log(f"    Progress: {idx + 1:,}/{len(records):,} processed, {dup_count:,} duplicates found")

    log(f"  Removed {dup_count:,} exact-fingerprint duplicates → {len(deduped):,} remaining")
    return deduped


# ══════════════════════════════════════════════════════════════════════════
# STAGE 6: Pipeline Processing
# ══════════════════════════════════════════════════════════════════════════

def process_and_refine_split(parquet_path: str, split_name: str) -> tuple:
    """
    Full M10.3 processing pipeline for a single split:
    1. Deep clean (handles, URLs, signatures, phones, orphans)
    2. Quality filter (off-domain, low-quality, DM deflection)
    3. Augment with 4-tier templates for detected intent categories
    4. Deduplicate
    """
    log(f"\n  Processing {split_name} split: {parquet_path}")
    table = pq.read_table(parquet_path)
    contexts = table["context"].to_pylist()
    responses = table["response"].to_pylist()
    sources = table["source"].to_pylist() if "source" in table.column_names else ["twcs"] * len(contexts)
    group_ids = table["group_id"].to_pylist() if "group_id" in table.column_names else [f"grp_{i}" for i in range(len(contexts))]

    stats = {
        "input_total": len(contexts),
        "dropped_dm": 0,
        "dropped_off_domain": 0,
        "dropped_low_quality": 0,
        "dropped_contaminated": 0,
        "orphans_repaired": 0,
        "templates_added": 0,
    }

    clean_records = []

    for i in range(len(contexts)):
        raw_ctx = str(contexts[i])
        raw_resp = str(responses[i])
        src = str(sources[i])
        gid = group_ids[i] or f"grp_{i}"

        # Skip DM deflections first (before any cleaning)
        if is_dm_deflection(raw_resp):
            stats["dropped_dm"] += 1
            continue

        # Deep clean both context and response
        c = deep_clean_text(raw_ctx)
        r = deep_clean_text(raw_resp)

        # Check if orphans were repaired
        if raw_resp != r and RE_ORPHAN_TRAILING.search(raw_resp):
            stats["orphans_repaired"] += 1

        # Skip off-domain contexts
        if is_off_domain_context(c):
            stats["dropped_off_domain"] += 1
            continue

        # Skip low-quality responses
        if is_low_quality_response(r):
            stats["dropped_low_quality"] += 1
            continue

        # Final contamination check — anything still with handles/URLs/phones
        if has_residual_contamination(r):
            stats["dropped_contaminated"] += 1
            continue

        clean_records.append({
            "id": f"m10_3_{src}_{i}",
            "source": src,
            "context": c,
            "response": r,
            "group_id": str(gid),
            "intent_category": detect_intent_category(c),
        })

    # Add 4-tier template responses for training data only
    if split_name == "train":
        template_records = generate_template_augmentation(clean_records)
        stats["templates_added"] = len(template_records)
        clean_records.extend(template_records)

    # Deduplicate
    pre_dedup = len(clean_records)
    clean_records = deduplicate_responses(clean_records)
    stats["dedup_removed"] = pre_dedup - len(clean_records)
    stats["output_total"] = len(clean_records)

    return clean_records, stats


def generate_template_augmentation(existing_records: list) -> list:
    """
    Generates 4-tier template augmentation records.
    For each category with templates, samples existing contexts
    and pairs them with structured template responses.
    """
    log("  Generating 4-tier template augmentation...")

    # Group existing records by detected category
    category_contexts = defaultdict(list)
    for rec in existing_records:
        cat = rec["intent_category"]
        if cat in CATEGORY_TEMPLATES:
            category_contexts[cat].append(rec["context"])

    template_records = []
    aug_id = 0

    for cat, templates in CATEGORY_TEMPLATES.items():
        contexts = category_contexts.get(cat, [])
        if not contexts:
            continue

        # Sample contexts for template pairing (up to 500 per category)
        sample_size = min(500, len(contexts))
        sampled_contexts = random.sample(contexts, sample_size)

        for ctx in sampled_contexts:
            # Pick a random template for this category
            template = random.choice(templates)
            template_records.append({
                "id": f"m10_3_template_{cat}_{aug_id}",
                "source": "template_4tier",
                "context": ctx,
                "response": template,
                "group_id": f"template_grp_{aug_id}",
                "intent_category": cat,
            })
            aug_id += 1

    log(f"  Generated {len(template_records):,} 4-tier template records across {len(CATEGORY_TEMPLATES)} categories")
    return template_records


def save_m10_3_parquet(records: list, out_path: str, split_name: str):
    """Saves records to a parquet file with the M10.3 schema."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    schema = pa.schema([
        ("id", pa.string()),
        ("source", pa.string()),
        ("context", pa.string()),
        ("response", pa.string()),
        ("group_id", pa.string()),
        ("intent_category", pa.string()),
        ("split", pa.string()),
    ])

    table = pa.Table.from_pydict({
        "id": [r["id"] for r in records],
        "source": [r["source"] for r in records],
        "context": [r["context"] for r in records],
        "response": [r["response"] for r in records],
        "group_id": [r["group_id"] for r in records],
        "intent_category": [r.get("intent_category", "general") for r in records],
        "split": [split_name] * len(records),
    }, schema=schema)

    pq.write_table(table, out_path, compression="zstd")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log(f"  💾 Saved {split_name} ({len(records):,} rows) → {out_path} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 80)
    log("  [M10.3] 4-TIER ENTERPRISE RESPONSE DATASET SYNTHESIZER")
    log("=" * 80)

    all_stats = {}

    # ── Process Train Split ──
    log("\n[1/3] Processing Training Split...")
    train_records, train_stats = process_and_refine_split(CLEAN_TRAIN, "train")
    save_m10_3_parquet(train_records, M103_TRAIN, "train")
    all_stats["train"] = train_stats

    # ── Process Validation Split ──
    log("\n[2/3] Processing Validation Split...")
    val_records, val_stats = process_and_refine_split(CLEAN_VAL, "val")
    save_m10_3_parquet(val_records, M103_VAL, "val")
    all_stats["val"] = val_stats

    # ── Process Test Split ──
    log("\n[3/3] Processing Test Split...")
    test_records, test_stats = process_and_refine_split(CLEAN_TEST, "test")
    save_m10_3_parquet(test_records, M103_TEST, "test")
    all_stats["test"] = test_stats

    # ── Summary Statistics ──
    log("\n" + "=" * 80)
    log("  📊 M10.3 DATASET SYNTHESIS SUMMARY")
    log("=" * 80)

    for split_name, stats in all_stats.items():
        log(f"\n  [{split_name.upper()}]")
        log(f"    Input rows:          {stats['input_total']:>8,}")
        log(f"    Dropped DM:          {stats['dropped_dm']:>8,}")
        log(f"    Dropped Off-Domain:  {stats['dropped_off_domain']:>8,}")
        log(f"    Dropped Low-Quality: {stats['dropped_low_quality']:>8,}")
        log(f"    Dropped Contaminated:{stats['dropped_contaminated']:>8,}")
        log(f"    Orphans Repaired:    {stats['orphans_repaired']:>8,}")
        log(f"    4-Tier Templates:    {stats['templates_added']:>8,}")
        log(f"    Dedup Removed:       {stats.get('dedup_removed', 0):>8,}")
        log(f"    Final Output:        {stats['output_total']:>8,}")

    # Response length audit
    train_resp_lengths = [len(r["response"].split()) for r in train_records]
    log(f"\n  📏 Training Response Length Distribution:")
    log(f"    Mean:   {np.mean(train_resp_lengths):.1f} words")
    log(f"    Median: {np.median(train_resp_lengths):.1f} words")
    log(f"    Min:    {np.min(train_resp_lengths)} words")
    log(f"    Max:    {np.max(train_resp_lengths)} words")
    log(f"    Std:    {np.std(train_resp_lengths):.1f} words")

    # Source distribution
    src_dist = Counter(r["source"] for r in train_records)
    log(f"\n  📂 Training Source Distribution:")
    for src, count in src_dist.most_common():
        pct = count / len(train_records) * 100
        log(f"    {src:20s}: {count:>8,} ({pct:>5.1f}%)")

    # Intent category distribution
    cat_dist = Counter(r.get("intent_category", "general") for r in train_records)
    log(f"\n  🏷️  Training Intent Category Distribution:")
    for cat, count in cat_dist.most_common():
        pct = count / len(train_records) * 100
        log(f"    {cat:20s}: {count:>8,} ({pct:>5.1f}%)")

    # Sample display
    log("\n" + "=" * 80)
    log("  📋 M10.3 SAMPLE RESPONSES (First 10)")
    log("=" * 80)

    # Show a mix: 5 refined originals + 5 templates
    original_samples = [r for r in train_records if r["source"] != "template_4tier"][:5]
    template_samples = [r for r in train_records if r["source"] == "template_4tier"][:5]

    for i, rec in enumerate(original_samples):
        print(f"\n[Original {i+1}] (Source: {rec['source']} | Intent: {rec.get('intent_category', 'N/A')})")
        print(f"  Customer: \"{rec['context'][:150]}\"")
        print(f"  Response: \"{rec['response'][:200]}\"")

    print("\n" + "-" * 40)

    for i, rec in enumerate(template_samples):
        print(f"\n[Template {i+1}] (Source: {rec['source']} | Intent: {rec.get('intent_category', 'N/A')})")
        print(f"  Customer: \"{rec['context'][:150]}\"")
        print(f"  Response: \"{rec['response'][:200]}\"")

    # ── Save Report ──
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "timestamp": timestamp(),
        "pipeline": "M10.3",
        "splits": all_stats,
        "train_response_length": {
            "mean": round(float(np.mean(train_resp_lengths)), 1),
            "median": round(float(np.median(train_resp_lengths)), 1),
            "min": int(np.min(train_resp_lengths)),
            "max": int(np.max(train_resp_lengths)),
            "std": round(float(np.std(train_resp_lengths)), 1),
        },
        "source_distribution": dict(src_dist),
        "intent_distribution": dict(cat_dist),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"\n  💾 Saved M10.3 Dataset Report: {REPORT_PATH}")

    log("\n🎉 M10.3 Dataset Synthesis Complete!")


if __name__ == "__main__":
    main()
