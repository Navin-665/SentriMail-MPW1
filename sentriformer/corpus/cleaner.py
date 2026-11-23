"""
Non-Destructive Text Cleaner & Quality Filter
"""
import re
import html
import unicodedata

# Regex for unprintable control chars (preserve standard whitespace: \n, \t, \r)
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
WHITESPACE_RE = re.compile(r'[^\S\n]+')
NEWLINES_RE = re.compile(r'\n{3,}')
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
ALPHANUMERIC_RE = re.compile(r'[a-zA-Z0-9]')

# Boilerplate DM / deflection signatures in TWCS
DEFLECTION_PATTERNS = [
    re.compile(r'\b(please|kindly|pls)\s+(dm|direct\s*message|pm)\s+us\b', re.IGNORECASE),
    re.compile(r'\bsend\s+us\s+a\s+(dm|direct\s*message|pm)\b', re.IGNORECASE),
    re.compile(r'\b(reach\s*out|contact\s*us)\s+(via\s+dm|in\s+dm|through\s+dm)\b', re.IGNORECASE),
    re.compile(r'\bclick\s+here\s*:\s*https?://', re.IGNORECASE),
    re.compile(r'\bcheck\s+out\s+this\s+link\b', re.IGNORECASE),
    re.compile(r'\bvisit\s+(our\s+help\s+center|us\s+at)\b', re.IGNORECASE),
    re.compile(r'\bplease\s+give\s+us\s+a\s+call\s+at\b', re.IGNORECASE),
]


def clean_text(text: str) -> str:
    """
    Applies non-destructive cleaning:
    - Unescapes HTML entities (&amp; -> &, &lt; -> <)
    - Strips invalid control characters
    - Normalizes unicode (NFKC)
    - Collapses duplicate horizontal whitespace while preserving paragraph structure
    - Preserves ALL casing, punctuation, contractions, and emojis
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Unicode normalization (NFKC decomposes compatibility chars safely)
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Strip unprintable control characters
    text = CONTROL_CHAR_RE.sub('', text)
    
    # 4. Collapse excessive horizontal spaces
    text = WHITESPACE_RE.sub(' ', text)
    
    # 5. Limit excessive consecutive newlines to maximum 2
    text = NEWLINES_RE.sub('\n\n', text)
    
    return text.strip()


def normalize_for_dedup(text: str) -> str:
    """
    Produces a normalized representation strictly for deduplication hashing.
    Does NOT modify the text stored in the dataset.
    """
    if not text:
        return ""
    # Lowercase, collapse all whitespace, strip outer punctuation
    cleaned = text.lower().strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove CFPB redaction masks like "XXXX" or "XX/XX/XXXX" for better near-duplicate detection
    cleaned = re.sub(r'x{2,}', '', cleaned)
    return cleaned.strip()


def is_valid_text(text: str, min_chars: int = 4, min_words: int = 1) -> bool:
    """
    Validates that text contains meaningful content.
    Retains short high-intent phrases (e.g. "Card stolen", "Fraud on account").
    Drops empty, pure punctuation, pure whitespace, or single-character noise.
    """
    if not text or len(text) < min_chars:
        return False
    
    # Must contain at least one alphanumeric character
    if not ALPHANUMERIC_RE.search(text):
        return False
    
    words = text.split()
    if len(words) < min_words:
        return False
    
    # If text is exclusively a URL with nothing else
    if URL_RE.fullmatch(text.strip()):
        return False
        
    return True


def is_deflection(text: str) -> bool:
    """Checks if a brand tweet is a boilerplate deflection (DM/link request)."""
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in DEFLECTION_PATTERNS)
