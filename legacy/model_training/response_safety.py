"""
Post-processing safety filter for generated response text.

The model learned surface patterns from Twitter along with tone - short
links, "-HN" style agent sign-offs - and reproduces them confidently even
though they're meaningless (or dead) outside the platform they came from.
This catches that before it reaches a user, without needing retraining.
"""

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+", re.IGNORECASE)
DOLLAR_AMOUNT_PATTERN = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?")
AGENT_SIGNOFF_PATTERN = re.compile(r"[-–^]\s?[A-Z]{1,3}$|\^[a-zA-Z]+$")  # e.g. "-HN", "- JW", "^DM"


def contains_unsafe_content(response_text: str) -> bool:
    """True if the generated text has a link, a dollar figure, or a
    Twitter-style agent sign-off - none of these can be trusted as real
    without a verified source behind them."""
    if not response_text:
        return True
    return bool(
        URL_PATTERN.search(response_text)
        or DOLLAR_AMOUNT_PATTERN.search(response_text)
        or AGENT_SIGNOFF_PATTERN.search(response_text.strip())
    )


def sanitize_or_fallback(response_text: str, fallback_text: str) -> str:
    """Returns the generated text if it's clean, otherwise the safe
    fallback (a verified template) - never a partially-cleaned version,
    since stripping just the URL out of a sentence built around it often
    leaves broken, half-finished phrasing behind."""
    if contains_unsafe_content(response_text):
        return fallback_text
    return response_text
