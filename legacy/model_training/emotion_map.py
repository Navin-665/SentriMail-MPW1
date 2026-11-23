"""
Emotion mapping for SentriMail dataset preparation.

Maps GoEmotions 28-label taxonomy into SentriMail's 7 canonical emotions:
- anger
- fear
- sadness
- disgust
- surprise
- joy
- neutral
"""

# Official GoEmotions label ordering (28 classes)
GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Mapping from GoEmotions 28 fine-grained labels to SentriMail's 7 canonical emotions
GOEMOTIONS_TO_SENTRI = {
    # Anger cluster
    "anger": "anger",
    "annoyance": "anger",

    # Fear cluster
    "fear": "fear",
    "nervousness": "fear",

    # Sadness cluster
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",

    # Disgust cluster
    "disgust": "disgust",
    "disapproval": "disgust",

    # Surprise cluster
    "surprise": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",

    # Joy / Positive cluster
    "joy": "joy",
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",

    # Neutral
    "neutral": "neutral",
}
