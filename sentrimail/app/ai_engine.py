"""
SentriMail AI Engine
--------------------
Uses transformer-based models for:
  1. Sentiment analysis        → polarity + confidence
  2. Emotion detection         → dominant emotion + intensity
  3. Priority scoring          → CRITICAL / HIGH / MEDIUM / LOW
  4. Root cause suggestion     → LLM-style heuristic summary
  5. Actionable response draft → templated AI suggestion

Falls back gracefully to rule-based scoring when models can't load.
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ─── Model Loading ─────────────────────────────────────────────

_sentiment_pipeline = None
_emotion_pipeline = None
_models_loaded = False
_use_transformers = True


def _load_models():
    global _sentiment_pipeline, _emotion_pipeline, _models_loaded, _use_transformers
    if _models_loaded:
        return

    try:
        from transformers import pipeline

        logger.info("Loading sentiment model (distilbert)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )

        logger.info("Loading emotion model (j-hartmann)...")
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            truncation=True,
            max_length=512
        )

        _models_loaded = True
        _use_transformers = True
        logger.info("✅ Transformer models loaded successfully.")

    except Exception as e:
        logger.warning(f"⚠️  Could not load transformers: {e}")
        logger.info("→ Falling back to rule-based AI engine.")
        _use_transformers = False
        _models_loaded = True


# ─── Rule-Based Fallback ───────────────────────────────────────

NEGATIVE_KEYWORDS = [
    "angry", "furious", "terrible", "horrible", "awful", "disgusting", "unacceptable",
    "worst", "broken", "failed", "fraud", "scam", "useless", "incompetent", "lied",
    "cheated", "stolen", "damaged", "defective", "ruined", "lawsuit", "legal",
    "never", "outraged", "disgusted", "appalled", "ridiculous", "pathetic",
    "disaster", "catastrophic", "urgent", "immediately", "asap", "emergency"
]

POSITIVE_KEYWORDS = [
    "good", "great", "fine", "okay", "thanks", "appreciate", "help", "resolved"
]

EMOTION_PATTERNS = {
    "anger": ["angry", "furious", "outraged", "mad", "infuriated", "livid", "rage"],
    "fear": ["scared", "afraid", "worried", "anxious", "terrified", "concern", "panic"],
    "sadness": ["sad", "disappointed", "upset", "devastated", "miserable", "heartbroken"],
    "disgust": ["disgusting", "disgusted", "revolting", "nasty", "appalled", "appalling"],
    "surprise": ["shocked", "unbelievable", "incredible", "unexpected", "sudden"],
    "joy": ["happy", "pleased", "delighted", "glad", "satisfied", "grateful"],
    "neutral": []
}


def _rule_based_sentiment(text: str) -> Dict:
    text_lower = text.lower()
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)

    if neg_count > pos_count:
        score = min(0.5 + (neg_count * 0.08), 0.99)
        return {"label": "NEGATIVE", "score": round(score, 3)}
    elif pos_count > neg_count:
        return {"label": "POSITIVE", "score": round(0.5 + (pos_count * 0.1), 3)}
    else:
        return {"label": "NEUTRAL", "score": 0.55}


def _rule_based_emotion(text: str) -> Dict:
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_PATTERNS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "neutral"
    total = sum(scores.values()) or 1
    confidence = round(min((scores.get(best, 1) / total) + 0.3, 0.99), 3)
    return {"label": best, "score": confidence}


# ─── Priority Scoring ──────────────────────────────────────────

def _compute_priority(sentiment: Dict, emotion: Dict, text: str) -> Dict:
    """
    Priority matrix:
      CRITICAL → NEGATIVE + (anger/fear/disgust) + intensity > 0.85
      HIGH     → NEGATIVE + strong emotion + intensity > 0.70
      MEDIUM   → NEGATIVE or neutral with some emotion
      LOW      → POSITIVE or weak signals
    """
    sentiment_label = sentiment["label"]
    sentiment_score = sentiment["score"]
    emotion_label = emotion["label"].lower()
    emotion_score = emotion["score"]

    high_intensity_emotions = {"anger", "fear", "disgust"}
    
    # Urgency keywords boost
    urgency_boost = any(kw in text.lower() for kw in [
        "urgent", "immediately", "asap", "emergency", "legal action", "lawsuit",
        "police", "critical", "danger", "life", "health", "injury"
    ])

    score = 0

    # Sentiment contribution (0-40 pts)
    if sentiment_label == "NEGATIVE":
        score += int(sentiment_score * 40)
    elif sentiment_label == "NEUTRAL":
        score += 10
    else:
        score += 0

    # Emotion contribution (0-40 pts)
    if emotion_label in high_intensity_emotions:
        score += int(emotion_score * 40)
    elif emotion_label in {"sadness", "surprise"}:
        score += int(emotion_score * 20)
    else:
        score += 0

    # Urgency boost (+20 pts)
    if urgency_boost:
        score += 20

    # Text length (long complaints signal more detail/seriousness)
    word_count = len(text.split())
    if word_count > 100:
        score += 5

    # Determine priority
    if score >= 75:
        priority = "CRITICAL"
        priority_color = "#ef4444"
        priority_desc = "Immediate attention required — high negativity and emotional distress detected."
    elif score >= 50:
        priority = "HIGH"
        priority_color = "#f97316"
        priority_desc = "Elevated concern — significant negative sentiment with strong emotion signals."
    elif score >= 25:
        priority = "MEDIUM"
        priority_color = "#eab308"
        priority_desc = "Moderate concern — some negative indicators present, monitor and respond."
    else:
        priority = "LOW"
        priority_color = "#22c55e"
        priority_desc = "Low urgency — neutral or positive tone, standard processing."

    return {
        "priority": priority,
        "priority_color": priority_color,
        "priority_score": score,
        "priority_description": priority_desc
    }


# ─── Root Cause & Response Generation ─────────────────────────

CATEGORY_ROOT_CAUSES = {
    "billing": "The complaint likely stems from an unexpected charge, billing discrepancy, or lack of transparency in the invoicing process.",
    "technical": "The issue appears to be a product malfunction, service outage, or software defect affecting the user's core experience.",
    "delivery": "Delivery delays or logistics failures have disrupted the expected service timeline, leading to customer frustration.",
    "customer_service": "A negative support interaction — possibly long wait times, unhelpful responses, or poor communication — has escalated the issue.",
    "product": "A product quality issue — defect, mismatch, or performance gap — is the likely root cause of dissatisfaction.",
    "refund": "Unresolved refund requests or unclear refund policies appear to be the source of escalating frustration.",
    "other": "The complaint reflects a broader dissatisfaction that may require cross-department review to identify the root cause."
}

RESPONSE_TEMPLATES = {
    "CRITICAL": (
        "Dear {username},\n\n"
        "We sincerely apologize for the distress this situation has caused you. Your complaint has been flagged as CRITICAL "
        "and assigned to our senior resolution team. We are treating this with the highest urgency.\n\n"
        "A dedicated team member will contact you within 2 business hours with a resolution plan. "
        "We take full responsibility for your experience and are committed to making this right.\n\n"
        "Reference ID: {id}\n\nSincerely,\nSentriMail Resolution Team"
    ),
    "HIGH": (
        "Dear {username},\n\n"
        "Thank you for bringing this to our attention. We understand your frustration and want to assure you that "
        "your complaint has been escalated to our priority queue. Our team will review your case within 4 business hours.\n\n"
        "Reference ID: {id}\n\nSincerely,\nSentriMail Support Team"
    ),
    "MEDIUM": (
        "Dear {username},\n\n"
        "Thank you for submitting your complaint. We have received your case and our support team will "
        "review it within 1–2 business days. We are committed to resolving your concern.\n\n"
        "Reference ID: {id}\n\nBest regards,\nSentriMail Support Team"
    ),
    "LOW": (
        "Dear {username},\n\n"
        "Thank you for reaching out. We have logged your feedback and will address it in due course. "
        "If your situation changes, please don't hesitate to submit an updated complaint.\n\n"
        "Reference ID: {id}\n\nBest regards,\nSentriMail Support Team"
    )
}


def _generate_root_cause(category: str, emotion_label: str, text: str) -> str:
    base = CATEGORY_ROOT_CAUSES.get(category.lower(), CATEGORY_ROOT_CAUSES["other"])
    emotion_note = ""
    if emotion_label in ("anger", "disgust"):
        emotion_note = " The emotional tone indicates this has reached a boiling point — swift resolution is essential."
    elif emotion_label == "fear":
        emotion_note = " An element of fear or anxiety is present, suggesting the user feels vulnerable or threatened."
    elif emotion_label == "sadness":
        emotion_note = " The user expresses disappointment and sadness — empathetic communication will be key."
    return base + emotion_note


def _generate_ai_response(priority: str, username: str, complaint_id: str) -> str:
    template = RESPONSE_TEMPLATES.get(priority, RESPONSE_TEMPLATES["LOW"])
    return template.format(username=username, id=complaint_id)


# ─── Main Public Function ──────────────────────────────────────

def analyze_complaint(text: str, category: str = "other", username: str = "Customer", complaint_id: str = "N/A") -> Dict[str, Any]:
    """
    Full AI analysis pipeline.
    Returns sentiment, emotion, priority, root cause, and suggested response.
    """
    _load_models()

    # 1. Sentiment
    if _use_transformers and _sentiment_pipeline:
        try:
            raw = _sentiment_pipeline(text[:512])[0]
            sentiment = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            sentiment = _rule_based_sentiment(text)
    else:
        sentiment = _rule_based_sentiment(text)

    # 2. Emotion
    if _use_transformers and _emotion_pipeline:
        try:
            raw = _emotion_pipeline(text[:512])[0]
            emotion = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            emotion = _rule_based_emotion(text)
    else:
        emotion = _rule_based_emotion(text)

    # 3. Priority
    priority_data = _compute_priority(sentiment, emotion, text)

    # 4. Root cause
    root_cause = _generate_root_cause(category, emotion["label"], text)

    # 5. AI suggested response
    ai_response = _generate_ai_response(priority_data["priority"], username, complaint_id)

    return {
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "emotion_label": emotion["label"].capitalize(),
        "emotion_score": emotion["score"],
        **priority_data,
        "root_cause_summary": root_cause,
        "ai_suggested_response": ai_response,
        "model_used": "transformer" if _use_transformers else "rule-based"
    }
