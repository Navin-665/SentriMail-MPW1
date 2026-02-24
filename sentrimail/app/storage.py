import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

COMPLAINTS_FILE = os.path.join(DATA_DIR, "complaints.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


# --------------------------------------------------
# Init Files
# --------------------------------------------------

def _ensure_files():
    if not os.path.exists(COMPLAINTS_FILE):
        with open(COMPLAINTS_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)


def _load_json(path: str):
    _ensure_files()

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# --------------------------------------------------
# Complaints
# --------------------------------------------------

def get_all_complaints() -> List[Dict]:
    return _load_json(COMPLAINTS_FILE)


def get_user_complaints(username: str) -> List[Dict]:
    complaints = _load_json(COMPLAINTS_FILE)

    return sorted(
        [c for c in complaints if c.get("username") == username],
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )


def get_complaint_by_id(complaint_id: str) -> Dict | None:
    complaints = _load_json(COMPLAINTS_FILE)

    return next(
        (c for c in complaints if c.get("id") == complaint_id),
        None,
    )


def save_complaint(data: Dict[str, Any]) -> Dict:

    complaints = _load_json(COMPLAINTS_FILE)

    complaint = {
        "id": str(uuid.uuid4())[:8].upper(),
        "title": data.get("title", ""),
        "category": data.get("category", "other"),
        "description": data.get("description", ""),
        "username": data.get("username", ""),
        "email": data.get("email", ""),
        "status": data.get("status", "pending"),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),

        # AI fields
        "sentiment_label": data.get("sentiment_label", "NEUTRAL"),
        "sentiment_score": data.get("sentiment_score", 0.5),
        "emotion_label": data.get("emotion_label", "Neutral"),
        "emotion_score": data.get("emotion_score", 0.5),
        "priority": data.get("priority", "LOW"),
        "priority_color": data.get("priority_color", "#22c55e"),
        "priority_score": data.get("priority_score", 0),
        "priority_description": data.get("priority_description", ""),
        "root_cause_summary": data.get("root_cause_summary", ""),
        "ai_suggested_response": data.get("ai_suggested_response", ""),
        "model_used": data.get("model_used", "lightweight"),
        "admin_response": data.get("admin_response", ""),
    }

    complaints.append(complaint)

    _save_json(COMPLAINTS_FILE, complaints)

    return complaint


def update_complaint_status(complaint_id: str, status: str) -> bool:

    complaints = _load_json(COMPLAINTS_FILE)

    for c in complaints:
        if c["id"] == complaint_id:
            c["status"] = status
            c["updated_at"] = datetime.now().isoformat()

            _save_json(COMPLAINTS_FILE, complaints)
            return True

    return False


def update_complaint_response(
    complaint_id: str,
    response: str,
    status: str = "resolved",
) -> bool:

    complaints = _load_json(COMPLAINTS_FILE)

    for c in complaints:
        if c["id"] == complaint_id:
            c["admin_response"] = response
            c["status"] = status
            c["updated_at"] = datetime.now().isoformat()

            _save_json(COMPLAINTS_FILE, complaints)
            return True

    return False