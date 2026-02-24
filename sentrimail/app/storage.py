"""
SentriMail Storage
-------------------
JSON-based persistent storage for complaints.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COMPLAINTS_FILE = os.path.join(DATA_DIR, "complaints.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_complaints() -> List[Dict]:
    _ensure_data_dir()
    if not os.path.exists(COMPLAINTS_FILE):
        return []
    with open(COMPLAINTS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_complaints(complaints: List[Dict]):
    _ensure_data_dir()
    with open(COMPLAINTS_FILE, "w") as f:
        json.dump(complaints, f, indent=2, default=str)


def save_complaint(complaint_data: Dict[str, Any]) -> Dict:
    complaints = _load_complaints()

    complaint_id = str(uuid.uuid4())[:8].upper()

    status = complaint_data.get("status", "pending")

    complaint = {
        "id": complaint_id,
        "title": complaint_data.get("title", "Untitled"),
        "category": complaint_data.get("category", "other"),
        "description": complaint_data.get("description", ""),
        "username": complaint_data.get("username", "anonymous"),
        "email": complaint_data.get("email", ""),
        "status": status,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        # AI analysis fields
        "sentiment_label": complaint_data.get("sentiment_label", "NEUTRAL"),
        "sentiment_score": complaint_data.get("sentiment_score", 0.5),
        "emotion_label": complaint_data.get("emotion_label", "Neutral"),
        "emotion_score": complaint_data.get("emotion_score", 0.5),
        "priority": complaint_data.get("priority", "LOW"),
        "priority_color": complaint_data.get("priority_color", "#22c55e"),
        "priority_score": complaint_data.get("priority_score", 0),
        "priority_description": complaint_data.get("priority_description", ""),
        "root_cause_summary": complaint_data.get("root_cause_summary", ""),
        "ai_suggested_response": complaint_data.get("ai_suggested_response", ""),
        "model_used": complaint_data.get("model_used", "rule-based"),
        "admin_response": complaint_data.get("admin_response", ""),
    }

    complaints.append(complaint)
    _save_complaints(complaints)
    return complaint


def get_all_complaints() -> List[Dict]:
    return _load_complaints()


def get_user_complaints(username: str) -> List[Dict]:
    complaints = _load_complaints()
    user_complaints = [c for c in complaints if c.get("username") == username]
    return sorted(user_complaints, key=lambda x: x.get("created_at", ""), reverse=True)


def update_complaint_status(complaint_id: str, status: str) -> bool:
    complaints = _load_complaints()
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            complaint["status"] = status
            complaint["updated_at"] = datetime.now().isoformat()
            _save_complaints(complaints)
            return True
    return False


def get_complaint_by_id(complaint_id: str) -> Dict | None:
    complaints = _load_complaints()
    return next((c for c in complaints if c["id"] == complaint_id), None)


def update_complaint_response(complaint_id: str, response: str, status: str = "resolved") -> bool:
    complaints = _load_complaints()
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            complaint["admin_response"] = response
            complaint["status"] = status
            complaint["updated_at"] = datetime.now().isoformat()
            _save_complaints(complaints)
            return True
    return False
