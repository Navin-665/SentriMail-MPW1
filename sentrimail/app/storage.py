import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from app.firebase_config import db
import uuid


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

def get_all_complaints():

    docs = db.collection("complaints").stream()

    complaints = []

    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        complaints.append(data)

    return complaints

def get_user_complaints(username: str) -> List[Dict]:
    complaints = _load_json(COMPLAINTS_FILE)

    return sorted(
        [c for c in complaints if c.get("username") == username],
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )


def get_complaint_by_id(cid):

    doc = db.collection("complaints").document(cid).get()

    if doc.exists:
        data = doc.to_dict()
        data["id"] = doc.id
        return data

    return None


def save_complaint(data):

    complaint_id = str(uuid.uuid4())

    db.collection("complaints").document(complaint_id).set(data)

    data["id"] = complaint_id

    return data

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