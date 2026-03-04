"""
SentriMail Authentication
--------------------------
Session-cookie based auth. No JWT for MVP simplicity.
Users stored in data/users.json
"""

import json
import os
import hashlib
import secrets
from typing import Optional, Dict
from fastapi import Request
from fastapi.responses import Response
from app.firebase_config import db

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS: Dict[str, dict] = {}  # In-memory session store

SESSION_COOKIE = "sentrimail_session"


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_users() -> list:
    _ensure_data_dir()
    if not os.path.exists(USERS_FILE):
        # Seed default users
        default_users = [
            {
                "username": "admin",
                "password": _hash_password("admin123"),
                "email": "admin@sentrimail.io",
                "role": "admin"
            },
            {
                "username": "alice",
                "password": _hash_password("alice123"),
                "email": "alice@example.com",
                "role": "user"
            },
            {
                "username": "bob",
                "password": _hash_password("bob123"),
                "email": "bob@example.com",
                "role": "user"
            }
        ]
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f, indent=2)
        return default_users

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: list):
    _ensure_data_dir()
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate_user(username, password):

    users = db.collection("users").where("username", "==", username).stream()

    for user in users:
        data = user.to_dict()

        if data["password"] == password:
            return data

    return None

def register_user(username, password, email):

    users_ref = db.collection("users")

    existing = users_ref.where("username", "==", username).stream()

    for user in existing:
        return {"success": False, "message": "Username already exists"}

    users_ref.add({
        "username": username,
        "password": password,
        "email": email,
        "role": "user"
    })

    return {"success": True}
def create_session(response: Response, user: dict):
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = user
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax"
    )


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return SESSIONS.get(token)


def logout_user(response: Response):
    response.delete_cookie(SESSION_COOKIE)


def get_all_users() -> list:
    users = _load_users()
    return [{k: v for k, v in u.items() if k != "password"} for u in users]
