import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

firebase_key = json.loads(os.getenv("FIREBASE_KEY"))

cred = credentials.Certificate(firebase_key)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()