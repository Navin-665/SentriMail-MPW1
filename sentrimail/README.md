# 🛡️ SentriMail — AI Complaint Management System

> Enterprise-grade, role-based complaint management powered by transformer-based sentiment & emotion AI.

---

## 🚀 Quick Start

### 1. Install dependencies

**Option A — Full (with transformer models, ~2GB download):**
```bash
pip install -r requirements.txt
```

**Option B — Minimal (rule-based AI fallback, instant start):**
```bash
pip install -r requirements_minimal.txt
```

### 2. Run the server
```bash
python run.py
```

### 3. Open in browser
```
http://localhost:8000
```

---

## 👥 Demo Accounts

| Role  | Username | Password   | Access |
|-------|----------|------------|--------|
| Admin | `admin`  | `admin123` | Full inbox, all complaints, user management |
| User  | `alice`  | `alice123` | Submit complaints, view own history |
| User  | `bob`    | `bob123`   | Submit complaints, view own history |

---

## 📁 Project Structure

```
sentrimail/
├── run.py                        # Entry point
├── requirements.txt              # Full deps (with transformers + torch)
├── requirements_minimal.txt      # Minimal deps (rule-based AI only)
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI routes & app config
│   ├── ai_engine.py              # 🧠 AI pipeline (sentiment + emotion + priority)
│   ├── auth.py                   # Session-based authentication
│   └── storage.py                # JSON-based persistence layer
│
├── templates/
│   ├── base.html                 # Shared layout, nav, styles
│   ├── login.html                # Login page (split-panel design)
│   ├── register.html             # Registration page
│   ├── user_dashboard.html       # User's complaint history
│   ├── submit_complaint.html     # Submit form + live AI analysis
│   ├── admin_dashboard.html      # Admin prioritized inbox
│   ├── admin_users.html          # User management
│   └── complaint_detail.html     # Full complaint + AI detail view
│
├── data/
│   ├── users.json                # Persistent user store (auto-created)
│   └── complaints.json           # Persistent complaint store (auto-created)
│
└── static/                       # CSS/JS assets (reserved for future use)
```

---

## 🧠 AI Pipeline

### Models Used

| Model | Purpose | Library |
|-------|---------|---------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment Analysis (POSITIVE / NEGATIVE) | HuggingFace Transformers |
| `j-hartmann/emotion-english-distilroberta-base` | Emotion Detection (anger, fear, sadness, joy, disgust, surprise, neutral) | HuggingFace Transformers |

### Fallback
When transformer models are unavailable (e.g., no internet or minimal install), the system automatically falls back to a **rule-based keyword scoring engine** that replicates the same output structure.

### Priority Scoring Matrix

| Score | Priority | Trigger |
|-------|----------|---------|
| ≥ 75 | 🔴 CRITICAL | NEGATIVE + anger/fear/disgust + intensity > 85% |
| ≥ 50 | 🟠 HIGH | NEGATIVE + strong emotion + intensity > 70% |
| ≥ 25 | 🟡 MEDIUM | Moderate negativity or some emotion |
| < 25 | 🟢 LOW | Positive/neutral tone |

Urgency keywords (urgent, emergency, legal, ASAP) add +20 to the score.

---

## ✨ Features

- **Role-Based Access Control** — User and Admin portals are fully separated
- **AI Sentiment Analysis** — DistilBERT fine-tuned on SST-2 dataset
- **Emotion Detection** — 7-class emotion classifier (anger, fear, sadness, disgust, joy, surprise, neutral)
- **Priority Scoring** — Weighted matrix combining sentiment + emotion + urgency signals
- **Root Cause Analysis** — AI-generated explanations per complaint category
- **Suggested Response** — Priority-based draft response generation
- **Persistent JSON Storage** — No database setup required
- **Admin Inbox** — Prioritized, filterable complaint table with click-through detail
- **Status Management** — Admin can mark complaints as Pending / In Progress / Resolved / Rejected
- **Graceful Fallback** — Works without GPU or internet (rule-based engine)

---

## 🌐 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Root (redirect to dashboard) |
| GET/POST | `/login` | Login page |
| GET/POST | `/register` | Registration |
| GET | `/logout` | Logout |
| GET | `/user/dashboard` | User complaint history |
| GET/POST | `/user/submit` | Submit + AI analyze complaint |
| GET | `/admin/dashboard` | Admin prioritized inbox |
| GET | `/admin/complaint/{id}` | Full complaint detail |
| POST | `/admin/complaint/{id}/status` | Update complaint status |
| GET | `/admin/users` | User management |
| GET | `/api/complaints` | JSON API (admin only) |
| POST | `/api/analyze` | Analyze text on-demand |

---

## 🔧 Tech Stack

- **Backend:** FastAPI + Uvicorn
- **AI/ML:** HuggingFace Transformers (DistilBERT, DistilRoBERTa)
- **Frontend:** Jinja2 templates, vanilla CSS/JS (no build step)
- **Storage:** JSON files (no database required)
- **Auth:** Cookie-based sessions (SHA-256 password hashing)
- **Fonts:** Syne + DM Sans + DM Mono (Google Fonts)
