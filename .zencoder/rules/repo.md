---
description: Repository Information Overview
alwaysApply: true
---

# SentriMail Information

## Summary

SentriMail is an enterprise-grade, AI-powered complaint management system built with FastAPI. It leverages transformer-based sentiment and emotion analysis to automatically prioritize complaints, detect patterns, and assist with resolution workflows. The system features role-based access control, persistent JSON storage, and graceful fallback to rule-based AI when transformer models are unavailable. No database setup is required for deployment.

## Structure

```
sentrimail/
├── run.py                        # Entry point (uvicorn server startup)
├── requirements.txt              # Full dependencies (with transformers + torch)
├── runtime.txt                   # Python version specification
├── Procfile                      # Procfile for deployment (Heroku)
│
├── app/                          # Application logic
│   ├── main.py                   # FastAPI routes & application configuration
│   ├── ai_engine.py              # AI pipeline (sentiment, emotion, priority scoring)
│   ├── auth.py                   # Session-based authentication & user management
│   └── storage.py                # JSON-based persistence layer
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                 # Base layout with navigation & styles
│   ├── login.html                # Login page (split-panel design)
│   ├── register.html             # User registration page
│   ├── user_dashboard.html       # User complaint history view
│   ├── submit_complaint.html     # Complaint submission form with live AI analysis
│   ├── admin_dashboard.html      # Admin prioritized complaint inbox
│   ├── admin_users.html          # User management interface
│   └── user_complaint_detail.html # Detailed complaint view with AI insights
│
├── static/                       # CSS/JS assets (reserved for future use)
│
└── data/                         # Persistent data storage
    ├── users.json                # User accounts (auto-created on first run)
    └── complaints.json           # Complaint records (auto-created on first run)
```

## Language & Runtime

**Language**: Python  
**Version**: 3.11.7  
**Build System**: None (direct Python execution)  
**Package Manager**: pip

## Dependencies

**Main Dependencies**:
- `fastapi==0.111.0` — Web framework for building APIs and routing
- `uvicorn[standard]==0.29.0` — ASGI server for running FastAPI application
- `jinja2==3.1.4` — Template engine for rendering HTML
- `python-multipart==0.0.9` — Handling multipart form data

**Optional AI Dependencies** (included in full `requirements.txt`):
- `transformers` — HuggingFace transformer models for sentiment & emotion analysis
- `torch` — PyTorch backend for transformer inference

**Fallback Mode**: System gracefully degrades to rule-based keyword scoring when transformer models cannot load (no internet, minimal install).

## Build & Installation

**Installation (Full with AI models)**:
```bash
pip install -r requirements.txt
```

**Installation (Minimal without transformers)**:
```bash
pip install -r requirements_minimal.txt
```

**Run the application**:
```bash
python run.py
```

Application starts on `http://localhost:8000` by default. Port can be customized via `PORT` environment variable.

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Root (redirects to dashboard) |
| GET/POST | `/login` | Login page & authentication |
| GET/POST | `/register` | User registration |
| GET | `/logout` | Session logout |
| GET | `/user/dashboard` | User complaint history |
| GET/POST | `/user/submit` | Submit complaint with live AI analysis |
| GET | `/admin/dashboard` | Admin prioritized inbox |
| GET | `/admin/complaint/{id}` | Full complaint detail with AI insights |
| POST | `/admin/complaint/{id}/status` | Update complaint status |
| GET | `/admin/users` | User management interface |
| GET | `/api/complaints` | JSON API (admin only) |
| POST | `/api/analyze` | On-demand text analysis |

## Main Files & Resources

**Entry Point**: `run.py`
- Initializes Uvicorn server, configures logging, creates data directory
- Starts FastAPI app at configurable port (default 8000)

**Application Configuration**: `app/main.py`
- FastAPI application setup with CORS middleware
- Mount static files and template directories
- Define all HTTP routes and handlers

**AI Engine**: `app/ai_engine.py`
- Loads sentiment model: `distilbert-base-uncased-finetuned-sst-2-english`
- Loads emotion model: `j-hartmann/emotion-english-distilroberta-base`
- Calculates priority scores based on sentiment + emotion + urgency keywords
- Generates AI-powered root cause analysis and response suggestions
- Implements rule-based fallback for offline/minimal deployments

**Authentication**: `app/auth.py`
- Session-cookie based authentication (no JWT)
- Password hashing with SHA-256
- In-memory session store with 24-hour expiration
- Default users seeded on first run (admin, alice, bob)

**Storage**: `app/storage.py`
- JSON-file based persistence for users and complaints
- Auto-creates `data/users.json` and `data/complaints.json` on first run
- No external database required

**Configuration Files**:
- `runtime.txt` — Specifies Python 3.11.7 for platform deployments
- `Procfile` — Web dyno configuration for Heroku deployment

## Demo Accounts

| Role  | Username | Password   |
|-------|----------|------------|
| Admin | `admin`  | `admin123` |
| User  | `alice`  | `alice123` |
| User  | `bob`    | `bob123`   |

## Key Features

- **Role-Based Access Control** — Separate user and admin portals
- **Transformer-Based AI** — Sentiment analysis (POSITIVE/NEGATIVE) and 7-class emotion detection (anger, fear, sadness, disgust, joy, surprise, neutral)
- **Priority Scoring Matrix** — CRITICAL (≥75), HIGH (≥50), MEDIUM (≥25), LOW (<25) based on sentiment + emotion + urgency
- **Root Cause Analysis** — LLM-style heuristic summaries for complaint categories
- **Status Management** — Mark complaints as Pending, In Progress, Resolved, or Rejected
- **Persistent JSON Storage** — No database setup required
- **Admin Complaint Inbox** — Prioritized, filterable, searchable table with click-through details
- **Graceful Fallback** — Works offline or without GPU using rule-based keyword engine
