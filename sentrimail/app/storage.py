from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.auth import (
    authenticate_user,
    create_session,
    get_current_user,
    logout_user,
    get_all_users,
    register_user,
)

from app.storage import (
    get_all_complaints,
    get_user_complaints,
    save_complaint,
    get_complaint_by_id,
    update_complaint_response,
    update_complaint_status,
)

from app.ai_engine import analyze_complaint


# --------------------------------------------------
# App Setup
# --------------------------------------------------

app = FastAPI(title="SentriMail", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Static + Templates
# --------------------------------------------------

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# --------------------------------------------------
# Health Check (For Railway)
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if user:
        role = user.get("role")

        if role == "admin":
            return RedirectResponse("/admin/dashboard", status_code=302)

        if role == "user":
            return RedirectResponse("/user/dashboard", status_code=302)

    return RedirectResponse("/login", status_code=302)


# --------------------------------------------------
# Auth Routes
# --------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if user:
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": error},
    )


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):

    user = authenticate_user(username, password)

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid credentials",
            },
        )

    response = RedirectResponse("/", status_code=302)
    create_session(response, user)

    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    error: str = None,
    success: str = None,
):

    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "error": error,
            "success": success,
        },
    )


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
):

    result = register_user(username, password, email)

    if not result["success"]:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": result["message"],
            },
        )

    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "success": "Account created! You can login now.",
        },
    )


@app.get("/logout")
async def logout(request: Request):

    response = RedirectResponse("/login", status_code=302)
    logout_user(response)

    return response


# --------------------------------------------------
# User Routes
# --------------------------------------------------

@app.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "user":
        return RedirectResponse("/login", status_code=302)

    complaints = get_user_complaints(user["username"])

    return templates.TemplateResponse(
        "user_dashboard.html",
        {
            "request": request,
            "user": user,
            "complaints": complaints,
            "total": len(complaints),
            "pending": len([c for c in complaints if c["status"] == "pending"]),
            "resolved": len([c for c in complaints if c["status"] == "resolved"]),
        },
    )


@app.get("/user/submit", response_class=HTMLResponse)
async def submit_page(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "user":
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        "submit_complaint.html",
        {
            "request": request,
            "user": user,
        },
    )


@app.post("/user/submit")
async def submit_complaint(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "user":
        return RedirectResponse("/login", status_code=302)

    analysis = analyze_complaint(
        description,
        category="other",
        username=user["username"],
    )

    is_low = analysis.get("priority") == "LOW"

    complaint_data = {
        "title": title,
        "category": "other",
        "description": description,
        "username": user["username"],
        "email": user.get("email", ""),
        **analysis,
    }

    if is_low:
        complaint_data["status"] = "resolved"
        complaint_data["admin_response"] = analysis.get(
            "ai_suggested_response", ""
        )

    complaint = save_complaint(complaint_data)

    return templates.TemplateResponse(
        "submit_complaint.html",
        {
            "request": request,
            "user": user,
            "success": True,
            "analysis": analysis,
            "title": title,
            "complaint": complaint,
            "auto_resolved": is_low,
        },
    )


# --------------------------------------------------
# Admin Routes
# --------------------------------------------------

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaints = get_all_complaints()

    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    complaints_sorted = sorted(
        complaints,
        key=lambda x: priority_order.get(
            x.get("priority", "LOW"), 3
        ),
    )

    stats = {
        "total": len(complaints),
        "critical": len([c for c in complaints if c.get("priority") == "CRITICAL"]),
        "high": len([c for c in complaints if c.get("priority") == "HIGH"]),
        "medium": len([c for c in complaints if c.get("priority") == "MEDIUM"]),
        "low": len([c for c in complaints if c.get("priority") == "LOW"]),
        "pending": len([c for c in complaints if c.get("status") == "pending"]),
    }

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "user": user,
            "complaints": complaints_sorted,
            "stats": stats,
        },
    )


@app.get("/admin/complaint/{complaint_id}", response_class=HTMLResponse)
async def complaint_detail(
    request: Request,
    complaint_id: str,
):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = get_complaint_by_id(complaint_id)

    if not complaint:
        raise HTTPException(status_code=404)

    return templates.TemplateResponse(
        "complaint_detail.html",
        {
            "request": request,
            "user": user,
            "complaint": complaint,
        },
    )


@app.post("/admin/complaint/{complaint_id}/status")
async def update_status(
    request: Request,
    complaint_id: str,
    status: str = Form(...),
):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    update_complaint_status(complaint_id, status)

    return RedirectResponse(
        f"/admin/complaint/{complaint_id}",
        status_code=302,
    )


@app.post("/admin/complaint/{complaint_id}/response")
async def update_response(
    request: Request,
    complaint_id: str,
    response: str = Form(...),
    status: str = Form("resolved"),
):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    update_complaint_response(
        complaint_id,
        response,
        status,
    )

    return RedirectResponse(
        f"/admin/complaint/{complaint_id}",
        status_code=302,
    )


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    users = get_all_users()

    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "user": user,
            "users": users,
        },
    )


# --------------------------------------------------
# API
# --------------------------------------------------

@app.get("/api/complaints")
async def api_complaints(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403)

    return get_all_complaints()


@app.post("/api/analyze")
async def api_analyze(request: Request):

    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if not user:
        raise HTTPException(status_code=401)

    body = await request.json()
    text = body.get("text", "")

    return analyze_complaint(text)