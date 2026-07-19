"""
SentriMail — Entry Point
Run with: python run.py
"""

import uvicorn
import os
import sys

# Make console output resilient on Windows code pages (cp1252, etc.)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure the project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load .env file if it exists in the project root
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()
    except Exception as e:
        print(f"Could not load .env file: {e}")

# Create data directory with absolute path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8088))
    print("\n" + "="*50)
    try:
        print("  🛡️  SentriMail — AI Complaint Management")
    except UnicodeEncodeError:
        print("  SentriMail - AI Complaint Management")
    print("="*50)
    print(f" -> Running on port {port}")
    print(f"  → Open in browser: http://localhost:{port}")
    print("  → Admin:  admin / admin123")
    print("  → User:   alice / alice123  |  bob / bob123")
    print("="*50 + "\n")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

