"""
Backend Launcher Script for AI-CDSS
Starts the FastAPI server with uvicorn on http://localhost:8000
"""

import sys
import os

# ── macOS fork-safety fix ────────────────────────────────────────────────────
# XGBoost / OpenMP libraries crash when used inside a process that was
# fork()ed by uvicorn's reloader on macOS (Objective-C runtime restriction).
# Setting these env vars before any import resolves the crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")               # single-threaded OpenMP
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # ObjC fork guard
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    # HuggingFace safety
# ─────────────────────────────────────────────────────────────────────────────

import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.chdir(current_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("AI-CDSS: Starting FastAPI Backend Server")
    print("API Documentation: http://localhost:8000/docs")
    print("Health check:      http://localhost:8000/api/health")
    print("=" * 60)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

