"""
Backend Launcher Script for AI-CDSS
Starts the FastAPI server with uvicorn on http://localhost:8000
"""

import sys
import os

# ── macOS fork-safety fix ────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ─────────────────────────────────────────────────────────────────────────────

import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root is the parent of backend/ — it contains the top-level 'ml/' package
repo_root = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.chdir(current_dir)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug_mode = os.environ.get("DEBUG", "true").lower() in ("true", "1", "yes")
    
    print("=" * 60)
    print(f"AI-CDSS: Starting FastAPI Backend Server on {host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health check:      http://{host}:{port}/api/health")
    print("=" * 60)
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug_mode,
        reload_dirs=["app"] if debug_mode else None
    )
