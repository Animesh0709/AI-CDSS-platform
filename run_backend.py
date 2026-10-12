"""
Backend Launcher Script for AI-CDSS
Starts the FastAPI server with uvicorn on http://localhost:8000
"""

import sys
import os
import uvicorn


current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = current_dir if os.path.basename(current_dir) == "backend" else os.path.join(current_dir, "backend")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


os.chdir(backend_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("AI-CDSS: Starting FastAPI Backend Server")
    print("API Documentation: http://localhost:8000/docs")
    print("Health check:      http://localhost:8000/api/health")
    print("=" * 60)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
