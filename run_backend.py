"""
Root-level launcher — delegates to backend/run_backend.py
Run from the repo root: python run_backend.py
"""
import subprocess
import sys
import os

backend_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "run_backend.py")
subprocess.run([sys.executable, backend_script], check=True)
