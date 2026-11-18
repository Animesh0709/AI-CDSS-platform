#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  AI-CDSS — Mac Launcher (Backend + Frontend)
#  Usage: bash start_mac.sh
# ═══════════════════════════════════════════════════════════════
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO/.venv-mac/bin/python3"

if [ ! -f "$VENV_PY" ]; then
  echo "ERROR: .venv-mac not found. Run setup first:"
  echo "  python3 -m venv .venv-mac && .venv-mac/bin/pip install -r backend/requirements.txt"
  exit 1
fi

export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

echo "════════════════════════════════════════════════════════"
echo "  AI-CDSS Backend   →  http://localhost:8000"
echo "  Swagger UI         →  http://localhost:8000/docs"
echo "  Frontend (dev)     →  http://localhost:5173"
echo "════════════════════════════════════════════════════════"

# Start backend in background
cd "$REPO/backend"
"$VENV_PY" run_backend.py &
BACKEND_PID=$!

# Start frontend dev server
cd "$REPO/frontend"
node_modules/.bin/vite &
FRONTEND_PID=$!

echo "Backend PID:  $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
