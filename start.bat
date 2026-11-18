@echo off
title AI-CDSS Launcher
echo ========================================================
echo   Starting Explainable AI Clinical Decision Support System
echo ========================================================
echo.

start "AI-CDSS Backend (FastAPI)" cmd /k "cd /d %~dp0 && .\backend\.venv\Scripts\python.exe run_backend.py"
start "AI-CDSS Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo [OK] Both servers are starting in separate windows.
echo - Backend API & Docs: http://localhost:8000/docs
echo - Frontend Web UI:    http://localhost:5173
echo.
pause
