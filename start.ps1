# AI-CDSS Windows PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Explainable AI Clinical Decision Support System" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir'; .\backend\.venv\Scripts\python.exe run_backend.py"

# Start Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\frontend'; npm run dev"

Write-Host ""
Write-Host "[OK] Both backend and frontend servers are launching in dedicated windows!" -ForegroundColor Green
Write-Host "Backend API & Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Frontend Portal:    http://localhost:5173" -ForegroundColor Yellow
