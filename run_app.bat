@echo off
title Credit Card Fraud Detection Dashboard
cd /d "%~dp0credit-card-fraud-detection"

echo ======================================================================
echo Starting Credit Card Fraud Detection Dashboard...
echo Directory: %CD%
echo ======================================================================

:: Launch Python app directly (automatically selects free port and opens browser)
python app\app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Trying alternative launcher with py...
    py app\app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Trying uvicorn on port 8080...
    start "" powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8080'"
    python -m uvicorn app.app:app --host 127.0.0.1 --port 8080
)

echo.
echo Server stopped.
pause
