@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM 1. Make sure we're running from the folder where this script lives
cd /d "%~dp0"
REM ─────────────────────────────────────────────────────────────────────────────

REM ─────────────────────────────────────────────────────────────────────────────
REM 2. Launch backend
start "Backend" /D "%~dp0backend" cmd /k ^
    "call venv\Scripts\activate && python main.py"
REM ─────────────────────────────────────────────────────────────────────────────

REM ─────────────────────────────────────────────────────────────────────────────
REM 3. Launch frontend
start "Frontend" /D "%~dp0frontend" cmd /k ^
    "npm run dev"
REM ─────────────────────────────────────────────────────────────────────────────

REM 4. Close launcher
exit