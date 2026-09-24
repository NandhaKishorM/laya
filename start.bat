@echo off
rem Start the Laya web UI + API on http://localhost:8055
rem Usage: start.bat [port]      (default port 8055)
setlocal
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8055"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment ^(Python 3.12^)...
    py -3.12 -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install -q --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
    ".venv\Scripts\python.exe" -m pip install -q -e ".[serve]" || goto :fail
)

set PYTHONUTF8=1
echo Laya UI: http://localhost:%PORT%   (Ctrl+C to stop)
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:%PORT%"
".venv\Scripts\python.exe" examples\server.py --host 127.0.0.1 --port %PORT%
goto :eof

:fail
echo [error] Setup failed. See the messages above.
exit /b 1
