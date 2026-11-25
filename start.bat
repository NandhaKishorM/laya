@echo off
rem Start the Laya web UI + API on http://localhost:8000
rem Usage: start.bat [port]      (default port 8000, same as examples/server.py)
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

if not exist ".venv\Scripts\python.exe" (
    rem Any interpreter the package supports, newest first. Pinning one version fails on a
    rem machine that has a different supported one: pyproject declares >=3.10 and LOCAL_SETUP.md
    rem documents a 3.11 venv, so a hard py -3.12 could not set up the repo it ships in.
    echo [setup] Creating virtual environment...
    set "PYEXE="
    for %%V in (3.13 3.12 3.11 3.10) do if not defined PYEXE (py -%%V -c "pass" >nul 2>&1 && set "PYEXE=py -%%V")
    if not defined PYEXE (
        echo [error] No supported Python found. Install Python 3.10 or newer from python.org.
        exit /b 1
    )
    !PYEXE! -m venv .venv || goto :fail
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
