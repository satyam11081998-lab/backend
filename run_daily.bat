@echo off
setlocal
cd /d "%~dp0"

set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"

"%PY%" -m bot %*
set "RC=%ERRORLEVEL%"
exit /b %RC%
