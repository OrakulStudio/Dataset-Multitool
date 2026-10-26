@echo off
title ORAKUL STUDIO — SYSTEM UPDATER
cd /d "%~dp0"

echo [SYSTEM] Checking for updates from GitHub...
call orakul_env\Scripts\python.exe updater.py

echo.
pause