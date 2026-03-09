@echo off
start "" python Display.py
timeout /t 3 >nul
start http://127.0.0.1:7860
