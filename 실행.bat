@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "PetTriage_Launcher\__main__.py"
) else (
    python "PetTriage_Launcher\__main__.py"
)
if errorlevel 1 pause
