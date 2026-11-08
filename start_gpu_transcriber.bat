@echo off
rem Starts the GPU Transcriber window. Double-click this file.
rem Runs from the folder this file is in, so it works wherever the project is.
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo The project's Python environment was not found at:
    echo     %~dp0.venv
    echo Set it up first, then try again. See README.md, "For developers".
    pause
    exit /b 1
)

rem pythonw opens no console window. Problems are written to the app's log file.
start "" ".venv\Scripts\pythonw.exe" -m app.main
