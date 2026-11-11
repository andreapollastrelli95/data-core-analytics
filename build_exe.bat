@echo off
REM Crea dist\DataCoreAnalytics.exe (file unico, senza console).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente virtuale non trovato. Crealo con:
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name DataCoreAnalytics ^
    --collect-data customtkinter ^
    --hidden-import matplotlib.backends.backend_tkagg ^
    --collect-submodules google.genai ^
    --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide6 ^
    --exclude-module IPython --exclude-module pytest ^
    main.py
echo.
echo Fatto: dist\DataCoreAnalytics.exe
