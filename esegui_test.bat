@echo off
REM Esegue i test automatici (motori, analisi guidata, interfaccia).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente virtuale non trovato: vedi README.md, sezione Avvio.
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q "pytest>=8.3"
".venv\Scripts\python.exe" -m pytest -q
pause
