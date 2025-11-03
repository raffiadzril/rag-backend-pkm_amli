@echo off
echo ========================================
echo   GATA Backend API - Unified Server
echo ========================================
echo.
echo Services:
echo   - Menu Generation (Gemini + ChromaDB)
echo   - Chatbot (Gemini + RAG)
echo.
echo Server: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
)

REM Load environment variables
if exist ".env" (
    echo Loading .env file...
) else if exist "..\.env" (
    echo Loading ..\.env file...
)

echo.
echo Starting FastAPI server...
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
