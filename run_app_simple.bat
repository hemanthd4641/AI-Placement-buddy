@echo off
echo 🤖 AI Placement Mentor Bot - Direct Run Script
echo ========================================
echo.

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Start the application directly
echo 🚀 Starting AI Placement Mentor Bot...
echo.
streamlit run app.py

pause