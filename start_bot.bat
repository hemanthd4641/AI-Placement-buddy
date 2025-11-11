@echo off
echo 🤖 AI Placement Mentor Bot - Startup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if pip is installed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed
    echo Please install pip
    pause
    exit /b 1
)

REM Install requirements if not already installed
echo 📦 Checking and installing dependencies...
python -m pip install -r requirements.txt

REM Install spaCy model if not already installed
echo 📦 Checking and installing spaCy model...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing spaCy English model...
    python -m spacy download en_core_web_sm
)

REM Install NLTK data if not already installed
echo 📦 Checking and installing NLTK data...
python -c "import nltk; nltk.data.find('tokenizers/punkt')" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing NLTK data...
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
)

REM Start the application
echo 🚀 Starting AI Placement Mentor Bot...
echo.
streamlit run app.py

pause