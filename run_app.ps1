# PowerShell script to activate virtual environment and run the Placement Bot application

Write-Host "🤖 AI Placement Mentor Bot - Startup Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the Placement Bot directory" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
try {
    .\venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Check if Python is available
Write-Host "🔍 Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher" -ForegroundColor Yellow
    exit 1
}

# Install/update requirements
Write-Host "📦 Checking and installing dependencies..." -ForegroundColor Cyan
try {
    python -m pip install -r requirements.txt --upgrade
    Write-Host "✅ Dependencies installed/updated successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Install spaCy model if not already installed
Write-Host "📦 Checking and installing spaCy model..." -ForegroundColor Cyan
try {
    python -c "import spacy; spacy.load('en_core_web_sm')"
    Write-Host "✅ spaCy English model already installed!" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing spaCy English model..." -ForegroundColor Cyan
    python -m spacy download en_core_web_sm
    Write-Host "✅ spaCy English model installed successfully!" -ForegroundColor Green
}

# Install NLTK data if not already installed
Write-Host "📦 Checking and installing NLTK data..." -ForegroundColor Cyan
try {
    python -c "import nltk; nltk.data.find('tokenizers/punkt')"
    Write-Host "✅ NLTK data already installed!" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing NLTK data..." -ForegroundColor Cyan
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    Write-Host "✅ NLTK data installed successfully!" -ForegroundColor Green
}

# Start the application
Write-Host "🚀 Starting AI Placement Mentor Bot..." -ForegroundColor Green
Write-Host ""
streamlit run app.py