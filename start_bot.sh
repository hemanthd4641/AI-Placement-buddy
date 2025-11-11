#!/bin/bash

echo "🤖 AI Placement Mentor Bot - Startup Script"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "❌ pip3 is not installed"
    echo "Please install pip"
    exit 1
fi

# Install requirements if not already installed
echo "📦 Checking and installing dependencies..."
pip3 install -r requirements.txt

# Install spaCy model if not already installed
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null
then
    echo "📦 Installing spaCy English model..."
    python3 -m spacy download en_core_web_sm
fi

# Install NLTK data if not already installed
if ! python3 -c "import nltk; nltk.data.find('tokenizers/punkt')" &> /dev/null
then
    echo "📦 Installing NLTK data..."
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
fi

# Start the application
echo "🚀 Starting AI Placement Mentor Bot..."
echo
streamlit run app.py