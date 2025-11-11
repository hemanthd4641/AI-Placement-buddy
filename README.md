# 🤖 AI Placement Mentor Bot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered mentor bot to help students prepare for placements with resume analysis, interview preparation, and career planning - built entirely with **free/open-source AI tools**.

The Placement Bot is an intelligent career guidance system that leverages state-of-the-art AI models and Retrieval-Augmented Generation (RAG) to provide personalized assistance for job seekers. With offline capabilities and efficient model caching, it offers a comprehensive suite of tools for resume optimization, skill gap analysis, career roadmap generation, and interview preparation.

## 📚 Documentation

- [LLM and Vector DB Integration Guide](LLM_VECTOR_DB_INTEGRATION.md)
- [Feature Workflow Documentation](FEATURE_WORKFLOW.md)
- [Technical Architecture](TECHNICAL_ARCHITECTURE.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Integration Summary](INTEGRATION_SUMMARY.md)
- [Project Issues and Solutions](PROJECT_ISSUES_AND_SOLUTIONS.md)
- [Technical Challenges Summary](TECHNICAL_CHALLENGES_SUMMARY.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Model Caching Improvements](documentation/model_caching_improvements.md)

## 🎯 Features

### 📄 **Resume Analyzer**
- **PDF/DOCX Support**: Upload resumes in multiple formats
- **ATS Scoring**: Get compatibility scores with Applicant Tracking Systems
- **Skills Extraction**: Automatic identification of technical and soft skills using NER
- **Gap Analysis**: Compare your skills with job descriptions
- **Improvement Suggestions**: Actionable recommendations for resume enhancement
- **Interview Questions**: Generate personalized technical and HR interview questions

### 🔍 **Skill Gap Analyzer**
- **Comprehensive Analysis**: Compare resume skills with job requirements
- **Missing Skills Identification**: Detailed breakdown of skill gaps
- **Learning Recommendations**: Personalized course and resource suggestions
- **Priority Ranking**: Essential vs. nice-to-have skills
- **Progress Tracking**: Monitor skill development over time

### 🗺️ **Career Roadmap Generator**
- **Personalized Roadmaps**: Custom learning paths based on target role and experience
- **Phase-based Learning**: Structured progression from foundations to advanced skills
- **Resource Integration**: Curated courses, tutorials, and documentation
- **Progress Tracking**: Milestone completion and overall progress monitoring
- **Export Functionality**: Download roadmaps as JSON or PDF

### 📄 **PDF Analyzer**
- **Document Analysis**: Extract and analyze content from PDF documents
- **AI Summarization**: Generate concise summaries of lengthy documents
- **Q&A Bot**: Ask questions about the document content
- **Vector Database Integration**: Store and retrieve document information

### 💬 **Placement Chatbot (RAG)**
- **Retrieval-Augmented Generation**: Answers based on knowledge base and internet sources
- **Technical Skills**: Guidance on programming languages, frameworks, and tools
- **DSA Preparation**: Help with data structures and algorithms concepts
- **Software Engineering**: Best practices and system design principles
- **HR Interview Prep**: Common HR questions and recommended responses

## 🛠️ Tech Stack (100% Free & Open Source)

### Core Technologies
- **Frontend Framework**: Streamlit - For creating interactive web applications
- **AI/ML Framework**: PyTorch - Deep learning framework for model inference
- **NLP Libraries**: Hugging Face Transformers - For state-of-the-art language models
- **Vector Database**: FAISS - For efficient similarity search and clustering
- **Natural Language Processing**: spaCy, NLTK - For text processing and named entity recognition
- **Document Processing**: PyMuPDF, python-docx - For handling PDF and DOCX files
- **Data Visualization**: Matplotlib, Plotly - For creating charts and visualizations

### AI Models
- **Primary LLM**: Microsoft Phi-3 Mini (4k) - Lightweight yet powerful language model
- **Fallback LLM**: DistilGPT-2 - Smaller language model for text generation
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 - For generating text embeddings
- **Sentiment Analysis**: cardiffnlp/twitter-roberta-base-sentiment-latest - For sentiment classification
- **Named Entity Recognition**: en_core_web_sm (spaCy) - For extracting entities from text

### Infrastructure
- **Model Caching**: Local storage with intelligent caching to avoid repeated downloads
- **Offline Operation**: All models can be downloaded and run locally
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Bot**:
   Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
placement-bot/
├── app.py                    # Main Streamlit application
├── modules/                  # Core modules
│   ├── resume_analyzer.py    # Resume parsing and analysis
│   ├── skill_gap_analyzer.py # Skill gap identification
│   ├── career_roadmap.py     # Career roadmap creation
│   ├── pdf_analyzer.py       # PDF document analysis
│   └── rag_chatbot.py        # RAG-powered chatbot
├── data/                     # Datasets and knowledge base
│   ├── placement_pipeline.db # SQLite database for application data
│   └── vector databases      # FAISS indexes for semantic search
├── models/                   # Model cache and embeddings
│   ├── sentence_transformers/ # Cached sentence transformer models
│   └── transformers/         # Cached transformer models
├── utils/                    # Utility functions
│   ├── text_processing.py
│   ├── model_manager.py
│   ├── vector_database.py
│   ├── vector_db_manager.py
│   └── pipeline_manager.py
├── vector_db/                # Vector database indexes and metadata
├── scripts/                  # Utility scripts
│   └── smoke_test_roadmap.py # Testing script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🔧 Configuration

The bot uses free/open-source models that are automatically downloaded and cached locally:

### Language Models
- **Primary LLM**: microsoft/Phi-3-mini-4k-instruct - Advanced conversational AI model
- **Fallback LLM**: distilgpt2 - Lightweight text generation model

### NLP Models
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 - For semantic similarity
- **NER**: en_core_web_sm (spaCy) - For named entity recognition
- **Sentiment Analysis**: cardiffnlp/twitter-roberta-base-sentiment-latest - For sentiment classification

### Model Caching
- Models are downloaded once and cached locally in the `models/` directory
- Intelligent caching system prevents redundant downloads
- Model status tracked in `models/model_cache.json`

## 🎯 Usage Examples

1. **Resume Analysis**: Upload your resume and get detailed feedback with ATS compatibility scoring, skill extraction, and improvement suggestions
2. **Skill Gap Analysis**: Identify missing skills by comparing your resume with job descriptions and get personalized learning recommendations
3. **Career Roadmaps**: Generate personalized learning paths with structured phases, resources, and project ideas based on your target role
4. **PDF Analysis**: Upload documents to extract content, generate summaries, and ask questions about the material
5. **Interview Preparation**: Get personalized technical and HR interview questions based on your resume and target role
6. **Chatbot Assistance**: Ask questions about technical skills, DSA, software engineering, and HR interviews with answers augmented by a knowledge base

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

### Hugging Face Spaces
1. Create new Space
2. Upload code and requirements
3. Configure as Streamlit app

## 🧠 How It Works

The Placement Bot leverages several advanced AI techniques to provide personalized career guidance:

### Retrieval-Augmented Generation (RAG)
- Combines language models with a vector database for context-aware responses
- Retrieves relevant information from a knowledge base before generating answers
- Provides accurate and up-to-date information without hallucination

### Model Caching and Offline Operation
- All AI models are downloaded once and cached locally
- Subsequent runs use cached models for faster startup
- Works completely offline after initial setup

### Feature Integration
- **Resume Analysis**: Uses NER to extract skills and experiences, compares with job requirements
- **Skill Gap Analysis**: Leverages embeddings to find semantic similarities between skills
- **Career Roadmaps**: Combines template-based approaches with LLM-generated personalized content
- **PDF Analysis**: Extracts text, stores in vector database, and enables question-answering
- **Chatbot**: Uses RAG to provide accurate answers based on knowledge base and conversation history

### Data Flow
1. User interacts with Streamlit frontend
2. Application loads cached AI models on first run
3. User inputs (resumes, job descriptions, questions) are processed
4. Relevant data retrieved from vector database using FAISS
5. LLM generates personalized responses based on context
6. Results displayed with visualizations and recommendations