# Technical Architecture & Codebase Documentation

## Project Overview
The Placement Bot is an AI-powered career guidance system that helps users with resume analysis, skill gap identification, career roadmap generation, PDF analysis, and interview preparation through a RAG-enabled chatbot.

## System Architecture

### Core Components
1. **Model Manager** - Centralized AI model management using Hugging Face API
2. **Vector Database** - Local FAISS+SQLite storage for embeddings and metadata
3. **RAG Modules** - Resume analyzer, PDF analyzer, and chatbot with retrieval capabilities
4. **Specialized Analyzers** - Career roadmap generator and skill gap analyzer

### Technology Stack
- **Language**: Python 3.x
- **AI/ML**: Hugging Face Transformers, Sentence Transformers
- **Vector Storage**: FAISS (Facebook AI Similarity Search)
- **Database**: SQLite (metadata), FAISS (embeddings)
- **NLP Libraries**: spaCy, NLTK
- **PDF Processing**: PyMuPDF (fitz)
- **Data Visualization**: matplotlib, plotly

## Codebase Structure

```
Placement_Bot/
├── app.py                 # Main application entry point
├── modules/               # Core functional modules
│   ├── resume_analyzer.py    # Resume parsing and analysis
│   ├── pdf_analyzer.py       # PDF content extraction and analysis
│   ├── career_roadmap.py     # Career path generation
│   ├── skill_gap_analyzer.py # Skill assessment and recommendations
│   └── rag_chatbot.py        # RAG-enabled Q&A system
├── utils/                 # Utility functions and managers
│   ├── model_manager.py      # AI model loading and API management
│   ├── vector_db_manager.py  # Vector database interface
│   ├── vector_database.py    # FAISS/SQLite implementation
│   ├── question_bank.py      # Interview question repository
│   └── text_processing.py    # Text cleaning and processing
├── vector_db/             # Local vector database storage
│   ├── *.index              # FAISS embedding indexes
│   └── *.db                 # SQLite metadata databases
└── models/                # Model configuration cache
```

## Key Features

### 1. Resume Analysis
- Parses and analyzes resume content
- Provides ATS compatibility scoring
- Suggests improvements and keyword optimization
- Stores embeddings for similarity search

### 2. PDF Analysis
- Extracts text from PDF documents
- Analyzes content and identifies key topics
- Enables question-answering on PDF content
- Stores content embeddings for retrieval

### 3. Career Roadmap Generation
- Creates personalized career development plans
- Recommends learning resources and projects
- Adapts to user's experience level and goals
- Integrates with skill gap analysis

### 4. Skill Gap Analysis
- Identifies missing skills for target roles
- Recommends learning resources
- Provides project ideas for skill development
- Tracks progress and suggests next steps

### 5. RAG Chatbot
- Answers technical and HR interview questions
- Retrieves relevant context from knowledge base
- Provides citations for answers
- Maintains conversation history

## Vector Database Implementation

### Storage Structure
- **FAISS Indexes**: Store document embeddings for similarity search
- **SQLite Database**: Store document metadata and content references
- **Separate Indexes**: Resumes, jobs, knowledge items, PDF content

### Data Types
1. **Knowledge Items** - General information and templates
2. **Resumes** - Candidate resume embeddings
3. **Job Descriptions** - Position requirements
4. **PDF Content** - Document section embeddings
5. **Interview Questions** - Technical and HR Q&A pairs

## AI Model Integration

### API-Based Approach
- All models accessed via Hugging Face Inference API
- No local model downloads to minimize storage
- Supports Phi-3 Mini, sentence transformers, and specialized models
- Fallback mechanisms for API failures

### Model Types
1. **Language Models** - Text generation (Phi-3 Mini)
2. **Embedding Models** - Semantic similarity (all-MiniLM-L6-v2)
3. **Sentiment Analysis** - Text sentiment scoring
4. **Named Entity Recognition** - Information extraction

## RAG Implementation

### Retrieval Process
1. **Query Embedding** - Convert user query to vector
2. **Similarity Search** - Find relevant documents in FAISS
3. **Metadata Retrieval** - Fetch document details from SQLite
4. **Context Assembly** - Format retrieved information for LLM

### Modules with RAG
1. **Resume Analyzer** - Template retrieval and comparison
2. **PDF Analyzer** - Content-based question answering
3. **RAG Chatbot** - Knowledge base querying

## Data Flow

1. **Input Processing** - User data (resume, PDF, questions) is processed
2. **Embedding Generation** - Content converted to vectors using sentence transformers
3. **Storage** - Embeddings stored in FAISS, metadata in SQLite
4. **Retrieval** - Queries matched against stored embeddings
5. **Generation** - Context combined with prompts for LLM response
6. **Output** - Structured responses with citations and recommendations

## Deployment

### Requirements
- Python 3.8+
- Required packages in `requirements.txt`
- Hugging Face API key for model access
- FAISS for vector similarity search

### Startup
- `start_bot.bat` - Windows execution script
- `start_bot.sh` - Linux/Mac execution script
- `Procfile` - Heroku deployment configuration

## Security & Privacy
- No local storage of user data
- API keys managed through environment variables
- Temporary vector database storage only
- No persistent user profiling