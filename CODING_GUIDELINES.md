# Codebase Overview & Development Guidelines

## Project Structure

The Placement Bot follows a modular architecture with clear separation of concerns:

```
Placement_Bot/
├── app.py              # Main application orchestrator
├── modules/            # Feature-specific implementations
├── utils/              # Shared utilities and managers
├── vector_db/          # Local vector database storage
└── models/             # Model configuration cache
```

## Core Modules

### 1. Model Manager (`utils/model_manager.py`)
Centralized AI model management that uses Hugging Face API instead of local downloads.

**Key Methods:**
- `load_language_model()` - Configures Phi-3 Mini API access
- `generate_text()` - Generates text using API-based Phi-3 Mini
- `load_embedding_model()` - Configures embedding model API access
- `get_embeddings()` - Generates embeddings via API
- `load_sentiment_model()` - Configures sentiment analysis API access
- `analyze_sentiment()` - Performs sentiment analysis via API
- `load_ner_model()` - Configures NER model API access
- `extract_entities()` - Extracts entities via API

### 2. Vector Database (`utils/vector_database.py`)
Local storage system using FAISS for embeddings and SQLite for metadata.

**Key Methods:**
- `add_resume()` - Store resume embeddings
- `add_knowledge_item()` - Store general knowledge embeddings
- `search_knowledge()` - Retrieve similar knowledge items
- `search_resumes()` - Find similar resumes

### 3. Vector Database Manager (`utils/vector_db_manager.py`)
Unified interface for all vector database operations.

**Key Methods:**
- `add_knowledge_item()` - Add knowledge to database
- `search_knowledge()` - Search knowledge base
- `add_resume()` - Add resume to database
- `search_similar_resumes()` - Find similar resumes

## RAG Modules

### 1. Resume Analyzer (`modules/resume_analyzer.py`)
Analyzes resumes with vector database integration for template matching.

**Key Features:**
- Resume parsing and text extraction
- ATS scoring and optimization suggestions
- Template-based formatting recommendations
- Vector database storage and retrieval

### 2. PDF Analyzer (`modules/pdf_analyzer.py`)
Extracts and analyzes PDF content with RAG capabilities.

**Key Features:**
- PDF text extraction using PyMuPDF
- Content storage in vector database
- Question-answering on PDF content
- Relevant section retrieval

### 3. RAG Chatbot (`modules/rag_chatbot.py`)
Retrieval-augmented chatbot for technical and HR interview preparation.

**Key Features:**
- Knowledge base querying
- Context-aware response generation
- Citation tracking
- Conversation history management

## Development Guidelines

### 1. API-First Approach
All AI models should be accessed via Hugging Face API rather than local downloads to minimize storage requirements.

### 2. Vector Database Usage
RAG functionality is limited to three modules:
- Resume Analyzer
- PDF Analyzer  
- RAG Chatbot

Other modules should not use vector database features.

### 3. Error Handling
All modules should gracefully handle:
- API failures with fallback responses
- Missing vector database with warning messages
- Missing dependencies with informative errors

### 4. Resource Management
- Clear cache when appropriate using `clear_cache()`
- Unload models when not needed using `unload_models()`
- Close database connections properly

## Key Dependencies

### Runtime Requirements
- Python 3.8+
- FAISS with AVX2 support
- Sentence Transformers (all-MiniLM-L6-v2)
- PyMuPDF for PDF processing
- Hugging Face API key

### Installation
All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

## Environment Configuration

### Required Variables
- `HUGGINGFACE_API_KEY` - Hugging Face API key for model access

### Configuration Files
- `.env` - Environment variables
- `.gitignore` - Excludes sensitive and temporary files

## Testing

### Unit Testing
Create test files for new functionality:
- Test API integrations with mock responses
- Test vector database operations
- Test error handling scenarios

### Integration Testing
- Verify RAG modules work with vector database
- Confirm API-based model access functions
- Test cross-module interactions

## Performance Considerations

### Memory Management
- Use `clear_cache()` to free memory when needed
- Unload models after use with `unload_models()`
- Process large documents in chunks

### Storage Optimization
- Vector database files are stored locally in `vector_db/`
- Index files can be deleted to reset knowledge base
- Metadata database can be cleared for fresh start

## Security Practices

### Data Handling
- No persistent storage of user data
- Temporary vector database storage only
- Environment variables for API keys

### Access Control
- API keys stored in `.env` file
- `.gitignore` prevents credential exposure
- No hardcoded sensitive information