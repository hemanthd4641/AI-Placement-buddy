# AI Placement Mentor Bot

A comprehensive AI-powered placement preparation assistant built with 100% free and open-source tools.

## 🎯 Overview

This project helps students and job seekers prepare for placements with advanced tools for resume analysis, skill development, and job application preparation. It uses state-of-the-art AI models while being completely free and open-source.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Placement_Bot
   ```

2. **Set up environment variables**
   Create a `.env` file in the project root and add your Hugging Face API key:
   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```
   
   To get your API key:
   - Visit [Hugging Face](https://huggingface.co)
   - Sign up or log in to your account
   - Go to your profile settings
   - Navigate to "Access Tokens"
   - Create a new token and copy it

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Features

### 📄 **Resume Analyzer**
- **ATS Compatibility Scoring**: Evaluates how well your resume matches Applicant Tracking Systems
- **Skill Extraction**: Automatically identifies technical and soft skills from your resume
- **Experience Analysis**: Analyzes your work experience and projects
- **Improvement Suggestions**: Provides actionable feedback to improve your resume
- **Industry Match**: Suggests the best matching industry roles based on your resume
- **Visualization**: Creates charts and graphs to visualize your skills and experience
- **Export Options**: Export analysis results as JSON or PDF

### 🎯 **Skill Gap Analysis**
- **Resume vs Job Description**: Compares your resume skills with job requirements
- **Missing Skills Identification**: Identifies skills you need to develop
- **Personalized Recommendations**: AI-powered learning path suggestions
- **Resource Suggestions**: Recommended courses, tutorials, and practice platforms
- **Industry-Specific Analysis**: Tailored recommendations based on target industry
- **Progress Tracking**: Track your skill development over time

### 🛣️ **Career Roadmap Generator**
- **Personalized Learning Paths**: Creates customized roadmaps based on your target role
- **Phase-based Learning**: Divides learning into manageable phases
- **Resource Recommendations**: Suggests relevant courses, books, and tutorials
- **Project Ideas**: Provides hands-on project suggestions to build your portfolio
- **Timeline Customization**: Adjust roadmap duration to fit your schedule
- **Industry-Specific Templates**: Role-specific learning paths
- **Progress Tracking**: Track your learning progress with completion checkboxes
- **Export Options**: Export your roadmap as JSON or Markdown

### 📚 **PDF Analyzer**
- **Document Analysis**: Extracts and analyzes content from PDF documents
- **AI-powered Summarization**: Generates concise summaries of lengthy documents
- **Q&A Bot**: Ask questions about the PDF content and get AI-powered answers
- **Vector Database Storage**: Stores document content for semantic search
- **Keyword Extraction**: Identifies important keywords and concepts
- **Content Visualization**: Visual representation of document structure

### 💬 **Placement Chatbot (RAG)**
- **Retrieval-Augmented Generation**: Answers based on knowledge base and internet sources
- **Technical Skills**: Guidance on programming languages, frameworks, and tools
- **DSA Preparation**: Help with data structures and algorithms concepts
- **Software Engineering**: Best practices and system design principles
- **HR Interview Prep**: Common HR questions and recommended responses
- **Industry-Specific Advice**: Role-specific career guidance
- **Real-time Learning**: Continuously updated knowledge base

## 🛠️ Tech Stack (100% Free & Open Source)

### Core Technologies
- **Frontend Framework**: Streamlit - For creating interactive web applications
- **AI/ML Framework**: PyTorch - Deep learning framework for model inference
- **NLP Libraries**: Hugging Face Transformers - For state-of-the-art language models
- **Vector Database**: FAISS - For efficient similarity search and clustering
- **Natural Language Processing**: spaCy, NLTK - For text processing and named entity recognition
- **Document Processing**: PyMuPDF, python-docx - For handling PDF and DOCX files
- **Data Visualization**: Matplotlib, Plotly - For creating charts and visualizations

### AI Models (All accessed via Hugging Face API)
- **Primary LLM**: Microsoft Phi-3 Mini (4k) - Lightweight yet powerful language model
- **Fallback LLM**: DistilGPT-2 - Smaller language model for text generation
- **Embedding Model**: all-MiniLM-L6-v2 - For sentence embeddings and similarity calculations
- **Sentiment Analysis**: twitter-roberta-base-sentiment-latest - For sentiment classification
- **Named Entity Recognition**: bert-large-cased-finetuned-conll03-english - For entity extraction

## 📁 Project Structure

```
Placement_Bot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (not in git)
├── models/               # Model configuration cache
├── modules/              # Feature modules
│   ├── resume_analyzer.py
│   ├── skill_gap_analyzer.py
│   ├── career_roadmap.py
│   ├── pdf_analyzer.py
│   └── rag_chatbot.py
├── utils/                # Utility functions
│   ├── model_manager.py
│   ├── vector_db_manager.py
│   ├── vector_database.py
│   ├── text_processing.py
│   ├── templates.py
│   ├── question_bank.py
│   └── question_bank_bulk.py
├── vector_db/            # Vector database files
└── documentation/        # Additional documentation
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Placement_Bot
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up Hugging Face API key**
   Create a `.env` file in the project root and add your Hugging Face API key:
   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎮 Usage Guide

### Resume Analyzer
1. Navigate to the "Resume Analyzer" section
2. Upload your resume (PDF or DOCX format)
3. Click "Analyze Resume" to get detailed feedback
4. Review the analysis results and improvement suggestions
5. Export results if needed

### Skill Gap Analysis
1. Go to the "Skill Gap Analysis" section
2. Upload your resume
3. Paste the job description you're targeting
4. Click "Analyze Skills Gap with AI"
5. Review the missing skills and recommendations

### Career Roadmap Generator
1. Visit the "Career Roadmap Generator" section
2. Enter your target role and experience level
3. Provide your current skills (comma-separated)
4. Adjust the timeline preference
5. Click "Generate Career Roadmap"
6. Follow the personalized learning path

### PDF Analyzer
1. Navigate to the "PDF Analyzer" section
2. Upload a PDF document
3. Click "Analyze PDF" to process the document
4. Use the Q&A bot to ask questions about the document
5. Review the summary and key points

### Placement Chatbot
1. Go to the "Placement Chatbot" section
2. Type your question in the input field
3. Click "Ask" or press Enter
4. Review the AI-generated response with citations

## 🔍 RAG Implementation

The Placement Bot implements Retrieval-Augmented Generation (RAG) in specific modules to enhance functionality with contextual information:

### RAG-Enabled Modules
- **Resume Analyzer**: Stores and retrieves resume embeddings for semantic search
- **PDF Analyzer**: Analyzes and stores document content for question answering
- **Placement Chatbot**: Retrieves relevant information from knowledge base for contextual responses

### Non-RAG Modules
- **Skill Gap Analysis**: Uses AI-powered analysis without vector database retrieval
- **Career Roadmap Generator**: Generates personalized roadmaps without vector database retrieval

## 📊 Model Information

### Language Models (API-based)
- **Primary**: Microsoft Phi-3 Mini (4k) - Accessed via Hugging Face API
- **Fallback**: DistilGPT-2 - Accessed via Hugging Face API

### Embedding Models (API-based)
- **Sentence Embeddings**: all-MiniLM-L6-v2 - Accessed via Hugging Face API

### Specialized Models (API-based)
- **Sentiment Analysis**: twitter-roberta-base-sentiment-latest - Accessed via Hugging Face API
- **Named Entity Recognition**: bert-large-cased-finetuned-conll03-english - Accessed via Hugging Face API

### Vector Database
- **FAISS**: For efficient similarity search and clustering
- **SQLite**: For metadata storage

## 🔒 Privacy and Security

- **API Processing**: All AI model inference is performed via Hugging Face API
- **No Local Storage**: Models are not downloaded or stored locally
- **Data Privacy**: Only text prompts are sent to Hugging Face API for inference
- **No Data Storage**: Hugging Face does not store your prompts or responses
- **Environment Variables**: API keys are stored locally in `.env` file (not committed to git)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
