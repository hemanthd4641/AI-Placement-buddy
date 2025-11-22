# AI Placement Mentor Bot


An AI-powered placement preparation assistant focused on resume analysis, skill-gap recommendations, career roadmaps, and robust PDF document Q&A. Built with 100% free and open-source tools. Uses Google Gemini as llm

## 🎯 Overview

This project helps students and job seekers prepare for placements with advanced tools for resume analysis, skill development, and job application preparation. It uses state-of-the-art AI models while being completely free and open-source.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Placement_Bot
   ```

2. **Set up environment variables (optional)**
   Create a `.env` file in the project root if you want to use Google Gemini:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # Optional: EMBEDDING_PROVIDER=local|gemini|auto (default: local)
   ```
   If you do not set a Gemini API key, the project will use local sentence-transformers for embeddings and LLM fallback logic.

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


## 🛠️ Tech Stack (100% Free & Open Source)

### Core Technologies
- **Frontend Framework**: Streamlit - For creating interactive web applications
- **AI/ML Framework**: PyTorch - Deep learning framework for model inference
- **LLM/Embeddings**: Google Gemini API (preferred if configured), or local sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS - For efficient similarity search and clustering
- **Natural Language Processing**: spaCy, NLTK - For text processing and named entity recognition
- **Document Processing**: PyMuPDF, python-docx - For handling PDF and DOCX files
- **Data Visualization**: Matplotlib, Plotly - For creating charts and visualizations

### AI Models
- **Primary LLM**: Google Gemini (model: gemini-2.5-flash) via API, or local fallback
- **Embedding Model**: all-MiniLM-L6-v2 (local sentence-transformers) or Gemini embeddings
- **Sentiment/NER**: Provided by Gemini or local NLP libraries (spaCy, NLTK)

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
│   
├── utils/                # Utility functions
│   ├── model_manager.py
│   ├── vector_db_manager.py
│   ├── vector_database.py
│   ├── text_processing.py
│   ├── templates.py
│   ├── question_bank.py
│   └── question_bank_bulk.py
├── vector_db/            # Vector database files
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone <your-fork-url>
   cd Placement_Bot
   ```
3. **Create a new feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** (code, docs, tests, etc.).
5. **Test your changes**:
   - Run the app and/or tests to ensure nothing is broken.
   - If you add new features, please add or update tests if possible.
6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Describe your changes"
   ```
7. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request** from your branch to the main repository's `main` branch.
9. **Describe your changes** clearly in the PR description and reference any related issues.

**Code style & review:**
- Please follow the existing code style and structure.
- Ensure your code passes before submitting a PR.
- Be responsive to review feedback and update your PR as needed.

## 📄 License

This project is licensed under the MIT License.

**Owner:** hemanthd4641  
**GitHub:** https://github.com/hemanthd4641  
**Contact:** hemanthd4641@gmail.com

See the [LICENSE](LICENSE) file for details.
