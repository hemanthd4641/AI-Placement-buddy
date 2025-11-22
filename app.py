"""
AI Placement Mentor Bot - Main Application
A comprehensive AI-powered placement preparation assistant
"""

import streamlit as st
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path to ensure modules can be imported
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import modules
try:
    from modules.resume_analyzer import ResumeAnalyzer
except ImportError as e:
    ResumeAnalyzer = None
    st.warning("ResumeAnalyzer module not available.")

try:
    from modules.skill_gap_analyzer import SkillGapAnalyzer
except ImportError as e:
    SkillGapAnalyzer = None
    st.warning("SkillGapAnalyzer module not available.")

try:
    from modules.career_roadmap import CareerRoadmapGenerator
except ImportError as e:
    CareerRoadmapGenerator = None
    st.warning("CareerRoadmapGenerator module not available.")

try:
    from modules.pdf_analyzer import PDFAnalyzer
except ImportError as e:
    PDFAnalyzer = None
    st.warning("PDFAnalyzer module not available.")

try:
    from modules.rag_chatbot import RAGChatbot
except ImportError as e:
    RAGChatbot = None
    st.warning("RAGChatbot module not available.")

try:
    from utils.model_manager import ModelManager
except ImportError as e:
    ModelManager = None
    st.warning("ModelManager module not available.")

# Page configuration
st.set_page_config(
    page_title="AI Placement Mentor Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern glass design
st.markdown("""
<style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 20px;
    }
    
    /* Glass morphism effect */
    .glass-card {
        background: rgba(44, 62, 80, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(52, 73, 94, 0.8);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 25px;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        background: rgba(52, 73, 94, 0.8);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem 0;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #ecf0f1, #bdc3c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(44, 62, 80, 0.6);
        backdrop-filter: blur(5px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(52, 73, 94, 0.6);
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        background: rgba(52, 73, 94, 0.8);
        transform: translateY(-3px);
        border: 1px solid rgba(52, 73, 94, 0.9);
    }
    
    .feature-card h4 {
        color: #ecf0f1;
        margin-top: 0;
        font-size: 1.3rem;
    }
    
    .feature-card p {
        color: #bdc3c7;
    }
    
    .feature-card ul {
        color: #95a5a6;
        padding-left: 1.2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.8), rgba(52, 73, 94, 0.8));
        color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(52, 73, 94, 0.9);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-card h3 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    /* Navigation styling */
    [data-testid="stSidebar"] {
        background: rgba(44, 62, 80, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(52, 73, 94, 0.8);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
        background: linear-gradient(135deg, #3d566e 0%, #34495e 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(44, 62, 80, 0.6);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        border: 1px dashed rgba(52, 73, 94, 0.8);
        padding: 2rem;
        text-align: center;
    }
    
    /* Chat interface */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
        background: rgba(44, 62, 80, 0.7);
        border: 1px solid rgba(52, 73, 94, 0.8);
    }
    
    .user-message {
        background: rgba(44, 62, 80, 0.8);
        border: 1px solid rgba(52, 73, 94, 0.9);
        margin-left: 20%;
    }
    
    .assistant-message {
        background: rgba(52, 73, 94, 0.8);
        border: 1px solid rgba(74, 96, 118, 0.9);
        margin-right: 20%;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(44, 62, 80, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(52, 73, 94, 0.8);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(52, 73, 94, 0.8);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #34495e, #2c3e50);
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stTextArea, .stNumberInput {
        background: rgba(44, 62, 80, 0.7);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        border: 1px solid rgba(52, 73, 94, 0.8);
        color: white;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ecf0f1;
    }
    
    p, li, div {
        color: #bdc3c7;
    }
    
    /* Animation for home logo */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .home-logo {
        animation: pulse 2s infinite;
        display: inline-block;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card h3 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}


def _linkify(text: str) -> str:
    """Convert plain URLs in text to HTML anchor tags for clickable links in chat messages."""
    if not text:
        return ""
    # Simple URL regex
    url_re = re.compile(r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)")
    def _replace(m):
        url = m.group(1)
        return f'<a href="{url}" target="_blank">{url}</a>'
    # Escape angle brackets lightly
    safe = text.replace('<', '&lt;').replace('>', '&gt;')
    return url_re.sub(_replace, safe)


def _stream_text(container, text: str, delay: float = 0.08):
    """Simple pseudo-streaming: show text sentence-by-sentence in the given Streamlit container."""
    if not text:
        return
    parts = re.split(r'(?<=[\.\?!])\s+', text)
    out = ""
    for p in parts:
        out = (out + " " + p).strip()
        container.markdown(f"<div class=\"chat-message assistant-message\">{_linkify(out)}</div>", unsafe_allow_html=True)
        time.sleep(delay)

def load_models():
    """Load and cache models"""
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            try:
                # Show detailed progress
                progress_text = "Loading AI models..."
                my_bar = st.progress(0, text=progress_text)
                
                # Initialize model manager
                my_bar.progress(10, text="Initializing Model Manager...")
                if ModelManager is not None:
                    st.session_state.model_manager = ModelManager()
                else:
                    st.warning("ModelManager not available.")
                
                # Initialize core components
                my_bar.progress(30, text="Loading Resume Analyzer...")
                if ResumeAnalyzer is not None:
                    st.session_state.resume_analyzer = ResumeAnalyzer()
                else:
                    st.warning("ResumeAnalyzer not available.")
                
                my_bar.progress(50, text="Loading PDF Analyzer...")
                if PDFAnalyzer is not None:
                    st.session_state.pdf_analyzer = PDFAnalyzer()
                else:
                    st.warning("PDFAnalyzer not available.")
                
                # Initialize advanced features
                # CoverLetterGenerator initialization removed as per user request
                my_bar.progress(70, text="Loading Skill Gap Analyzer...")
                if SkillGapAnalyzer is not None:
                    st.session_state.skill_gap_analyzer = SkillGapAnalyzer()
                else:
                    st.warning("SkillGapAnalyzer not available.")
                
                my_bar.progress(90, text="Loading Career Roadmap Generator...")
                if CareerRoadmapGenerator is not None:
                    st.session_state.career_roadmap_generator = CareerRoadmapGenerator()
                else:
                    st.warning("CareerRoadmapGenerator not available.")

                # Placement Chatbot (RAG) removed — no initialization performed

                # Ensure core templates and interview questions are present in the vector DB
                try:
                    my_bar.progress(97, text="Seeding core templates & interview Qs (one-time)...")
                    from utils.vector_db_manager import ensure_templates_seeded, ensure_interview_questions_seeded, is_vector_db_available
                    if is_vector_db_available():
                        ensure_templates_seeded()
                        # Seed interview questions (technical + HR)
                        ensure_interview_questions_seeded()
                except Exception:
                    pass

                st.session_state.models_loaded = True
                my_bar.progress(100, text="All models loaded successfully!")
                
                # Show success message
                st.success("✅ All models loaded successfully!")
                
                # Remove progress bar after a short delay
                time.sleep(1)
                my_bar.empty()
                
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                return False
    else:
        st.info("✅ Models already loaded from cache")
    
    return True

def show_home_page():
    """Display home page with project overview"""
    st.markdown('<h1 class="main-header"><span class="home-logo">🤖</span> AI Placement Mentor Bot</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: white; text-align: center;">Welcome to Your Personal Placement Preparation Assistant!</h2>
        <p style="text-align: center; font-size: 1.2rem;">
            This comprehensive AI-powered bot helps you prepare for placements with advanced tools for resume analysis, 
            skill development, and job application preparation - all built with free, open-source AI models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature categories with glass cards
    st.markdown("### Core Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Resume Analyzer</h4>
            <p>Upload your resume and get detailed analysis including:</p>
            <ul>
                <li>ATS compatibility score</li>
                <li>Skill extraction and gap analysis</li>
                <li>Improvement suggestions</li>
                <li>Job description matching</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>PDF Analyzer</h4>
            <p>Analyze PDF documents with advanced features:</p>
            <ul>
                <li>Text extraction and storage</li>
                <li>AI-powered summarization</li>
                <li>Q&A bot for document content</li>
                <li>Vector database integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Features Section
    st.markdown("### Advanced Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Skill Gap Analysis</h4>
            <p>Identify missing skills:</p>
            <ul>
                <li>Resume vs job description comparison</li>
                <li>Personalized learning recommendations</li>
                <li>Resource suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Career Roadmap Generator</h4>
            <p>Create personalized career plans:</p>
            <ul>
                <li>Role-specific skill development paths</li>
                <li>Learning resources and project ideas</li>
                <li>Progress tracking</li>
                <li>Timeline-based learning paths</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### Project Highlights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>100%</h3>
            <p>Free & Open Source</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Core Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>∞</h3>
            <p>Practice Sessions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>24/7</h3>
            <p>Availability</p>
        </div>
        """, unsafe_allow_html=True)

def show_resume_analyzer():
    """Display resume analyzer interface with enhanced UI/UX"""
    st.markdown('<h2 class="main-header">Resume Analyzer</h2>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><p>Upload your resume to get comprehensive analysis and improvement suggestions.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your resume file", 
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX file containing your resume",
            key="resume_analyzer_uploader"
        )
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>💡 Tips for Best Results</h4>
            <ul>
                <li>Use clear section headings</li>
                <li>Include relevant keywords</li>
                <li>Quantify achievements with numbers</li>
                <li>Keep it 1-2 pages long</li>
                <li>Save as PDF for consistency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        if st.button("Analyze Resume", type="primary"):
            with st.spinner("Analyzing your resume..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Analyze resume
                    analyzer = st.session_state.resume_analyzer
                    results = analyzer.analyze_resume(temp_path)
                    
                    # Display results in enhanced glass cards with tabs
                    st.markdown('<div class="glass-card"><h2>Resume Analysis Results</h2></div>', unsafe_allow_html=True)
                    
                    # Export options
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        if st.button("Export as JSON"):
                            try:
                                filename = analyzer.export_analysis_json(results)
                                st.success(f"Exported to {filename}")
                            except Exception as e:
                                st.error(f"Export failed: {str(e)}")
                    with col_exp2:
                        if st.button("Export as PDF"):
                            try:
                                filename = analyzer.export_analysis_pdf(results)
                                st.success(f"Exported to {filename}")
                            except Exception as e:
                                st.error(f"Export failed: {str(e)}")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "ATS Analysis", "Skills", "Industry Match", "Projects", "Recommendations", "Interview Questions"])
                    
                    with tab1:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{results['ats_score']}/100</h3>
                                <p>ATS Compatibility</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{len(results['skills'])}</h3>
                                <p>Skills Found</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{results.get('experience_years', 'N/A')}</h3>
                                <p>Years Experience</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Industry suggestion
                        st.markdown(f"""
                        <div class="glass-card">
                            <h3>Industry Match</h3>
                            <p>Your resume best matches the <strong>{results.get('industry_suggestion', 'software_engineer').replace('_', ' ').title()}</strong> role.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Skill radar chart
                        if results.get('skill_radar'):
                            st.image(f"data:image/png;base64,{results['skill_radar']}", use_column_width=True)
                        else:
                            st.info("Visualization not available. This feature requires matplotlib and numpy libraries. Please install them with: pip install matplotlib numpy")

                        # Skill visualization
                        if results.get('skill_chart'):
                            st.markdown('<div class="glass-card"><h3>Skills Distribution</h3>', unsafe_allow_html=True)
                            st.image(f"data:image/png;base64,{results['skill_chart']}", use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="glass-card"><h3>Skills Distribution</h3>', unsafe_allow_html=True)
                            st.info("Visualization not available. This feature requires matplotlib and numpy libraries. Please install them with: pip install matplotlib numpy")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="glass-card">
                                <h3>ATS Score: {results['ats_score']}/100 ({results['ats_grade']})</h3>
                                <p>Detailed breakdown of your ATS compatibility score</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # ATS breakdown chart
                            if results.get('ats_chart'):
                                st.image(f"data:image/png;base64,{results['ats_chart']}", use_column_width=True)
                        
                        with col2:
                            st.markdown('<div class="glass-card"><h3>Detailed Feedback</h3>', unsafe_allow_html=True)
                            ats_feedback = results.get('ats_detailed_feedback', {})
                            
                            for category, feedback_list in ats_feedback.items():
                                if feedback_list:
                                    st.markdown(f"**{category.replace('_', ' ').title()}:**")
                                    for feedback in feedback_list:
                                        st.markdown(f"• {feedback}")
                                    st.markdown("---")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown('<div class="glass-card"><h3>Skills Analysis</h3>', unsafe_allow_html=True)
                        
                        # Skill radar chart
                        if results.get('skill_radar'):
                            st.image(f"data:image/png;base64,{results['skill_radar']}", use_column_width=True)
                        
                        # Detailed skills by category
                        detailed_skills = results.get('detailed_skills', {})
                        if detailed_skills:
                            st.markdown("### Skills by Category")
                            for category, skills in detailed_skills.items():
                                if skills:
                                    st.markdown(f"**{category.replace('_', ' ').title()}:**")
                                    st.markdown(", ".join(skills))
                                    st.markdown("---")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab4:
                        st.markdown('<div class="glass-card"><h3>Industry Comparison</h3>', unsafe_allow_html=True)
                        
                        industry_comparison = results.get('industry_comparison', {})
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{industry_comparison.get('comparison_score', 0)}/100</h3>
                            <p>Industry Match Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"**Industry:** {industry_comparison.get('industry', 'software_engineer').replace('_', ' ').title()}")
                        
                        # Experience analysis
                        exp_analysis = industry_comparison.get('experience_analysis', {})
                        st.markdown(f"**Experience:** You have {exp_analysis.get('user_experience', 0)} years, industry average is {exp_analysis.get('industry_avg', 0)} years")
                        
                        # Skills analysis
                        skills_analysis = industry_comparison.get('skills_analysis', {})
                        st.markdown(f"**Skills Found:** {skills_analysis.get('user_skills', 0)}, Industry minimum: {skills_analysis.get('industry_min', 0)}")
                        
                        # Missing key skills
                        missing_skills = skills_analysis.get('missing_key_skills', [])
                        if missing_skills:
                            st.markdown("**Missing Key Skills:**")
                            st.markdown(", ".join(missing_skills))
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab5:
                        st.markdown('<div class="glass-card"><h3>Projects Analysis</h3>', unsafe_allow_html=True)
                        
                        # Projects information
                        projects = results.get('projects', {})
                        if projects.get('has_projects'):
                            st.markdown("### 📁 Projects Found")
                            st.markdown(f"**Project Count:** {projects.get('project_count', 0)}")
                            
                            # Display structured projects if available
                            structured_projects = projects.get('structured_projects', [])
                            if structured_projects:
                                st.markdown("### Project Details:")
                                for i, project in enumerate(structured_projects, 1):
                                    with st.expander(f"Project {i}: {project.get('title', 'Untitled Project')}"):
                                        st.markdown(f"**Description:** {project.get('description', 'No description available')}")
                                        
                                        technologies = project.get('technologies', [])
                                        if technologies:
                                            st.markdown(f"**Technologies Used:** {', '.join(technologies)}")
                                        
                                        keywords = project.get('keywords', [])
                                        if keywords:
                                            st.markdown(f"**Keywords:** {', '.join(keywords)}")
                            else:
                                # Display projects text using the existing method
                                projects_text = projects.get('projects_text', '')
                                if projects_text:
                                    st.markdown("### Project Details:")
                                    # Only show lines that look like actual projects
                                    project_lines = projects_text.split('\n')
                                    project_lines = [line for line in project_lines if line.strip() and 
                                                   any(keyword in line.lower() for keyword in [
                                                       'project', 'developed', 'built', 'created', 'designed', 
                                                       'implemented', 'application', 'system', 'website', 'software'
                                                   ])]
                                    
                                    if project_lines:
                                        filtered_projects_text = '\n'.join(project_lines)
                                        st.text_area("Extracted Projects", value=filtered_projects_text, height=300, key="projects_display")
                                    else:
                                        st.info("No clear project information found in the extracted content.")
                            
                            # Recommendations for projects
                            st.markdown("### 💡 Project Improvement Suggestions")
                            st.markdown("""
                            - Quantify your project achievements with numbers (e.g., "Increased performance by 40%")
                            - Include specific technologies and tools used
                            - Add links to GitHub repositories or live demos
                            - Describe the problem your project solved
                            - Mention any challenges faced and how you overcame them
                            """)
                        else:
                            st.info("No projects section found in your resume. Consider adding projects to showcase your practical skills.")
                            st.markdown("### 📝 Why Projects Matter")
                            st.markdown("""
                            Projects are crucial for demonstrating your practical skills and hands-on experience. They show:
                            - Your ability to apply theoretical knowledge
                            - Problem-solving capabilities
                            - Familiarity with real-world tools and technologies
                            - Initiative and passion for learning
                            
                            ### 🚀 Tips for Adding Projects
                            1. **Include 2-4 relevant projects** that showcase different skills
                            2. **Provide specific details** like technologies used, project duration, and your role
                            3. **Quantify achievements** with metrics when possible
                            4. **Add links** to GitHub repositories or live demos
                            5. **Describe the impact** of your projects
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab6:
                        st.markdown('<div class="glass-card"><h3>Recommendations</h3>', unsafe_allow_html=True)
                        
                        # Template suggestion
                        template_suggestion = results.get('template_suggestion', {})
                        st.markdown("###Recommended Resume Template")
                        st.markdown(template_suggestion.get('template', 'No template suggestion available'))
                        
                        st.markdown("### 💡 Improvement Suggestions")
                        recommendations = results.get('recommendations', [])
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(f"• {rec}")
                        else:
                            st.markdown("Your resume is well-optimized! No major improvements needed.")
                        
                        # AI-powered rewriting suggestions
                        st.markdown("### AI-Powered Rewriting Suggestions")
                        rewrite_suggestions = results.get('rewrite_suggestions', [])
                        if rewrite_suggestions:
                            for suggestion in rewrite_suggestions:
                                st.markdown(f"• {suggestion}")
                        else:
                            st.markdown("No AI rewriting suggestions available at this time.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab7:
                        st.markdown('<div class="glass-card"><h3>Interview Questions</h3>', unsafe_allow_html=True)
                        
                        # Get interview questions
                        interview_questions = results.get('interview_questions', {})
                        technical_questions = interview_questions.get('technical_questions', [])
                        hr_questions = interview_questions.get('hr_questions', [])
                        
                        # Technical Questions
                        st.markdown("### 💻 Technical Questions")
                        if technical_questions:
                            for i, question in enumerate(technical_questions, 1):
                                st.markdown(f"{i}. {question}")
                        else:
                            st.markdown("No technical questions generated.")
                        
                        st.markdown("---")
                        
                        # HR Questions
                        st.markdown("### 👥 HR Questions")
                        if hr_questions:
                            for i, question in enumerate(hr_questions, 1):
                                st.markdown(f"{i}. {question}")
                        else:
                            st.markdown("No HR questions generated.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Clean up temp file
                    import os
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error analyzing resume: {str(e)}")

def show_preparation_dashboard():
    """Display preparation dashboard"""
    st.markdown('<h2 class="main-header">🎯 Interview Preparation Dashboard</h2>', unsafe_allow_html=True)
    
    # Create tabs for the different preparation tools
    st.markdown('<div class="glass-card"><p>This section has been simplified as per your requirements.</p></div>', unsafe_allow_html=True)

def show_skill_gap_analysis():
    """Display skill gap analysis with enhanced LLM-powered recommendations"""
    st.markdown('<h2 class="main-header">Enhanced Skill Gap Analysis</h2>', unsafe_allow_html=True)

    
    st.markdown("""
    <div class="glass-card">
        <p>Upload your resume and job description to identify missing skills and get AI-powered personalized learning recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card"><h3>Upload Resume</h3>', unsafe_allow_html=True)
        resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'], key="skill_gap_resume")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><h3>💼 Job Description</h3>', unsafe_allow_html=True)
        job_description = st.text_area("Paste job description", height=200, key="skill_gap_job_desc")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if resume_file and job_description:
        if st.button("Analyze Skills Gap with AI", type="primary"):
            with st.spinner("Analyzing skills gap with AI-powered recommendations..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{resume_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(resume_file.getvalue())
                    
                    # Analyze resume to extract skills
                    analyzer = st.session_state.resume_analyzer
                    resume_data = analyzer.analyze_resume(temp_path)
                    
                    # Clean up temp file
                    import os
                    os.remove(temp_path)
                    
                    # Compare resume skills with job description using enhanced analyzer
                    skill_analyzer = st.session_state.skill_gap_analyzer
                    
                    # Perform comprehensive skill gap analysis with LLM
                    target_role = "General Position"  # This could be extracted from job description
                    comprehensive_analysis = skill_analyzer.analyze_skill_gap(
                        resume_data.get('skills', []), 
                        target_role
                    )
                    
                    # Also get the basic comparison for immediate feedback
                    comparison_result = skill_analyzer.compare_resume_with_job_description(
                        resume_data.get('skills', []), 
                        job_description
                    )
                    
                    # Display results
                    st.markdown('<div class="glass-card"><h3>Enhanced Skills Gap Analysis Results</h3>', unsafe_allow_html=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{comparison_result['match_percentage']}%</h3>
                            <p>Skill Match</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{comparison_result['matching_skills_count']}</h3>
                            <p>Matching Skills</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{comparison_result['missing_skills_count']}</h3>
                            <p>Missing Skills</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{comprehensive_analysis['analysis']['readiness_score']}%</h3>
                            <p>Readiness Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Missing Skills", "Matching Skills", "AI Recommendations", "Learning Roadmap"])
                    
                    with tab1:
                        st.markdown("### ❌ Missing Skills")
                        if comparison_result['missing_skills']:
                            # Categorize missing skills
                            essential_missing = comprehensive_analysis['analysis']['missing_essential']
                            important_missing = comprehensive_analysis['analysis']['missing_important']
                            nice_to_have_missing = comprehensive_analysis['analysis']['missing_nice_to_have']
                            
                            if essential_missing:
                                st.markdown("#### Essential Skills (Must-have):")
                                for skill in essential_missing:
                                    st.markdown(f"- 🔴 {skill.replace('_', ' ').title()}")
                            
                            if important_missing:
                                st.markdown("#### Important Skills (Should-have):")
                                for skill in important_missing:
                                    st.markdown(f"- 🟡 {skill.replace('_', ' ').title()}")
                            
                            if nice_to_have_missing:
                                st.markdown("#### Nice-to-have Skills:")
                                for skill in nice_to_have_missing:
                                    st.markdown(f"- 🟢 {skill.replace('_', ' ').title()}")
                        else:
                            st.success("🎉 Great! Your resume covers all the skills mentioned in the job description.")
                    
                    with tab2:
                        st.markdown("###  Matching Skills")
                        if comparison_result['matching_skills']:
                            cols = st.columns(3)
                            for i, skill in enumerate(comparison_result['matching_skills']):
                                with cols[i % 3]:
                                    st.markdown(f"""
                                    <div class="glass-card" style="margin: 10px 0;">
                                        <p style="text-align: center; font-weight: bold;">{skill.replace('_', ' ').title()}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No matching skills found.")
                    
                    with tab3:
                        st.markdown("###  AI-Powered Recommendations")
                        
                        # Display AI-generated recommendations
                        recommendations = comprehensive_analysis['recommendations']
                        if recommendations:
                            st.markdown(f"#### Personalized Learning Path (Estimated Timeline: {comprehensive_analysis['estimated_timeline']})")
                            
                            for i, rec in enumerate(recommendations):
                                with st.expander(f"{i+1}. {rec['skill']} ({rec['priority'].title()})"):
                                    st.markdown(f"**Priority:** {rec['priority'].title()}")
                                    st.markdown(f"**Difficulty:** {rec['difficulty'].title()}")
                                    st.markdown(f"**Estimated Time:** {rec['estimated_time']}")
                                    
                                    # Display learning path
                                    st.markdown("**Learning Path:**")
                                    learning_path = rec.get('learning_path', [])
                                    for step in learning_path:
                                        st.markdown(f"- {step}")
                                    
                                    # Display learning resources
                                    learning_resources = rec.get('learning_resources', {})
                                    free_courses = learning_resources.get('free_courses', [])
                                    practice_platforms = learning_resources.get('practice_platforms', [])
                                    youtube_channels = learning_resources.get('youtube_channels', [])
                                    
                                    if free_courses:
                                        st.markdown("**📚 Free Courses:**")
                                        for course in free_courses[:3]:  # Show up to 3 courses
                                            st.markdown(f"- [{course['name']}]({course['url']}) ({course.get('platform', 'Online')}) - {course.get('duration', 'Self-paced')}")
                                    
                                    if practice_platforms:
                                        st.markdown("**💻 Practice Platforms:**")
                                        for platform in practice_platforms[:2]:  # Show up to 2 platforms
                                            st.markdown(f"- [{platform['name']}]({platform['url']}) ({platform.get('platform', 'Online')}")
                                    
                                    if youtube_channels:
                                        st.markdown("**🎥 YouTube Tutorials:**")
                                        for video in youtube_channels[:2]:  # Show up to 2 videos
                                            st.markdown(f"- [{video['name']}]({video['url']}) ({video.get('platform', 'YouTube')}")
                        else:
                            st.info("No AI-powered recommendations available at this time.")
                    
                    with tab4:
                        st.markdown("###  Personalized Learning Roadmap")
                        
                        # Display learning roadmap
                        roadmap = comprehensive_analysis['learning_roadmap']
                        phases = roadmap.get('phases', [])
                        
                        if phases:
                            st.markdown(f"#### Target Role: {roadmap.get('target_role', 'General Position')}")
                            
                            for phase in phases:
                                with st.expander(f"Phase {phase.get('phase', 1)}: {phase.get('title', 'Learning Phase')}"):
                                    st.markdown(f"**Description:** {phase.get('description', 'Learning phase')}")
                                    st.markdown(f"**Duration:** {phase.get('duration_weeks', 4)} weeks")
                                    st.markdown(f"**Priority:** {phase.get('priority', 'medium').title()}")
                                    
                                    skills = phase.get('skills', [])
                                    if skills:
                                        st.markdown("**Skills to Focus On:**")
                                        for skill in skills:
                                            st.markdown(f"- {skill.replace('_', ' ').title()}")
                        else:
                            st.info("No learning roadmap available at this time.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error analyzing skills gap: {str(e)}")
    else:
        st.info("Please upload your resume and paste the job description to begin analysis.")

def show_career_roadmap():
    """Display career roadmap generator"""
    st.markdown('<h2 class="main-header">Career Roadmap Generator</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p>Generate a personalized career development roadmap based on your target role, experience level, and timeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chatbot-style interface for Career Roadmap (NLP prompt)
    st.markdown('<div class="glass-card"><h3>Career Roadmap Chatbot</h3><p>Ask for a role roadmap or how to learn a specific skill. Example: "Roadmap to become a backend developer in 8 months" or "How to learn React".</p></div>', unsafe_allow_html=True)

    # Ensure chat history and roadmap result exist in session state
    if 'career_chat_history' not in st.session_state:
        st.session_state.career_chat_history = []  # list of {role, content}
    if 'roadmap_result' not in st.session_state:
        st.session_state.roadmap_result = None

    # Input area: add Send, Regenerate, Clarify buttons
    col1, col2, col3 = st.columns([4,1,1])
    with col1:
        user_input = st.text_input("Ask about career roadmap or learning a skill", key="career_chat_input", placeholder="e.g., Roadmap to become a data scientist in 6 months")
    with col2:
        send = st.button("Send", type="primary", key="career_chat_send")
    with col3:
        regenerate = st.button("Regenerate Last", key="career_chat_regen")
        clarify = st.button("Ask Clarifying Qs", key="career_chat_clarify")

    # Display recent chat messages (linkify URLs)
    if st.session_state.career_chat_history:
        for msg in st.session_state.career_chat_history[-12:]:
            css_class = 'assistant-message' if msg.get('role') == 'assistant' else 'user-message'
            content_html = _linkify(msg.get("content"))
            st.markdown(f'<div class="chat-message {css_class}">{content_html}</div>', unsafe_allow_html=True)

    def _handle_user_prompt(prompt_text: str):
        st.session_state.career_chat_history.append({'role': 'user', 'content': prompt_text})
        try:
            generator = st.session_state.get('career_roadmap_generator')
            if generator is None:
                st.warning("Career roadmap generator not initialized.")
                return
            with st.spinner("Thinking..."):
                resp = generator.process_nlp_prompt(prompt_text, chat_history=st.session_state.career_chat_history)

            if resp.get('type') == 'roadmap':
                # store roadmap for detailed display
                st.session_state.roadmap_result = resp.get('content')
                target = st.session_state.roadmap_result.get('metadata', {}).get('target_role', 'your goal')
                assistant_text = f"I've generated a roadmap for {target}. Use the detailed view below to explore phases, resources, and projects."
            else:
                assistant_text = resp.get('content', '(No answer)')

            # Show assistant text with pseudo-streaming for nicer UX
            st.session_state.career_chat_history.append({'role': 'assistant', 'content': assistant_text})
        except Exception as e:
            st.error(f"Error processing prompt: {e}")

    # Handle Send
    if send and user_input:
        _handle_user_prompt(user_input)

    # Handle Regenerate: find last user prompt and resend
    if regenerate:
        last_user = None
        for msg in reversed(st.session_state.career_chat_history):
            if msg.get('role') == 'user':
                last_user = msg.get('content')
                break
        if last_user:
            _handle_user_prompt(last_user)
        else:
            st.info('No previous user prompt to regenerate.')

    # Handle Clarify: ask LLM to propose clarifying questions for the last user prompt
    if clarify:
        last_user = None
        for msg in reversed(st.session_state.career_chat_history):
            if msg.get('role') == 'user':
                last_user = msg.get('content')
                break
        if last_user:
            generator = st.session_state.get('career_roadmap_generator')
            if generator is None:
                st.warning("Career roadmap generator not initialized.")
            else:
                try:
                    prompt = f"Given the user's request: \"{last_user}\", propose 2 short clarifying questions to better understand the user's goal, timeline, or context. Return them as short bullet points."
                    clarifying = generator.model_manager.generate_text(prompt, max_length=200)
                    st.session_state.career_chat_history.append({'role': 'assistant', 'content': clarifying.strip()})
                except Exception as e:
                    st.error(f"Error generating clarifying questions: {e}")
    
    # Display results
    if st.session_state.roadmap_result:
        result = st.session_state.roadmap_result
        
        st.markdown('<div class="glass-card"><h3>Generated Career Roadmap</h3>', unsafe_allow_html=True)
        
        # Metadata
        metadata = result.get('metadata', {})
        st.markdown(f"""
        <div class="glass-card">
            <h4>Roadmap Overview</h4>
            <p><strong>Target Role:</strong> {metadata.get('target_role', 'N/A')}</p>
            <p><strong>Experience Level:</strong> {metadata.get('experience_level', 'N/A').title()}</p>
            <p><strong>Timeline:</strong> {metadata.get('timeline_months', 'N/A')} months</p>
            <p><strong>Primary Goal:</strong> {metadata.get('primary_goal', 'N/A')}</p>
            <p><strong>Generated:</strong> {metadata.get('generated_date', 'N/A')}</p>
            <p><strong>Source:</strong> {metadata.get('source', 'template_based').replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Current skills if provided
        if metadata.get('current_skills'):
            st.markdown(f"""
            <div class="glass-card">
                <h4>Your Current Skills</h4>
                <p>{', '.join(metadata['current_skills'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress tracking
        if 'completed_activities' not in st.session_state:
            st.session_state.completed_activities = []
        
        # Display phases with enhanced UI
        st.markdown("### 📚 Learning Phases", unsafe_allow_html=True)
        
        # Create tabs for each phase
        phase_names = [phase['name'] for phase in result['phases']]
        if phase_names:
            tabs = st.tabs(phase_names)
            
            for i, (tab, phase) in enumerate(zip(tabs, result['phases'])):
                with tab:
                    st.markdown(f"<h4>{phase['name']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Duration:</strong> {phase['duration']}</p>", unsafe_allow_html=True)
                    
                    # Skills with enhanced UI
                    st.markdown("<h5>🎯 Skills to Develop</h5>", unsafe_allow_html=True)
                    cols = st.columns(2)
                    for j, skill in enumerate(phase['skills']):
                        with cols[j % 2]:
                            # Create a unique key for each checkbox
                            checkbox_key = f"skill_{i}_{hash(skill)}"
                            completed = skill in st.session_state.completed_activities
                            
                            if st.checkbox(skill, value=completed, key=checkbox_key):
                                if skill not in st.session_state.completed_activities:
                                    st.session_state.completed_activities.append(skill)
                            else:
                                if skill in st.session_state.completed_activities:
                                    st.session_state.completed_activities.remove(skill)
                    
                    # Resources with enhanced UI
                    st.markdown("<h5>📚 Learning Resources</h5>", unsafe_allow_html=True)
                    for resource in phase['resources'][:5]:  # Limit to 5 resources
                        st.markdown(f"""
                        <div class="glass-card" style="margin: 10px 0;">
                            <p><strong>{resource['name']}</strong></p>
                            <p>📁 Type: {resource['type'].title()} | 🏢 Platform: {resource['platform']} | ⏱️ Duration: {resource['duration']}</p>
                            <p>📊 Difficulty: {resource['difficulty']} | 💰 Cost: {resource['cost']}</p>
                            <a href="{resource['url']}" target="_blank">🔗 View Resource</a>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Projects with enhanced UI
                    if 'projects' in phase and phase['projects']:
                        st.markdown("<h5>🏗️ Project Ideas</h5>", unsafe_allow_html=True)
                        for project in phase['projects']:
                            # Create a unique key for each project checkbox
                            project_checkbox_key = f"project_{i}_{hash(project['name'])}"
                            project_completed = project['name'] in st.session_state.completed_activities
                            
                            st.markdown(f"""
                            <div class="glass-card" style="margin: 10px 0;">
                                <p><strong>{project['name']}</strong> ({project['difficulty']})</p>
                                <p>{project['description']}</p>
                                <p>🔧 Technologies: {', '.join(project['technologies'])}</p>
                                <p>⏱️ Estimated Hours: {project['estimated_hours']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.checkbox(f"✅ Mark project as completed: {project['name']}", value=project_completed, key=project_checkbox_key):
                                if project['name'] not in st.session_state.completed_activities:
                                    st.session_state.completed_activities.append(project['name'])
                            else:
                                if project['name'] in st.session_state.completed_activities:
                                    st.session_state.completed_activities.remove(project['name'])
        
        # Progress tracking with enhanced UI
        st.markdown("### 📈 Progress Tracking", unsafe_allow_html=True)
        
        # Calculate progress
        progress_data = st.session_state.career_roadmap_generator.get_progress(
            result, 
            st.session_state.completed_activities
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{progress_data['progress_percentage']}%</h3>
                <p>Overall Progress</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{progress_data['completed_count']}/{progress_data['total_activities']}</h3>
                <p>Completed Activities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{progress_data['next_activity']}</h3>
                <p>Next Activity</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(progress_data['progress_percentage'] / 100)
        
        # Suggestions with enhanced UI
        suggestions = st.session_state.career_roadmap_generator.suggest_next_steps(
            result, 
            st.session_state.completed_activities
        )
        
        if suggestions:
            st.markdown("### 💡 Personalized Suggestions", unsafe_allow_html=True)
            for suggestion in suggestions:
                st.markdown(f"""
                <div class="glass-card" style="margin: 10px 0;">
                    <p>👉 {suggestion}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("### 📤 Export Options", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                try:
                    json_data = st.session_state.career_roadmap_generator.export_roadmap(result, "json")
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"career_roadmap_{metadata.get('target_role', 'roadmap').replace(' ', '_')}.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting as JSON: {str(e)}")
        
        with col2:
            if st.button("Export as Markdown"):
                try:
                    markdown_data = st.session_state.career_roadmap_generator.export_roadmap(result, "text")
                    st.download_button(
                        "Download Markdown",
                        markdown_data,
                        f"career_roadmap_{metadata.get('target_role', 'roadmap').replace(' ', '_')}.md",
                        "text/markdown"
                    )
                except Exception as e:
                    st.error(f"Error exporting as Markdown: {str(e)}")
            # Compact chat summary button
            st.markdown("", unsafe_allow_html=True)
            if st.button("Summarize Roadmap (Chat)"):
                try:
                    generator = st.session_state.get('career_roadmap_generator')
                    if generator is not None and st.session_state.roadmap_result:
                        summary = generator.summarize_roadmap(st.session_state.roadmap_result)
                        # Append into career chat history so it's visible in chat UI
                        if 'career_chat_history' not in st.session_state:
                            st.session_state.career_chat_history = []
                        st.session_state.career_chat_history.append({'role': 'assistant', 'content': summary})
                        st.success("Summary added to chat history")
                    else:
                        st.warning("Roadmap or generator not available")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

def show_pdf_analyzer():
    """Display PDF analyzer interface"""
    st.markdown('<h2 class="main-header">PDF Analyzer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p>Upload a PDF document to analyze its content, generate summaries, and ask questions about it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📝 Summary", "❓ Q&A Bot"])
    
    with tab1:
        st.markdown('<div class="glass-card"><h3>Upload PDF Document</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pdf_file = st.file_uploader(
                "Choose a PDF file", 
                type=['pdf'],
                help="Upload a PDF document for analysis",
                key="pdf_analyzer_uploader"
            )
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4>💡 Tips for Best Results</h4>
                <ul>
                    <li>Use clear, well-formatted PDFs</li>
                    <li>Documents with structured content work best</li>
                    <li>Keep file size reasonable for faster processing</li>
                    <li>Text-based PDFs are preferred over scanned images</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize session state for results
        if 'pdf_analysis_result' not in st.session_state:
            st.session_state.pdf_analysis_result = None
        
        if pdf_file:
            if st.button("Analyze PDF", type="primary"):
                with st.spinner("Analyzing your PDF document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{pdf_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(pdf_file.getvalue())
                        
                        # Analyze PDF
                        pdf_analyzer = st.session_state.pdf_analyzer
                        result = pdf_analyzer.analyze_pdf(temp_path)
                        
                        # Store result in session state
                        st.session_state.pdf_analysis_result = result
                        
                        # Clean up temp file
                        import os
                        os.remove(temp_path)
                        
                        st.success("✅ PDF analysis completed successfully!")
                        
                        # Display basic information
                        st.markdown('<div class="glass-card"><h3>Document Information</h3>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{result['word_count']}</h3>
                                <p>Words</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{result['character_count']}</h3>
                                <p>Characters</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{result['document_id'][:8] if result['document_id'] else 'N/A'}</h3>
                                <p>Document ID</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error analyzing PDF: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="glass-card"><h3>Document Summary</h3>', unsafe_allow_html=True)
        if st.session_state.get('pdf_analysis_result'):
            result = st.session_state.pdf_analysis_result
            summary = result.get('summary', '')
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"<h4>Summary</h4><p>{_linkify(summary)}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("Regenerate Summary"):
                try:
                    pdf_analyzer = st.session_state.pdf_analyzer
                    with st.spinner("Regenerating summary..."):
                        new_summary = pdf_analyzer.summarize_pdf(result.get('text',''))
                        st.session_state.pdf_analysis_result['summary'] = new_summary
                        st.success("Summary regenerated")
                except Exception as e:
                    st.error(f"Error regenerating summary: {e}")
        else:
            st.info("No analyzed document available. Upload and analyze a PDF first.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card"><h3>Q&A Chatbot (RAG + LLM)</h3><p>Ask questions about the analyzed document content. The bot will use retrieval-augmented generation (RAG) with the vector database and the LLM to answer.</p></div>', unsafe_allow_html=True)

        # Ensure chat history store exists
        if 'pdf_chat_history' not in st.session_state:
            st.session_state.pdf_chat_history = {}

        # Select document to chat about (allow selecting previously stored PDFs)
        current_doc_id = None
        current_doc_text = ''
        try:
            from utils.vector_db_manager import list_documents, get_document_by_id
            docs = list_documents(limit=50)
        except Exception:
            docs = []

        # Build options for selectbox: prefer recent documents
        select_options = []
        id_map = {}
        # Include currently analyzed document first (if present)
        if st.session_state.get('pdf_analysis_result'):
            doc_res = st.session_state.pdf_analysis_result
            doc_name = doc_res.get('file_name', 'Current Document')
            doc_id = doc_res.get('document_id')
            label = f"Current: {doc_name}"
            select_options.append(label)
            id_map[label] = doc_id
            # keep text available for current session doc
            current_doc_text = doc_res.get('text', '')

        # Add persisted docs
        for d in docs:
            title = d.get('title') or d.get('metadata', {}).get('file_name') or d.get('doc_id')[:8]
            did = d.get('doc_id')
            label = f"{title} ({did[:8]})"
            if label in id_map:
                continue
            select_options.append(label)
            id_map[label] = did

        if select_options:
            chosen_label = st.selectbox("Choose document to chat about:", select_options, key="pdf_doc_selector")
            current_doc_id = id_map.get(chosen_label)
            # If chosen doc is the current session doc, keep its full text; otherwise leave empty and rely on RAG excerpts
            if st.session_state.get('pdf_analysis_result') and current_doc_id and st.session_state.pdf_analysis_result.get('document_id') == current_doc_id:
                current_doc_text = st.session_state.pdf_analysis_result.get('text','')
            else:
                # Try to fetch metadata/excerpt for display
                try:
                    md = get_document_by_id(current_doc_id) or {}
                    excerpt = md.get('metadata', {}).get('excerpt', '')
                    if excerpt:
                        st.markdown(f"<div class=\"glass-card\"><strong>Excerpt:</strong><p>{_linkify(excerpt[:1000])}</p></div>", unsafe_allow_html=True)
                except Exception:
                    pass
        else:
            current_doc_id = None
            current_doc_text = ''

        # Chat UI
        if not current_doc_id:
            st.info("No analyzed document available for chat. Upload and analyze a PDF first.")
        else:
            if current_doc_id not in st.session_state.pdf_chat_history:
                st.session_state.pdf_chat_history[current_doc_id] = []

            # Display chat history
            for msg in st.session_state.pdf_chat_history[current_doc_id][-20:]:
                css_class = 'assistant-message' if msg.get('role') == 'assistant' else 'user-message'
                st.markdown(f'<div class="chat-message {css_class}">{_linkify(msg.get("content"))}</div>', unsafe_allow_html=True)

            # Input
            q_col1, q_col2 = st.columns([4,1])
            with q_col1:
                user_q = st.text_input("Ask a question about the document", key="pdf_chat_input")
            with q_col2:
                send_q = st.button("Send", key="pdf_chat_send")

            if send_q and user_q:
                # Append user message
                st.session_state.pdf_chat_history[current_doc_id].append({'role':'user','content':user_q})
                try:
                    analyzer = st.session_state.pdf_analyzer
                    with st.spinner("Searching & generating answer..."):
                        answer = analyzer.answer_question(current_doc_text, user_q)
                    st.session_state.pdf_chat_history[current_doc_id].append({'role':'assistant','content':answer})
                except Exception as e:
                    st.error(f"Error answering question: {e}")

            # Regenerate last answer
            if st.button("Regenerate Last Answer", key="pdf_chat_regen"):
                # find last user message
                history = st.session_state.pdf_chat_history[current_doc_id]
                last_user = None
                for m in reversed(history):
                    if m.get('role') == 'user':
                        last_user = m.get('content')
                        break
                if last_user:
                    try:
                        analyzer = st.session_state.pdf_analyzer
                        with st.spinner("Regenerating answer..."):
                            answer = analyzer.answer_question(current_doc_text, last_user)
                        st.session_state.pdf_chat_history[current_doc_id].append({'role':'assistant','content':answer})
                    except Exception as e:
                        st.error(f"Error regenerating answer: {e}")

def show_chatbot():
    """Placement Chatbot feature has been removed from this project."""
    st.markdown('<h2 class="main-header">Placement Chatbot (Removed)</h2>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><p>The Retrieval-Augmented (RAG) Placement Chatbot has been removed from this project. PDF Analyzer and other core features remain available.</p></div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Load models on first run
    if not load_models():
        st.stop()
    
    # Sidebar navigation with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: white; font-size: 2rem;">🤖</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # All in One dropdown for navigation
        st.markdown("### All in One")
        page = st.selectbox(
            "Choose a feature:",
            [
                "Home", 
                "Resume Analyzer", 
                "PDF Analyzer",
                # "Cover Letter Generator" removed as per user request
                "Skill Gap Analysis",
                "Career Roadmap Generator"
            ],
            key="main_navigation"  # Add unique key to prevent duplicate ID error
        )

    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Resume Analyzer":
        show_resume_analyzer()
    elif page == "PDF Analyzer":
        show_pdf_analyzer()
    # elif page == "Cover Letter Generator": removed as per user request
    elif page == "Skill Gap Analysis":
        show_skill_gap_analysis()
    elif page == "Career Roadmap Generator":
        show_career_roadmap()
    
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="glass-card">
        <p style="color: #a0ffa0;"><strong>100% Free & Open Source</strong> 🎉</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()