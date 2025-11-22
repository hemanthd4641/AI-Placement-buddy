"""
Resume Analyzer Module
Extracts text from resumes, analyzes content, and provides ATS scoring
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

# Handle ModelManager import gracefully
try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: ModelManager not available. Some features will be limited.")
    ModelManager = None
    MODEL_MANAGER_AVAILABLE = False

# Handle fitz import gracefully
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("Warning: PyMuPDF (fitz) not available. PDF processing will be limited.")

# Handle docx import gracefully
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. DOCX processing will be limited.")

# Handle text processing utilities import
try:
    from utils.text_processing import extract_projects_section
    TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    try:
        from text_processing import extract_projects_section
        TEXT_PROCESSING_AVAILABLE = True
    except ImportError:
        TEXT_PROCESSING_AVAILABLE = False
        print("Warning: text_processing utilities not available. Projects extraction will be limited.")

import spacy
from collections import Counter

# Handle matplotlib and numpy imports gracefully to avoid DLL errors
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    
    # Additional check to ensure key functions are available
    if not hasattr(plt, 'subplots') or plt.subplots is None:
        print("Warning: matplotlib available but subplots function is missing.")
        MATPLOTLIB_AVAILABLE = False
        plt = None
        np = None
except (ImportError, OSError, AttributeError) as e:
    print(f"Warning: matplotlib/numpy not available due to: {e}. Visualization features will be limited.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

# Handle plotly import for alternative visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Alternative visualization features will be limited.")
    PLOTLY_AVAILABLE = False
    go = None
    px = None

from io import BytesIO
import base64

# Vector database components removed - RAG should only be used in PDF Analyzer and Chatbot
VectorDBManager = None
is_vector_db_available = lambda: False

class ResumeAnalyzer:
    """Analyzes resumes for ATS compatibility and provides feedback
    This analyzer provides resume analysis without using vector database.
    """
    
    def __init__(self):
        # Check if ModelManager is available
        if MODEL_MANAGER_AVAILABLE and ModelManager:
            try:
                self.model_manager = ModelManager()
                self.nlp = self.model_manager.get_model('ner')
            except Exception as e:
                print(f"Warning: Failed to initialize ModelManager: {e}. Some features will be limited.")
                self.model_manager = None
                self.nlp = None
        else:
            self.model_manager = None
            self.nlp = None
            print("Warning: ModelManager not available. NLP features will be limited.")
        
        # Vector database manager removed - RAG should only be used in PDF Analyzer and Chatbot
        # Vector DB removed: not used in ResumeAnalyzer

        # Helper cache for template texts
        self._template_cache: Dict[str, str] = {}
        
        # More comprehensive check for matplotlib availability
        self.matplotlib_available = False
        if MATPLOTLIB_AVAILABLE and plt is not None and np is not None:
            # Additional check to ensure key functions are available
            try:
                # Test if we can create a simple plot
                if hasattr(plt, 'subplots'):
                    fig, ax = plt.subplots()
                    plt.close(fig)  # Clean up
                    self.matplotlib_available = True
                else:
                    print("Warning: matplotlib available but subplots function is missing.")
            except Exception as e:
                print(f"Warning: matplotlib available but not functional: {e}")
        else:
            print("Warning: matplotlib/numpy not available. Visualization features will be limited.")
        
        # Check for plotly availability as alternative
        self.plotly_available = PLOTLY_AVAILABLE and go is not None and px is not None
        if not self.matplotlib_available and self.plotly_available:
            print("Using plotly for visualizations as matplotlib is not available.")
        
        # Common skills database (expanded)
        self.skills_db = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'kotlin',
                'swift', 'php', 'ruby', 'scala', 'r', 'matlab', 'typescript',
                'dart', 'objective-c', 'perl', 'lua', 'haskell'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
                'django', 'flask', 'spring', 'laravel', 'bootstrap', 'jquery',
                'webpack', 'babel', 'sass', 'less', 'gatsby', 'next.js', 'nuxt.js'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
                'sql server', 'cassandra', 'elasticsearch', 'neo4j', 'firebase',
                'dynamodb', 'mariadb', 'couchdb', 'influxdb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes',
                'heroku', 'digitalocean', 'ibm cloud', 'oracle cloud',
                'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
                'keras', 'matplotlib', 'seaborn', 'plotly', 'jupyter',
                'apache spark', 'hadoop', 'tableau', 'power bi', 'excel'
            ],
            'mobile_development': [
                'android', 'ios', 'react native', 'flutter', 'xamarin',
                'ionic', 'cordova', 'swift', 'kotlin', 'objective-c'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
                'slack', 'trello', 'asana', 'notion', 'figma', 'sketch',
                'adobe xd', 'photoshop', 'illustrator', 'postman', 'insomnia'
            ]
        }
        
        # ATS keywords for scoring
        self.ats_keywords = [
            'experience', 'skills', 'education', 'projects', 'achievements',
            'responsibilities', 'developed', 'implemented', 'managed', 'led',
            'created', 'designed', 'optimized', 'improved', 'collaborated',
            'analyzed', 'tested', 'deployed', 'maintained', 'troubleshot'
        ]
        
        # Skill category colors for visualization
        self.skill_colors = {
            'programming_languages': '#FF6B6B',
            'web_technologies': '#4ECDC4',
            'databases': '#45B7D1',
            'cloud_platforms': '#96CEB4',
            'data_science': '#FFEAA7',
            'mobile_development': '#DDA0DD',
            'tools': '#98D8C8'
        }
        
        # Industry-specific resume templates and suggestions
        self.industry_templates = {
            'software_engineer': {
                'template': '''SOFTWARE ENGINEER RESUME TEMPLATE
                
Header:
- Full Name
- Phone Number | Email | LinkedIn | GitHub | Portfolio (if applicable)

Summary:
- 2-3 sentences highlighting years of experience, key skills, and what you bring to the table

Technical Skills:
- Programming Languages: List relevant languages
- Frameworks & Libraries: List relevant frameworks
- Databases: List relevant databases
- Tools & Platforms: List relevant tools
- Cloud & DevOps: List relevant platforms

Professional Experience:
- Job Title, Company Name, Location, Dates
- 3-5 bullet points using STAR method (Situation, Task, Action, Result)
- Quantify achievements with metrics

Education:
- Degree, University Name, Graduation Date
- Relevant coursework (optional)
- GPA (if above 3.5)

Projects (Optional but recommended):
- Project Name
- Technologies used
- Brief description with impact/results

Certifications (if applicable):
- Certification Name, Issuing Organization, Date''',
                'keywords': ['software development', 'programming', 'coding', 'debugging', 'testing', 'agile', 'scrum'],
                'suggestions': [
                    'Highlight programming languages and frameworks relevant to the job',
                    'Quantify your achievements with specific metrics',
                    'Use action verbs like developed, implemented, optimized, etc.',
                    'Include links to GitHub repositories or portfolio',
                    'Mention any open-source contributions'
                ]
            },
            'data_scientist': {
                'template': '''DATA SCIENTIST RESUME TEMPLATE
                
Header:
- Full Name
- Phone Number | Email | LinkedIn | GitHub | Kaggle (if applicable)

Summary:
- 2-3 sentences highlighting years of experience, key skills, and what you bring to the table

Technical Skills:
- Programming Languages: Python, R, SQL, etc.
- Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Tableau, etc.
- Machine Learning: Scikit-learn, TensorFlow, PyTorch, etc.
- Big Data Technologies: Spark, Hadoop, etc.
- Tools & Platforms: Jupyter, Git, Docker, etc.

Professional Experience:
- Job Title, Company Name, Location, Dates
- 3-5 bullet points using STAR method
- Focus on data-driven results and business impact

Education:
- Degree, University Name, Graduation Date
- Relevant coursework: Machine Learning, Statistics, Data Mining, etc.

Projects:
- Project Name
- Technologies and methodologies used
- Problem statement and solution approach
- Results with metrics

Certifications:
- Relevant certifications like AWS Certified Machine Learning, etc.''',
                'keywords': ['machine learning', 'data analysis', 'statistics', 'predictive modeling', 'data visualization'],
                'suggestions': [
                    'Emphasize statistical analysis and machine learning skills',
                    'Include specific tools and technologies used in projects',
                    'Quantify the business impact of your data science work',
                    'Mention any Kaggle competitions or data science projects',
                    'Highlight experience with big data technologies if applicable'
                ]
            },
            'product_manager': {
                'template': '''PRODUCT MANAGER RESUME TEMPLATE
                
Header:
- Full Name
- Phone Number | Email | LinkedIn | Portfolio (if applicable)

Summary:
- 2-3 sentences highlighting years of experience, key skills, and what you bring to the table

Skills:
- Product Management: Roadmap development, requirement gathering, etc.
- Technical: Technical background relevant to products managed
- Analytics: Tools used for data analysis and decision making
- Leadership: Team management, stakeholder communication

Professional Experience:
- Job Title, Company Name, Location, Dates
- 3-5 bullet points focusing on product outcomes and business impact
- Use metrics to show product success

Education:
- Degree, University Name, Graduation Date

Projects/Products Managed:
- Product Name
- Your role and responsibilities
- Key achievements and metrics

Certifications (if applicable):
- Product management certifications, Agile, Scrum, etc.''',
                'keywords': ['product roadmap', 'user stories', 'agile', 'scrum', 'stakeholder management', 'product lifecycle'],
                'suggestions': [
                    'Focus on product outcomes and business impact',
                    'Quantify success with metrics like user growth, revenue, etc.',
                    'Highlight cross-functional collaboration experience',
                    'Mention experience with product management tools',
                    'Include any successful product launches'
                ]
            }
        }
    
    def _get_template_text(self, template_for: str) -> str:
        """Stub for template text retrieval (vector DB removed)."""
        # Always return empty string or a static template if needed
        return ""
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        if not FITZ_AVAILABLE:
            raise Exception("PDF processing not available: PyMuPDF (fitz) library not installed. Please install it with 'pip install PyMuPDF'")
        
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise Exception("DOCX processing not available: python-docx library not installed. Please install it with 'pip install python-docx'")
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text(self, file_path):
        """Extract text from resume file"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            raise Exception(f"Unsupported file format: {file_extension}. Please upload a PDF or DOCX file.")
    
    def extract_contact_info(self, text):
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone regex (various formats)
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
        
        # LinkedIn profile
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        # GitHub profile
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, text, re.IGNORECASE)
        if github:
            contact_info['github'] = github[0]
        
        return contact_info
    
    def extract_skills(self, text):
        """Enhanced skills extraction with better pattern matching"""
        text_lower = text.lower()
        found_skills = []
        
        # Check each skill category with improved matching
        for category, skills in self.skills_db.items():
            for skill in skills:
                # Exact word boundary matching
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
                    continue
                
                # Handle common variations
                if skill == 'javascript' and re.search(r'\bjs\b', text_lower):
                    found_skills.append(skill)
                elif skill == 'react' and 'reactjs' in text_lower:
                    found_skills.append(skill)
                elif skill == 'node.js' and re.search(r'\bnode\b.*\bjs\b|\bnodejs\b', text_lower):
                    found_skills.append(skill)
        
        # Use NER to find additional technical skills and organizations
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE'] and len(ent.text) > 2:
                    skill_candidate = ent.text.lower().strip()
                    # Filter out common words that aren't skills
                    if (skill_candidate not in ['university', 'college', 'company', 'team', 'project'] and
                        skill_candidate not in found_skills):
                        found_skills.append(skill_candidate)
        
        # Extract from common skill sections
        skill_sections = re.findall(r'(?:skills?|technologies?|tools?):\s*([^\n]+)', text_lower)
        for section in skill_sections:
            # Split by common delimiters
            skills_in_section = re.split(r'[,;|•·]', section)
            for skill in skills_in_section:
                skill = skill.strip()
                if len(skill) > 1 and skill not in found_skills:
                    found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_education(self, text):
        """Extract education information"""
        education_info = {}
        
        # Common degree patterns
        degree_patterns = [
            r'(bachelor|master|phd|doctorate|diploma|certificate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?com|m\.?com|mba|bba)',
            r'(engineering|computer science|information technology|software|electronics|mechanical|civil)',
            r'(university|college|institute|school)'
        ]
        
        education_matches = []
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education_matches.extend(matches)
        
        if education_matches:
            education_info['degrees'] = list(set(education_matches))
        
        # Extract graduation years
        year_pattern = r'(19|20)\d{2}'
        years = re.findall(year_pattern, text)
        if years:
            education_info['graduation_years'] = sorted(list(set(years)))
        
        return education_info
    
    def extract_experience(self, text):
        """Extract work experience information"""
        experience_info = {}
        
        # Look for experience indicators
        exp_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience\s*:?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        years_found = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        if years_found:
            experience_info['years'] = max(years_found)
        
        # Extract job titles
        job_titles = []
        title_patterns = [
            r'(software engineer|developer|programmer|analyst|manager|lead|senior|junior)',
            r'(intern|trainee|associate|consultant|specialist|architect)',
            r'(data scientist|ml engineer|devops|full stack|backend|frontend)'
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_titles.extend(matches)
        
        if job_titles:
            experience_info['titles'] = list(set(job_titles))
        
        return experience_info
    
    def extract_projects(self, text):
        """Extract projects information from resume text with enhanced detection"""
        if not TEXT_PROCESSING_AVAILABLE:
            print("Warning: Text processing utilities not available for projects extraction")
            return {}
        
        try:
            # Use the text processing utility to extract projects section
            projects_text = extract_projects_section(text)
            
            if not projects_text:
                # Try alternative extraction method
                projects_text = self._extract_projects_alternative(text)
            
            if not projects_text:
                return {}
            
            # Parse projects into structured format with better filtering
            project_lines = projects_text.split('\n')
            filtered_lines = []
            
            # Enhanced filtering to identify actual project descriptions
            exclude_keywords = [
                'college', 'university', 'school', 'institute', 'tumkur', 
                'karnataka', 'address', 'phone', 'email', 'grade',
                'dormitory management system', 'vijaya pu college', 'isro',
                'department of aiml', 'visual studio code', 'nss', 'cmr institute'
            ]
            
            # Keywords that indicate project content
            project_indicators = [
                'project', 'developed', 'built', 'created', 'designed', 
                'implemented', 'application', 'system', 'website', 'software',
                'tool', 'platform', 'api', 'database', 'algorithm', 'model',
                'framework', 'library', 'mobile app', 'web app', 'dashboard',
                'solution', 'prototype', 'automation', 'machine learning',
                'data analysis', 'visualization', 'integration', 'deployment',
                'engineered', 'architected', 'forecast', 'visualize', 'integrated',
                'authentication', 'payment', 'inventory', 'synchronization',
                'performance', 'metrics', 'accuracy'
            ]
            
            for line in project_lines:
                line_lower = line.lower().strip()
                # Skip empty lines
                if not line_lower:
                    continue
                    
                # Skip lines containing personal/institutional information
                if any(exclude_keyword in line_lower for exclude_keyword in exclude_keywords):
                    continue
                
                # Include lines that seem to be actual project descriptions
                if any(indicator in line_lower for indicator in project_indicators):
                    filtered_lines.append(line)
                # Also include lines that contain technical terms or tools
                elif any(skill.lower() in line_lower for skill in self.skills_db['programming_languages'] + 
                         self.skills_db['web_technologies'] + self.skills_db['databases'] + 
                         self.skills_db['cloud_platforms'] + self.skills_db['data_science']):
                    filtered_lines.append(line)
                # Include lines that look like project titles (contain colon)
                elif ':' in line and len(line.split()) <= 20:
                    filtered_lines.append(line)
            
            # If we filtered everything out, try a more lenient approach
            if not filtered_lines and project_lines:
                # Look for lines that start with common project prefixes
                for line in project_lines:
                    line_stripped = line.strip()
                    if line_stripped and not any(exclude in line_stripped.lower() for exclude in exclude_keywords):
                        # Check if line looks like a project title (starts with capital letter, contains colon or dash)
                        if (line_stripped[0].isupper() and 
                            (':' in line_stripped[:50] or '-' in line_stripped[:50] or 
                             len(line_stripped.split()) <= 12)):
                            filtered_lines.append(line_stripped)
            
            # Structure projects into detailed format
            structured_projects = self._structure_projects(filtered_lines)
            
            projects_info = {
                'projects_text': '\n'.join(filtered_lines) if filtered_lines else projects_text,
                'project_count': len(structured_projects) if structured_projects else (len([line for line in filtered_lines if line.strip()]) if filtered_lines else 0),
                'has_projects': bool(filtered_lines or structured_projects),
                'structured_projects': structured_projects
            }
            
            return projects_info
        except Exception as e:
            print(f"Warning: Failed to extract projects information: {e}")
            return {}
    
    def _extract_projects_alternative(self, text):
        """Alternative method to extract projects when standard method fails"""
        try:
            # Look for "Projects" section with more variations
            project_section_patterns = [
                r'projects?\s*:?',
                r'personal projects?\s*:?',
                r'key projects?\s*:?',
                r'side projects?\s*:?',
                r'technical projects?\s*:?'
            ]
            
            lines = text.split('\n')
            project_lines = []
            
            # Look for project section header first
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Check if this is a project section header
                if any(re.search(pattern, line_lower) for pattern in project_section_patterns):
                    project_lines.append(line)
                    # Continue to collect lines until we hit another section
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        # Stop if we hit another section header
                        section_headers = [
                            'experience', 'education', 'skills', 'achievements', 
                            'certifications', 'awards', 'contact', 'summary', 
                            'objective', 'profile'
                        ]
                        if next_line.lower() in [header + ':' for header in section_headers] or \
                           next_line.lower() in section_headers or \
                           (next_line.isupper() and len(next_line.split()) <= 4 and next_line.endswith(':')):
                            break
                        # Add non-empty lines
                        if next_line:
                            project_lines.append(next_line)
                    break
            
            return '\n'.join(project_lines)
        except Exception as e:
            print(f"Warning: Alternative project extraction failed: {e}")
            return ""
    
    def _structure_projects(self, project_lines):
        """Structure project lines into detailed project objects"""
        if not project_lines:
            return []
        
        structured_projects = []
        current_project = None
        
        # Common project-related keywords
        project_keywords = [
            'project', 'developed', 'built', 'created', 'designed', 
            'implemented', 'application', 'system', 'website', 'software',
            'tool', 'platform', 'api', 'database', 'algorithm', 'model',
            'engineered', 'architected', 'forecast', 'visualize', 'integrated'
        ]
        
        for line in project_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line looks like a project title (contains colon or dash, or has project keywords)
            is_project_title = (
                (':' in line_stripped[:50] or '-' in line_stripped[:50]) or
                any(keyword in line_stripped.lower() for keyword in project_keywords) or
                (line_stripped[0].isupper() and len(line_stripped.split()) <= 12)
            )
            
            if is_project_title:
                # Save previous project if exists
                if current_project:
                    structured_projects.append(current_project)
                
                # Start new project
                title = line_stripped.split(':')[0].split('-')[0].strip()
                current_project = {
                    'title': title if title else 'Untitled Project',
                    'description': line_stripped,
                    'technologies': [],
                    'keywords': []
                }
                
                # Extract technologies from the line
                tech_keywords = []
                for category in self.skills_db.values():
                    for tech in category:
                        if tech.lower() in line_stripped.lower():
                            tech_keywords.append(tech)
                
                if tech_keywords:
                    current_project['technologies'] = list(set(tech_keywords))
                
                # Extract keywords
                found_keywords = [kw for kw in project_keywords if kw in line_stripped.lower()]
                if found_keywords:
                    current_project['keywords'] = found_keywords
            elif current_project:
                # Add to current project description (limit description length)
                if len(current_project['description'] + ' ' + line_stripped) < 500:
                    current_project['description'] += ' ' + line_stripped
                    
                # Also extract additional technologies from this line
                tech_keywords = []
                for category in self.skills_db.values():
                    for tech in category:
                        if tech.lower() in line_stripped.lower():
                            tech_keywords.append(tech)
                
                if tech_keywords:
                    current_project['technologies'].extend(tech_keywords)
                    current_project['technologies'] = list(set(current_project['technologies']))
            else:
                # Create a new project with this line if it looks project-like
                if any(keyword in line_stripped.lower() for keyword in project_keywords):
                    current_project = {
                        'title': line_stripped[:30] + "..." if len(line_stripped) > 30 else line_stripped,
                        'description': line_stripped,
                        'technologies': [],
                        'keywords': []
                    }
                    structured_projects.append(current_project)
                    current_project = None  # Reset for next project
        
        # Add last project
        if current_project:
            structured_projects.append(current_project)
        
        return structured_projects
    
    def calculate_ats_score(self, text, skills, contact_info, education, experience):
        """Enhanced ATS compatibility score calculation with LLM-powered detailed feedback"""
        score = 0
        max_score = 100
        feedback = []
        detailed_feedback = {
            'contact_info': [],
            'skills': [],
            'education': [],
            'experience': [],
            'formatting': []
        }
        
        # Use LLM to generate detailed feedback if available
        if self.model_manager and hasattr(self.model_manager, 'generate_text'):
            try:
                # Generate detailed feedback using LLM
                detailed_feedback = self._generate_llm_detailed_feedback(text, skills, contact_info, education, experience)
            except Exception as e:
                print(f"LLM detailed feedback generation failed: {e}")
                # Fallback to original detailed feedback generation
                detailed_feedback = self._generate_static_detailed_feedback(text, skills, contact_info, education, experience)
        else:
            # Fallback to original detailed feedback generation
            detailed_feedback = self._generate_static_detailed_feedback(text, skills, contact_info, education, experience)
        
        # Contact information scoring (15 points)
        contact_score = 0
        if contact_info.get('email'):
            contact_score += 4
        else:
            feedback.append("Missing professional email address")
        
        if contact_info.get('phone'):
            contact_score += 3
        else:
            feedback.append("Missing phone number")
        
        if contact_info.get('linkedin'):
            contact_score += 4
        else:
            feedback.append("Missing LinkedIn profile")
        
        if contact_info.get('github'):
            contact_score += 4
        else:
            feedback.append("Consider adding GitHub profile")
        
        score += contact_score
        
        # Skills section scoring (25 points)
        skills_score = 0
        if len(skills) >= 8:
            skills_score += 15
        elif len(skills) >= 5:
            skills_score += 12
        elif len(skills) >= 3:
            skills_score += 8
        else:
            feedback.append("Add more relevant skills")
        
        # Technical skills depth
        tech_skills = [s for s in skills if s in 
                      self.skills_db['programming_languages'] + 
                      self.skills_db['web_technologies'] + 
                      self.skills_db['databases'] + 
                      self.skills_db['cloud_platforms']]
        
        if len(tech_skills) >= 5:
            skills_score += 10
        elif len(tech_skills) >= 3:
            skills_score += 7
        elif len(tech_skills) >= 1:
            skills_score += 4
        else:
            feedback.append("Add more technical skills")
        
        score += skills_score
        
        # Education section scoring (15 points)
        education_score = 0
        if education.get('degrees'):
            education_score += 10
        else:
            feedback.append("Include education details")
        
        if education.get('graduation_years'):
            education_score += 5
        else:
            feedback.append("Include graduation years")
        
        score += education_score
        
        # Experience section scoring (25 points)
        experience_score = 0
        if experience.get('years'):
            years = experience['years']
            if years >= 5:
                experience_score += 15
            elif years >= 3:
                experience_score += 12
            elif years >= 1:
                experience_score += 8
            else:
                experience_score += 4
        else:
            feedback.append("Clearly mention years of experience")
        
        if experience.get('titles'):
            experience_score += 5
        else:
            feedback.append("Include specific job titles")
        
        # Quantifiable achievements check
        numbers_pattern = r'\d+[%]?|\$\d+|improved|increased|decreased|reduced'
        if re.search(numbers_pattern, text, re.IGNORECASE):
            experience_score += 5
        else:
            feedback.append("Add quantifiable achievements with numbers")
        
        score += experience_score
        
        # ATS keywords and formatting (20 points)
        formatting_score = 0
        text_lower = text.lower()
        
        # Check for ATS keywords
        keyword_matches = 0
        for keyword in self.ats_keywords[:20]:  # Check top 20 keywords
            if keyword in text_lower:
                keyword_matches += 1
        
        keyword_score = min(15, (keyword_matches / 20) * 15)  # Max 15 points
        formatting_score += keyword_score
        
        # Check formatting (simple checks)
        if '\t' in text or '    ' in text:  # Tabs or multiple spaces
            formatting_score += 5
        else:
            feedback.append("Consider using consistent formatting with tabs or spaces")
        
        score += formatting_score
        
        # Ensure score is within bounds
        final_score = min(max_score, max(0, score))
        grade = self._get_grade(final_score)
        
        return {
            'score': round(final_score, 2),
            'grade': grade,
            'feedback': feedback,
            'detailed_feedback': detailed_feedback,
            'breakdown': {
                'contact_info': round((contact_score / 15) * 100, 2),
                'skills': round((skills_score / 25) * 100, 2),
                'education': round((education_score / 15) * 100, 2),
                'experience': round((experience_score / 25) * 100, 2),
                'formatting': round((formatting_score / 20) * 100, 2)
            }
        }
    
    def _generate_llm_detailed_feedback(self, text, skills, contact_info, education, experience):
        """Generate detailed feedback using LLM"""
        try:
            # Include template-driven instructions if available
            template_instructions = self._get_template_text('resume_analysis')
            # Create a comprehensive prompt for detailed feedback
            prompt = f"""
{template_instructions}

You are an expert resume analyst. Provide detailed, personalized feedback on each section of this resume.
Be specific and actionable in your feedback. Respond with a simple list format.

Resume Information:
- Skills: {', '.join(skills[:20]) if skills else 'Not specified'}
- Experience: {experience.get('years', 'Not specified')} years
- Education: {', '.join(education.get('degrees', ['Not specified']))}
- Contact Info: Email: {'Yes' if contact_info.get('email') else 'No'}, 
                 Phone: {'Yes' if contact_info.get('phone') else 'No'}, 
                 LinkedIn: {'Yes' if contact_info.get('linkedin') else 'No'}

Provide exactly 1-2 specific feedback points for each section in this simple format:
Contact Info: [feedback point 1], [feedback point 2]
Skills: [feedback point 1], [feedback point 2]
Education: [feedback point 1], [feedback point 2]
Experience: [feedback point 1], [feedback point 2]
Formatting: [feedback point 1], [feedback point 2]
"""

            # Generate detailed feedback using LLM
            feedback_text = self.model_manager.generate_text(prompt, max_length=300)
            
            # Parse the feedback text
            if feedback_text and len(feedback_text.strip()) > 30:
                return self._parse_llm_feedback_text(feedback_text)
            else:
                # Fallback to static feedback
                return self._generate_static_detailed_feedback(text, skills, contact_info, education, experience)
                
        except Exception as e:
            print(f"LLM detailed feedback generation failed: {e}")
            # Fallback to static feedback
            return self._generate_static_detailed_feedback(text, skills, contact_info, education, experience)
    
    def _parse_llm_feedback_text(self, feedback_text):
        """Parse LLM feedback text into structured format"""
        # Initialize with fallback feedback
        feedback_dict = {
            'contact_info': ["Consider reviewing your contact information for completeness and professionalism"],
            'skills': ["Review your skills section for relevance and proper categorization"],
            'education': ["Ensure your education section includes all relevant qualifications and dates"],
            'experience': ["Make sure your experience section highlights achievements with quantifiable results"],
            'formatting': ["Check for consistent formatting, clear section headings, and proper spacing"]
        }
        
        try:
            lines = feedback_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                    
                # Split on first colon only
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                    
                section_name = parts[0].strip().lower()
                feedback_content = parts[1].strip()
                
                # Map section names to our categories
                section_mapping = {
                    'contact info': 'contact_info',
                    'contact': 'contact_info',
                    'skills': 'skills',
                    'education': 'education',
                    'experience': 'experience',
                    'formatting': 'formatting',
                    'format': 'formatting'
                }
                
                # Get our internal section name
                internal_section = section_mapping.get(section_name)
                if not internal_section:
                    continue
                
                # Parse feedback points (split on comma, but be careful of commas in the content)
                if ',' in feedback_content:
                    # Simple split for now
                    points = [p.strip() for p in feedback_content.split(',') if p.strip()]
                    if points:
                        feedback_dict[internal_section] = points[:2]  # Limit to 2 points
                elif feedback_content:
                    feedback_dict[internal_section] = [feedback_content]
                    
        except Exception as e:
            print(f"Error parsing LLM feedback: {e}")
            # Return default feedback if parsing fails
            pass
        
        return feedback_dict
    
    def _generate_static_detailed_feedback(self, text, skills, contact_info, education, experience):
        """Original static detailed feedback generation (fallback method)"""
        detailed_feedback = {
            'contact_info': [],
            'skills': [],
            'education': [],
            'experience': [],
            'formatting': []
        }
        
        # Contact info feedback
        if not contact_info.get('email'):
            detailed_feedback['contact_info'].append("Add a professional email address for better ATS compatibility")
        if not contact_info.get('phone'):
            detailed_feedback['contact_info'].append("Include your phone number for easier contact")
        if not contact_info.get('linkedin'):
            detailed_feedback['contact_info'].append("Add your LinkedIn profile to showcase your professional network")
        if not contact_info.get('github'):
            detailed_feedback['contact_info'].append("Include your GitHub profile to showcase your coding projects")
        if not detailed_feedback['contact_info']:
            detailed_feedback['contact_info'].append("Your contact information is well-presented")
        
        # Skills feedback
        if len(skills) < 8:
            detailed_feedback['skills'].append(f"You currently have {len(skills)} skills. Consider adding 3-5 more relevant technical skills.")
        else:
            detailed_feedback['skills'].append("Good variety of skills listed")
            
        tech_skills = [s for s in skills if s in 
                      self.skills_db['programming_languages'] + 
                      self.skills_db['web_technologies'] + 
                      self.skills_db['databases']]
        
        if len(tech_skills) < 5:
            detailed_feedback['skills'].append(f"Technical skills count: {len(tech_skills)}. Recommended: 5+ technical skills.")
        else:
            detailed_feedback['skills'].append("Strong technical skills section")
        
        # Education feedback
        if not education.get('degrees'):
            detailed_feedback['education'].append("Add your educational qualifications including degree and institution name")
        if not education.get('graduation_years'):
            detailed_feedback['education'].append("Include graduation years to show your experience timeline")
        if not detailed_feedback['education']:
            detailed_feedback['education'].append("Education section is well-detailed")
        
        # Experience feedback
        if not experience.get('years'):
            detailed_feedback['experience'].append("Specify your total years of relevant work experience")
        if not experience.get('titles'):
            detailed_feedback['experience'].append("List your specific job titles to show career progression")
        
        numbers_pattern = r'\d+[%]?|\$\d+|improved|increased|decreased|reduced'
        if not re.search(numbers_pattern, text, re.IGNORECASE):
            detailed_feedback['experience'].append("Include metrics and numbers to quantify your achievements (e.g., 'Increased sales by 25%')")
        else:
            detailed_feedback['experience'].append("Good use of quantifiable achievements")
        
        # Formatting feedback
        if len(text.split()) < 200:
            detailed_feedback['formatting'].append("Resume seems brief. Consider expanding with more relevant details")
        detailed_feedback['formatting'].append("Use consistent formatting with clear section headings")
        
        return detailed_feedback
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'
    
    def generate_recommendations(self, text, skills, contact_info, education, experience, ats_score):
        """Generate improvement recommendations using LLM for more personalized suggestions"""
        # Use LLM to generate personalized recommendations if available
        if self.model_manager and hasattr(self.model_manager, 'generate_text'):
            try:
                template_instructions = self._get_template_text('resume_recommendations') or self._get_template_text('resume_analysis')
                # Create a comprehensive prompt for LLM-based recommendations
                prompt = f"""
{template_instructions}
You are an expert resume consultant. Analyze this resume and provide specific, actionable recommendations for improvement.
Focus on making the resume more ATS-friendly and industry-competitive. Provide exactly 5 recommendations.

Resume Information:
- Skills: {', '.join(skills[:20]) if skills else 'Not specified'}
- Experience: {experience.get('years', 'Not specified')} years
- Education: {', '.join(education.get('degrees', ['Not specified']))}
- Contact Info: Email: {'Yes' if contact_info.get('email') else 'No'}, 
                 Phone: {'Yes' if contact_info.get('phone') else 'No'}, 
                 LinkedIn: {'Yes' if contact_info.get('linkedin') else 'No'}

ATS Score: {ats_score}/100

Provide exactly 5 specific recommendations in this simple numbered format:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]
4. [Recommendation 4]
5. [Recommendation 5]

Focus on:
- ATS optimization
- Skills presentation
- Experience quantification
- Contact information completeness
"""

                # Generate recommendations using LLM
                llm_recommendations = self.model_manager.generate_text(prompt, max_length=300)
                
                # Parse the LLM response into a list
                if llm_recommendations and len(llm_recommendations.strip()) > 30:
                    # Extract numbered recommendations
                    lines = llm_recommendations.strip().split('\n')
                    llm_recs = []
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                            # Remove numbering/markers
                            clean_line = re.sub(r'^[0-9\-\•\.\s]+', '', line).strip()
                            if clean_line and len(clean_line) > 10:
                                llm_recs.append(clean_line)
                    
                    # If we successfully extracted recommendations, use them
                    if len(llm_recs) >= 3:
                        return llm_recs[:7]  # Limit to 7 recommendations
                    else:
                        # Fallback to original approach if not enough recommendations
                        return self._generate_static_recommendations(text, skills, contact_info, education, experience, ats_score)
                else:
                    # Fallback to original approach if LLM response is too short
                    return self._generate_static_recommendations(text, skills, contact_info, education, experience, ats_score)
                    
            except Exception as e:
                print(f"LLM recommendation generation failed: {e}")
                # Fallback to original static recommendations
                return self._generate_static_recommendations(text, skills, contact_info, education, experience, ats_score)
        else:
            # Fallback to original static recommendations if LLM not available
            return self._generate_static_recommendations(text, skills, contact_info, education, experience, ats_score)
    
    def _generate_static_recommendations(self, text, skills, contact_info, education, experience, ats_score):
        """Original static recommendation generation (fallback method)"""
        recommendations = []
        
        # Contact info recommendations
        if not contact_info.get('email'):
            recommendations.append("Add a professional email address")
        if not contact_info.get('phone'):
            recommendations.append("Include your phone number")
        if not contact_info.get('linkedin'):
            recommendations.append("Add your LinkedIn profile URL")
        if not contact_info.get('github'):
            recommendations.append("Include your GitHub profile if you're a developer")
        
        # Skills recommendations
        if len(skills) < 5:
            recommendations.append("Add more relevant technical skills")
        
        tech_skills = [s for s in skills if s in self.skills_db['programming_languages'] + 
                      self.skills_db['web_technologies'] + self.skills_db['databases']]
        if len(tech_skills) < 3:
            recommendations.append("Include more technical/programming skills")
        
        # Experience recommendations
        if not experience.get('years'):
            recommendations.append("Clearly mention your years of experience")
        if not experience.get('titles'):
            recommendations.append("Include specific job titles and roles")
        
        # ATS optimization
        text_lower = text.lower()
        missing_keywords = [kw for kw in self.ats_keywords[:10] if kw not in text_lower]
        if len(missing_keywords) > 5:
            recommendations.append(f"Include action keywords like: {', '.join(missing_keywords[:5])}")
        
        # Format recommendations
        if len(text.split()) < 200:
            recommendations.append("Expand your resume content - it seems too brief")
        
        # Specific score-based recommendations
        if ats_score < 50:
            recommendations.append("Your resume needs significant improvement for ATS compatibility")
        elif ats_score < 70:
            recommendations.append("Good start! A few improvements can boost your ATS score")
        else:
            recommendations.append("Great job! Your resume is well-optimized for ATS systems")
        
        return recommendations
    
    def generate_rewrite_suggestions(self, resume_results):
        """Generate AI-powered rewriting suggestions using LLM"""
        if not self.model_manager or not hasattr(self.model_manager, 'generate_text'):
            # Fallback to static suggestions if LLM not available
            return self._generate_static_rewrite_suggestions(resume_results)
        
        try:
            text = resume_results.get('text', '')
            skills = resume_results.get('skills', [])
            experience_years = resume_results.get('experience_years', 0)
            
            # Create a prompt for rewriting suggestions
            prompt = f"""
You are an expert resume writer. Analyze this resume and provide specific rewriting suggestions to make it more impactful.
Focus on improving clarity, impact, and ATS compatibility. Provide exactly 4 suggestions.

Resume Information:
- Skills: {', '.join(skills[:15]) if skills else 'Not specified'}
- Experience: {experience_years} years
- ATS Score: {resume_results.get('ats_score', 0)}/100

Provide exactly 4 specific rewriting suggestions in this simple numbered format:
1. [Suggestion 1]
2. [Suggestion 2]
3. [Suggestion 3]
4. [Suggestion 4]

Focus on:
- Making achievements more quantifiable
- Improving action verbs
- Enhancing skills presentation
- Making experience descriptions more impactful
"""

            # Generate rewriting suggestions using LLM
            suggestions_text = self.model_manager.generate_text(prompt, max_length=250)
            
            # Parse the LLM response into a list
            if suggestions_text and len(suggestions_text.strip()) > 30:
                # Extract numbered suggestions
                lines = suggestions_text.strip().split('\n')
                suggestions = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Remove numbering/markers
                        clean_line = re.sub(r'^[0-9\-\•\.\s]+', '', line).strip()
                        if clean_line and len(clean_line) > 10:
                            suggestions.append(clean_line)
                
                # Return suggestions or fallback if not enough
                if len(suggestions) >= 2:
                    return suggestions[:5]
                else:
                    return self._generate_static_rewrite_suggestions(resume_results)
            else:
                # Fallback to static suggestions if LLM response is too short
                return self._generate_static_rewrite_suggestions(resume_results)
                
        except Exception as e:
            print(f"LLM rewriting suggestions generation failed: {e}")
            # Fallback to static suggestions
            return self._generate_static_rewrite_suggestions(resume_results)
    
    def _generate_static_rewrite_suggestions(self, resume_results):
        """Original static rewriting suggestions (fallback method)"""
        suggestions = []
        text = resume_results.get('text', '')
        skills = resume_results.get('skills', [])
        experience_years = resume_results.get('experience_years', 0)
        ats_score = resume_results.get('ats_score', 0)
        
        # Add general rewriting suggestions based on resume analysis
        if ats_score < 60:
            suggestions.append("Rewrite job descriptions to include more quantifiable achievements with numbers and metrics")
        
        if experience_years > 0 and "years" not in text.lower():
            suggestions.append("Clearly state your years of experience in the summary or experience section")
        
        if len(skills) > 5:
            suggestions.append("Organize skills into categories (e.g., Programming Languages, Frameworks, Tools) for better readability")
        
        # General improvement suggestions
        suggestions.append("Use strong action verbs to start each bullet point (e.g., 'Developed,' 'Implemented,' 'Managed')")
        suggestions.append("Quantify achievements with specific numbers, percentages, or timeframes when possible")
        
        return suggestions[:5]
    
    def create_skill_visualization(self, skills):
        """Create a bar chart visualization of skills by category"""
        # Check if we have visualization libraries available
        if not self.matplotlib_available and not self.plotly_available:
            print("Visualization not available: neither matplotlib/numpy nor plotly properly configured")
            return None
            
        try:
            # Categorize skills
            category_counts = {category: 0 for category in self.skills_db.keys()}
            for skill in skills:
                for category, skill_list in self.skills_db.items():
                    if skill.lower() in [s.lower() for s in skill_list]:
                        category_counts[category] += 1
                        break
            
            # Filter out categories with zero skills
            categories = [cat.replace('_', ' ').title() for cat, count in category_counts.items() if count > 0]
            counts = [count for count in category_counts.values() if count > 0]
            colors = [self.skill_colors.get(cat.lower().replace(' ', '_'), '#666666') for cat in categories]
            
            # Check if we have data to plot
            if not categories or not counts:
                print("No data to plot for skill visualization")
                return None
            
            # Use plotly if matplotlib is not available
            if not self.matplotlib_available and self.plotly_available:
                # Create plotly bar chart
                fig = go.Figure(data=[go.Bar(
                    x=categories,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title='Skills Distribution by Category',
                    xaxis_title='Skill Categories',
                    yaxis_title='Number of Skills',
                    xaxis_tickangle=-45,
                    height=500
                )
                
                # Convert to base64 for embedding in HTML
                img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return image_base64
            else:
                # Use matplotlib as fallback
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(categories)), counts, color=colors)
                
                # Customize chart
                ax.set_xlabel('Skill Categories')
                ax.set_ylabel('Number of Skills')
                ax.set_title('Skills Distribution by Category')
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Convert to base64 for embedding in HTML
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                return image_base64
        except Exception as e:
            print(f"Warning: Failed to create skill visualization: {e}")
            return None
    
    def create_detailed_skill_analysis(self, skills):
        """Create a detailed breakdown of skills by category with specific skill listings"""
        # Categorize skills
        categorized_skills = {}
        for skill in skills:
            found = False
            for category, skill_list in self.skills_db.items():
                if skill.lower() in [s.lower() for s in skill_list]:
                    if category not in categorized_skills:
                        categorized_skills[category] = []
                    categorized_skills[category].append(skill)
                    found = True
                    break
            if not found:
                # For uncategorized skills, put them in 'other'
                if 'other' not in categorized_skills:
                    categorized_skills['other'] = []
                categorized_skills['other'].append(skill)
        
        return categorized_skills
    
    def create_skill_radar_chart(self, skills):
        """Create a radar chart for skill proficiency visualization"""
        # Check if we have visualization libraries available
        if not self.matplotlib_available and not self.plotly_available:
            print("Visualization not available: neither matplotlib/numpy nor plotly properly configured")
            return None
            
        try:
            # Categorize skills
            category_counts = {category: 0 for category in self.skills_db.keys()}
            for skill in skills:
                for category, skill_list in self.skills_db.items():
                    if skill.lower() in [s.lower() for s in skill_list]:
                        category_counts[category] += 1
                        break
            
            # Prepare data for radar chart
            categories = list(self.skills_db.keys())
            values = [category_counts[cat] for cat in categories]
            
            # Check if we have data to plot
            if sum(values) == 0:
                print("No data to plot for radar chart")
                return None
            
            # Use plotly if matplotlib is not available
            if not self.matplotlib_available and self.plotly_available:
                # Normalize values to 0-10 scale for better visualization
                max_value = max(values) if max(values) > 0 else 1
                normalized_values = [v / max_value * 10 for v in values]
                
                # Create plotly radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=[cat.replace('_', ' ').title() for cat in categories],
                    fill='toself',
                    name='Skill Proficiency'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    title='Skill Proficiency Radar Chart',
                    showlegend=False,
                    height=500
                )
                
                # Convert to base64 for embedding in HTML
                img_bytes = fig.to_image(format="png", width=600, height=500, scale=2)
                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return image_base64
            else:
                # Use matplotlib
                # Normalize values to 0-10 scale for better visualization
                max_value = max(values) if max(values) > 0 else 1
                normalized_values = [v / max_value * 10 for v in values]
                
                # Add the first value to the end to close the circle
                normalized_values += normalized_values[:1]
                
                # Set up angles for radar chart
                angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
                angles += angles[:1]
                
                # Create radar chart
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], [cat.replace('_', ' ').title() for cat in categories], color='grey', size=10)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=7)
                plt.ylim(0, 10)
                
                # Plot data
                ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', color='#4ECDC4')
                ax.fill(angles, normalized_values, alpha=0.4, color='#4ECDC4')
                
                # Add title
                plt.title('Skill Proficiency Radar Chart', size=16, pad=20)
                
                # Convert to base64 for embedding in HTML
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                return image_base64
        except Exception as e:
            print(f"Warning: Failed to create skill radar chart: {e}")
            return None
    
    def create_ats_breakdown_chart(self, ats_breakdown):
        """Create a pie chart visualization of ATS score breakdown"""
        # Check if we have visualization libraries available
        if not self.matplotlib_available and not self.plotly_available:
            print("Visualization not available: neither matplotlib/numpy nor plotly properly configured")
            return None
            
        try:
            # Prepare data
            categories = list(ats_breakdown.keys())
            scores = list(ats_breakdown.values())
            
            # Check if we have data to plot
            if not categories or not scores or sum(scores) == 0:
                print("No data to plot for ATS breakdown chart")
                return None
            
            # Use plotly if matplotlib is not available
            if not self.matplotlib_available and self.plotly_available:
                # Create plotly pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=categories,
                    values=scores,
                    textinfo='label+percent',
                    textfont_size=12,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                )])
                
                fig.update_layout(
                    title='ATS Score Breakdown',
                    height=500
                )
                
                # Convert to base64 for embedding in HTML
                img_bytes = fig.to_image(format="png", width=600, height=500, scale=2)
                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return image_base64
            else:
                # Use matplotlib
                # Colors for each category
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(scores, labels=categories, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
                
                # Customize chart
                ax.set_title('ATS Score Breakdown', fontsize=16, pad=20)
                
                # Make percentage text white and bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.tight_layout()
                
                # Convert to base64 for embedding in HTML
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                return image_base64
        except Exception as e:
            print(f"Warning: Failed to create ATS breakdown chart: {e}")
            return None
    
    def analyze_resume(self, file_path):
        """Perform complete resume analysis"""
        try:
            # Extract text
            text = self.extract_text(file_path)
            
            if not text.strip():
                raise Exception("No text could be extracted from the resume")
            
            # Extract components
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience = self.extract_experience(text)
            projects = self.extract_projects(text)  # Add projects extraction
            
            # Calculate enhanced ATS score
            ats_result = self.calculate_ats_score(text, skills, contact_info, education, experience)
            ats_score = ats_result['score']
            
            # Generate recommendations based on ATS feedback
            recommendations = self.generate_recommendations(
                text, skills, contact_info, education, experience, ats_score
            )
            
            # Add ATS-specific recommendations
            recommendations.extend(ats_result['feedback'])
            
            # Create visualizations
            skill_chart = self.create_skill_visualization(skills)
            ats_chart = self.create_ats_breakdown_chart(ats_result['breakdown'])
            skill_radar = self.create_skill_radar_chart(skills)
            detailed_skills = self.create_detailed_skill_analysis(skills)
            
            # Suggest industry template
            industry, template_data = self.suggest_industry_template(skills)
            
            # Compile results
            results = {
                'text': text,
                'contact_info': contact_info,
                'skills': skills,
                'education': education,
                'experience': experience,
                'projects': projects,  # Add projects to results
                'ats_score': ats_score,
                'ats_grade': ats_result['grade'],
                'ats_breakdown': ats_result['breakdown'],
                'ats_detailed_feedback': ats_result['detailed_feedback'],
                'recommendations': list(set(recommendations)),  # Remove duplicates
                'word_count': len(text.split()),
                'character_count': len(text),
                'experience_years': experience.get('years', 0),
                'education_level': ', '.join(education.get('degrees', ['Not specified'])),
                'skills_count': len(skills),
                'tech_skills_count': len([s for s in skills if s in 
                    self.skills_db['programming_languages'] + 
                    self.skills_db['web_technologies'] + 
                    self.skills_db['databases'] + 
                    self.skills_db['cloud_platforms']]),
                'skill_chart': skill_chart,
                'ats_chart': ats_chart,
                'skill_radar': skill_radar,
                'detailed_skills': detailed_skills,
                'industry_suggestion': industry,
                'template_suggestion': template_data
            }
            
            # Compare with industry standards
            results['industry_comparison'] = self.compare_with_industry_standards(results)
            
            # Generate rewriting suggestions
            results['rewrite_suggestions'] = self.generate_rewrite_suggestions(results)
            
            # Generate interview questions
            results['interview_questions'] = self.generate_interview_questions(
                skills, 
                experience.get('years', 0), 
                industry
            )
            
            # Vector database enhancement removed - RAG only for PDF Analyzer and Chatbot
            
            return results
            
        except Exception as e:
            raise Exception(f"Resume analysis failed: {str(e)}")
    
    def compare_with_job_description(self, resume_results, job_description):
        """Enhanced resume-JD comparison with semantic similarity"""
        if not job_description:
            return {"match_score": 0, "missing_skills": [], "matching_skills": [], "similarity_score": 0}
        
        jd_text = job_description.lower()
        resume_text = resume_results['text'].lower()
        resume_skills = [skill.lower() for skill in resume_results['skills']]
        
        # Extract skills from job description using enhanced pattern matching
        jd_skills = []
        for category, skills in self.skills_db.items():
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, jd_text):
                    jd_skills.append(skill)
        
        # Extract requirements and qualifications from JD
        jd_requirements = self._extract_jd_requirements(job_description)
        
        # Find matches and gaps
        matching_skills = [skill for skill in resume_skills if skill in jd_skills]
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        # Calculate skills match score
        if len(jd_skills) > 0:
            skills_match_score = (len(matching_skills) / len(jd_skills)) * 100
        else:
            skills_match_score = 0
        
        # Calculate semantic similarity using embeddings
        try:
            if hasattr(self.model_manager, 'get_embeddings'):
                resume_embedding = self.model_manager.get_embeddings(resume_text[:1000])  # Limit text length
                jd_embedding = self.model_manager.get_embeddings(jd_text[:1000])
                
                # Calculate cosine similarity
                import numpy as np
                cosine_similarity = np.dot(resume_embedding, jd_embedding) / (
                    np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
                )
                similarity_score = max(0, cosine_similarity * 100)  # Convert to percentage
            else:
                similarity_score = 0
        except Exception:
            similarity_score = 0
        
        # Combined match score (weighted average)
        combined_score = (skills_match_score * 0.7) + (similarity_score * 0.3)
        
        # Extract missing keywords from JD
        missing_keywords = self._extract_missing_keywords(resume_text, jd_text)
        
        return {
            'match_score': round(combined_score, 2),
            'skills_match_score': round(skills_match_score, 2),
            'similarity_score': round(similarity_score, 2),
            'matching_skills': matching_skills,
            'missing_skills': missing_skills[:15],  # Top 15 missing skills
            'missing_keywords': missing_keywords[:10],  # Top 10 missing keywords
            'jd_skills_total': len(jd_skills),
            'resume_skills_total': len(resume_skills),
            'jd_requirements': jd_requirements
        }
    
    def compare_with_industry_standards(self, resume_results):
        """Compare resume against industry standards for the suggested role"""
        industry = resume_results.get('industry_suggestion', 'software_engineer')
        skills = resume_results.get('skills', [])
        experience_years = resume_results.get('experience_years', 0)
        
        # Industry standards (these would typically come from a database)
        industry_standards = {
            'software_engineer': {
                'min_skills': 8,
                'min_tech_skills': 5,
                'avg_experience': 3,
                'key_skills': ['python', 'java', 'javascript', 'sql', 'git', 'html', 'css', 'react', 'node.js'],
                'sections': ['summary', 'skills', 'experience', 'education', 'projects']
            },
            'data_scientist': {
                'min_skills': 10,
                'min_tech_skills': 7,
                'avg_experience': 4,
                'key_skills': ['python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'matplotlib', 'jupyter'],
                'sections': ['summary', 'skills', 'experience', 'education', 'projects']
            },
            'product_manager': {
                'min_skills': 6,
                'min_tech_skills': 3,
                'avg_experience': 5,
                'key_skills': ['product roadmap', 'user stories', 'agile', 'scrum', 'analytics', 'stakeholder management'],
                'sections': ['summary', 'skills', 'experience', 'education', 'projects', 'products']
            }
        }
        
        # Get standards for the industry
        standards = industry_standards.get(industry, industry_standards['software_engineer'])
        
        # Compare skills
        skills_count = len(skills)
        tech_skills_count = resume_results.get('tech_skills_count', 0)
        
        # Find missing key skills
        missing_key_skills = [skill for skill in standards['key_skills'] if skill not in [s.lower() for s in skills]]
        
        # Compare experience
        experience_comparison = {
            'user_experience': experience_years,
            'industry_avg': standards['avg_experience'],
            'difference': experience_years - standards['avg_experience']
        }
        
        # Compare sections (this would need text analysis to be fully accurate)
        text_lower = resume_results['text'].lower()
        missing_sections = [section for section in standards['sections'] if section not in text_lower]
        
        # Generate comparison score
        score = 0
        max_score = 100
        
        # Skills scoring (30 points)
        skills_score = min(30, (skills_count / standards['min_skills']) * 30) if standards['min_skills'] > 0 else 0
        score += skills_score
        
        # Tech skills scoring (25 points)
        tech_skills_score = min(25, (tech_skills_count / standards['min_tech_skills']) * 25) if standards['min_tech_skills'] > 0 else 0
        score += tech_skills_score
        
        # Experience scoring (20 points)
        if experience_years >= standards['avg_experience']:
            experience_score = 20
        else:
            experience_score = max(0, 20 - (abs(experience_comparison['difference']) * 5))
        score += experience_score
        
        # Key skills scoring (15 points)
        key_skills_found = len(standards['key_skills']) - len(missing_key_skills)
        key_skills_score = (key_skills_found / len(standards['key_skills'])) * 15 if len(standards['key_skills']) > 0 else 0
        score += key_skills_score
        
        # Sections scoring (10 points)
        sections_found = len(standards['sections']) - len(missing_sections)
        sections_score = (sections_found / len(standards['sections'])) * 10 if len(standards['sections']) > 0 else 0
        score += sections_score
        
        # Ensure score is within bounds
        final_score = min(max_score, max(0, score))
        
        return {
            'industry': industry,
            'comparison_score': round(final_score, 2),
            'skills_analysis': {
                'user_skills': skills_count,
                'industry_min': standards['min_skills'],
                'tech_skills': tech_skills_count,
                'industry_tech_min': standards['min_tech_skills'],
                'missing_key_skills': missing_key_skills[:10]
            },
            'experience_analysis': experience_comparison,
            'sections_analysis': {
                'missing_sections': missing_sections,
                'required_sections': standards['sections']
            },
            'grade': self._get_grade(final_score)
        }
    
    def export_analysis_json(self, results, filename=None):
        """Export resume analysis results to JSON format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resume_analysis_{timestamp}.json"
        
        # Create a clean version of results for export (remove binary data)
        export_results = results.copy()
        
        # Remove binary data that can't be JSON serialized
        keys_to_remove = ['skill_chart', 'ats_chart', 'skill_radar', 'text']
        for key in keys_to_remove:
            export_results.pop(key, None)
        
        # Add metadata
        export_results['export_date'] = datetime.now().isoformat()
        export_results['analysis_type'] = 'resume_analysis'
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        return filename
    
    def export_analysis_pdf(self, results, filename=None):
        """Export resume analysis results to PDF format"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"resume_analysis_{timestamp}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Resume Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_data = [
                ['Metric', 'Value'],
                ['ATS Score', f"{results['ats_score']}/100 ({results['ats_grade']})"],
                ['Skills Found', str(len(results['skills']))],
                ['Experience Years', str(results.get('experience_years', 'N/A'))],
                ['Industry Match', results.get('industry_suggestion', 'N/A').replace('_', ' ').title()],
                ['Industry Match Score', f"{results.get('industry_comparison', {}).get('comparison_score', 'N/A')}/100"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Detailed Sections
            story.append(Paragraph("Detailed Analysis", styles['Heading2']))
            
            # Skills Section
            story.append(Paragraph("Skills Analysis", styles['Heading3']))
            skills_text = ", ".join(results['skills'][:30]) if results['skills'] else "No skills found"
            story.append(Paragraph(f"Extracted Skills: {skills_text}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Recommendations
            story.append(Paragraph("Recommendations", styles['Heading3']))
            for rec in results.get('recommendations', [])[:15]:  # Limit to 15 recommendations
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            return filename
            
        except ImportError:
            # Fallback to text-based PDF if reportlab is not available
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"resume_analysis_{timestamp}.pdf"
            
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            
            # Title
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(width/2, height-50, "Resume Analysis Report")
            
            # Content
            c.setFont("Helvetica", 12)
            y_position = height - 100
            
            c.drawString(50, y_position, f"ATS Score: {results['ats_score']}/100 ({results['ats_grade']})")
            y_position -= 20
            c.drawString(50, y_position, f"Skills Found: {len(results['skills'])}")
            y_position -= 20
            c.drawString(50, y_position, f"Experience Years: {results.get('experience_years', 'N/A')}")
            y_position -= 20
            
            # Recommendations
            c.drawString(50, y_position, "Key Recommendations:")
            y_position -= 20
            
            for rec in results.get('recommendations', [])[:10]:  # Limit to 10 recommendations
                if y_position < 50:  # Start new page if needed
                    c.showPage()
                    y_position = height - 50
                c.drawString(70, y_position, f"• {rec}")
                y_position -= 15
            
            c.save()
            return filename
        except Exception as e:
            # If PDF generation fails, create a simple text file as fallback
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"resume_analysis_{timestamp}.txt"
            else:
                filename = filename.replace('.pdf', '.txt')
            
            with open(filename, 'w') as f:
                f.write("Resume Analysis Report\n")
                f.write("=====================\n\n")
                f.write(f"ATS Score: {results['ats_score']}/100 ({results['ats_grade']})\n")
                f.write(f"Skills Found: {len(results['skills'])}\n")
                f.write(f"Experience Years: {results.get('experience_years', 'N/A')}\n\n")
                f.write("Recommendations:\n")
                for rec in results.get('recommendations', []):
                    f.write(f"• {rec}\n")
            
            return filename
    
    def generate_rewrite_suggestions(self, results):
        """Generate AI-powered resume rewriting suggestions"""
        try:
            # Get the language model
            if not hasattr(self.model_manager, 'generate_text'):
                return ["AI-powered rewriting suggestions require a language model. Please ensure the model is loaded correctly."]
            
            # Prepare prompt for resume improvements
            prompt = f"""
            Based on the following resume analysis, provide specific rewriting suggestions to improve the resume:
            
            ATS Score: {results['ats_score']}/100
            Industry Match: {results.get('industry_suggestion', 'N/A')}
            Skills Found: {', '.join(results['skills'][:10])}
            Experience Years: {results.get('experience_years', 'N/A')}
            
            Key Issues Identified:
            {chr(10).join(results.get('recommendations', [])[:5])}
            
            Please provide 5 specific rewriting suggestions to improve this resume, focusing on:
            1. Better use of action verbs and keywords
            2. Quantifying achievements with metrics
            3. Improving section structure and clarity
            4. Industry-specific improvements
            5. Overall formatting and readability
            
            Format your response as a numbered list of actionable suggestions.
            """
            
            # Generate suggestions using the language model
            response = self.model_manager.generate_text(prompt, max_length=300)
            
            # Parse the response into a list
            suggestions = []
            lines = response.split('\n')
            for line in lines:
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    suggestions.append(line.strip())
            
            # If we couldn't parse specific suggestions, return the whole response
            if not suggestions:
                suggestions = [response.strip()]
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            return [f"Error generating rewriting suggestions: {str(e)}"]
    
    def _extract_jd_requirements(self, job_description):
        """Extract requirements and qualifications from job description"""
        requirements = []
        jd_lower = job_description.lower()
        
        # Common requirement patterns
        requirement_patterns = [
            r'(?:required|must have|essential)\s*:?\s*([^\n]+)',
            r'(?:qualifications?)\s*:?\s*([^\n]+)',
            r'(?:experience in|proficiency in|knowledge of)\s*([^\n]+)',
            r'(\d+\+?\s*years?\s*(?:of\s*)?experience)',
            r'(?:bachelor|master|degree)\s*(?:in)?\s*([^\n]+)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, jd_lower)
            requirements.extend(matches)
        
        return [req.strip() for req in requirements if len(req.strip()) > 5]
    
    def _extract_missing_keywords(self, resume_text, jd_text):
        """Extract important keywords missing from resume"""
        # Split texts into words
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        jd_words = re.findall(r'\b\w+\b', jd_text.lower())
        
        # Count word frequency in JD
        from collections import Counter
        jd_word_freq = Counter(jd_words)
        
        # Filter important words (length > 3, appears multiple times)
        important_jd_words = {
            word for word, freq in jd_word_freq.items() 
            if len(word) > 3 and freq > 1 and word not in resume_words
        }
        
        # Remove common words
        common_words = {
            'will', 'work', 'team', 'company', 'role', 'position', 'candidate',
            'opportunity', 'responsibilities', 'duties', 'requirements'
        }
        missing_keywords = list(important_jd_words - common_words)
        
        # Sort by frequency in JD
        missing_keywords.sort(key=lambda x: jd_word_freq[x], reverse=True)
        
        return missing_keywords
    
    def suggest_industry_template(self, skills):
        """Suggest industry-specific resume template based on skills"""
        # Count matches with industry keywords
        industry_scores = {}
        for industry, data in self.industry_templates.items():
            score = 0
            for skill in skills:
                skill_lower = skill.lower()
                # Check if skill matches any industry keywords
                for keyword in data['keywords']:
                    if keyword in skill_lower or skill_lower in keyword:
                        score += 1
                        break
            industry_scores[industry] = score
        
        # Find industry with highest score
        if industry_scores:
            best_industry = max(industry_scores, key=industry_scores.get)
            if industry_scores[best_industry] > 0:
                return best_industry, self.industry_templates[best_industry]
        
        # Default to software engineer if no strong match
        return 'software_engineer', self.industry_templates['software_engineer']
    
    def generate_interview_questions(self, skills, experience_years, industry_suggestion):
        """Generate technical and HR interview questions based on resume content"""
        # Technical questions based on skills
        technical_questions = []
        
        # Map skills to question categories
        skill_question_mapping = {
            'python': [
                "What are Python decorators and how do you use them?",
                "Explain the difference between lists and tuples in Python.",
                "How does garbage collection work in Python?",
                "What is the difference between deep copy and shallow copy?",
                "Explain Python's GIL (Global Interpreter Lock).",
                "What are Python generators and how do they differ from iterators?",
                "How do you handle exceptions in Python?",
                "Explain the use of *args and **kwargs in Python.",
                "What is the difference between a module and a package in Python?",
                "How do you manage memory in Python?"
            ],
            'java': [
                "What is the difference between == and .equals() in Java?",
                "Explain the concept of JVM, JRE, and JDK.",
                "What are the differences between ArrayList and LinkedList?",
                "Explain the importance of the 'static' keyword in Java.",
                "What is multithreading in Java and how do you implement it?",
                "What is the difference between an interface and an abstract class?",
                "Explain the concept of method overloading and overriding.",
                "What are Java annotations and how are they used?",
                "How does garbage collection work in Java?",
                "What is the purpose of the 'final' keyword in Java?"
            ],
            'javascript': [
                "Explain the difference between let, const, and var in JavaScript.",
                "What is closure in JavaScript and how is it used?",
                "Explain event delegation in JavaScript.",
                "What are promises and how do they differ from callbacks?",
                "What is the difference between null and undefined?",
                "What is the event loop in JavaScript?",
                "Explain prototypal inheritance in JavaScript.",
                "What are arrow functions and how do they differ from regular functions?",
                "How does the 'this' keyword work in JavaScript?",
                "What is the difference between synchronous and asynchronous code?"
            ],
            'sql': [
                "What is the difference between INNER JOIN and OUTER JOIN?",
                "Explain ACID properties in databases.",
                "What is normalization and why is it important?",
                "How do you optimize a slow-running SQL query?",
                "What is the difference between DELETE and TRUNCATE commands?",
                "Explain the difference between WHERE and HAVING clauses.",
                "What are indexes and how do they improve query performance?",
                "What is a subquery and when would you use it?",
                "Explain the difference between UNION and UNION ALL.",
                "What are SQL constraints and how are they used?"
            ],
            'react': [
                "What are the differences between state and props in React?",
                "Explain the React component lifecycle methods.",
                "What is the virtual DOM and how does it work?",
                "What are React hooks and how do you use them?",
                "Explain the concept of lifting state up in React.",
                "What is the difference between controlled and uncontrolled components?",
                "How do you optimize React component performance?",
                "Explain the use of React Context.",
                "What are React Fragments and why are they useful?",
                "How do you handle forms in React?"
            ],
            'node.js': [
                "What is the event loop in Node.js?",
                "Explain middleware in Express.js.",
                "How do you handle errors in Node.js?",
                "What is the difference between synchronous and asynchronous code?",
                "Explain the module system in Node.js.",
                "How do you handle file uploads in Node.js?",
                "What is clustering in Node.js and why is it used?",
                "Explain the concept of streams in Node.js.",
                "How do you secure a Node.js application?",
                "What is the purpose of package.json?"
            ],
            'docker': [
                "What is the difference between an image and a container in Docker?",
                "Explain Docker Compose and its benefits.",
                "How do you optimize Docker images for production?",
                "What are Docker volumes and when would you use them?",
                "Explain the concept of multi-stage builds in Docker.",
                "How do you network containers in Docker?",
                "What is the difference between CMD and ENTRYPOINT in Docker?",
                "Explain Docker layers and how they work.",
                "How do you manage Docker containers in production?",
                "What are Docker registries and how do they work?"
            ],
            'aws': [
                "What are the different types of EC2 instances?",
                "Explain the difference between S3 and EBS storage.",
                "What is auto-scaling in AWS and how does it work?",
                "How do you secure data in AWS?",
                "Explain the difference between IAM roles and users.",
                "What is the difference between AMI and snapshot in AWS?",
                "Explain the concept of VPC in AWS.",
                "How do you monitor applications in AWS?",
                "What are the different types of load balancers in AWS?",
                "Explain the difference between EBS and instance store."
            ]
        }
        
        # Add technical questions based on skills found
        skill_specific_questions = []
        for skill in skills[:15]:  # Increase limit to top 15 skills
            skill_lower = skill.lower()
            for key, questions in skill_question_mapping.items():
                if key in skill_lower or skill_lower in key:
                    # Add questions specific to this skill
                    skill_specific_questions.extend(questions[:5])  # Add top 5 questions per skill
                    break
        
        # Add skill-specific questions to the beginning of the list
        technical_questions.extend(skill_specific_questions)
        
        # Add general technical questions based on experience level
        general_tech_questions = [
            "Explain the difference between TCP and UDP.",
            "What is the difference between authentication and authorization?",
            "How do you ensure code quality in your projects?",
            "Explain the concept of RESTful APIs.",
            "What are the principles of object-oriented programming?",
            "How do you handle debugging in your development process?",
            "Explain the difference between unit testing and integration testing.",
            "What is version control and why is it important?",
            "How do you stay updated with the latest technologies in your field?",
            "Describe a challenging technical problem you solved and how you approached it.",
            "What is the difference between a stack and a queue?",
            "Explain the concept of caching and when to use it.",
            "What is the difference between synchronous and asynchronous programming?",
            "How do you approach debugging a production issue?",
            "Explain the concept of microservices architecture.",
            "What are design patterns and why are they useful?",
            "How do you handle security in your applications?",
            "Explain the difference between horizontal and vertical scaling.",
            "What is the CAP theorem and how does it apply to distributed systems?",
            "How do you handle data consistency in distributed systems?"
        ]
        
        # Adjust technical questions based on experience level
        if experience_years >= 5:
            technical_questions.extend([
                "How do you approach architectural decisions in large-scale systems?",
                "Describe your experience with mentoring junior developers.",
                "How do you balance technical debt with feature development?",
                "Explain your approach to system design and scalability.",
                "How do you evaluate and integrate new technologies into existing systems?",
                "How do you handle performance optimization in large applications?",
                "Describe your experience with DevOps practices.",
                "How do you ensure high availability and fault tolerance in systems?",
                "Explain your approach to disaster recovery planning.",
                "How do you handle capacity planning for growing systems?"
            ])
        elif experience_years >= 2:
            technical_questions.extend([
                "How do you approach debugging complex issues in production?",
                "Describe a time when you had to learn a new technology quickly.",
                "How do you ensure your code is maintainable and readable?",
                "Explain your experience with code reviews.",
                "How do you handle disagreements with team members on technical decisions?",
                "How do you approach testing in your projects?",
                "Describe a time when you had to refactor legacy code.",
                "How do you handle technical documentation?",
                "Explain your experience with CI/CD pipelines.",
                "How do you approach code optimization?"
            ])
        else:
            technical_questions.extend([
                "Walk me through a project you've worked on from start to finish.",
                "How do you approach solving a coding problem you've never encountered before?",
                "Describe your experience working in a team environment.",
                "How do you handle feedback on your code?",
                "What steps do you take to ensure your code is bug-free?",
                "How do you approach learning new programming concepts?",
                "Describe a time when you had to debug someone else's code.",
                "How do you handle tight deadlines in projects?",
                "Explain your approach to version control.",
                "How do you ensure your code follows best practices?"
            ])
        
        # Use LLM to generate additional personalized technical questions based on skills and experience
        if self.model_manager and hasattr(self.model_manager, 'generate_text'):
            try:
                # Create a prompt for generating personalized technical questions
                skills_str = ", ".join(skills[:10])
                prompt = f"""Based on the following skills and experience level, generate 10 additional technical interview questions that would be relevant for a candidate with {experience_years} years of experience in these areas: {skills_str}.
                
                Requirements:
                1. Questions should be specific to the skills mentioned
                2. Questions should match the experience level
                3. Include a mix of conceptual, practical, and problem-solving questions
                4. Format each question on a new line without any numbering or bullet points
                5. Do not include any additional text, just the questions
                """
                
                llm_response = self.model_manager.generate_text(prompt, max_length=500)
                # Parse the response to extract questions
                if llm_response:
                    llm_questions = [q.strip() for q in llm_response.split('\n') if q.strip() and '?' in q]
                    technical_questions.extend(llm_questions[:10])  # Add up to 10 LLM-generated questions
            except Exception as e:
                print(f"Error generating LLM technical questions: {e}")
        
        # Combine general and specific technical questions
        technical_questions.extend(general_tech_questions[:10])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_technical_questions = []
        for question in technical_questions:
            if question not in seen:
                seen.add(question)
                unique_technical_questions.append(question)
        
        # Increase limit to 30 technical questions
        technical_questions = unique_technical_questions[:30]
        
        # Common HR questions (expanded to 40+ questions)
        hr_questions = [
            "Tell me about yourself and your background.",
            "Why do you want to work for our company?",
            "What are your strengths and weaknesses?",
            "Describe a time when you faced a significant challenge at work and how you overcame it.",
            "Where do you see yourself in 5 years?",
            "Why are you leaving your current job?",
            "How do you handle stress and pressure?",
            "Describe a situation where you had to work with a difficult team member.",
            "What motivates you in your work?",
            "How do you prioritize your work when you have multiple deadlines?",
            "Tell me about a time when you had to make a difficult decision.",
            "How do you handle failure or setbacks?",
            "What are your salary expectations?",
            "Do you have any questions for us?",
            "What interests you most about this role?",
            "How do you define success?",
            "What would your previous supervisor say about you?",
            "Describe your ideal work environment.",
            "How do you handle criticism?",
            "What type of work culture do you prefer?",
            "Tell me about a time you demonstrated leadership skills.",
            "How do you stay organized?",
            "What would you do if you disagreed with your manager?",
            "Describe a time when you went above and beyond your job responsibilities.",
            "How do you handle working under tight deadlines?",
            "What are your hobbies and interests outside of work?",
            "How do you continue to educate yourself in your field?",
            "Tell me about a time you had to adapt to a major change at work.",
            "What are three words your colleagues would use to describe you?",
            "How do you measure your performance at work?",
            "What steps do you take to build relationships with colleagues?",
            "Describe a situation where you had to persuade someone to see things your way.",
            "How do you handle conflicts in the workplace?",
            "What would you do if you noticed a colleague was struggling?",
            "How do you balance work and personal life?",
            "Describe a time when you had to work with a team that was not performing well.",
            "What would you do if you were given a task outside your comfort zone?",
            "How do you handle feedback from peers?",
            "Tell me about a time when you had to work with limited resources.",
            "How do you approach continuous learning and professional development?",
            "What role do you typically play in a team setting?",
            "How do you handle ambiguity in tasks or projects?",
            "Describe a situation where you had to influence someone without authority.",
            "How do you approach setting and achieving goals?"
        ]
        
        # Industry-specific HR questions
        industry_hr_questions = {
            'software_engineer': [
                "How do you approach learning new programming languages or technologies?",
                "Describe your ideal development environment.",
                "How do you ensure the quality of your code?",
                "What software development methodologies are you familiar with?",
                "How do you handle tight deadlines and scope changes?",
                "How do you approach code reviews?",
                "Describe your experience with pair programming.",
                "How do you handle technical disagreements with team members?",
                "What tools do you use for debugging and profiling?",
                "How do you approach technical documentation?"
            ],
            'data_scientist': [
                "How do you ensure the ethical use of data in your projects?",
                "Describe a time when your analysis influenced a business decision.",
                "How do you handle missing or dirty data?",
                "What's your approach to explaining complex technical findings to non-technical stakeholders?",
                "How do you stay current with developments in data science and machine learning?",
                "How do you validate your models and ensure they generalize well?",
                "Describe a challenging data science project you worked on.",
                "How do you handle imbalanced datasets?",
                "What's your approach to feature selection and engineering?",
                "How do you communicate uncertainty in your findings?"
            ],
            'product_manager': [
                "How do you prioritize features when resources are limited?",
                "Describe a time when you had to convince a team to pursue a different direction.",
                "How do you measure the success of a product?",
                "How do you handle conflicts between engineering and business teams?",
                "Describe your approach to user research and feedback collection.",
                "How do you handle scope creep in projects?",
                "Describe a product you worked on that didn't meet expectations.",
                "How do you approach A/B testing and experimentation?",
                "How do you balance short-term needs with long-term vision?",
                "How do you handle feedback from customers and stakeholders?"
            ]
        }
        
        # Add industry-specific HR questions
        if industry_suggestion in industry_hr_questions:
            hr_questions.extend(industry_hr_questions[industry_suggestion])
        
        # Use LLM to generate additional personalized HR questions based on industry and experience
        if self.model_manager and hasattr(self.model_manager, 'generate_text'):
            try:
                # Create a prompt for generating personalized HR questions
                prompt = f"""Based on the {industry_suggestion} industry and {experience_years} years of experience, generate 10 additional HR interview questions that would be relevant for a candidate.
                
                Requirements:
                1. Questions should be relevant to the industry
                2. Questions should match the experience level
                3. Include a mix of behavioral, situational, and cultural fit questions
                4. Format each question on a new line without any numbering or bullet points
                5. Do not include any additional text, just the questions
                """
                
                llm_response = self.model_manager.generate_text(prompt, max_length=500)
                # Parse the response to extract questions
                if llm_response:
                    llm_questions = [q.strip() for q in llm_response.split('\n') if q.strip() and '?' in q]
                    hr_questions.extend(llm_questions[:10])  # Add up to 10 LLM-generated questions
            except Exception as e:
                print(f"Error generating LLM HR questions: {e}")
        
        # Randomize the order of questions but keep some structure
        import random
        if len(technical_questions) > 10:
            technical_questions = random.sample(technical_questions, min(30, len(technical_questions)))
        if len(hr_questions) > 10:
            hr_questions = random.sample(hr_questions, min(30, len(hr_questions)))
        
        return {
            'technical_questions': technical_questions,
            'hr_questions': hr_questions
        }
    
    # Vector database methods removed - RAG only for PDF Analyzer and Chatbot