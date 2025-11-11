"""
Text Processing Utilities
Common text processing functions for the placement bot
"""

import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class TextProcessor:
    """Text processing utilities"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords from text"""
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            return ' '.join(filtered_tokens)
        except:
            # Fallback without NLTK
            words = text.split()
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
    
    def extract_keywords(self, text, top_k=10):
        """Extract top keywords from text"""
        # Clean text
        cleaned_text = self.clean_text(text)
        cleaned_text = self.remove_stopwords(cleaned_text)
        
        # Tokenize and count
        words = cleaned_text.split()
        words = [word for word in words if len(word) > 2]  # Remove very short words
        
        # Count frequency
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, freq in word_freq.most_common(top_k)]
    
    def calculate_similarity(self, text1, text2):
        """Calculate basic text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        # Clean and tokenize both texts
        words1 = set(self.clean_text(text1).split())
        words2 = set(self.clean_text(text2).split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def word_count(self, text):
        """Count words in text"""
        if not text:
            return 0
        return len(text.split())
    
    def character_count(self, text):
        """Count characters in text"""
        if not text:
            return 0
        return len(text)
    
    def readability_score(self, text):
        """Calculate basic readability score"""
        if not text:
            return 0
        
        sentences = self.extract_sentences(text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        # Simple readability metric
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Lower scores are better (easier to read)
        score = (avg_words_per_sentence * 0.4) + (avg_chars_per_word * 0.6)
        
        # Normalize to 1-10 scale (lower is better)
        normalized_score = min(10, max(1, score / 2))
        
        return round(normalized_score, 1)

def preprocess_resume_text(text):
    """Preprocess resume text for analysis"""
    processor = TextProcessor()
    
    # Clean the text
    cleaned = processor.clean_text(text)
    
    # Extract key information sections
    sections = {
        'education': extract_education_section(text),
        'experience': extract_experience_section(text),
        'skills': extract_skills_section(text),
        'projects': extract_projects_section(text)
    }
    
    return {
        'cleaned_text': cleaned,
        'sections': sections,
        'word_count': processor.word_count(text),
        'character_count': processor.character_count(text),
        'readability': processor.readability_score(text)
    }

def extract_education_section(text):
    """Extract education-related content"""
    education_keywords = [
        'education', 'degree', 'bachelor', 'master', 'phd', 'university',
        'college', 'school', 'graduated', 'gpa', 'coursework'
    ]
    
    lines = text.split('\n')
    education_lines = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in education_keywords):
            education_lines.append(line.strip())
    
    return '\n'.join(education_lines)

def extract_experience_section(text):
    """Extract work experience content"""
    experience_keywords = [
        'experience', 'work', 'job', 'position', 'role', 'employment',
        'intern', 'internship', 'company', 'organization', 'responsibilities'
    ]
    
    lines = text.split('\n')
    experience_lines = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in experience_keywords):
            experience_lines.append(line.strip())
    
    return '\n'.join(experience_lines)

def extract_skills_section(text):
    """Extract skills-related content"""
    skills_keywords = [
        'skills', 'technologies', 'programming', 'languages', 'tools',
        'software', 'frameworks', 'libraries', 'technical', 'proficient'
    ]
    
    lines = text.split('\n')
    skills_lines = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in skills_keywords):
            skills_lines.append(line.strip())
    
    return '\n'.join(skills_lines)

def extract_projects_section(text):
    """Extract projects-related content with enhanced detection"""
    # Split text into lines
    lines = text.split('\n')
    project_lines = []
    
    # Look for project section header first with more variations
    project_section_headers = [
        'projects:', 'project:', 'personal projects:', 'key projects:', 
        'technical projects:', 'side projects:', 'relevant projects:'
    ]
    
    project_section_found = False
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        # Check if this is a project section header
        if any(header in line_lower for header in project_section_headers):
            project_section_found = True
            # Continue to collect lines until we hit another section
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                # Stop if we hit another section header
                section_headers = [
                    'experience:', 'education:', 'skills:', 'achievements:', 
                    'certifications:', 'awards:', 'contact:', 'summary:', 
                    'objective:', 'profile:', 'work experience:', 'employment:'
                ]
                # Check if next line is a new section (ends with colon or is a known section header)
                if (next_line.lower() in section_headers or 
                    next_line.lower() in [h + ':' for h in section_headers] or
                    (':' in next_line and next_line.isupper()) or
                    (next_line.endswith(':') and len(next_line.split()) <= 3)):
                    break
                # Add non-empty lines that are not just whitespace
                if next_line and len(next_line) > 5:
                    project_lines.append(next_line)
            break
    
    # If no project section found, look for project-related sentences
    if not project_section_found and not project_lines:
        # More specific project patterns to avoid extracting personal information
        project_indicators = [
            r'project\s*:',
            r'developed\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'built\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'created\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'designed\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'implemented\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'engineered\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
            r'architected\s+.*(?:application|system|website|software|tool|platform|api|dashboard)',
        ]
        
        # Look for lines with project indicators
        for line in lines:
            line_lower = line.lower().strip()
            # Check if line contains project indicators
            if any(re.search(pattern, line_lower) for pattern in project_indicators):
                # Avoid lines that are clearly not projects (like addresses, institutions, etc.)
                exclude_keywords = [
                    'college', 'university', 'school', 'institute', 'tumkur', 
                    'karnataka', 'address', 'phone', 'email', 'grade',
                    'dormitory', 'management system', 'vijaya pu college'
                ]
                if not any(exclude in line_lower for exclude in exclude_keywords):
                    project_lines.append(line)
        
        # If still no projects found, look for lines that start with project-like patterns
        if not project_lines:
            for line in lines:
                line_stripped = line.strip()
                # Look for lines that start with capital letters and contain technical terms
                if (line_stripped and line_stripped[0].isupper() and 
                    len(line_stripped) > 10 and
                    any(tech_term in line_stripped.lower() for tech_term in [
                        'python', 'java', 'javascript', 'react', 'node', 'sql', 
                        'docker', 'aws', 'machine learning', 'data analysis',
                        'api', 'web', 'mobile', 'application', 'system', 'dashboard',
                        'platform', 'tool', 'model', 'algorithm'
                    ])):
                    # Avoid personal information lines
                    exclude_keywords = [
                        'college', 'university', 'school', 'institute', 'tumkur', 
                        'karnataka', 'address', 'phone', 'email', 'grade'
                    ]
                    if not any(exclude in line_stripped.lower() for exclude in exclude_keywords):
                        project_lines.append(line_stripped)
    
    # Additional filtering to remove experience lines that might have been captured
    filtered_project_lines = []
    experience_indicators = [
        'experience:', 'work experience:', 'employment:', 'job:', 'position:',
        'responsibilities:', 'duties:', 'role:', 'title:'
    ]
    
    for line in project_lines:
        line_lower = line.lower()
        # Skip lines that are clearly experience-related
        if not any(indicator in line_lower for indicator in experience_indicators):
            # Skip lines that contain experience verbs in experience context
            experience_verbs = ['worked', 'responsible', 'handled', 'managed', 'led']
            if not (any(verb in line_lower for verb in experience_verbs) and 
                   any(word in line_lower for word in ['team', 'client', 'project', 'company'])):
                filtered_project_lines.append(line)
    
    return '\n'.join(filtered_project_lines if filtered_project_lines else project_lines)

def format_for_display(text, max_length=500):
    """Format text for display in UI"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Truncate and add ellipsis
    return text[:max_length-3] + "..."

def extract_contact_patterns(text):
    """Extract common contact information patterns"""
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'linkedin': r'linkedin\.com/in/[\w-]+',
        'github': r'github\.com/[\w-]+',
        'website': r'https?://[\w\.-]+\.[\w]+'
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            results[pattern_name] = matches
    
    return results