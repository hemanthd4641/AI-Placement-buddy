# Career Roadmap Generator Feature Documentation

## Overview
The Career Roadmap Generator creates personalized, timeline-based career development plans tailored to a user's target role, experience level, and goals. It provides structured learning phases with specific skills, resources, project ideas, and milestones to guide professional growth.

## Technology Stack
- **Python**: Core programming language
- **Natural Language Processing**: Role and skill analysis
- **Vector Database**: Storage and retrieval of roadmap templates
- **Machine Learning Models**: AI-powered roadmap generation (when available)
- **JSON**: Data storage and configuration
- **Streamlit**: Web interface for user interaction
- **Sentence Transformers**: Embedding generation for semantic analysis
- **FAISS**: Vector similarity search engine
- **SQLite**: Metadata storage for vector database

## Input Processing
1. **Target Role**: User's desired career position
2. **Experience Level**: Current proficiency (Beginner, Intermediate, Advanced)
3. **Timeline**: Preferred duration for achieving goals (3-24 months)
4. **Primary Goal**: Career change, promotion, or skill development
5. **Current Skills** (optional): User's existing skill set

## Core Processing Components

### 1. Role Analysis
- **Skill Extraction**: Identifies required skills from the target role
- **Industry Mapping**: Determines appropriate industry template
- **Experience Alignment**: Matches roadmap complexity to user's level

### 2. Template System
Predefined roadmap templates for various roles:
- **Software Development**: Programming fundamentals, web development, system design
- **Data Science**: Mathematics, programming, machine learning
- **Custom Roles**: Dynamically generated roadmaps using LLM

### 3. Phase Generation
Creates learning phases with:
- **Duration**: Time allocation based on complexity
- **Skills**: Specific competencies to develop
- **Resources**: Courses, tutorials, and learning materials
- **Projects**: Hands-on project ideas with technologies and time estimates
- **Milestones**: Measurable achievements for progress tracking

### 4. Timeline Adjustment
- **Proportional Scaling**: Adjusts phase durations to match user's timeline
- **Experience-based Pacing**: Modifies content depth based on proficiency level
- **Goal Alignment**: Tailors content to career change, promotion, or skill development

## RAG and Vector Database Integration

### How LLM and Vector Database Work Together

The Career Roadmap Generator implements Retrieval-Augmented Generation (RAG) to enhance its roadmap creation capabilities by combining the power of Large Language Models (LLMs) with semantic search from a vector database:

1. **Template Retrieval**: The system searches the vector database for career roadmaps similar to the user's target role and experience level.

2. **Context Enhancement**: Retrieved professional roadmaps and transition examples are used to enhance the LLM's understanding of effective learning paths.

3. **Personalized Roadmap Generation**: The LLM generates a customized roadmap based on the enhanced context and professional examples.

4. **Resource Enhancement**: The system searches for learning resources that match the skills and phases in the generated roadmap.

### Detailed Processing Flow

1. **User Input Analysis**:
   - User specifies target role, experience level, and timeline
   - System analyzes input to determine roadmap requirements
   - Experience level and goals are categorized

2. **Vector Database Roadmap Search**:
   - Target role and experience level are converted to embeddings
   - System searches FAISS vector index for similar career roadmaps
   - Top-k roadmaps are retrieved with metadata from SQLite
   - Roadmaps are filtered by role, experience level, and specialization

3. **Professional Transition Analysis**:
   - System retrieves career transition examples from vector database
   - Extracts learning paths, timelines, and phase structures
   - Identifies common challenges and solutions for the target role
   - Analyzes resource effectiveness from successful transitions

4. **LLM Prompt Enhancement**:
   - Retrieved roadmaps and transition examples are incorporated into LLM prompt
   - Professional roadmap structures guide generation
   - Industry-specific skills and phases are prioritized
   - Realistic timelines are established based on successful examples

5. **Personalized Roadmap Generation**:
   - Enhanced prompt is sent to LLM with user's requirements
   - LLM generates customized roadmap with phases and skills
   - Roadmap incorporates best practices from professional examples
   - Timeline is adjusted based on user's specified duration

6. **Resource Integration**:
   - Generated roadmap phases are converted to embeddings
   - System searches for learning resources matching each phase
   - Resources are retrieved and linked to roadmap milestones
   - Resource diversity is ensured (courses, tutorials, practice)

7. **Project Idea Integration**:
   - System generates relevant project ideas for each phase
   - Projects include descriptions, technologies, difficulty levels, and time estimates
   - Project ideas are tailored to the skills being learned in each phase

8. **Quality Assurance**:
   - Generated roadmap is compared with professional examples
   - Completeness and coherence are verified
   - Timeline realism is validated against successful transitions
   - Resource availability is confirmed

### Benefits of RAG Integration

1. **Evidence-Based Roadmaps**:
   - Learning paths are based on successful career transitions
   - Phase structures are validated by professional outcomes
   - Timeline estimates are grounded in real experiences

2. **Comprehensive Coverage**:
   - Roadmaps include all essential skills for the target role
   - Professional specializations are incorporated
   - Industry-specific requirements are addressed

3. **Personalized Structure**:
   - Roadmaps are tailored to user's experience level
   - Phase durations are adjusted to user's timeline
   - Learning preferences are accommodated

4. **Resource Integration**:
   - High-quality learning resources are automatically linked
   - Resource diversity ensures comprehensive learning
   - Free and accessible resources are prioritized

5. **Project-Based Learning**:
   - Hands-on project ideas reinforce theoretical knowledge
   - Projects are tailored to specific skills and technologies
   - Time estimates help with planning and progress tracking

## Output Generation
1. **Structured Roadmap**: Phased learning plan with clear objectives
2. **Resource Recommendations**: Links to courses, tutorials, and practice platforms
3. **Project Ideas**: Hands-on projects with technologies and time estimates
4. **Progress Tracking**: Milestone-based achievement system
5. **Export Options**: JSON and Markdown formats for external use
6. **Next Steps Suggestions**: Personalized guidance based on current progress
7. **Professional Examples**: Successful roadmaps similar to user's situation
8. **Resource Diversity**: Multiple learning formats for each phase
9. **Timeline Validation**: Realistic duration based on professional transitions

## Workflow Process
```
User inputs (role, experience, timeline, goals) → 
Role analysis → Vector database roadmap search → 
Professional transition analysis → LLM prompt enhancement → 
Roadmap generation → Resource integration → 
Project idea generation → Quality assurance → Output presentation
```

## Key Features
- **Personalized Planning**: Custom roadmaps based on individual parameters
- **Flexible Timeline**: Adjustable duration to fit user constraints
- **Comprehensive Resources**: Curated learning materials for each phase
- **Project Ideas**: Hands-on projects with detailed specifications
- **Progress Tracking**: Milestone-based achievement system
- **Export Functionality**: JSON and Markdown export options
- **AI Enhancement**: LLM-powered dynamic roadmap generation
- **RAG-Enhanced Generation**: Professional examples guide roadmap creation
- **Semantic Template Matching**: Context-aware roadmap selection

## Technical Implementation Details
- **Template-based Approach**: Predefined roadmaps for common roles
- **Dynamic Generation**: AI-created roadmaps for custom roles
- **Vector Database Integration**: Semantic search for similar roadmaps
- **Modular Architecture**: Separate components for each processing stage
- **Fallback Mechanisms**: Template-based generation when AI is unavailable
- **Vector Database Integration**: Seamless RAG implementation with FAISS and Sentence Transformers
- **Quality Assurance**: Roadmap completeness validated through professional examples

## Dependencies
- spaCy (for NLP)
- Vector database libraries (faiss-cpu, sentence-transformers)
- Machine learning model manager (optional)
- sqlite3 (for metadata storage)