"""
Script to populate the vector database with professional content from internet sources
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "utils"))
sys.path.append(str(Path(__file__).parent / "modules"))

# Import vector database manager
try:
    from utils.vector_db_manager import VectorDBManager
    print("✅ Vector database manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VectorDBManager: {e}")
    VectorDBManager = None

def populate_professional_resumes():
    """Add professional resume examples to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Professional resume examples (simplified versions)
    professional_resumes = [
        {
            "text": """John Doe - Senior Software Engineer
            Email: john.doe@email.com | Phone: (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe
            
            SUMMARY
            Senior Software Engineer with 8+ years of experience in developing scalable web applications 
            and leading engineering teams. Expertise in Python, JavaScript, and cloud technologies.
            
            TECHNICAL SKILLS
            Languages: Python, JavaScript, Java, SQL
            Frameworks: Django, React, Node.js, Express
            Databases: PostgreSQL, MongoDB, Redis
            Cloud: AWS, Docker, Kubernetes
            Tools: Git, Jenkins, Jira
            
            PROFESSIONAL EXPERIENCE
            Senior Software Engineer | TechCorp | Jan 2020 - Present
            - Led a team of 5 developers to build a microservices-based e-commerce platform
            - Implemented CI/CD pipeline reducing deployment time by 60%
            - Developed RESTful APIs serving 1M+ daily active users
            - Reduced system downtime by 95% through proactive monitoring
            
            Software Engineer | InnovateX | Jun 2016 - Dec 2019
            - Developed and maintained customer-facing web applications
            - Collaborated with UX team to implement responsive designs
            - Optimized database queries improving performance by 40%
            
            EDUCATION
            M.S. Computer Science | University of Technology | 2016
            B.S. Software Engineering | State University | 2014
            
            CERTIFICATIONS
            AWS Certified Solutions Architect
            Google Professional Cloud Developer""",
            "metadata": {
                "type": "professional_resume",
                "industry": "software_engineering",
                "experience_level": "senior",
                "title": "Senior Software Engineer Resume Example"
            }
        },
        {
            "text": """Sarah Johnson - Data Scientist
            Email: sarah.j@datasci.com | Location: San Francisco, CA
            
            PROFESSIONAL SUMMARY
            Data Scientist with 5 years of experience in machine learning and statistical analysis. 
            Specialized in predictive modeling and data visualization. Proficient in Python, R, and SQL.
            
            CORE COMPETENCIES
            Machine Learning: Scikit-learn, TensorFlow, PyTorch
            Data Analysis: Pandas, NumPy, SciPy
            Visualization: Tableau, Power BI, Matplotlib
            Big Data: Spark, Hadoop
            Cloud: AWS, GCP
            Databases: PostgreSQL, MongoDB
            
            PROFESSIONAL EXPERIENCE
            Data Scientist | DataDriven Inc. | Mar 2021 - Present
            - Built recommendation engine increasing user engagement by 35%
            - Developed predictive models for customer churn with 89% accuracy
            - Created automated reporting dashboards saving 10 hours/week
            
            Data Analyst | AnalyticsPro | Aug 2018 - Feb 2021
            - Analyzed sales data to identify market trends and opportunities
            - Built statistical models to forecast demand with 92% accuracy
            - Presented findings to executive team influencing strategic decisions
            
            EDUCATION
            M.S. Data Science | Stanford University | 2018
            B.S. Mathematics | UC Berkeley | 2016
            
            PROJECTS
            Customer Segmentation Analysis
            - Used K-means clustering to segment customers into 8 distinct groups
            - Increased marketing campaign effectiveness by 28%""",
            "metadata": {
                "type": "professional_resume",
                "industry": "data_science",
                "experience_level": "mid_level",
                "title": "Data Scientist Resume Example"
            }
        }
    ]
    
    # Add resumes to vector database
    for i, resume in enumerate(professional_resumes):
        try:
            doc_id = vector_db.add_resume(resume["text"], resume["metadata"])
            print(f"✅ Added professional resume {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding professional resume {i+1}: {e}")

def populate_cover_letter_templates():
    """Add professional cover letter templates to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Professional cover letter templates
    cover_letter_templates = [
        {
            "text": """Dear Hiring Manager,
            
I am writing to express my strong interest in the Software Engineer position at [Company Name]. 
With [X] years of experience in developing scalable web applications and leading engineering teams, 
I am confident in my ability to contribute to your organization's success.

In my current role at [Current Company], I have successfully:
- Led a team of 5 developers to build a microservices-based platform serving 1M+ daily users
- Implemented CI/CD pipeline reducing deployment time by 60%
- Reduced system downtime by 95% through proactive monitoring

I am particularly drawn to [Company Name] because of [specific reason related to company]. 
I would welcome the opportunity to bring my expertise in [relevant skill] to your team.

Thank you for your time and consideration. I look forward to discussing how I can contribute to [Company Name]'s continued success.

Sincerely,
[Your Name]""",
            "metadata": {
                "type": "cover_letter_template",
                "industry": "technology",
                "role": "software_engineer",
                "title": "Software Engineer Cover Letter Template"
            }
        },
        {
            "text": """Dear [Hiring Manager Name],
            
I am excited to apply for the Data Scientist position at [Company Name]. 
With a Master's degree in Data Science and [X] years of experience in machine learning 
and statistical analysis, I am eager to contribute to your data-driven initiatives.

My recent achievements include:
- Built recommendation engine increasing user engagement by 35%
- Developed predictive models for customer churn with 89% accuracy
- Created automated reporting dashboards saving 10 hours/week

I am impressed by [Company Name]'s commitment to [specific company value/initiative] 
and would be thrilled to apply my skills in [relevant skill] to support your goals.

Thank you for considering my application. I look forward to the opportunity to discuss 
how my analytical skills can benefit [Company Name].

Best regards,
[Your Name]""",
            "metadata": {
                "type": "cover_letter_template",
                "industry": "data_science",
                "role": "data_scientist",
                "title": "Data Scientist Cover Letter Template"
            }
        }
    ]
    
    # Add cover letter templates to vector database
    for i, template in enumerate(cover_letter_templates):
        try:
            doc_id = vector_db.add_knowledge_item(template["text"], template["metadata"])
            print(f"✅ Added cover letter template {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding cover letter template {i+1}: {e}")

def populate_career_roadmaps():
    """Add career roadmaps for different job roles to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Career roadmaps for different roles
    career_roadmaps = [
        {
            "text": """Career Roadmap: Software Engineer (0-3 years experience)
            
Phase 1: Foundation Building (Months 1-6)
Skills to Master: Programming fundamentals, Data structures, Algorithms
Key Milestones: Complete online courses, Build 3 personal projects, Contribute to open source

Phase 2: Web Development (Months 7-12)
Skills to Master: HTML/CSS, JavaScript, Frontend frameworks, Backend development
Key Milestones: Build full-stack web application, Deploy to cloud platform, Learn version control

Phase 3: Specialization (Months 13-18)
Skills to Master: Database design, API development, Testing, Security
Key Milestones: Lead a small project, Implement automated testing, Learn about system design

Resources:
- freeCodeCamp (Free)
- CS50: Introduction to Computer Science (Free)
- The Odin Project (Free)
- Python for Everybody (Free on Coursera)

Projects:
1. Personal Portfolio Website - Create a responsive website to showcase your skills and projects
   Technologies: HTML, CSS, JavaScript, React
   Difficulty: Beginner
   Estimated Hours: 20

2. E-commerce Landing Page - Design and build a responsive e-commerce landing page
   Technologies: HTML, CSS, JavaScript, Bootstrap, React
   Difficulty: Beginner
   Estimated Hours: 30

Certifications:
- AWS Certified Developer - Associate
- Google Professional Cloud Developer
- Microsoft Certified: Azure Developer Associate""",
            "metadata": {
                "type": "career_roadmap",
                "role": "software_engineer",
                "experience_level": "entry_level",
                "title": "Software Engineer Entry Level Roadmap"
            }
        },
        {
            "text": """Career Roadmap: Data Scientist (0-3 years experience)
            
Phase 1: Mathematics & Statistics (Months 1-4)
Skills to Master: Linear Algebra, Calculus, Probability, Statistics
Key Milestones: Complete Mathematics for Machine Learning course, Apply statistical concepts to datasets

Phase 2: Programming & Data Analysis (Months 5-8)
Skills to Master: Python, Pandas, NumPy, Data Cleaning, Data Visualization
Key Milestones: Clean and preprocess real-world dataset, Perform exploratory data analysis

Phase 3: Machine Learning Fundamentals (Months 9-12)
Skills to Master: Scikit-learn, Supervised Learning, Model Evaluation
Key Milestones: Build and evaluate machine learning models, Complete Kaggle competition

Resources:
- Khan Academy Statistics (Free)
- Python for Data Science (Free on Coursera)
- Kaggle Learn (Free)
- Andrew Ng's Machine Learning Course (Free on Coursera)

Projects:
1. Descriptive Statistics Analysis - Analyze a real-world dataset and create comprehensive statistical reports
   Technologies: Python, Pandas, NumPy, Matplotlib
   Difficulty: Beginner
   Estimated Hours: 20

2. Predictive Model for Housing Prices - Build a regression model to predict housing prices
   Technologies: Python, Scikit-learn, Pandas, NumPy
   Difficulty: Intermediate
   Estimated Hours: 40

Certifications:
- Google Professional Data Engineer
- Microsoft Certified: Azure Data Scientist Associate
- IBM Data Science Professional Certificate""",
            "metadata": {
                "type": "career_roadmap",
                "role": "data_scientist",
                "experience_level": "entry_level",
                "title": "Data Scientist Entry Level Roadmap"
            }
        }
    ]
    
    # Add career roadmaps to vector database
    for i, roadmap in enumerate(career_roadmaps):
        try:
            doc_id = vector_db.add_knowledge_item(roadmap["text"], roadmap["metadata"])
            print(f"✅ Added career roadmap {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding career roadmap {i+1}: {e}")

def populate_skill_gap_analyses():
    """Add skill gap analyses to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Skill gap analyses for different roles
    skill_gap_analyses = [
        {
            "text": """Skill Gap Analysis: Transitioning from Java Developer to Full Stack Developer
            
Current Skills:
- Java (Expert)
- Spring Framework (Expert)
- SQL (Intermediate)
- HTML/CSS (Beginner)

Required Skills for Full Stack Developer:
- Frontend: JavaScript, React, Vue.js
- Backend: Node.js, Python, Ruby on Rails
- Databases: MongoDB, PostgreSQL
- DevOps: Docker, Kubernetes, AWS

Learning Path:
1. Learn JavaScript fundamentals (2 months)
2. Master React framework (2 months)
3. Practice building full-stack applications (3 months)
4. Learn Node.js for backend development (2 months)

Resources:
- FreeCodeCamp Full Stack Curriculum
- The Odin Project
- Udemy courses with lifetime access""",
            "metadata": {
                "type": "skill_gap_analysis",
                "from_role": "java_developer",
                "to_role": "full_stack_developer",
                "title": "Java Developer to Full Stack Developer Transition"
            }
        },
        {
            "text": """Skill Gap Analysis: Moving from Data Analyst to Machine Learning Engineer
            
Current Skills:
- SQL (Expert)
- Python (Intermediate)
- Statistics (Expert)
- Tableau/Power BI (Expert)
- Excel (Expert)

Required Skills for Machine Learning Engineer:
- Machine Learning Algorithms
- Deep Learning Frameworks (TensorFlow/PyTorch)
- Model Deployment
- Cloud Platforms (AWS/GCP)
- Software Engineering Practices

Learning Path:
1. Master scikit-learn and statistical modeling (2 months)
2. Learn deep learning with TensorFlow (3 months)
3. Practice with Kaggle competitions (2 months)
4. Learn model deployment and MLOps (2 months)

Resources:
- Andrew Ng's Machine Learning Course
- Fast.ai Practical Deep Learning
- Google Machine Learning Crash Course""",
            "metadata": {
                "type": "skill_gap_analysis",
                "from_role": "data_analyst",
                "to_role": "ml_engineer",
                "title": "Data Analyst to ML Engineer Career Transition"
            }
        }
    ]
    
    # Add skill gap analyses to vector database
    for i, analysis in enumerate(skill_gap_analyses):
        try:
            doc_id = vector_db.add_knowledge_item(analysis["text"], analysis["metadata"])
            print(f"✅ Added skill gap analysis {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding skill gap analysis {i+1}: {e}")

def populate_project_ideas():
    """Add project ideas for different skills and roles to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Project ideas for different skills and roles
    project_ideas = [
        # Software Development Projects
        {
            "text": """Project Idea: Personal Portfolio Website
Description: Create a responsive website to showcase your skills, projects, and experience
Technologies: HTML, CSS, JavaScript, React
Difficulty: Beginner
Estimated Hours: 20
Learning Outcomes: Frontend development, responsive design, UI/UX principles
This project helps beginners demonstrate their frontend skills and create a professional online presence.""",
            "metadata": {
                "type": "project_idea",
                "category": "web_development",
                "skills": ["HTML", "CSS", "JavaScript", "React"],
                "difficulty": "Beginner",
                "estimated_hours": 20,
                "title": "Personal Portfolio Website",
                "description": "Create a responsive website to showcase your skills and projects"
            }
        },
        {
            "text": """Project Idea: E-commerce Landing Page
Description: Design and build a responsive e-commerce landing page with product listings and a shopping cart
Technologies: HTML, CSS, JavaScript, Bootstrap, React
Difficulty: Beginner
Estimated Hours: 30
Learning Outcomes: Frontend frameworks, responsive design, state management
This project helps beginners practice building complex UI components and managing state in web applications.""",
            "metadata": {
                "type": "project_idea",
                "category": "web_development",
                "skills": ["HTML", "CSS", "JavaScript", "Bootstrap", "React"],
                "difficulty": "Beginner",
                "estimated_hours": 30,
                "title": "E-commerce Landing Page",
                "description": "Design and build a responsive e-commerce landing page"
            }
        },
        {
            "text": """Project Idea: Blog Platform
Description: Create a full-stack blog platform with user authentication, post creation, and comment functionality
Technologies: Node.js, Express, MongoDB, JWT, React
Difficulty: Intermediate
Estimated Hours: 40
Learning Outcomes: Full-stack development, database design, authentication, REST APIs
This project helps intermediate developers practice building complete web applications with user management.""",
            "metadata": {
                "type": "project_idea",
                "category": "web_development",
                "skills": ["Node.js", "Express", "MongoDB", "JWT", "React"],
                "difficulty": "Intermediate",
                "estimated_hours": 40,
                "title": "Blog Platform",
                "description": "Create a full-stack blog platform with user authentication"
            }
        },
        {
            "text": """Project Idea: Social Media Dashboard
Description: Create a responsive dashboard with real-time data visualization and social media integration
Technologies: React, Redux, Chart.js, Firebase, REST APIs
Difficulty: Intermediate
Estimated Hours: 50
Learning Outcomes: State management, data visualization, real-time updates, API integration
This project helps intermediate developers practice building complex frontend applications with real-time data.""",
            "metadata": {
                "type": "project_idea",
                "category": "web_development",
                "skills": ["React", "Redux", "Chart.js", "Firebase"],
                "difficulty": "Intermediate",
                "estimated_hours": 50,
                "title": "Social Media Dashboard",
                "description": "Create a responsive dashboard with real-time data visualization"
            }
        },
        {
            "text": """Project Idea: Microservices Architecture
Description: Design and implement a microservices architecture with Docker and Kubernetes
Technologies: Docker, Kubernetes, Node.js, MongoDB, REST APIs
Difficulty: Advanced
Estimated Hours: 60
Learning Outcomes: Containerization, orchestration, distributed systems, DevOps practices
This project helps advanced developers practice building and deploying scalable distributed systems.""",
            "metadata": {
                "type": "project_idea",
                "category": "devops",
                "skills": ["Docker", "Kubernetes", "Node.js", "MongoDB"],
                "difficulty": "Advanced",
                "estimated_hours": 60,
                "title": "Microservices Architecture",
                "description": "Design and implement a microservices architecture with Docker and Kubernetes"
            }
        },
        
        # Data Science Projects
        {
            "text": """Project Idea: Descriptive Statistics Analysis
Description: Analyze a real-world dataset and create comprehensive statistical reports with visualizations
Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn
Difficulty: Beginner
Estimated Hours: 20
Learning Outcomes: Data cleaning, statistical analysis, data visualization
This project helps beginners practice fundamental data analysis skills with real datasets.""",
            "metadata": {
                "type": "project_idea",
                "category": "data_science",
                "skills": ["Python", "Pandas", "NumPy", "Matplotlib"],
                "difficulty": "Beginner",
                "estimated_hours": 20,
                "title": "Descriptive Statistics Analysis",
                "description": "Analyze a real-world dataset and create comprehensive statistical reports"
            }
        },
        {
            "text": """Project Idea: Predictive Model for Housing Prices
Description: Build a regression model to predict housing prices based on property features
Technologies: Python, Scikit-learn, Pandas, NumPy, Matplotlib
Difficulty: Intermediate
Estimated Hours: 40
Learning Outcomes: Machine learning, regression analysis, feature engineering, model evaluation
This project helps intermediate data scientists practice building and evaluating predictive models.""",
            "metadata": {
                "type": "project_idea",
                "category": "data_science",
                "skills": ["Python", "Scikit-learn", "Pandas", "NumPy"],
                "difficulty": "Intermediate",
                "estimated_hours": 40,
                "title": "Predictive Model for Housing Prices",
                "description": "Build a regression model to predict housing prices"
            }
        },
        {
            "text": """Project Idea: Image Classification System
Description: Build a deep learning model to classify images from a dataset using convolutional neural networks
Technologies: Python, TensorFlow, CNN, OpenCV
Difficulty: Intermediate
Estimated Hours: 50
Learning Outcomes: Deep learning, computer vision, neural networks, model training
This project helps intermediate data scientists practice building computer vision applications.""",
            "metadata": {
                "type": "project_idea",
                "category": "data_science",
                "skills": ["Python", "TensorFlow", "CNN", "OpenCV"],
                "difficulty": "Intermediate",
                "estimated_hours": 50,
                "title": "Image Classification System",
                "description": "Build a deep learning model to classify images from a dataset"
            }
        },
        {
            "text": """Project Idea: Real-time Data Pipeline
Description: Build a real-time data processing pipeline using Apache Spark and Kafka
Technologies: Apache Spark, Kafka, Python, SQL
Difficulty: Advanced
Estimated Hours: 60
Learning Outcomes: Big data processing, stream processing, distributed computing, data engineering
This project helps advanced data scientists practice building scalable data processing systems.""",
            "metadata": {
                "type": "project_idea",
                "category": "data_engineering",
                "skills": ["Apache Spark", "Kafka", "Python", "SQL"],
                "difficulty": "Advanced",
                "estimated_hours": 60,
                "title": "Real-time Data Pipeline",
                "description": "Build a real-time data processing pipeline using Spark"
            }
        },
        
        # Additional Project Ideas
        {
            "text": """Project Idea: Chatbot Application
Description: Create a chatbot using natural language processing techniques to answer user queries
Technologies: Python, NLTK, TensorFlow, Flask
Difficulty: Advanced
Estimated Hours: 70
Learning Outcomes: Natural language processing, deep learning, API development, user interaction
This project helps advanced developers practice building conversational AI applications.""",
            "metadata": {
                "type": "project_idea",
                "category": "ai",
                "skills": ["Python", "NLTK", "TensorFlow", "Flask"],
                "difficulty": "Advanced",
                "estimated_hours": 70,
                "title": "Chatbot Application",
                "description": "Create a chatbot using NLP techniques"
            }
        },
        {
            "text": """Project Idea: Sales Forecasting System
Description: Build a time series forecasting model for sales prediction with visualization dashboard
Technologies: Python, Statsmodels, Time Series, Plotly, Dash
Difficulty: Intermediate
Estimated Hours: 50
Learning Outcomes: Time series analysis, forecasting, data visualization, dashboard development
This project helps intermediate data scientists practice building forecasting models for business applications.""",
            "metadata": {
                "type": "project_idea",
                "category": "data_science",
                "skills": ["Python", "Statsmodels", "Time Series", "Plotly"],
                "difficulty": "Intermediate",
                "estimated_hours": 50,
                "title": "Sales Forecasting System",
                "description": "Build a time series forecasting model for sales prediction"
            }
        }
    ]
    
    # Add project ideas to vector database
    for i, project in enumerate(project_ideas):
        try:
            doc_id = vector_db.add_knowledge_item(project["text"], project["metadata"])
            print(f"✅ Added project idea {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding project idea {i+1}: {e}")

def populate_certification_resources():
    """Add certification resources and courses to vector database"""
    if not VectorDBManager:
        return
    
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    
    # Certification resources and courses
    certification_resources = [
        # Cloud Certifications
        {
            "text": """AWS Certified Solutions Architect - Associate
Provider: Amazon Web Services
Description: Validate your ability to design and deploy scalable, highly available, and fault-tolerant systems on AWS
Duration: 130 minutes
Cost: $150
Prerequisites: None
Recommended Experience: 1+ years of hands-on experience with AWS
Learning Resources:
- AWS Official Training: AWS Technical Essentials, Architecting on AWS
- Free Resources: AWS Free Tier, AWS Documentation
- Paid Resources: A Cloud Guru, Linux Academy courses
Preparation Time: 2-3 months of study
Difficulty: Intermediate
Skills Validated: 
- Design resilient architectures
- Define performant architectures
- Specify secure applications and architectures
- Design cost-optimized architectures
- Define operationally-excellent architectures""",
            "metadata": {
                "type": "certification",
                "provider": "AWS",
                "category": "cloud",
                "level": "associate",
                "title": "AWS Certified Solutions Architect - Associate",
                "description": "Validate your ability to design and deploy scalable, highly available, and fault-tolerant systems on AWS"
            }
        },
        {
            "text": """Google Professional Cloud Developer
Provider: Google Cloud
Description: Demonstrate your proficiency in designing, building, and managing applications on Google Cloud Platform
Duration: 120 minutes
Cost: $200
Prerequisites: None
Recommended Experience: 2+ years of experience with Google Cloud
Learning Resources:
- Google Cloud Official Training: Developing Applications with Google Cloud
- Free Resources: Google Cloud Free Tier, Qwiklabs
- Paid Resources: Coursera, A Cloud Guru courses
Preparation Time: 2-3 months of study
Difficulty: Professional
Skills Validated:
- Designing highly scalable and available applications
- Building and testing applications
- Deploying applications
- Integrating Google Cloud services
- Managing application performance monitoring""",
            "metadata": {
                "type": "certification",
                "provider": "Google Cloud",
                "category": "cloud",
                "level": "professional",
                "title": "Google Professional Cloud Developer",
                "description": "Demonstrate your proficiency in designing, building, and managing applications on Google Cloud Platform"
            }
        },
        
        # Data Science Certifications
        {
            "text": """Google Professional Data Engineer
Provider: Google Cloud
Description: Prove your expertise in designing and building data processing systems on Google Cloud
Duration: 120 minutes
Cost: $200
Prerequisites: None
Recommended Experience: 2+ years of experience with Google Cloud
Learning Resources:
- Google Cloud Official Training: Data Engineering on Google Cloud
- Free Resources: Google Cloud Free Tier, Qwiklabs
- Paid Resources: Coursera, A Cloud Guru courses
Preparation Time: 2-3 months of study
Difficulty: Professional
Skills Validated:
- Designing and archiving data processing systems
- Building and operationalizing storage systems and databases
- Ingesting and transforming data
- Configuring and managing data pipelines
- Analyzing and optimizing technical and business processes""",
            "metadata": {
                "type": "certification",
                "provider": "Google Cloud",
                "category": "data_science",
                "level": "professional",
                "title": "Google Professional Data Engineer",
                "description": "Prove your expertise in designing and building data processing systems on Google Cloud"
            }
        },
        
        # Software Development Certifications
        {
            "text": """Microsoft Certified: Azure Developer Associate
Provider: Microsoft
Description: Validate your expertise in designing, building, testing, and maintaining cloud applications and services on Microsoft Azure
Duration: 120 minutes
Cost: $165
Prerequisites: None
Recommended Experience: 1+ years of experience with Azure
Learning Resources:
- Microsoft Learn: Azure Developer modules
- Free Resources: Azure Free Account, Microsoft Learn sandbox
- Paid Resources: Pluralsight, Udemy courses
Preparation Time: 2-3 months of study
Difficulty: Associate
Skills Validated:
- Develop Azure compute solutions
- Develop for Azure storage
- Implement Azure security
- Monitor, troubleshoot, and optimize Azure solutions
- Connect to and consume Azure services and third-party services""",
            "metadata": {
                "type": "certification",
                "provider": "Microsoft",
                "category": "software_development",
                "level": "associate",
                "title": "Microsoft Certified: Azure Developer Associate",
                "description": "Validate your expertise in designing, building, testing, and maintaining cloud applications and services on Microsoft Azure"
            }
        }
    ]
    
    # Add certification resources to vector database
    for i, cert in enumerate(certification_resources):
        try:
            doc_id = vector_db.add_knowledge_item(cert["text"], cert["metadata"])
            print(f"✅ Added certification resource {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding certification resource {i+1}: {e}")

def populate_interview_questions():
    """Seed curated technical and HR interview questions into the vector DB."""
    if not VectorDBManager:
        return
    vector_db = VectorDBManager()
    if not vector_db.is_available():
        print("❌ Vector database not available")
        return
    try:
        count = vector_db.seed_interview_questions()
        print(f"✅ Seeded {count} interview questions into vector DB")
    except Exception as e:
        print(f"❌ Error seeding interview questions: {e}")

def main():
    """Main function to populate all professional content"""
    print("🚀 Starting vector database population...")
    
    # Populate different types of professional content
    print("\n📁 Populating professional resumes...")
    populate_professional_resumes()
    
    print("\n📄 Populating cover letter templates...")
    populate_cover_letter_templates()
    
    print("\n🗺️ Populating career roadmaps...")
    populate_career_roadmaps()
    
    print("\n🎯 Populating skill gap analyses...")
    populate_skill_gap_analyses()
    
    print("\n💻 Populating project ideas...")
    populate_project_ideas()
    
    print("\n📜 Populating certification resources...")
    populate_certification_resources()
    
    print("\n❓ Seeding interview questions (technical + HR)...")
    populate_interview_questions()
    
    print("\n✅ Vector database population completed!")

if __name__ == "__main__":
    main()