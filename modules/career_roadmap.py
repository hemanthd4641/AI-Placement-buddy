import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import requests
from urllib.parse import quote

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

# Import required components
from utils.model_manager import ModelManager

class CareerRoadmapGenerator:
    """Generates personalized career roadmaps with essential features"""
    
    def __init__(self):
        # Initialize model manager for LLM capabilities
        self.model_manager = ModelManager()
        
        # Set vector database manager to None since RAG should only be used in specific modules
        self.vector_db_manager = None
        
        
        # Define role-specific roadmap templates (fallback)
        self.roadmap_templates = {
            'software_developer': {
                'beginner': {
                    'phases': [
                        {
                            'name': 'Programming Fundamentals (Months 1-4)',
                            'duration': '4 months',
                            'skills': ['Python', 'Java', 'JavaScript', 'Data Structures', 'Algorithms'],
                            'resources': [
                                {
                                    'name': 'Python for Everybody',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/specializations/python',
                                    'duration': '4 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'CS50: Introduction to Computer Science',
                                    'type': 'course',
                                    'platform': 'edX',
                                    'url': 'https://cs50.harvard.edu/x/',
                                    'duration': '12 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Personal Portfolio Website',
                                    'description': 'Create a responsive website to showcase your skills and projects',
                                    'technologies': ['HTML', 'CSS', 'JavaScript'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 20
                                },
                                {
                                    'name': 'To-Do List Application',
                                    'description': 'Build a full-stack to-do list app with database storage',
                                    'technologies': ['Python', 'Flask', 'SQLite'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 25
                                }
                            ]
                        },
                        {
                            'name': 'Web Development Basics (Months 5-8)',
                            'duration': '4 months',
                            'skills': ['HTML/CSS', 'JavaScript', 'Frontend Frameworks', 'Git'],
                            'resources': [
                                {
                                    'name': 'The Web Developer Bootcamp',
                                    'type': 'course',
                                    'platform': 'Udemy',
                                    'url': 'https://www.udemy.com/course/the-web-developer-bootcamp/',
                                    'duration': '8 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Paid (often on sale)'
                                },
                                {
                                    'name': 'freeCodeCamp Responsive Web Design',
                                    'type': 'certification',
                                    'platform': 'freeCodeCamp',
                                    'url': 'https://www.freecodecamp.org/learn/responsive-web-design/',
                                    'duration': '6 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'E-commerce Landing Page',
                                    'description': 'Design and build a responsive e-commerce landing page',
                                    'technologies': ['HTML', 'CSS', 'JavaScript', 'Bootstrap'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 30
                                },
                                {
                                    'name': 'Weather Dashboard',
                                    'description': 'Create a weather dashboard that fetches data from an API',
                                    'technologies': ['JavaScript', 'API', 'CSS'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 20
                                }
                            ]
                        },
                        {
                            'name': 'Backend & Databases (Months 9-12)',
                            'duration': '4 months',
                            'skills': ['Node.js/Python Backend', 'SQL/NoSQL', 'APIs', 'Docker'],
                            'resources': [
                                {
                                    'name': 'The Complete Node.js Developer Course',
                                    'type': 'course',
                                    'platform': 'Udemy',
                                    'url': 'https://www.udemy.com/course/the-complete-nodejs-developer-course-2/',
                                    'duration': '6 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Paid (often on sale)'
                                },
                                {
                                    'name': 'SQL for Data Science',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/learn/sql-for-data-science',
                                    'duration': '4 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free with audit'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Blog Platform',
                                    'description': 'Create a full-stack blog platform with user authentication',
                                    'technologies': ['Node.js', 'Express', 'MongoDB', 'JWT'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 40
                                },
                                {
                                    'name': 'Task Management API',
                                    'description': 'Build a RESTful API for task management with full CRUD operations',
                                    'technologies': ['Python', 'Flask', 'PostgreSQL'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 35
                                }
                            ]
                        }
                    ]
                },
                'intermediate': {
                    'phases': [
                        {
                            'name': 'Advanced Frontend (Months 1-4)',
                            'duration': '4 months',
                            'skills': ['React/Vue', 'State Management', 'Testing', 'TypeScript'],
                            'resources': [
                                {
                                    'name': 'React - The Complete Guide',
                                    'type': 'course',
                                    'platform': 'Udemy',
                                    'url': 'https://www.udemy.com/course/react-the-complete-guide-incl-redux/',
                                    'duration': '6 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Paid (often on sale)'
                                },
                                {
                                    'name': 'Frontend Masters',
                                    'type': 'subscription',
                                    'platform': 'Frontend Masters',
                                    'url': 'https://frontendmasters.com/',
                                    'duration': 'Ongoing',
                                    'difficulty': 'Intermediate-Advanced',
                                    'cost': 'Paid'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Social Media Dashboard',
                                    'description': 'Create a responsive dashboard with real-time data visualization',
                                    'technologies': ['React', 'Redux', 'Chart.js', 'Firebase'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 50
                                },
                                {
                                    'name': 'E-commerce Platform',
                                    'description': 'Build a full-featured e-commerce platform with payment integration',
                                    'technologies': ['React', 'Node.js', 'MongoDB', 'Stripe'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 80
                                }
                            ]
                        },
                        {
                            'name': 'DevOps & Cloud (Months 5-8)',
                            'duration': '4 months',
                            'skills': ['Docker', 'AWS/Azure', 'CI/CD', 'Kubernetes'],
                            'resources': [
                                {
                                    'name': 'Docker Mastery',
                                    'type': 'course',
                                    'platform': 'Udemy',
                                    'url': 'https://www.udemy.com/course/docker-mastery/',
                                    'duration': '4 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Paid (often on sale)'
                                },
                                {
                                    'name': 'AWS Cloud Practitioner',
                                    'type': 'certification',
                                    'platform': 'AWS',
                                    'url': 'https://aws.amazon.com/certification/certified-cloud-practitioner/',
                                    'duration': '6 weeks',
                                    'difficulty': 'Beginner-Intermediate',
                                    'cost': 'Paid exam'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Microservices Architecture',
                                    'description': 'Design and implement a microservices architecture with Docker and Kubernetes',
                                    'technologies': ['Docker', 'Kubernetes', 'Node.js', 'MongoDB'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 60
                                },
                                {
                                    'name': 'CI/CD Pipeline',
                                    'description': 'Set up a complete CI/CD pipeline with automated testing and deployment',
                                    'technologies': ['Jenkins', 'GitHub Actions', 'Docker', 'AWS'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 40
                                }
                            ]
                        },
                        {
                            'name': 'System Design & Architecture (Months 9-12)',
                            'duration': '4 months',
                            'skills': ['System Design', 'Microservices', 'Security', 'Performance Optimization'],
                            'resources': [
                                {
                                    'name': 'Grokking the System Design Interview',
                                    'type': 'course',
                                    'platform': 'Educative',
                                    'url': 'https://www.educative.io/courses/grokking-the-system-design-interview',
                                    'duration': '8 weeks',
                                    'difficulty': 'Advanced',
                                    'cost': 'Paid'
                                },
                                {
                                    'name': 'Designing Data-Intensive Applications',
                                    'type': 'book',
                                    'platform': 'O\'Reilly',
                                    'url': 'https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/',
                                    'duration': '12 weeks',
                                    'difficulty': 'Advanced',
                                    'cost': 'Paid'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Scalable Chat Application',
                                    'description': 'Design and build a scalable real-time chat application',
                                    'technologies': ['WebSocket', 'Redis', 'Node.js', 'MongoDB'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 70
                                },
                                {
                                    'name': 'Video Streaming Platform',
                                    'description': 'Create a video streaming platform with content delivery network',
                                    'technologies': ['AWS S3', 'CloudFront', 'Node.js', 'React'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 90
                                }
                            ]
                        }
                    ]
                }
            },
            'data_scientist': {
                'beginner': {
                    'phases': [
                        {
                            'name': 'Mathematics & Statistics (Months 1-3)',
                            'duration': '3 months',
                            'skills': ['Linear Algebra', 'Calculus', 'Probability', 'Statistics'],
                            'resources': [
                                {
                                    'name': 'Mathematics for Machine Learning',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/specializations/mathematics-machine-learning',
                                    'duration': '6 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Statistics with Python',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/specializations/statistics-with-python',
                                    'duration': '6 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free with audit'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Descriptive Statistics Analysis',
                                    'description': 'Analyze a real-world dataset and create comprehensive statistical reports',
                                    'technologies': ['Python', 'Pandas', 'Matplotlib'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 20
                                },
                                {
                                    'name': 'A/B Testing Simulator',
                                    'description': 'Create a simulator to understand A/B testing concepts and statistical significance',
                                    'technologies': ['Python', 'SciPy', 'Statsmodels'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 25
                                }
                            ]
                        },
                        {
                            'name': 'Programming & Data Analysis (Months 4-7)',
                            'duration': '4 months',
                            'skills': ['Python', 'Pandas', 'NumPy', 'Data Cleaning', 'Data Visualization'],
                            'resources': [
                                {
                                    'name': 'Python for Data Science and AI',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/learn/python-for-applied-data-science-ai',
                                    'duration': '4 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Pandas Course',
                                    'type': 'course',
                                    'platform': 'Kaggle',
                                    'url': 'https://www.kaggle.com/learn/pandas',
                                    'duration': '3 weeks',
                                    'difficulty': 'Beginner',
                                    'cost': 'Free'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Data Cleaning and Analysis',
                                    'description': 'Clean and analyze a messy real-world dataset',
                                    'technologies': ['Python', 'Pandas', 'NumPy'],
                                    'difficulty': 'Beginner',
                                    'estimated_hours': 30
                                },
                                {
                                    'name': 'Interactive Dashboard',
                                    'description': 'Create an interactive dashboard to visualize key metrics from a dataset',
                                    'technologies': ['Python', 'Plotly', 'Dash'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 35
                                }
                            ]
                        },
                        {
                            'name': 'Machine Learning Fundamentals (Months 8-12)',
                            'duration': '5 months',
                            'skills': ['Scikit-learn', 'Supervised Learning', 'Model Evaluation', 'Feature Engineering'],
                            'resources': [
                                {
                                    'name': 'Machine Learning',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/learn/machine-learning',
                                    'duration': '11 weeks',
                                    'difficulty': 'Beginner-Intermediate',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Applied Data Science with Python',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/specializations/data-science-python',
                                    'duration': '8 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Free with audit'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Predictive Model for Housing Prices',
                                    'description': 'Build a regression model to predict housing prices',
                                    'technologies': ['Python', 'Scikit-learn', 'Pandas'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 40
                                },
                                {
                                    'name': 'Customer Segmentation',
                                    'description': 'Implement clustering algorithms to segment customers',
                                    'technologies': ['Python', 'Scikit-learn', 'K-Means'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 35
                                }
                            ]
                        }
                    ]
                },
                'intermediate': {
                    'phases': [
                        {
                            'name': 'Deep Learning & Neural Networks (Months 1-4)',
                            'duration': '4 months',
                            'skills': ['TensorFlow/PyTorch', 'Neural Networks', 'Computer Vision', 'NLP'],
                            'resources': [
                                {
                                    'name': 'Deep Learning Specialization',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/specializations/deep-learning',
                                    'duration': '12 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Practical Deep Learning for Coders',
                                    'type': 'course',
                                    'platform': 'Fast.ai',
                                    'url': 'https://course.fast.ai/',
                                    'duration': '7 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Free'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Image Classification System',
                                    'description': 'Build a deep learning model to classify images from a dataset',
                                    'technologies': ['Python', 'TensorFlow', 'CNN'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 50
                                },
                                {
                                    'name': 'Sentiment Analysis Tool',
                                    'description': 'Create a tool that analyzes sentiment in text using NLP techniques',
                                    'technologies': ['Python', 'PyTorch', 'NLP'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 45
                                }
                            ]
                        },
                        {
                            'name': 'Big Data & Cloud Platforms (Months 5-8)',
                            'duration': '4 months',
                            'skills': ['Spark', 'Hadoop', 'Cloud ML Services', 'Databricks'],
                            'resources': [
                                {
                                    'name': 'Google Cloud Big Data and Machine Learning Fundamentals',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/learn/gcp-big-data-ml-fundamentals',
                                    'duration': '4 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Databricks Lakehouse Fundamentals',
                                    'type': 'course',
                                    'platform': 'Databricks',
                                    'url': 'https://www.databricks.com/learn/training/lakehouse-fundamentals',
                                    'duration': '3 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Free'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Real-time Data Pipeline',
                                    'description': 'Build a real-time data processing pipeline using Spark',
                                    'technologies': ['Apache Spark', 'Kafka', 'Python'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 60
                                },
                                {
                                    'name': 'Cloud ML Model Deployment',
                                    'description': 'Deploy a machine learning model on a cloud platform',
                                    'technologies': ['AWS SageMaker', 'Python', 'Scikit-learn'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 40
                                }
                            ]
                        },
                        {
                            'name': 'Advanced Topics & Specialization (Months 9-12)',
                            'duration': '4 months',
                            'skills': ['NLP', 'Time Series Analysis', 'A/B Testing', 'MLOps'],
                            'resources': [
                                {
                                    'name': 'Natural Language Processing',
                                    'type': 'course',
                                    'platform': 'Coursera',
                                    'url': 'https://www.coursera.org/learn/language-processing',
                                    'duration': '4 weeks',
                                    'difficulty': 'Advanced',
                                    'cost': 'Free with audit'
                                },
                                {
                                    'name': 'Time Series Analysis and Forecasting',
                                    'type': 'course',
                                    'platform': 'Udemy',
                                    'url': 'https://www.udemy.com/course/time-series-analysis-and-forecasting/',
                                    'duration': '6 weeks',
                                    'difficulty': 'Intermediate',
                                    'cost': 'Paid (often on sale)'
                                }
                            ],
                            'projects': [
                                {
                                    'name': 'Chatbot Application',
                                    'description': 'Create a chatbot using NLP techniques',
                                    'technologies': ['Python', 'NLTK', 'TensorFlow'],
                                    'difficulty': 'Advanced',
                                    'estimated_hours': 70
                                },
                                {
                                    'name': 'Sales Forecasting System',
                                    'description': 'Build a time series forecasting model for sales prediction',
                                    'technologies': ['Python', 'Statsmodels', 'Time Series'],
                                    'difficulty': 'Intermediate',
                                    'estimated_hours': 50
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
        # Default roadmap for roles not in templates
        self.default_roadmap = {
            'beginner': {
                'phases': [
                    {
                        'name': 'Foundational Skills (Months 1-4)',
                        'duration': '4 months',
                        'skills': ['Industry Basics', 'Core Tools', 'Fundamental Concepts'],
                        'resources': [
                            {
                                'name': 'Industry Fundamentals Course',
                                'type': 'course',
                                'platform': 'Coursera',
                                'url': '#',
                                'duration': '4 weeks',
                                'difficulty': 'Beginner',
                                'cost': 'Variable'
                            }
                        ],
                        'projects': [
                            {
                                'name': 'Foundational Project',
                                'description': 'Create a basic project to demonstrate foundational skills',
                                'technologies': ['Core Tools'],
                                'difficulty': 'Beginner',
                                'estimated_hours': 20
                            }
                        ]
                    },
                    {
                        'name': 'Intermediate Skills (Months 5-8)',
                        'duration': '4 months',
                        'skills': ['Applied Skills', 'Problem Solving', 'Tools Proficiency'],
                        'resources': [
                            {
                                'name': 'Intermediate Course',
                                'type': 'course',
                                'platform': 'edX',
                                'url': '#',
                                'duration': '6 weeks',
                                'difficulty': 'Intermediate',
                                'cost': 'Variable'
                            }
                        ],
                        'projects': [
                            {
                                'name': 'Portfolio Project',
                                'description': 'Create a portfolio project to showcase intermediate skills',
                                'technologies': ['Applied Skills'],
                                'difficulty': 'Intermediate',
                                'estimated_hours': 30
                            }
                        ]
                    },
                    {
                        'name': 'Advanced Application (Months 9-12)',
                        'duration': '4 months',
                        'skills': ['Specialization', 'Leadership', 'Innovation'],
                        'resources': [
                            {
                                'name': 'Advanced Course',
                                'type': 'course',
                                'platform': 'Udemy',
                                'url': '#',
                                'duration': '8 weeks',
                                'difficulty': 'Advanced',
                                'cost': 'Variable'
                            }
                        ],
                        'projects': [
                            {
                                'name': 'Advanced Project',
                                'description': 'Lead an advanced project to demonstrate specialization',
                                'technologies': ['Specialization'],
                                'difficulty': 'Advanced',
                                'estimated_hours': 40
                            }
                        ]
                    }
                ]
            }
        }

    def _get_template_text(self, template_for: str) -> str:
        """Fetch a template (stub method since RAG is not used in this module)."""
        return ""
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using NER and keyword matching"""
        skills = []
        
        # Use spaCy NER if available
        if self.model_manager.models.get('ner'):
            try:
                doc = self.model_manager.models['ner'](text)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'TECH']:
                        skills.append(ent.text)
            except Exception as e:
                print(f"Error extracting skills with NER: {e}")
        
        # Use comprehensive keyword matching
        text_lower = text.lower()
        
        # Role-specific skill mappings
        role_skill_mapping = {
            'backend': ['Python', 'Java', 'Node.js', 'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Docker', 'Kubernetes', 'AWS', 'REST APIs', 'GraphQL', 'Microservices', 'Spring Boot', 'Express.js', 'Django', 'Flask', 'Nginx', 'Apache', 'Linux', 'Git', 'CI/CD', 'Jenkins', 'Testing', 'Security'],
            'frontend': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue.js', 'Angular', 'TypeScript', 'SASS/SCSS', 'Webpack', 'Responsive Design', 'Accessibility', 'Performance Optimization', 'Testing', 'State Management', 'REST APIs', 'Git'],
            'fullstack': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue.js', 'Angular', 'Python', 'Java', 'Node.js', 'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'REST APIs', 'GraphQL', 'Docker', 'Git', 'Testing', 'Security'],
            'data scientist': ['Python', 'R', 'SQL', 'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Machine Learning', 'Deep Learning', 'Statistics', 'Data Visualization', 'Tableau', 'Power BI', 'Jupyter', 'Git'],
            'data engineer': ['Python', 'Java', 'SQL', 'Spark', 'Hadoop', 'Kafka', 'Airflow', 'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes', 'Linux', 'Git', 'CI/CD'],
            'devops': ['Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure', 'Linux', 'Bash', 'Python', 'CI/CD', 'Jenkins', 'GitLab', 'Ansible', 'Terraform', 'Monitoring', 'Logging'],
            'mobile': ['Java', 'Kotlin', 'Swift', 'React Native', 'Flutter', 'iOS', 'Android', 'Firebase', 'REST APIs', 'Git', 'Testing'],
            'security': ['Network Security', 'Encryption', 'Firewalls', 'Penetration Testing', 'Compliance', 'Risk Assessment', 'Incident Response', 'Linux', 'Python', 'SIEM', 'IDS/IPS'],
            'cloud': ['AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes', 'Linux', 'Python', 'Terraform', 'CI/CD', 'Networking', 'Security'],
            'ai': ['Python', 'TensorFlow', 'PyTorch', 'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'Statistics', 'Mathematics', 'Git'],
            'ml': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Machine Learning', 'Deep Learning', 'Statistics', 'Mathematics', 'Data Visualization', 'Git']
        }
        
        # Check for role-specific skills first
        for role_keyword, role_skills in role_skill_mapping.items():
            if role_keyword in text_lower:
                skills.extend(role_skills)
                break
        
        # General skill keywords if no role-specific skills found
        if not skills:
            skill_keywords = [
                'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'aws', 'azure', 'gcp',
                'html', 'css', 'docker', 'kubernetes', 'tensorflow', 'pytorch', 'pandas', 'numpy',
                'scikit-learn', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'mongodb', 'postgresql',
                'mysql', 'redis', 'elasticsearch', 'git', 'jenkins', 'ansible', 'terraform',
                'machine learning', 'deep learning', 'data science', 'artificial intelligence',
                'natural language processing', 'computer vision', 'data analysis', 'data visualization',
                'web development', 'mobile development', 'api development', 'rest', 'graphql',
                'devops', 'cloud computing', 'microservices', 'system design', 'agile', 'scrum'
            ]
            
            for skill in skill_keywords:
                if skill in text_lower:
                    skills.append(skill.title())
        
        return list(set(skills))  # Remove duplicates

    def _role_to_categories(self, target_role: str) -> List[str]:
        """Map a free-text role to one or more project categories used in the knowledge base."""
        r = target_role.lower()
        cats = []
        if any(k in r for k in ["frontend", "react", "vue", "angular", "web"]):
            cats.append("web_development")
        if any(k in r for k in ["backend", "api", "server", "microservices"]):
            cats.append("web_development")
        if any(k in r for k in ["data", "ml", "machine learning", "ai", "analytics", "scientist"]):
            cats.append("data_science")
        if any(k in r for k in ["data engineer", "pipeline", "spark", "kafka"]):
            cats.append("data_engineering")
        if any(k in r for k in ["devops", "sre", "platform", "kubernetes", "docker"]):
            cats.append("devops")
        if any(k in r for k in ["ai", "nlp", "cv", "vision", "llm", "chatbot"]):
            cats.append("ai")
        # Fallback
        if not cats:
            cats.append("web_development")
        return list(dict.fromkeys(cats))

    def _generate_role_specific_projects(self, target_role: str, experience_level: str, phase_skills: List[str], count: int = 3) -> List[Dict[str, Any]]:
        """Generate role-specific project ideas using RAG + LLM.

        Contract:
        - Inputs: role string, experience level, up to ~5 phase skills, desired count
        - Output: list[project] with keys: name, description, technologies[list], difficulty, estimated_hours
        - Fallbacks: vector retrieval only, then generic if models unavailable
        """
        projects: List[Dict[str, Any]] = []

        # Step 1: No RAG retrieval since it's not used in this module
        retrieved_context = []

        # If no LLM, try to adapt retrieved items or fallback generics
        if not self.model_manager.models.get('lm'):
            if retrieved_context:
                for md in retrieved_context[:count]:
                    projects.append({
                        'name': md.get('title', 'Project'),
                        'description': md.get('description', f"A project related to {target_role}"),
                        'technologies': md.get('technologies') or md.get('skills') or phase_skills[:3] or [target_role],
                        'difficulty': md.get('difficulty', experience_level.title()),
                        'estimated_hours': md.get('estimated_hours', 30)
                    })
                return projects
            # Fallback to generic
            return self._get_generic_projects(phase_skills[:3])

        # Step 2: Ask LLM to synthesize role-specific projects using retrieved context
        try:
            tmpl = self._get_template_text('project_ideas')
            context_lines = []
            for md in retrieved_context:
                context_lines.append(f"- {md.get('title','Idea')}: {md.get('description','')} | tech: {', '.join(md.get('technologies') or md.get('skills') or [])} | diff: {md.get('difficulty','')}")
            context_block = "\n".join(context_lines) if context_lines else "(no prior ideas)"
            prompt = f"""
{tmpl}
You are generating role-specific project ideas.
Role: {target_role}
Experience level: {experience_level}
Key skills for this phase: {', '.join(phase_skills[:6]) if phase_skills else 'N/A'}
Known example ideas (use as inspiration, avoid verbatim duplication):
{context_block}

Task: Propose {count} distinct portfolio-ready project ideas tailored to the role and skills. Ensure:
- Each idea is specific to the role and practical to build
- Use realistic, modern stacks and align with listed skills
- Provide increasing difficulty across items
- Include estimated hours appropriate for {experience_level}

Return ONLY valid JSON as:
{{
  "projects": [
    {{
      "name": "",
      "description": "",
      "technologies": [""],
      "difficulty": "Beginner|Intermediate|Advanced",
      "estimated_hours": 0
    }}
  ]
}}
"""
            resp = self.model_manager.generate_text(prompt, max_length=900)
            json_start = resp.find('{')
            json_end = resp.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(resp[json_start:json_end])
                for p in data.get('projects', [])[:count]:
                    # Normalize and append
                    projects.append({
                        'name': p.get('name', 'Project').strip(),
                        'description': p.get('description', '').strip() or f"A project for {target_role}",
                        'technologies': [t for t in (p.get('technologies') or []) if isinstance(t, str)] or (phase_skills[:3] or [target_role]),
                        'difficulty': p.get('difficulty', experience_level.title()),
                        'estimated_hours': int(p.get('estimated_hours', 30))
                    })
        except Exception as e:
            print(f"Error generating role-specific projects: {e}")

        # If still not enough, top up from retrieved context
        if len(projects) < count and retrieved_context:
            for md in retrieved_context:
                if len(projects) >= count:
                    break
                projects.append({
                    'name': md.get('title', 'Project'),
                    'description': md.get('description', f"A project related to {target_role}"),
                    'technologies': md.get('technologies') or md.get('skills') or phase_skills[:3] or [target_role],
                    'difficulty': md.get('difficulty', experience_level.title()),
                    'estimated_hours': md.get('estimated_hours', 30)
                })

        # Final fallback if empty
        if not projects:
            projects = self._get_generic_projects(phase_skills[:3])

        return projects[:count]
    
    def _fetch_youtube_tutorials(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Fetch YouTube tutorials for a given query"""
        try:
            # Since we can't use the YouTube API without an API key, we'll create placeholder links
            # In a real implementation, you would use the YouTube Data API
            search_query = query.replace(' ', '+')
            tutorials = []
            
            # Create placeholder YouTube tutorial entries
            tutorials.append({
                'name': f'{query} - Complete Tutorial',
                'type': 'video',
                'platform': 'YouTube',
                'url': f'https://www.youtube.com/results?search_query={search_query}',
                'duration': 'Variable',
                'difficulty': 'Beginner',
                'cost': 'Free'
            })
            
            tutorials.append({
                'name': f'{query} - Crash Course',
                'type': 'video',
                'platform': 'YouTube',
                'url': f'https://www.youtube.com/results?search_query={search_query}+crash+course',
                'duration': '1-2 hours',
                'difficulty': 'Beginner',
                'cost': 'Free'
            })
            
            tutorials.append({
                'name': f'{query} - Advanced Guide',
                'type': 'video',
                'platform': 'YouTube',
                'url': f'https://www.youtube.com/results?search_query={search_query}+advanced',
                'duration': '2-4 hours',
                'difficulty': 'Intermediate',
                'cost': 'Free'
            })
            
            return tutorials[:max_results]
        except Exception as e:
            print(f"Error fetching YouTube tutorials: {e}")
            return []
    
    def _fetch_free_resources(self, skill: str) -> List[Dict[str, str]]:
        """Fetch free learning resources for a skill"""
        resources = []
        
        try:
            # Create placeholder free resource entries
            skill_query = skill.replace(' ', '+')
            
            # FreeCodeCamp resources
            resources.append({
                'name': f'{skill} - FreeCodeCamp Curriculum',
                'type': 'course',
                'platform': 'FreeCodeCamp',
                'url': f'https://www.freecodecamp.org/news/search?query={skill_query}',
                'duration': 'Self-paced',
                'difficulty': 'Beginner',
                'cost': 'Free'
            })
            
            # Khan Academy resources
            resources.append({
                'name': f'{skill} - Khan Academy',
                'type': 'course',
                'platform': 'Khan Academy',
                'url': f'https://www.khanacademy.org/search?page_search_query={skill_query}',
                'duration': 'Self-paced',
                'difficulty': 'Beginner',
                'cost': 'Free'
            })
            
            # Coursera audit resources
            resources.append({
                'name': f'{skill} - Coursera (Audit)',
                'type': 'course',
                'platform': 'Coursera',
                'url': f'https://www.coursera.org/search?query={skill_query}',
                'duration': '4-12 weeks',
                'difficulty': 'Beginner',
                'cost': 'Free with audit'
            })
            
            # YouTube tutorials
            youtube_tutorials = self._fetch_youtube_tutorials(skill)
            resources.extend(youtube_tutorials)
            
        except Exception as e:
            print(f"Error fetching free resources: {e}")
        
        return resources
    
    def _generate_dynamic_phase(self, phase_name: str, skills: List[str], duration: str) -> Dict[str, Any]:
        """Generate a dynamic phase using LLM"""
        if not self.model_manager.models.get('lm'):
            # Fallback to template-based generation
            return {
                'name': phase_name,
                'duration': duration,
                'skills': skills[:5],  # Limit to 5 skills
                'resources': self._get_generic_resources(skills),
                'projects': self._get_generic_projects(skills[:3])
            }
        
        try:
            # Create prompt for LLM to generate phase details
            prompt = f"""
            Generate a detailed learning phase for a career roadmap with the following parameters:
            
            Phase Name: {phase_name}
            Duration: {duration}
            Skills to Cover: {', '.join(skills)}
            
            Please provide:
            1. A brief description of what will be learned in this phase
            2. 3-5 specific skills that will be developed
            3. 2-3 recommended learning resources (courses, books, tutorials)
            4. 2-3 project ideas with descriptions, technologies, difficulty level, and estimated hours
            
            Format the response as JSON with the following structure:
            {{
                "name": "Phase Name",
                "duration": "Duration",
                "skills": ["Skill 1", "Skill 2", ...],
                "resources": [
                    {{
                        "name": "Resource Name",
                        "type": "course/book/tutorial/etc",
                        "platform": "Platform Name",
                        "link": "URL if available",
                        "duration": "Estimated time",
                        "difficulty": "Beginner/Intermediate/Advanced",
                        "cost": "Free/Paid/Variable"
                    }}
                ],
                "projects": [
                    {{
                        "name": "Project Name",
                        "description": "Project description",
                        "technologies": ["Tech 1", "Tech 2", ...],
                        "difficulty": "Beginner/Intermediate/Advanced",
                        "estimated_hours": 20
                    }}
                ]
            }}
            """
            
            # Generate response using LLM
            response = self.model_manager.generate_text(prompt, max_length=800)
            
            # Try to parse JSON from response
            import json
            try:
                # Extract JSON from response if it contains other text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    phase_data = json.loads(json_str)
                    return phase_data
            except json.JSONDecodeError:
                pass  # Fall back to template-based approach
                
        except Exception as e:
            print(f"Error generating dynamic phase with LLM: {e}")
        
        # Fallback to template-based generation
        return {
            'name': phase_name,
            'duration': duration,
            'skills': skills[:5],  # Limit to 5 skills
            'resources': self._get_generic_resources(skills),
            'projects': self._get_generic_projects(skills[:3])
        }
    
    def _get_generic_resources(self, skills: List[str]) -> List[Dict[str, str]]:
        """Get generic learning resources for skills without using vector database"""
        resources = []
        
        # Since RAG is not used in this module, directly use fallback resources
        for skill in skills[:3]:
            # Get free resources for each skill
            free_resources = self._fetch_free_resources(skill)
            resources.extend(free_resources[:2])  # Limit to 2 resources per skill
        
        # If still no resources, provide generic ones
        if not resources:
            for skill in skills[:3]:
                resources.append({
                    'name': f'{skill} Fundamentals Course',
                    'type': 'course',
                    'platform': 'Coursera',
                    'url': '#',
                    'duration': '4-8 weeks',
                    'difficulty': 'Beginner',
                    'cost': 'Free with audit'
                })
        
        return resources
    
    def _get_generic_projects(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Get generic project ideas for skills without using vector database"""
        projects = []
        
        # Since RAG is not used in this module, directly use fallback project ideas
        for i, skill in enumerate(skills[:2]):
            projects.append({
                'name': f'{skill} Practice Project',
                'description': f'A hands-on project to practice {skill} skills',
                'technologies': [skill],
                'difficulty': 'Beginner' if i == 0 else 'Intermediate',
                'estimated_hours': 20 + (i * 10)
            })
        
        return projects
    
    def _search_related_roadmaps(self, target_role: str, experience_level: str) -> List[Dict[str, Any]]:
        """Search for related roadmaps (stub method since RAG is not used in this module)"""
        # Since RAG is not used in this module, return empty list
        return []
    
    def generate_roadmap(self, target_role: str, experience_level: str, timeline: int, primary_goal: str, current_skills: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a personalized career roadmap based on user inputs"""
        # Normalize target role for template matching
        normalized_role = target_role.lower().replace(' ', '_').replace('-', '_')
        
        # Extract skills from target role if not provided
        if not current_skills:
            current_skills = self._extract_skills_from_text(target_role)
        
        # Try to generate dynamic roadmap using LLM and vector database
        if self.model_manager.models.get('lm'):
            try:
                # Search for related roadmaps in vector database
                related_roadmaps = self._search_related_roadmaps(target_role, experience_level)
                
                # Create enhanced prompt for LLM to generate roadmap
                template_instructions = self._get_template_text('career_roadmap')
                prompt = f"""
{template_instructions}
                You are an expert career advisor. Generate a highly personalized career roadmap for the following:
                
                Target Role: {target_role}
                Experience Level: {experience_level}
                Timeline: {timeline} months
                Primary Goal: {primary_goal}
                Current Skills: {', '.join(current_skills) if current_skills else 'Not specified'}
                
                {f'Related roadmaps found: {related_roadmaps[:2]}' if related_roadmaps else 'No related roadmaps found.'}
                
                Requirements for the roadmap:
                1. Create exactly 3-4 learning phases appropriate for the timeline
                2. Each phase should have:
                   - A clear name with time range (e.g., "Phase Name (Months X-Y)")
                   - Duration in months that fits within the total timeline
                   - 5-8 specific skills to learn
                   - 3-5 high-quality learning resources (courses, books, tutorials)
                   - 2-3 project ideas with detailed descriptions, technologies, difficulty level, and estimated hours
                3. Total timeline should match the requested duration
                4. Include links to free learning resources and YouTube tutorials where possible
                5. Ensure progression from beginner to advanced concepts
                6. Tailor content to the user's experience level ({experience_level})
                7. Align with the primary goal ({primary_goal})
                
                Format the response as JSON with the following structure:
                {{
                    "phases": [
                        {{
                            "name": "Phase Name (Months X-Y)",
                            "duration": "Duration in months",
                            "skills": ["Skill 1", "Skill 2", ...],
                            "resources": [
                                {{
                                    "name": "Resource Name",
                                    "type": "course/book/tutorial/etc",
                                    "platform": "Platform Name",
                                    "url": "URL if available",
                                    "duration": "Estimated time",
                                    "difficulty": "Beginner/Intermediate/Advanced",
                                    "cost": "Free/Paid/Variable"
                                }}
                            ],
                            "projects": [
                                {{
                                    "name": "Project Name",
                                    "description": "Project description",
                                    "technologies": ["Tech 1", "Tech 2", ...],
                                    "difficulty": "Beginner/Intermediate/Advanced",
                                    "estimated_hours": 20
                                }}
                            ]
                        }}
                    ]
                }}
                
                Important: Return ONLY valid JSON. Do not include any other text.
                """
                
                # Generate response using LLM
                response = self.model_manager.generate_text(prompt, max_length=2000)
                
                # Try to parse JSON from response
                import json
                try:
                    # Extract JSON from response if it contains other text
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        roadmap_data = json.loads(json_str)
                        
                        # Enhance resources with additional free resources and YouTube tutorials
                        for phase in roadmap_data.get('phases', []):
                            phase_skills = phase.get('skills', [])
                            # Add free resources for each phase
                            free_resources = []
                            for skill in phase_skills[:3]:  # Limit to 3 skills
                                free_resources.extend(self._fetch_free_resources(skill)[:2])  # 2 resources per skill
                            phase['resources'].extend(free_resources)
                            
                            # Regenerate projects with role-specific focus via LLM + RAG
                            phase['projects'] = self._generate_role_specific_projects(
                                target_role=target_role,
                                experience_level=experience_level,
                                phase_skills=phase_skills,
                                count=3
                            )
                        
                        # Adjust timeline if needed
                        roadmap = self._adjust_timeline(roadmap_data, timeline, experience_level)
                        
                        # Add metadata
                        roadmap['metadata'] = {
                            'target_role': target_role,
                            'experience_level': experience_level,
                            'timeline_months': timeline,
                            'primary_goal': primary_goal,
                            'current_skills': current_skills,
                            'generated_date': datetime.now().isoformat(),
                            'source': 'llm_generated'
                        }
                        
                        # Enhance roadmap without vector database context since RAG is not used in this module                        
                        return roadmap
                except json.JSONDecodeError:
                    pass  # Fall back to template-based approach
                    
            except Exception as e:
                print(f"Error generating roadmap with LLM: {e}")
        
        # Fallback to template-based approach
        # Select appropriate template
        if normalized_role in self.roadmap_templates and experience_level in self.roadmap_templates[normalized_role]:
            template = self.roadmap_templates[normalized_role][experience_level]
        else:
            # Create a role-specific default roadmap based on the target role
            template = self._create_role_specific_default(target_role, experience_level)
        
        # Enhance template resources with additional free resources and YouTube tutorials
        enhanced_template = json.loads(json.dumps(template))
        for phase in enhanced_template.get('phases', []):
            phase_skills = phase.get('skills', [])
            # Add free resources for each phase
            free_resources = []
            for skill in phase_skills[:3]:  # Limit to 3 skills
                free_resources.extend(self._fetch_free_resources(skill)[:2])  # 2 resources per skill
            phase['resources'].extend(free_resources)
            
            # Always generate role-specific projects in fallback path
            phase['projects'] = self._generate_role_specific_projects(
                target_role=target_role,
                experience_level=experience_level,
                phase_skills=phase_skills,
                count=2 if experience_level == 'beginner' else 3
            )
        
        # Adjust timeline if needed
        roadmap = self._adjust_timeline(enhanced_template, timeline, experience_level)
        
        # Add metadata
        roadmap['metadata'] = {
            'target_role': target_role,
            'experience_level': experience_level,
            'timeline_months': timeline,
            'primary_goal': primary_goal,
            'current_skills': current_skills or [],
            'generated_date': datetime.now().isoformat(),
            'source': 'template_based'
        }
        
        # Enhance roadmap without vector database context since RAG is not used in this module        
        return roadmap
    
    def _adjust_timeline(self, template: Dict, timeline: int, experience_level: str) -> Dict:
        """Adjust the roadmap timeline based on user preference"""
        # Create a copy of the template to modify
        adjusted_roadmap = json.loads(json.dumps(template))
        
        # Get the original total duration in months
        original_phases = len(adjusted_roadmap['phases'])
        original_duration_per_phase = 12 // original_phases if original_phases > 0 else 3
        
        # Adjust phase durations proportionally
        if original_phases > 0:
            adjustment_factor = timeline / (original_phases * original_duration_per_phase)
            
            for i, phase in enumerate(adjusted_roadmap['phases']):
                # Calculate new duration for this phase
                original_duration = int(phase['duration'].split()[0])
                new_duration = max(1, round(original_duration * adjustment_factor))
                adjusted_roadmap['phases'][i]['duration'] = f"{new_duration} months"
                
                # Update phase name with new time range
                start_month = sum(int(p['duration'].split()[0]) for p in adjusted_roadmap['phases'][:i]) + 1
                end_month = start_month + new_duration - 1
                adjusted_roadmap['phases'][i]['name'] = f"{phase['name'].split(' (')[0]} (Months {start_month}-{end_month})"
        
        return adjusted_roadmap
    
    def _create_role_specific_default(self, target_role: str, experience_level: str) -> Dict:
        """Create a role-specific default roadmap when no template is available"""
        # Extract skills from the target role
        role_skills = self._extract_skills_from_text(target_role)
        
        # If no skills extracted, use generic skills based on role keywords
        if not role_skills:
            role_keywords = target_role.lower().split()
            skill_mapping = {
                'software': ['Programming', 'Web Development', 'Databases', 'APIs'],
                'web': ['HTML/CSS', 'JavaScript', 'Frontend Frameworks', 'Backend Development'],
                'data': ['Data Analysis', 'Statistics', 'Python', 'SQL'],
                'machine': ['Machine Learning', 'Python', 'Statistics', 'Deep Learning'],
                'ai': ['Artificial Intelligence', 'Python', 'Machine Learning', 'Neural Networks'],
                'devops': ['Docker', 'Kubernetes', 'CI/CD', 'Cloud Platforms'],
                'cloud': ['AWS', 'Azure', 'GCP', 'Infrastructure'],
                'mobile': ['Mobile Development', 'iOS', 'Android', 'React Native'],
                'cybersecurity': ['Security Fundamentals', 'Network Security', 'Ethical Hacking', 'Compliance'],
                'network': ['Networking', 'Protocols', 'Security', 'Infrastructure'],
                'database': ['SQL', 'NoSQL', 'Database Design', 'Performance Tuning'],
                'ui': ['UI Design', 'UX Principles', 'Figma', 'Prototyping'],
                'ux': ['UX Research', 'User Testing', 'Prototyping', 'Analytics'],
                'product': ['Product Management', 'Agile', 'Scrum', 'Roadmapping'],
                'project': ['Project Management', 'Agile', 'Scrum', 'Risk Management'],
                'business': ['Business Analysis', 'Requirements Gathering', 'Stakeholder Management', 'Process Improvement'],
                'quality': ['Testing', 'Automation', 'Test Planning', 'Bug Tracking'],
                'system': ['System Administration', 'Linux', 'Scripting', 'Monitoring'],
                'backend': ['Python', 'Java', 'Node.js', 'SQL', 'REST APIs', 'Docker', 'Git'],
            }
            
            for keyword in role_keywords:
                for skill_key, skills in skill_mapping.items():
                    if skill_key in keyword:
                        role_skills.extend(skills)
                        break
        
        # If still no skills found, use generic ones
        if not role_skills:
            role_skills = ['Industry Basics', 'Core Tools', 'Fundamental Concepts']
        
        # Create role-specific phase names
        role_specific_names = {
            'backend': {
                'beginner': [
                    'Programming Fundamentals',
                    'Web Development Basics',
                    'Backend & Databases'
                ],
                'intermediate': [
                    'Advanced Backend Development',
                    'API Design & Security',
                    'DevOps & Deployment'
                ],
                'advanced': [
                    'System Design & Architecture',
                    'Performance Optimization',
                    'Leadership & Mentoring'
                ]
            },
            'frontend': {
                'beginner': [
                    'HTML/CSS Fundamentals',
                    'JavaScript Basics',
                    'Frontend Frameworks'
                ],
                'intermediate': [
                    'Advanced JavaScript & State Management',
                    'Performance & Accessibility',
                    'Testing & Tooling'
                ],
                'advanced': [
                    'Architecture & Patterns',
                    'Cross-platform Development',
                    'Team Leadership'
                ]
            },
            'data': {
                'beginner': [
                    'Mathematics & Statistics',
                    'Programming & Data Analysis',
                    'Data Visualization'
                ],
                'intermediate': [
                    'Machine Learning Fundamentals',
                    'Big Data Technologies',
                    'Cloud Platforms'
                ],
                'advanced': [
                    'Deep Learning & Neural Networks',
                    'Specialization & Research',
                    'Industry Leadership'
                ]
            }
        }
        
        # Determine role category for phase names
        role_category = 'other'
        target_role_lower = target_role.lower()
        if 'backend' in target_role_lower:
            role_category = 'backend'
        elif 'frontend' in target_role_lower:
            role_category = 'frontend'
        elif 'data' in target_role_lower:
            role_category = 'data'
        # Add more role categories as needed
        
        # Get phase names based on role category
        phase_names = role_specific_names.get(role_category, {}).get(experience_level, [
            f'{target_role.title()} Fundamentals',
            f'Intermediate {target_role.title()} Skills',
            f'Advanced {target_role.title()} Application'
        ])
        
        # Create phases based on experience level
        if experience_level == 'beginner':
            phases = [
                {
                    'name': f'{phase_names[0]} (Months 1-4)',
                    'duration': '4 months',
                    'skills': role_skills[:5] if len(role_skills) >= 5 else role_skills + ['Problem Solving', 'Communication', 'Git', 'Databases', 'APIs'][:5-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[0]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[:3], count=2)
                },
                {
                    'name': f'{phase_names[1]} (Months 5-8)',
                    'duration': '4 months',
                    'skills': role_skills[5:10] if len(role_skills) >= 10 else role_skills[5:] + ['Applied Skills', 'Tools Proficiency', 'Testing', 'Security', 'Performance'][:10-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[1]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[5:8] if len(role_skills) >= 8 else role_skills[3:5], count=2)
                },
                {
                    'name': f'{phase_names[2]} (Months 9-12)',
                    'duration': '4 months',
                    'skills': role_skills[10:15] if len(role_skills) >= 15 else role_skills[10:] + ['Specialization', 'Leadership', 'Architecture', 'Innovation', 'Strategy'][:15-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[2]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[10:13] if len(role_skills) >= 13 else role_skills[5:8], count=3)
                }
            ]
        elif experience_level == 'intermediate':
            phases = [
                {
                    'name': f'{phase_names[0]} (Months 1-4)',
                    'duration': '4 months',
                    'skills': role_skills[:5] if len(role_skills) >= 5 else role_skills + [skill for skill in ['Advanced Concepts', 'Architecture', 'Security', 'Performance', 'Testing'] if skill not in role_skills][:5-len(role_skills)],
                    'resources': [
                        {
                            'name': f'Advanced {target_role.title()} Course',
                            'type': 'course',
                            'platform': 'Coursera',
                            'url': '#',
                            'duration': '4 weeks',
                            'difficulty': 'Advanced',
                            'cost': 'Variable'
                        }
                    ],
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[:3], count=2)
                },
                {
                    'name': f'{phase_names[1]} (Months 5-8)',
                    'duration': '4 months',
                    'skills': role_skills[5:10] if len(role_skills) >= 10 else role_skills[5:] + [skill for skill in ['Specialization', 'Leadership', 'Innovation', 'Mentoring', 'Strategy'] if skill not in role_skills][:10-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[1]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[5:8] if len(role_skills) >= 8 else role_skills[3:5], count=2)
                },
                {
                    'name': f'{phase_names[2]} (Months 9-12)',
                    'duration': '4 months',
                    'skills': role_skills[10:15] if len(role_skills) >= 15 else role_skills[10:] + [skill for skill in ['Industry Leadership', 'Research', 'Speaking', 'Publishing', 'Innovation'] if skill not in role_skills][:15-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[2]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[10:13] if len(role_skills) >= 13 else role_skills[5:8], count=3)
                }
            ]
        else:  # advanced
            phases = [
                {
                    'name': f'{phase_names[0]} (Months 1-4)',
                    'duration': '4 months',
                    'skills': role_skills[:5] if len(role_skills) >= 5 else role_skills + [skill for skill in ['Expert Concepts', 'Research', 'Architecture', 'Innovation', 'Leadership'] if skill not in role_skills][:5-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[0]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[:3], count=2)
                },
                {
                    'name': f'{phase_names[1]} (Months 5-8)',
                    'duration': '4 months',
                    'skills': role_skills[5:10] if len(role_skills) >= 10 else role_skills[5:] + [skill for skill in ['Thought Leadership', 'Innovation', 'Mentoring', 'Strategy', 'Vision'] if skill not in role_skills][:10-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[1]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[5:8] if len(role_skills) >= 8 else role_skills[3:5], count=2)
                },
                {
                    'name': f'{phase_names[2]} (Months 9-12)',
                    'duration': '4 months',
                    'skills': role_skills[10:15] if len(role_skills) >= 15 else role_skills[10:] + [skill for skill in ['Vision', 'Strategy', 'Innovation', 'Research', 'Speaking'] if skill not in role_skills][:15-len(role_skills)],
                    'resources': self._fetch_free_resources(phase_names[2]),
                    'projects': self._generate_role_specific_projects(target_role, experience_level, role_skills[10:13] if len(role_skills) >= 13 else role_skills[5:8], count=3)
                }
            ]
        
        return {'phases': phases}
    
    def get_progress(self, roadmap: Dict, completed_activities: List[str]) -> Dict:
        """Calculate progress based on completed learning activities"""
        total_activities = 0
        completed_count = 0
        
        # Count total learning activities (skills + projects)
        for phase in roadmap['phases']:
            total_activities += len(phase['skills'])
            total_activities += len(phase.get('projects', []))
            
            # Count completed skills and projects
            for skill in phase['skills']:
                if skill in completed_activities:
                    completed_count += 1
            
            for project in phase.get('projects', []):
                if project['name'] in completed_activities:
                    completed_count += 1
        
        progress_percentage = (completed_count / total_activities * 100) if total_activities > 0 else 0
        
        return {
            'completed_count': completed_count,
            'total_activities': total_activities,
            'progress_percentage': round(progress_percentage, 1),
            'next_activity': self._get_next_activity(roadmap, completed_activities)
        }
    
    def _get_next_activity(self, roadmap: Dict, completed_activities: List[str]) -> str:
        """Get the next uncompleted learning activity"""
        for phase in roadmap['phases']:
            # Check for uncompleted skills
            for skill in phase['skills']:
                if skill not in completed_activities:
                    return f"Learn {skill}"
            
            # Check for uncompleted projects
            for project in phase.get('projects', []):
                if project['name'] not in completed_activities:
                    return f"Work on project: {project['name']}"
        
        return "All learning activities completed!"
    
    def suggest_next_steps(self, roadmap: Dict, completed_activities: List[str]) -> List[str]:
        """Suggest next steps based on current progress"""
        suggestions = []
        
        # Get current phase
        current_phase = None
        for phase in roadmap['phases']:
            # Check if there are any uncompleted activities in this phase
            uncompleted_activities = []
            for skill in phase['skills']:
                if skill not in completed_activities:
                    uncompleted_activities.append(f"skill: {skill}")
            
            for project in phase.get('projects', []):
                if project['name'] not in completed_activities:
                    uncompleted_activities.append(f"project: {project['name']}")
            
            if uncompleted_activities:
                current_phase = phase
                break
        
        if current_phase:
            # Suggest resources for current phase
            for resource in current_phase['resources'][:2]:
                suggestions.append(f"Consider starting with: {resource['name']} on {resource['platform']}")
            
            # Suggest next activity
            next_activity = self._get_next_activity(roadmap, completed_activities)
            if next_activity and next_activity != "All learning activities completed!":
                suggestions.append(f"Next activity to focus on: {next_activity}")
        
        # General suggestions based on progress
        progress = self.get_progress(roadmap, completed_activities)
        if progress['progress_percentage'] < 25:
            suggestions.append("Start with the foundational skills in the first phase")
        elif progress['progress_percentage'] < 50:
            suggestions.append("Keep up the good work! You're making steady progress")
        elif progress['progress_percentage'] < 75:
            suggestions.append("You're halfway there! Focus on the intermediate skills")
        else:
            suggestions.append("Almost there! Complete the remaining activities to finish your roadmap")
        
        # Add personalized suggestions based on experience level
        experience_level = roadmap.get('metadata', {}).get('experience_level', 'beginner')
        if experience_level == 'beginner':
            suggestions.append("As a beginner, focus on building a strong foundation before moving to advanced topics")
        elif experience_level == 'intermediate':
            suggestions.append("As an intermediate learner, try to build projects that combine multiple skills")
        else:  # advanced
            suggestions.append("As an advanced learner, consider mentoring others or contributing to open source projects")
        
        # Add goal-specific suggestions
        primary_goal = roadmap.get('metadata', {}).get('primary_goal', 'skill development')
        if 'career' in primary_goal.lower():
            suggestions.append("Since your goal is career change, focus on skills that are in high demand in your target industry")
        elif 'promotion' in primary_goal.lower():
            suggestions.append("Since your goal is promotion, emphasize leadership skills and project management alongside technical skills")
        else:
            suggestions.append("Since your goal is skill development, choose projects that allow you to apply what you've learned")
        
        return suggestions
    
    def export_roadmap(self, roadmap: Dict, format: str = "json") -> str:
        """Export roadmap in specified format"""
        if format.lower() == "json":
            return json.dumps(roadmap, indent=2)
        elif format.lower() == "text":
            markdown = f"# Career Roadmap for {roadmap['metadata']['target_role']}\n\n"
            markdown += f"**Experience Level:** {roadmap['metadata']['experience_level']}\n"
            markdown += f"**Timeline:** {roadmap['metadata']['timeline_months']} months\n"
            markdown += f"**Primary Goal:** {roadmap['metadata']['primary_goal']}\n\n"
            
            for i, phase in enumerate(roadmap['phases'], 1):
                markdown += f"## Phase {i}: {phase['name']}\n\n"
                markdown += f"**Duration:** {phase['duration']}\n\n"
                markdown += f"**Skills to Develop:**\n"
                for skill in phase['skills']:
                    markdown += f"- {skill}\n"
                markdown += "\n"
                
                markdown += f"**Learning Resources:**\n"
                for resource in phase['resources']:
                    markdown += f"- **{resource['name']}** ({resource['type']}) on {resource['platform']}\n"
                    markdown += f"  - Duration: {resource['duration']}\n"
                    markdown += f"  - Difficulty: {resource['difficulty']}\n"
                    markdown += f"  - Cost: {resource['cost']}\n"
                    if resource.get('url', '#') != '#' and resource.get('url'):
                        markdown += f"  - Link: [{resource['url']}]({resource['url']})\n"
                    elif resource.get('link', '#') != '#' and resource.get('link'):
                        markdown += f"  - Link: [{resource['link']}]({resource['link']})\n"
                markdown += "\n"
                
                markdown += f"**Projects:**\n"
                for project in phase.get('projects', []):
                    markdown += f"- **{project['name']}** ({project['difficulty']})\n"
                    markdown += f"  - Description: {project['description']}\n"
                    markdown += f"  - Technologies: {', '.join(project['technologies'])}\n"
                    markdown += f"  - Estimated Hours: {project['estimated_hours']}\n"
                markdown += "\n"
            
            return markdown
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def store_roadmap(self, roadmap_text: str, metadata: Dict[str, Any]) -> str:
        """Store generated roadmap (stub method since RAG is not used in this module)
        
        Args:
            roadmap_text (str): The generated roadmap text
            metadata (Dict[str, Any]): Metadata about the roadmap
            
        Returns:
            str: Empty string since RAG is not used in this module
        """
        print("Warning: Vector database not available for storing roadmap since RAG is not used in this module")
        return ""
    
    def search_similar_roadmaps(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar roadmaps (stub method since RAG is not used in this module)
        
        Args:
            query_text (str): Query text to search for similar roadmaps
            top_k (int): Number of similar roadmaps to return
            
        Returns:
            List[Dict[str, Any]]: Empty list since RAG is not used in this module
        """
        print("Warning: Vector database not available for searching similar roadmaps since RAG is not used in this module")
        return []
    
    def enhance_roadmap_with_vector_db(self, roadmap: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance roadmap (stub method since RAG is not used in this module)
        
        Args:
            roadmap (Dict[str, Any]): The generated roadmap
            user_profile (Dict[str, Any]): User profile information
            
        Returns:
            Dict[str, Any]: Unchanged roadmap since RAG is not used in this module
        """
        print("Warning: Vector database not available for enhancing roadmap since RAG is not used in this module")
        return roadmap