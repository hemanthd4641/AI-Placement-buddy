import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import quote
import json
import re

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

# Handle ModelManager import gracefully
try:
    from utils.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from model_manager import ModelManager
        MODEL_MANAGER_AVAILABLE = True
    except ImportError:
        print("Warning: ModelManager not available. Some features will be limited.")
        ModelManager = None
        MODEL_MANAGER_AVAILABLE = False

# Handle vector database components import
try:
    from utils.vector_db_manager import VectorDBManager, is_vector_db_available
    VECTOR_DB_AVAILABLE = True
except ImportError:
    try:
        from vector_db_manager import VectorDBManager, is_vector_db_available
        VECTOR_DB_AVAILABLE = True
    except ImportError:
        VectorDBManager = None
        is_vector_db_available = lambda: False
        VECTOR_DB_AVAILABLE = False

class SkillGapAnalyzer:
    """Analyzes skill gaps and provides learning recommendations
    This analyzer combines skill gap identification with LLM-powered recommendations
    and vector database storage to provide personalized learning paths.
    """
    
    def __init__(self):
        # Initialize the ModelManager to handle all AI models
        # This enables AI-powered recommendations and analysis
        if MODEL_MANAGER_AVAILABLE and ModelManager:
            try:
                self.model_manager = ModelManager()
            except Exception as e:
                print(f"Warning: Failed to initialize ModelManager: {e}. Some features will be limited.")
                self.model_manager = None
        else:
            self.model_manager = None
            print("Warning: ModelManager not available. Some features will be limited.")
        
        # Use VectorDBManager if available, otherwise set to None
        # This enables semantic search and storage capabilities for learning resources
        if VECTOR_DB_AVAILABLE and VectorDBManager and is_vector_db_available():
            try:
                self.vector_db_manager = VectorDBManager()
                self.vector_db = self.vector_db_manager.vector_db if hasattr(self.vector_db_manager, 'vector_db') else None
                print("✅ Vector database manager initialized for Skill Gap Analyzer")
            except Exception as e:
                print(f"Warning: Failed to initialize VectorDBManager: {e}. Some features will be limited.")
                self.vector_db_manager = None
                self.vector_db = None
        else:
            self.vector_db_manager = None
            self.vector_db = None
            print("Warning: VectorDBManager not available. Some features will be limited.")
        
        # Industry skill requirements database
        self.industry_skills = {
            'software_development': {
                'essential': ['python', 'java', 'javascript', 'git', 'sql', 'algorithms', 'data_structures'],
                'important': ['react', 'node.js', 'docker', 'aws', 'testing', 'agile', 'rest_apis'],
                'nice_to_have': ['kubernetes', 'mongodb', 'redis', 'microservices', 'ci/cd']
            },
            'data_science': {
                'essential': ['python', 'sql', 'statistics', 'pandas', 'numpy', 'machine_learning'],
                'important': ['scikit-learn', 'tensorflow', 'tableau', 'r', 'excel', 'data_visualization'],
                'nice_to_have': ['pytorch', 'spark', 'hadoop', 'deep_learning', 'nlp']
            },
            'web_development': {
                'essential': ['html', 'css', 'javascript', 'react', 'node.js', 'git'],
                'important': ['typescript', 'mongodb', 'express', 'rest_apis', 'responsive_design'],
                'nice_to_have': ['vue', 'angular', 'graphql', 'redux', 'webpack']
            },
            'mobile_development': {
                'essential': ['java', 'kotlin', 'swift', 'android', 'ios', 'git'],
                'important': ['react_native', 'flutter', 'firebase', 'api_integration'],
                'nice_to_have': ['xamarin', 'ionic', 'unity', 'ar/vr']
            },
            'cybersecurity': {
                'essential': ['networking', 'linux', 'python', 'security_fundamentals', 'encryption'],
                'important': ['penetration_testing', 'incident_response', 'compliance', 'risk_assessment'],
                'nice_to_have': ['ethical_hacking', 'forensics', 'malware_analysis', 'blockchain_security']
            }
        }
        
        # Learning resources database - Enhanced with more comprehensive resources
        self.learning_resources = {
            'python': {
                'free_courses': [
                    {'name': 'Python for Everybody Specialization', 'url': 'https://www.coursera.org/specializations/python', 'type': 'course', 'platform': 'Coursera', 'duration': '6 months', 'difficulty': 'Beginner'},
                    {'name': 'CS50: Introduction to Computer Science', 'url': 'https://cs50.harvard.edu/x/', 'type': 'course', 'platform': 'edX', 'duration': '3 months', 'difficulty': 'Beginner'},
                    {'name': 'Automate the Boring Stuff with Python', 'url': 'https://automatetheboringstuff.com/', 'type': 'book', 'platform': 'Online', 'duration': 'Self-paced', 'difficulty': 'Beginner'},
                    {'name': 'Python.org Official Tutorial', 'url': 'https://docs.python.org/3/tutorial/', 'type': 'documentation', 'platform': 'Python.org', 'duration': 'Self-paced', 'difficulty': 'Beginner'}
                ],
                'practice_platforms': [
                    {'name': 'HackerRank Python Track', 'url': 'https://www.hackerrank.com/domains/python', 'type': 'practice', 'platform': 'HackerRank', 'difficulty': 'All Levels'},
                    {'name': 'LeetCode Python Problems', 'url': 'https://leetcode.com/problemset/all/?difficulty=Easy&status=NOT_STARTED&tags=python', 'type': 'practice', 'platform': 'LeetCode', 'difficulty': 'All Levels'},
                    {'name': 'Codewars Python Challenges', 'url': 'https://www.codewars.com/?language=python', 'type': 'practice', 'platform': 'Codewars', 'difficulty': 'All Levels'}
                ],
                'youtube_channels': [
                    {'name': 'Corey Schafer Python Tutorials', 'url': 'https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Programming with Mosh', 'url': 'https://www.youtube.com/playlist?list=PLTjRvDozrdlw5En5v2xrBr_EqieHf7hGs', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Advanced'}
                ],
                'estimated_time': '2-4 months',
                'difficulty': 'beginner'
            },
            'javascript': {
                'free_courses': [
                    {'name': 'freeCodeCamp JavaScript Algorithms and Data Structures', 'url': 'https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/', 'type': 'course', 'platform': 'freeCodeCamp', 'duration': '300 hours', 'difficulty': 'Beginner'},
                    {'name': 'JavaScript30', 'url': 'https://javascript30.com/', 'type': 'course', 'platform': 'Wes Bos', 'duration': '30 days', 'difficulty': 'Intermediate'},
                    {'name': 'Eloquent JavaScript', 'url': 'https://eloquentjavascript.net/', 'type': 'book', 'platform': 'Online', 'duration': 'Self-paced', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'MDN JavaScript Guide', 'url': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide', 'type': 'documentation', 'platform': 'Mozilla', 'duration': 'Self-paced', 'difficulty': 'All Levels'}
                ],
                'practice_platforms': [
                    {'name': 'JavaScript.info Exercises', 'url': 'https://javascript.info/', 'type': 'tutorial', 'platform': 'JavaScript.info', 'difficulty': 'Beginner to Advanced'},
                    {'name': 'Exercism JavaScript Track', 'url': 'https://exercism.org/tracks/javascript', 'type': 'practice', 'platform': 'Exercism', 'difficulty': 'All Levels'},
                    {'name': 'JSFiddle', 'url': 'https://jsfiddle.net/', 'type': 'practice', 'platform': 'JSFiddle', 'difficulty': 'All Levels'}
                ],
                'youtube_channels': [
                    {'name': 'Traversy Media JavaScript Tutorials', 'url': 'https://www.youtube.com/playlist?list=PLillGF-RfqbZ7s3t6ZInY3NjEOOX7hsBv', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Advanced'},
                    {'name': 'The Net Ninja JavaScript Playlist', 'url': 'https://www.youtube.com/playlist?list=PL4cUxeGkcC9i9Ae2D9Ee1RvylH38dKuET', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'}
                ],
                'estimated_time': '2-3 months',
                'difficulty': 'beginner'
            },
            'react': {
                'free_courses': [
                    {'name': 'freeCodeCamp React Tutorial', 'url': 'https://www.freecodecamp.org/news/search?query=react', 'type': 'course', 'platform': 'freeCodeCamp', 'duration': '150 hours', 'difficulty': 'Beginner'},
                    {'name': 'React Official Tutorial', 'url': 'https://reactjs.org/tutorial/tutorial.html', 'type': 'tutorial', 'platform': 'React Docs', 'duration': 'Self-paced', 'difficulty': 'Beginner'},
                    {'name': 'Scrimba React Bootcamp', 'url': 'https://scrimba.com/learn/react', 'type': 'course', 'platform': 'Scrimba', 'duration': '7 hours', 'difficulty': 'Beginner'}
                ],
                'practice_platforms': [
                    {'name': 'React Challenges', 'url': 'https://github.com/alexgurr/react-coding-challenges', 'type': 'practice', 'platform': 'GitHub', 'difficulty': 'Intermediate'},
                    {'name': 'Frontend Mentor', 'url': 'https://www.frontendmentor.io/challenges?technologies=react', 'type': 'practice', 'platform': 'Frontend Mentor', 'difficulty': 'All Levels'},
                    {'name': 'React Hooks Exercises', 'url': 'https://github.com/mithun12000/react-hooks-exercises', 'type': 'practice', 'platform': 'GitHub', 'difficulty': 'Intermediate'}
                ],
                'youtube_channels': [
                    {'name': 'Codevolution React Tutorials', 'url': 'https://www.youtube.com/playlist?list=PLC3y8-rFHvwgg3vaYJgHGnModB54rxOk3', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Advanced'},
                    {'name': 'Ben Awad React Tutorials', 'url': 'https://www.youtube.com/user/99baddawg', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Intermediate to Advanced'}
                ],
                'estimated_time': '2-3 months',
                'difficulty': 'intermediate'
            },
            'machine_learning': {
                'free_courses': [
                    {'name': 'Machine Learning by Andrew Ng', 'url': 'https://www.coursera.org/learn/machine-learning', 'type': 'course', 'platform': 'Coursera', 'duration': '11 weeks', 'difficulty': 'Beginner'},
                    {'name': 'Fast.ai Practical Deep Learning', 'url': 'https://course.fast.ai/', 'type': 'course', 'platform': 'Fast.ai', 'duration': '7 weeks', 'difficulty': 'Beginner'},
                    {'name': 'Kaggle Learn Microcourses', 'url': 'https://www.kaggle.com/learn', 'type': 'course', 'platform': 'Kaggle', 'duration': 'Self-paced', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Hands-On Machine Learning Book', 'url': 'https://github.com/ageron/handson-ml2', 'type': 'book', 'platform': 'GitHub', 'duration': 'Self-paced', 'difficulty': 'Intermediate'}
                ],
                'practice_platforms': [
                    {'name': 'Kaggle Competitions', 'url': 'https://www.kaggle.com/competitions', 'type': 'competition', 'platform': 'Kaggle', 'difficulty': 'All Levels'},
                    {'name': 'Google Colab', 'url': 'https://colab.research.google.com/', 'type': 'practice', 'platform': 'Google', 'difficulty': 'All Levels'},
                    {'name': 'ML Zoomcamp', 'url': 'https://github.com/alexeygrigorev/mlbookcamp-code', 'type': 'course', 'platform': 'GitHub', 'difficulty': 'Intermediate'}
                ],
                'youtube_channels': [
                    {'name': 'StatQuest with Josh Starmer', 'url': 'https://www.youtube.com/user/joshstarmer', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Advanced'},
                    {'name': 'Sentdex Machine Learning Tutorials', 'url': 'https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG0YXM7ueo6', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'}
                ],
                'estimated_time': '4-6 months',
                'difficulty': 'advanced'
            },
            'sql': {
                'free_courses': [
                    {'name': 'SQL for Data Science', 'url': 'https://www.coursera.org/learn/sql-for-data-science', 'type': 'course', 'platform': 'Coursera', 'duration': '4 weeks', 'difficulty': 'Beginner'},
                    {'name': 'SQLBolt Interactive Tutorial', 'url': 'https://sqlbolt.com/', 'type': 'tutorial', 'platform': 'SQLBolt', 'duration': 'Self-paced', 'difficulty': 'Beginner'},
                    {'name': 'Mode Analytics SQL Tutorial', 'url': 'https://mode.com/sql-tutorial/', 'type': 'tutorial', 'platform': 'Mode Analytics', 'duration': 'Self-paced', 'difficulty': 'Beginner'}
                ],
                'practice_platforms': [
                    {'name': 'HackerRank SQL Track', 'url': 'https://www.hackerrank.com/domains/sql', 'type': 'practice', 'platform': 'HackerRank', 'difficulty': 'All Levels'},
                    {'name': 'LeetCode Database Problems', 'url': 'https://leetcode.com/problemset/database/', 'type': 'practice', 'platform': 'LeetCode', 'difficulty': 'All Levels'},
                    {'name': 'SQLZoo', 'url': 'https://sqlzoo.net/', 'type': 'practice', 'platform': 'SQLZoo', 'difficulty': 'All Levels'}
                ],
                'youtube_channels': [
                    {'name': 'TechTFQ SQL Tutorials', 'url': 'https://www.youtube.com/playlist?list=PLwV5aSqnEbwqwzH2hHHBOw7sZ2m_dBmOi', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Joey Blue SQL Tutorials', 'url': 'https://www.youtube.com/playlist?list=PL_c9BZzLwBRKn28X5x7PBHG_B6vCj3YRa', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner'}
                ],
                'estimated_time': '1-2 months',
                'difficulty': 'beginner'
            },
            'docker': {
                'free_courses': [
                    {'name': 'Docker Mastery', 'url': 'https://www.udemy.com/course/docker-mastery/', 'type': 'course', 'platform': 'Udemy', 'duration': '8 hours', 'difficulty': 'Beginner'},
                    {'name': 'Docker Curriculum', 'url': 'https://docker-curriculum.com/', 'type': 'tutorial', 'platform': 'Online', 'duration': 'Self-paced', 'difficulty': 'Beginner'},
                    {'name': 'Play with Docker', 'url': 'https://labs.play-with-docker.com/', 'type': 'hands-on', 'platform': 'Docker', 'duration': 'Self-paced', 'difficulty': 'Beginner'}
                ],
                'practice_platforms': [
                    {'name': 'Katacoda Docker Playground', 'url': 'https://www.katacoda.com/courses/docker', 'type': 'hands-on', 'platform': 'Katacoda', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Docker Labs', 'url': 'https://github.com/docker/labs', 'type': 'practice', 'platform': 'GitHub', 'difficulty': 'All Levels'}
                ],
                'youtube_channels': [
                    {'name': 'LearnLinuxTV Docker Tutorials', 'url': 'https://www.youtube.com/playlist?list=PLT98CRl2KxKEUHie1m24-wkyHpEsa4Y70', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Nana Docker Tutorial', 'url': 'https://www.youtube.com/watch?v=3c-iBn73dDE', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner'}
                ],
                'estimated_time': '1 month',
                'difficulty': 'intermediate'
            },
            'git': {
                'free_courses': [
                    {'name': 'Git and GitHub for Beginners', 'url': 'https://www.youtube.com/watch?v=RGOj5yH7evk', 'type': 'tutorial', 'platform': 'YouTube', 'duration': '1 hour', 'difficulty': 'Beginner'},
                    {'name': 'Learn Git Branching', 'url': 'https://learngitbranching.js.org/', 'type': 'interactive', 'platform': 'Online', 'duration': 'Self-paced', 'difficulty': 'Beginner to Intermediate'},
                    {'name': 'Pro Git Book', 'url': 'https://git-scm.com/book/en/v2', 'type': 'book', 'platform': 'Git', 'duration': 'Self-paced', 'difficulty': 'All Levels'}
                ],
                'practice_platforms': [
                    {'name': 'GitHub Learning Lab', 'url': 'https://lab.github.com/', 'type': 'interactive', 'platform': 'GitHub', 'duration': 'Self-paced', 'difficulty': 'All Levels'},
                    {'name': 'Git Exercises', 'url': 'https://github.com/ondrejsika/git-training-course', 'type': 'practice', 'platform': 'GitHub', 'difficulty': 'Beginner to Advanced'}
                ],
                'youtube_channels': [
                    {'name': 'Programming with Mosh Git Tutorial', 'url': 'https://www.youtube.com/watch?v=8JJ101D3knE', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner'},
                    {'name': 'The Net Ninja Git Playlist', 'url': 'https://www.youtube.com/playlist?list=PL4cUxeGkcC9goXbgTDQ0n_4TBzOO0ocPR', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'Beginner to Intermediate'}
                ],
                'estimated_time': '2-3 weeks',
                'difficulty': 'beginner'
            }
        }
        
        # Skill level mapping
        self.skill_levels = {
            'beginner': {'min_score': 0, 'max_score': 40},
            'intermediate': {'min_score': 40, 'max_score': 70},
            'advanced': {'min_score': 70, 'max_score': 100}
        }

    def _get_template_text(self, template_for: str) -> str:
        """Fetch a template from vector DB by feature key, if available."""
        try:
            if self.vector_db_manager and hasattr(self.vector_db_manager, 'search_templates'):
                res = self.vector_db_manager.search_templates(template_for, top_k=1)
                if res:
                    md = res[0].get('metadata', {})
                    return (md.get('template') or md.get('excerpt') or '').strip()
        except Exception:
            pass
        return ""
    
    def _fetch_youtube_tutorials(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Fetch YouTube tutorials for a given query"""
        try:
            # Since we can't use the YouTube API without an API key, we'll create more specific placeholder links
            search_query = query.replace(' ', '+')
            tutorials = []
            
            # Create more specific YouTube tutorial entries based on the skill
            tutorial_templates = [
                {
                    'name': f'{query} Complete Course for Beginners',
                    'type': 'video',
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={search_query}+complete+course+for+beginners',
                    'duration': '2-6 hours',
                    'difficulty': 'Beginner',
                    'cost': 'Free'
                },
                {
                    'name': f'{query} Tutorial - Crash Course',
                    'type': 'video',
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={search_query}+tutorial+crash+course',
                    'duration': '1-2 hours',
                    'difficulty': 'Beginner',
                    'cost': 'Free'
                },
                {
                    'name': f'{query} Full Course - Learn {query} in Few Hours',
                    'type': 'video',
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={search_query}+full+course',
                    'duration': '4-8 hours',
                    'difficulty': 'Beginner to Intermediate',
                    'cost': 'Free'
                },
                {
                    'name': f'{query} Advanced Tutorial',
                    'type': 'video',
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={search_query}+advanced+tutorial',
                    'duration': '2-4 hours',
                    'difficulty': 'Intermediate',
                    'cost': 'Free'
                },
                {
                    'name': f'{query} Project Tutorial',
                    'type': 'video',
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={search_query}+project+tutorial',
                    'duration': '1-3 hours',
                    'difficulty': 'Intermediate',
                    'cost': 'Free'
                }
            ]
            
            return tutorial_templates[:max_results]
        except Exception as e:
            print(f"Error fetching YouTube tutorials: {e}")
            return []
    
    def _fetch_free_resources(self, skill: str) -> List[Dict[str, str]]:
        """Fetch free learning resources for a skill"""
        resources = []
        
        try:
            # Create more specific free resource entries
            skill_query = skill.replace(' ', '+')
            
            # FreeCodeCamp resources
            resources.append({
                'name': f'{skill} - FreeCodeCamp Curriculum',
                'type': 'course',
                'platform': 'FreeCodeCamp',
                'url': f'https://www.freecodecamp.org/news/search?query={skill_query}',
                'duration': 'Self-paced',
                'difficulty': 'Beginner to Intermediate',
                'cost': 'Free'
            })
            
            # Khan Academy resources (if relevant)
            if skill.lower() in ['math', 'statistics', 'python', 'javascript']:
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
                'difficulty': 'Beginner to Intermediate',
                'cost': 'Free with audit'
            })
            
            # edX free resources
            resources.append({
                'name': f'{skill} - edX Free Courses',
                'type': 'course',
                'platform': 'edX',
                'url': f'https://www.edx.org/search?q={skill_query}',
                'duration': '4-12 weeks',
                'difficulty': 'Beginner to Intermediate',
                'cost': 'Free with audit'
            })
            
            # YouTube tutorials
            youtube_tutorials = self._fetch_youtube_tutorials(skill)
            resources.extend(youtube_tutorials)
            
        except Exception as e:
            print(f"Error fetching free resources: {e}")
        
        return resources
    
    def analyze_skill_gap(self, user_skills: List[str], target_role: str, 
                         current_scores: Optional[Dict] = None) -> Dict:
        """Perform comprehensive skill gap analysis"""
        try:
            # Get required skills for target role
            role_requirements = self.get_role_requirements(target_role)
            
            # Normalize user skills
            normalized_user_skills = [skill.lower().replace(' ', '_') for skill in user_skills]
            
            # Analyze gaps
            gaps = self.identify_skill_gaps(normalized_user_skills, role_requirements)
            
            # Get learning recommendations
            recommendations = self.generate_learning_recommendations(gaps, current_scores)
            
            # Create learning roadmap
            roadmap = self.create_learning_roadmap(gaps, target_role)
            
            # Calculate readiness score
            readiness_score = self.calculate_readiness_score(normalized_user_skills, role_requirements)
            
            # Enhance recommendations with vector database context
            if self.vector_db_manager:
                enhanced_recommendations = self.enhance_skill_recommendations_with_vector_db(
                    gaps['missing_essential'] + gaps['missing_important'] + gaps['missing_nice_to_have'], 
                    target_role
                )
                # Add enhanced context to recommendations
                for rec in recommendations:
                    skill = rec['skill']
                    if skill in enhanced_recommendations:
                        rec['vector_db_context'] = enhanced_recommendations[skill]
            
            return {
                'target_role': target_role,
                'user_skills': user_skills,
                'analysis': {
                    'matching_skills': gaps['matching_skills'],
                    'missing_essential': gaps['missing_essential'],
                    'missing_important': gaps['missing_important'],
                    'missing_nice_to_have': gaps['missing_nice_to_have'],
                    'readiness_score': readiness_score
                },
                'recommendations': recommendations,
                'learning_roadmap': roadmap,
                'estimated_timeline': self.calculate_learning_timeline(gaps)
            }
        except Exception as e:
            print(f"Error in skill gap analysis: {e}")
            # Return a basic analysis as fallback
            return {
                'target_role': target_role,
                'user_skills': user_skills,
                'analysis': {
                    'matching_skills': [],
                    'missing_essential': [],
                    'missing_important': [],
                    'missing_nice_to_have': [],
                    'readiness_score': 0
                },
                'recommendations': [],
                'learning_roadmap': {},
                'estimated_timeline': 'Unknown'
            }
    
    def get_role_requirements(self, target_role: str) -> Dict:
        """Get skill requirements for a target role"""
        # Normalize target role
        normalized_role = target_role.lower().replace(' ', '_').replace('-', '_')
        
        # Try to find exact match
        for role_key, requirements in self.industry_skills.items():
            if normalized_role in role_key or role_key in normalized_role:
                return requirements
        
        # If not found, try LLM to generate requirements for custom roles
        if self.model_manager and self.model_manager.models.get('lm'):
            try:
                prompt = f"""
                Generate a list of required skills for the role: {target_role}.
                Please categorize the skills into three levels:
                1. Essential (must-have skills)
                2. Important (should-have skills)
                3. Nice to have (good-to-have skills)
                
                Format your response as a JSON object with these three keys.
                Each category should contain a list of skills (5-8 skills per category).
                """
                
                response = self.model_manager.generate_text(prompt, max_length=300)
                
                # Try to parse JSON from response
                import json
                try:
                    # Extract JSON from response if it's wrapped in text
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        requirements = json.loads(json_match.group())
                        return requirements
                except:
                    pass
            except Exception as e:
                print(f"Error generating role requirements with LLM: {e}")
        
        # Fallback to general software development skills
        return self.industry_skills.get('software_development', {
            'essential': ['programming', 'problem_solving', 'communication'],
            'important': ['version_control', 'testing', 'documentation'],
            'nice_to_have': ['leadership', 'mentoring', 'project_management']
        })
    
    def identify_skill_gaps(self, user_skills: List[str], role_requirements: Dict) -> Dict:
        """Identify gaps between user skills and role requirements"""
        # Convert user skills to set for faster lookup
        user_skills_set = set(user_skills)
        
        # Identify matching and missing skills for each category
        matching_skills = []
        missing_essential = []
        missing_important = []
        missing_nice_to_have = []
        
        # Check essential skills
        for skill in role_requirements.get('essential', []):
            normalized_skill = skill.lower().replace(' ', '_')
            if normalized_skill in user_skills_set:
                matching_skills.append(skill)
            else:
                missing_essential.append(skill)
        
        # Check important skills
        for skill in role_requirements.get('important', []):
            normalized_skill = skill.lower().replace(' ', '_')
            if normalized_skill in user_skills_set:
                matching_skills.append(skill)
            else:
                missing_important.append(skill)
        
        # Check nice-to-have skills
        for skill in role_requirements.get('nice_to_have', []):
            normalized_skill = skill.lower().replace(' ', '_')
            if normalized_skill in user_skills_set:
                matching_skills.append(skill)
            else:
                missing_nice_to_have.append(skill)
        
        return {
            'matching_skills': matching_skills,
            'missing_essential': missing_essential,
            'missing_important': missing_important,
            'missing_nice_to_have': missing_nice_to_have
        }
    
    def generate_learning_recommendations(self, skill_gaps: Dict, current_scores: Optional[Dict] = None) -> List[Dict]:
        """Generate personalized learning recommendations with specific resources"""
        recommendations = []
        
        # Combine all missing skills with their priorities
        skills_with_priority = []
        for skill in skill_gaps.get('missing_essential', []):
            skills_with_priority.append((skill, 'essential'))
        for skill in skill_gaps.get('missing_important', []):
            skills_with_priority.append((skill, 'important'))
        for skill in skill_gaps.get('missing_nice_to_have', []):
            skills_with_priority.append((skill, 'nice_to_have'))
        
        # If no skills are missing, provide a general recommendation
        if not skills_with_priority:
            return [{
                'skill': 'General Development',
                'priority': 'personalized',
                'difficulty': 'intermediate',
                'estimated_time': 'Varies',
                'learning_resources': {
                    'free_courses': [
                        {'name': 'Learning How to Learn', 'url': 'https://www.coursera.org/learn/learning-how-to-learn', 'type': 'course', 'platform': 'Coursera', 'duration': '4 weeks', 'difficulty': 'Beginner'},
                        {'name': 'Mindset and Learning', 'url': 'https://www.edx.org/course/mindset-and-learning', 'type': 'course', 'platform': 'edX', 'duration': '4 weeks', 'difficulty': 'Beginner'}
                    ],
                    'practice_platforms': [
                        {'name': 'Khan Academy', 'url': 'https://www.khanacademy.org/', 'type': 'practice', 'platform': 'Khan Academy', 'difficulty': 'All Levels'}
                    ],
                    'youtube_channels': [
                        {'name': 'Thomas Frank - College Info Geek', 'url': 'https://www.youtube.com/user/collegeinfogeek', 'type': 'tutorial', 'platform': 'YouTube', 'difficulty': 'All Levels'}
                    ]
                },
                'learning_path': [
                    "Continue developing your existing skills",
                    "Explore new areas of interest",
                    "Build projects that challenge you",
                    "Stay updated with industry trends"
                ]
            }]
        
        # If LLM is available, generate personalized recommendations using LLM
        if self.model_manager and self.model_manager.models.get('lm'):
            try:
                llm_recommendations = self._generate_personalized_recommendations_with_llm(skill_gaps, current_scores)
                if llm_recommendations:
                    return llm_recommendations
            except Exception as e:
                print(f"Error generating LLM recommendations: {e}")
        
        # Generate recommendations for each missing skill with detailed resources
        for skill, priority in skills_with_priority:
            # Get learning resources for this skill
            resources = self.get_learning_resources_for_skill(skill)
            
            # Determine difficulty based on current scores or default
            difficulty = resources.get('difficulty', 'beginner')
            if current_scores and skill.lower() in current_scores:
                score = current_scores[skill.lower()]
                if score >= 70:
                    difficulty = 'advanced'
                elif score >= 40:
                    difficulty = 'intermediate'
                else:
                    difficulty = 'beginner'
            
            # Determine estimated time
            estimated_time = resources.get('estimated_time', '2-4 months')
            
            # Create a comprehensive recommendation with personalized learning path
            recommendation = {
                'skill': skill,
                'priority': priority,
                'difficulty': difficulty,
                'estimated_time': estimated_time,
                'learning_resources': {
                    'free_courses': resources.get('free_courses', [])[:3],  # Top 3 courses
                    'practice_platforms': resources.get('practice_platforms', [])[:2],  # Top 2 practice platforms
                    'youtube_channels': resources.get('youtube_channels', [])[:2]  # Top 2 YouTube channels
                },
                'learning_path': self._generate_learning_path(skill, priority, resources)
            }
            
            recommendations.append(recommendation)
        
        # Sort recommendations by priority (essential first, then important, then nice_to_have)
        priority_order = {'essential': 1, 'important': 2, 'nice_to_have': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations[:15]  # Limit to top 15 recommendations

    def _generate_learning_path(self, skill: str, priority: str, resources: Dict) -> List[str]:
        """Generate a step-by-step learning path for a skill"""
        learning_path = []
        
        # Get resources
        free_courses = resources.get('free_courses', [])
        practice_platforms = resources.get('practice_platforms', [])
        youtube_channels = resources.get('youtube_channels', [])
        
        # Add initial learning step
        learning_path.append(f"1. Start with foundational learning for {skill}")
        
        # Add course recommendation
        if free_courses:
            first_course = free_courses[0]
            learning_path.append(f"2. Take '{first_course.get('name', 'a beginner course')}' on {first_course.get('platform', 'a learning platform')} ({first_course.get('duration', 'self-paced')})")
        
        # Add practice step
        if practice_platforms:
            first_practice = practice_platforms[0]
            learning_path.append(f"3. Practice on '{first_practice.get('name', 'a practice platform')}' on {first_practice.get('platform', 'a practice platform')}")
        
        # Add intermediate step
        learning_path.append(f"4. Move to intermediate resources and build small projects")
        
        # Add YouTube resources if available
        if youtube_channels:
            first_youtube = youtube_channels[0]
            learning_path.append(f"5. Supplement with '{first_youtube.get('name', 'tutorial videos')}' on YouTube")
        
        # Add advanced step
        learning_path.append(f"6. Work on real-world projects to demonstrate proficiency")
        
        # Add final step based on priority
        if priority == 'essential':
            learning_path.append(f"7. Achieve mastery through consistent practice and portfolio building")
        elif priority == 'important':
            learning_path.append(f"7. Build a solid understanding through practice and application")
        else:
            learning_path.append(f"7. Gain familiarity through tutorials and basic projects")
        
        return learning_path
    
    def _generate_personalized_recommendations_with_llm(self, skill_gaps: Dict, current_scores: Optional[Dict] = None) -> List[Dict]:
        """Generate personalized learning recommendations using LLM"""
        # Check if model manager is available
        if not self.model_manager or not self.model_manager.models.get('lm'):
            return []  # Return empty list instead of None
            
        # Create a more comprehensive prompt for LLM
        template_instructions = self._get_template_text('skill_gap')
        prompt = f"""
{template_instructions}
        Based on the following skill gaps analysis, generate personalized learning recommendations:
        
        Missing Essential Skills: {', '.join(skill_gaps.get('missing_essential', []))}
        Missing Important Skills: {', '.join(skill_gaps.get('missing_important', []))}
        Missing Nice-to-have Skills: {', '.join(skill_gaps.get('missing_nice_to_have', []))}
        
        Current skill scores (if available): {current_scores if current_scores else 'Not provided'}
        
        Please provide detailed learning recommendations that:
        1. Prioritize essential skills first
        2. Suggest specific learning resources (courses, books, practice platforms, YouTube channels)
        3. Include estimated time commitment for each skill
        4. Consider the learner's current skill level
        5. Provide a logical learning sequence
        6. Include a step-by-step learning path for each skill
        7. Suggest practical projects to apply the skills
        
        Format your response as a JSON array of recommendation objects.
        Each recommendation should have the following structure:
        {{
            "skill": "skill_name",
            "priority": "essential|important|nice_to_have",
            "difficulty": "beginner|intermediate|advanced",
            "estimated_time": "time estimate",
            "learning_resources": {{
                "free_courses": [
                    {{
                        "name": "Course Name",
                        "url": "Course URL",
                        "type": "course|book|tutorial",
                        "platform": "Platform Name",
                        "duration": "Duration",
                        "difficulty": "Beginner|Intermediate|Advanced"
                    }}
                ],
                "practice_platforms": [
                    {{
                        "name": "Platform Name",
                        "url": "Platform URL",
                        "type": "practice|competition|interactive",
                        "platform": "Platform Name",
                        "difficulty": "Beginner|Intermediate|Advanced"
                    }}
                ],
                "youtube_channels": [
                    {{
                        "name": "Channel Name",
                        "url": "Channel URL",
                        "type": "tutorial|playlist",
                        "platform": "YouTube",
                        "difficulty": "Beginner|Intermediate|Advanced"
                    }}
                ]
            }},
            "learning_path": [
                "Step 1: Description",
                "Step 2: Description",
                ...
            ]
        }}
        
        Provide 3-5 detailed recommendations.
        """
        
        # Generate recommendations using LLM
        response = self.model_manager.generate_text(prompt, max_length=1500)
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response if it's wrapped in text
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
        except Exception as e:
            print(f"Error parsing LLM recommendations: {e}")
            # Fallback to basic parsing
            recommendations = []
            lines = response.split('\n')
            
            for line in lines:
                if line.strip() and not line.startswith('```'):
                    # Create a simple recommendation from each line
                    recommendations.append({
                        'skill': 'Personalized Recommendation',
                        'priority': 'personalized',
                        'difficulty': 'intermediate',
                        'estimated_time': 'Varies',
                        'learning_resources': {
                            'description': line.strip(),
                            'free_courses': [],
                            'practice_platforms': [],
                            'youtube_channels': []
                        }
                    })
            
            return recommendations[:5]  # Limit to top 5 personalized recommendations
        
        # Return empty list if all parsing attempts failed
        return []
    
    def get_learning_resources_for_skill(self, skill: str) -> Dict:
        """Get learning resources for a specific skill"""
        # Normalize skill name
        normalized_skill = skill.lower().replace(' ', '_').replace('-', '_')
        
        # Check if we have predefined resources
        if normalized_skill in self.learning_resources:
            resources = self.learning_resources[normalized_skill].copy()  # Make a copy to avoid modifying original
            return resources
        
        # If LLM is available, generate resources for custom skills
        if self.model_manager and self.model_manager.models.get('lm'):
            try:
                template_instructions = self._get_template_text('skill_gap')
                prompt = f"""
{template_instructions}
                Generate comprehensive learning resources for the skill: {skill}.
                Please provide:
                1. 3-4 free online courses or tutorials (include name, URL, type, platform, duration, difficulty)
                2. 2-3 practice platforms or exercises (include name, URL, type, platform, difficulty)
                3. 2 YouTube channels or playlists (include name, URL, type, platform, difficulty)
                4. Estimated time to learn (e.g., "2-4 months")
                5. Difficulty level (beginner, intermediate, advanced)
                
                Format as a JSON object with keys: free_courses, practice_platforms, youtube_channels, estimated_time, difficulty.
                Each course/platform should have name, url, type, platform, duration (for courses), and difficulty fields.
                """
                
                response = self.model_manager.generate_text(prompt, max_length=500)
                
                # Try to parse JSON from response
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        resources = json.loads(json_match.group())
                        return resources
                except:
                    pass
            except Exception as e:
                print(f"Error generating learning resources with LLM: {e}")
        
        # Fallback resources with enhanced free resources and YouTube tutorials
        free_resources = self._fetch_free_resources(skill)
        courses = [r for r in free_resources if r.get('type') in ['course', 'tutorial', 'book', 'documentation']]
        practice_platforms = [r for r in free_resources if r.get('type') in ['practice', 'interactive', 'competition', 'hands-on']]
        youtube_videos = [r for r in free_resources if r.get('platform') == 'YouTube']
        
        return {
            'free_courses': courses[:4],  # Limit to 4 courses
            'practice_platforms': practice_platforms[:3],  # Limit to 3 practice platforms
            'youtube_channels': youtube_videos[:3],  # Limit to 3 YouTube resources
            'estimated_time': '2-4 months',
            'difficulty': 'intermediate'
        }
    
    def create_learning_roadmap(self, skill_gaps: Dict, target_role: str) -> Dict:
        """Create a personalized learning roadmap"""
        roadmap = {
            'target_role': target_role,
            'phases': []
        }
        
        # Phase 1: Essential skills
        essential_skills = skill_gaps.get('missing_essential', [])
        if essential_skills:
            roadmap['phases'].append({
                'phase': 1,
                'title': 'Master Essential Skills',
                'description': 'Focus on the core skills required for your target role',
                'skills': essential_skills,
                'duration_weeks': len(essential_skills) * 4,  # 4 weeks per skill on average
                'priority': 'high'
            })
        
        # Phase 2: Important skills
        important_skills = skill_gaps.get('missing_important', [])
        if important_skills:
            roadmap['phases'].append({
                'phase': 2,
                'title': 'Develop Important Skills',
                'description': 'Build on your foundation with important complementary skills',
                'skills': important_skills,
                'duration_weeks': len(important_skills) * 3,  # 3 weeks per skill on average
                'priority': 'medium'
            })
        
        # Phase 3: Nice-to-have skills
        nice_to_have_skills = skill_gaps.get('missing_nice_to_have', [])
        if nice_to_have_skills:
            roadmap['phases'].append({
                'phase': 3,
                'title': 'Enhance with Nice-to-have Skills',
                'description': 'Round out your expertise with advanced skills',
                'skills': nice_to_have_skills,
                'duration_weeks': len(nice_to_have_skills) * 2,  # 2 weeks per skill on average
                'priority': 'low'
            })
        
        # If LLM is available, generate a more sophisticated roadmap
        if self.model_manager and self.model_manager.models.get('lm'):
            try:
                llm_roadmap = self._generate_roadmap_with_llm(skill_gaps, target_role)
                if llm_roadmap and 'phases' in llm_roadmap:
                    roadmap = llm_roadmap
            except Exception as e:
                print(f"Error generating roadmap with LLM: {e}")
        
        return roadmap
    
    def _generate_roadmap_with_llm(self, skill_gaps: Dict, target_role: str) -> Dict[str, Any]:
        """Generate a sophisticated learning roadmap using LLM"""
        # Check if model manager is available
        if not self.model_manager or not self.model_manager.models.get('lm'):
            return {}  # Return empty dict instead of None
            
        prompt = f"""
        Create a detailed learning roadmap for the role: {target_role}.
        
        Skill gaps analysis:
        - Essential missing skills: {', '.join(skill_gaps.get('missing_essential', []))}
        - Important missing skills: {', '.join(skill_gaps.get('missing_important', []))}
        - Nice-to-have missing skills: {', '.join(skill_gaps.get('missing_nice_to_have', []))}
        
        Please create a comprehensive learning roadmap with:
        1. 3-4 learning phases
        2. Specific skills to focus on in each phase
        3. Estimated duration for each phase (in weeks)
        4. Learning objectives for each phase
        5. Suggested resources for each phase
        
        Format the response as a JSON object with a 'phases' array.
        Each phase should have: phase (number), title, description, skills (array), duration_weeks (number), priority.
        """
        
        response = self.model_manager.generate_text(prompt, max_length=500)
        
        # Try to parse JSON from response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                roadmap = json.loads(json_match.group())
                roadmap['target_role'] = target_role
                return roadmap
        except:
            pass
        
        # Return empty dict if parsing failed
        return {}
    
    def calculate_readiness_score(self, user_skills: List[str], role_requirements: Dict) -> float:
        """Calculate readiness score based on skill match"""
        # Count total required skills
        essential_skills = role_requirements.get('essential', [])
        important_skills = role_requirements.get('important', [])
        nice_to_have_skills = role_requirements.get('nice_to_have', [])
        
        total_required_skills = len(essential_skills) + len(important_skills)
        
        if total_required_skills == 0:
            return 100.0  # No required skills, assume fully ready
        
        # Count matching skills
        user_skills_set = set(user_skills)
        matching_essential = sum(1 for skill in essential_skills if skill.lower().replace(' ', '_') in user_skills_set)
        matching_important = sum(1 for skill in important_skills if skill.lower().replace(' ', '_') in user_skills_set)
        
        # Essential skills weighted at 70%, important skills at 30%
        readiness_score = (
            (matching_essential / len(essential_skills) * 70) if essential_skills else 0 +
            (matching_important / len(important_skills) * 30) if important_skills else 0
        )
        
        return min(100.0, readiness_score)
    
    def calculate_learning_timeline(self, skill_gaps: Dict) -> str:
        """Calculate estimated learning timeline"""
        # Simple estimation: 4 weeks per essential skill, 3 per important, 2 per nice-to-have
        essential_count = len(skill_gaps.get('missing_essential', []))
        important_count = len(skill_gaps.get('missing_important', []))
        nice_to_have_count = len(skill_gaps.get('missing_nice_to_have', []))
        
        total_weeks = (essential_count * 4) + (important_count * 3) + (nice_to_have_count * 2)
        
        if total_weeks == 0:
            return "Ready now"
        elif total_weeks < 4:
            return "Less than 1 month"
        elif total_weeks < 12:
            return "1-3 months"
        elif total_weeks < 24:
            return "3-6 months"
        else:
            return "6+ months"
    
    def compare_resume_with_job_description(self, resume_skills: List[str], job_description: str) -> Dict:
        """Compare resume skills with job description requirements"""
        # Extract skills from job description
        job_skills = self._extract_skills_from_text(job_description)
        
        # Normalize skills
        normalized_resume_skills = [skill.lower().replace(' ', '_') for skill in resume_skills]
        normalized_job_skills = [skill.lower().replace(' ', '_') for skill in job_skills]
        
        # Find matching and missing skills
        resume_skills_set = set(normalized_resume_skills)
        job_skills_set = set(normalized_job_skills)
        
        matching_skills = list(resume_skills_set.intersection(job_skills_set))
        missing_skills = list(job_skills_set.difference(resume_skills_set))
        
        # Calculate match percentage
        match_percentage = (len(matching_skills) / len(job_skills_set) * 100) if job_skills_set else 100
        
        return {
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'match_percentage': round(match_percentage, 1),
            'matching_skills_count': len(matching_skills),
            'missing_skills_count': len(missing_skills),
            'total_job_skills': len(job_skills_set)
        }
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching"""
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
            'html', 'css', 'angular', 'vue', 'spring', 'django', 'flask', 'express', 'postgresql',
            'mysql', 'redis', 'rabbitmq', 'kafka', 'jenkins', 'git', 'github', 'gitlab', 'ci/cd',
            'machine learning', 'deep learning', 'data science', 'data analysis', 'data visualization',
            'tableau', 'power bi', 'excel', 'r', 'scala', 'go', 'rust', 'c++', 'c#', '.net',
            'communication', 'leadership', 'problem solving', 'teamwork', 'project management'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in tech_skills:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return found_skills
    
    def get_skill_difficulty(self, skill: str) -> str:
        """Get difficulty level for a skill"""
        # This would typically be more sophisticated, but for now we'll use a simple approach
        beginner_skills = ['html', 'css', 'excel', 'communication', 'teamwork']
        intermediate_skills = ['javascript', 'python', 'sql', 'git', 'react', 'node.js']
        advanced_skills = ['machine learning', 'tensorflow', 'kubernetes', 'aws', 'docker']
        
        skill_lower = skill.lower()
        
        if skill_lower in beginner_skills:
            return 'beginner'
        elif skill_lower in intermediate_skills:
            return 'intermediate'
        elif skill_lower in advanced_skills:
            return 'advanced'
        else:
            return 'intermediate'  # Default
    
    def store_learning_resource(self, resource_text: str, metadata: Dict[str, Any]) -> str:
        """Store learning resource in vector database
        
        Args:
            resource_text (str): The learning resource text (description, title, etc.)
            metadata (Dict[str, Any]): Metadata about the learning resource
            
        Returns:
            str: Document ID of the stored learning resource
        """
        if not self.vector_db_manager:
            print("Warning: Vector database not available for storing learning resource")
            return ""  # Return empty string instead of None
            
        try:
            # Add type identifier for learning resources
            metadata['type'] = 'learning_resource'
            # Store a small excerpt for RAG prompts later
            try:
                excerpt = resource_text.strip().replace('\n', ' ')
                metadata['excerpt'] = excerpt[:600]
            except Exception:
                metadata['excerpt'] = resource_text[:600]
            
            # Add learning resource to vector database as knowledge item
            doc_id = self.vector_db_manager.add_knowledge_item(resource_text, metadata)
            print(f"✅ Learning resource stored in vector database with ID: {doc_id}")
            return doc_id if doc_id else ""  # Ensure we return a string
        except Exception as e:
            print(f"Error storing learning resource in vector database: {e}")
            return ""  # Return empty string instead of None
    
    def search_learning_resources(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for learning resources using vector database
        
        Args:
            query_text (str): Query text to search for learning resources
            top_k (int): Number of learning resources to return
            
        Returns:
            List[Dict[str, Any]]: List of learning resources with scores and metadata
        """
        if not self.vector_db_manager:
            print("Warning: Vector database not available for searching learning resources")
            return []
            
        try:
            # Search for knowledge items (learning resources)
            results = self.vector_db_manager.search_knowledge(query_text, top_k)
            # Filter for learning resources
            learning_resource_results = [r for r in results if r.get('metadata', {}).get('type') == 'learning_resource']
            print(f"✅ Found {len(learning_resource_results)} learning resources")
            return learning_resource_results
        except Exception as e:
            print(f"Error searching learning resources in vector database: {e}")
            return []
    
    def enhance_skill_recommendations_with_vector_db(self, skill_gaps: List[str], industry: str) -> Dict[str, Any]:
        """Enhance skill recommendations by combining LLM results with vector database retrieval
        
        Args:
            skill_gaps (List[str]): List of skills the user needs to learn
            industry (str): Target industry for recommendations
            
        Returns:
            Dict[str, Any]: Enhanced recommendations combining LLM and vector database results
        """
        if not self.vector_db_manager:
            print("Warning: Vector database not available for enhancing skill recommendations")
            return {}
            
        try:
            # Search for professional skill gap analyses and learning resources
            enhanced_recommendations = {}
            
            for skill in skill_gaps:
                # Create query for professional learning resources
                query_text = f"best learning resources for {skill} in {industry}"
                
                # Search for professional resources
                resources = self.search_learning_resources(query_text, top_k=3)
                
                # Add professional resources to recommendations
                enhanced_recommendations[skill] = {
                    'professional_resources': resources,
                    'difficulty': self.get_skill_difficulty(skill)
                }
                
                # Also search for career transition examples if relevant
                transition_query = f"career transition to {skill}"
                transition_examples = self.search_learning_resources(transition_query, top_k=2)
                if transition_examples:
                    enhanced_recommendations[skill]['transition_examples'] = transition_examples
            
            print("✅ Skill recommendations enhanced with professional resources from vector database")
            return enhanced_recommendations
        except Exception as e:
            print(f"Error enhancing skill recommendations with vector database: {e}")
            return {}

# Global instance
skill_gap_analyzer = SkillGapAnalyzer()