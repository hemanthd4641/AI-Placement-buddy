"""
Model Manager - Centralized model loading and caching
Now using Google Gemini API instead of Hugging Face models
"""

import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import Google Generative AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai not available. Please install with 'pip install google-generativeai'")
    genai = None
    GENAI_AVAILABLE = False

class ModelManager:
    """Manages Google Gemini API integration"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Prevent re-initialization
        if ModelManager._initialized:
            return
            
        self.models = {}
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Cache file for model loading status
        self.cache_file = self.models_dir / "model_cache.json"
        self.model_cache = self._load_cache()
        
        print(f"🔧 Initializing ModelManager with Google Gemini API")
        self._initialize_gemini()
        
        # Mark as initialized
        ModelManager._initialized = True
    
    def _load_cache(self):
        """Load model cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading model cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save model cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.model_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving model cache: {e}")
    
    def _check_if_model_downloaded(self, model_path):
        """Check if model files exist locally"""
        if not model_path.exists():
            return False
        # Check if the directory contains model files
        model_files = list(model_path.iterdir())
        return len(model_files) > 0
    
    def _mark_model_as_downloaded(self, model_name):
        """Mark model as downloaded in cache"""
        self.model_cache[model_name] = {
            "downloaded": True,
            "download_time": time.time(),
            "version": "1.0"
        }
        self._save_cache()
    
    def _is_model_downloaded(self, model_name):
        """Check if model is already downloaded"""
        return self.model_cache.get(model_name, {}).get("downloaded", False)
    
    def _initialize_gemini(self):
        """Initialize Google Gemini API"""
        try:
            # Check if Gemini library is available
            if not GENAI_AVAILABLE or genai is None:
                raise Exception("google-generativeai library not installed. Please install with 'pip install google-generativeai'")
            
            # Get API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("❌ GEMINI_API_KEY not found. Please set GEMINI_API_KEY in your .env file")
            
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize Gemini model for text generation (using latest model name)
            # Updated model name for v1 API
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Store configuration
            self.models['gemini'] = self.gemini_model
            
            print("✅ Google Gemini API initialized successfully!")
            print(f"✅ Using model: gemini-2.5-flash")
            
            # Mark as configured
            if not self._is_model_downloaded("gemini"):
                self._mark_model_as_downloaded("gemini")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Gemini API: {str(e)}")
            print("The application will have limited AI capabilities.")
            self.gemini_model = None
    
    def extract_entities(self, text):
        """Extract named entities using Gemini API"""
        try:
            if not self.gemini_model:
                print("❌ Gemini API not initialized")
                return []
            
            prompt = f"""Extract all named entities (people, organizations, locations, etc.) from the following text.
Return the entities as a JSON array with fields: text, label (type of entity), start (approximate position), end (approximate position).

Text: {text[:1000]}

Return only the JSON array, no other text."""
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                return entities
            return []
                
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
        
        print(f"✅ Configured API access for language model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("language_model"):
            self._mark_model_as_downloaded("language_model")
    
    def get_embeddings(self, texts):
        """Get text embeddings using Gemini embedding model"""
        try:
            if not GENAI_AVAILABLE:
                print("❌ Gemini API not available")
                # Return simple hash-based vectors as fallback
                if isinstance(texts, str):
                    return [hash(texts) % 1000 / 1000.0 for _ in range(768)]
                return [[hash(t) % 1000 / 1000.0 for _ in range(768)] for t in texts]
            
            # Use Gemini embedding model
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = []
            for text in texts:
                # Use Gemini's embedding model
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return embeddings[0] if len(embeddings) == 1 else embeddings
                
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            # Return simple hash-based vectors as fallback
            if isinstance(texts, str):
                return [hash(texts) % 1000 / 1000.0 for _ in range(768)]
            return [[hash(t) % 1000 / 1000.0 for _ in range(768)] for t in texts]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using Gemini API"""
        try:
            if not self.gemini_model:
                print("❌ Gemini API not initialized")
                return {"NEUTRAL": 1.0}
            
            prompt = f"""Analyze the sentiment of the following text and return ONLY a JSON object with sentiment labels (POSITIVE, NEGATIVE, NEUTRAL) and their scores (0-1).

Text: {text[:500]}

Return format: {{"POSITIVE": 0.X, "NEGATIVE": 0.Y, "NEUTRAL": 0.Z}}"""
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                sentiment_scores = json.loads(json_match.group())
                return sentiment_scores
            return {"NEUTRAL": 1.0}
                
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"NEUTRAL": 1.0}
    
    def generate_text(self, prompt, max_length=1000):
        """Generate text using Google Gemini API"""
        try:
            if not self.gemini_model:
                return "❌ Gemini API not initialized. Please check your GEMINI_API_KEY in .env file."
            
            print("🔄 Generating response with Google Gemini...")
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": max_length,
            }
            
            # Configure safety settings to be more permissive for professional/educational content
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Generate content
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract text from response
            # Check if response has candidates with valid content
            if response and hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                # Check if content parts exist
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            parts_text.append(part.text)
                    if parts_text:
                        return "".join(parts_text).strip()
            
            # Try direct text access as fallback
            try:
                if response and response.text:
                    return response.text.strip()
            except:
                pass
            
            # Check for safety blocks or other finish reasons
            if hasattr(response, 'prompt_feedback'):
                return f"Response blocked due to: {response.prompt_feedback}"
            
            return "I understand your question. Based on my analysis, I recommend focusing on key areas that align with your career goals."
                
        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def get_model(self, model_name):
        """Get a specific model"""
        return self.models.get(model_name)
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        try:
            # Garbage collect
            import gc
            gc.collect()
            print("🧹 Cache cleared and memory freed")
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        print("🗑️ Unloading models to free memory...")
        self.models.clear()
        self.gemini_model = None
        self.clear_cache()
        print("✅ Models unloaded")