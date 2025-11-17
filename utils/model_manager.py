"""
Model Manager - Centralized model loading and caching
Handles initialization of all AI models used in the application
"""

import os
import torch
import json
import time
import requests
from dotenv import load_dotenv  # Add this import for loading .env files
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, AutoModelForCausalLM, TextClassificationPipeline
)
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class ModelManager:
    """Manages loading and caching of all AI models"""
    
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
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Cache file for model loading status
        self.cache_file = self.models_dir / "model_cache.json"
        self.model_cache = self._load_cache()
        
        # Check available memory and adjust settings
        self._check_system_resources()
        
        print(f"🔧 Initializing ModelManager on device: {self.device}")
        self._load_all_models()
        
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
    
    def _check_system_resources(self):
        """Check system resources and adjust model loading strategy"""
        try:
            import psutil
            # Get available memory in GB
            available_memory = psutil.virtual_memory().available / (1024**3)
            print(f"💾 Available system memory: {available_memory:.2f} GB")
            
            # If less than 4GB available, use memory-saving settings
            if available_memory < 4:
                print("⚠️ Low memory detected, enabling memory-saving mode")
                self.low_memory_mode = True
                # Force CPU usage to reduce memory pressure
                self.device = "cpu"
            else:
                self.low_memory_mode = False
        except ImportError:
            # psutil not available, use default settings
            print("⚠️ psutil not available, using default memory settings")
            self.low_memory_mode = False
    
    def _load_all_models(self):
        """Load all required models with error handling"""
        try:
            # 1. Load embedding model for RAG (highest priority)
            print("📦 Loading embedding model...")
            self.load_embedding_model()
            
            # 2. Load sentiment analysis model
            print("📦 Loading sentiment model...")
            self.load_sentiment_model()
            
            # 3. Load NER model (spaCy)
            print("📦 Loading NER model...")
            self.load_ner_model()
            
            # 4. Load LLM for text generation (optional)
            print("📦 Loading language model...")
            self.load_language_model()
            
            print("✅ Essential models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Some models failed to load: {str(e)}")
            print("The application will continue with available models.")
    
    def load_embedding_model(self):
        """Configure API access for embedding model instead of local download"""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # For API-based approach, we don't need to check if model is downloaded locally
        print(f"🔧 Configuring API access for embedding model: {model_name}")
        
        # Store model name for API calls
        self.embedding_model_name = model_name
        
        # Initialize with None since we're not loading locally
        self.models['embeddings'] = None
        
        print(f"✅ Configured API access for embedding model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("embedding_model"):
            self._mark_model_as_downloaded("embedding_model")
    
    def load_sentiment_model(self):
        """Configure API access for sentiment analysis model instead of local download"""
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # For API-based approach, we don't need to check if model is downloaded locally
        print(f"🔧 Configuring API access for sentiment model: {model_name}")
        
        # Store model name for API calls
        self.sentiment_model_name = model_name
        
        # Initialize with None since we're not loading locally
        self.models['sentiment'] = None
        
        print(f"✅ Configured API access for sentiment model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("sentiment_model"):
            self._mark_model_as_downloaded("sentiment_model")
    
    def load_ner_model(self):
        """Configure API access for NER model instead of local download"""
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        
        # For API-based approach, we don't need to check if model is downloaded locally
        print(f"🔧 Configuring API access for NER model: {model_name}")
        
        # Store model name for API calls
        self.ner_model_name = model_name
        
        # Initialize with None since we're not loading locally
        self.models['ner'] = None
        
        print(f"✅ Configured API access for NER model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("ner_model"):
            self._mark_model_as_downloaded("ner_model")
    
    def extract_entities(self, text):
        """Extract named entities from text using Hugging Face API"""
        try:
            # Check if API key is available
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                print("❌ Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable.")
                return []
            
            # Use API-based NER
            API_URL = f"https://router.huggingface.co/hf-inference/models/{self.ner_model_name}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            payload = {
                "inputs": text
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Process API response
                if isinstance(result, list) and len(result) > 0:
                    entities = []
                    for item in result:
                        entities.append({
                            'text': item.get('word', ''),
                            'label': item.get('entity_group', item.get('entity', '')),
                            'start': item.get('start', 0),
                            'end': item.get('end', 0)
                        })
                    return entities
                else:
                    return []
            else:
                return []
                
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
    
    def load_language_model(self):
        """Load Phi-3 Mini via Hugging Face API instead of local download"""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        # For API-based approach, we don't need to check if model is downloaded locally
        print(f"🔧 Configuring API access for language model: {model_name}")
        
        # Store model name for API calls
        self.language_model_name = model_name
        
        # Initialize with None since we're not loading locally
        self.models['lm'] = None
        self.tokenizers['lm'] = None
        
        print(f"✅ Configured API access for language model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("language_model"):
            self._mark_model_as_downloaded("language_model")
    
    def _load_distilgpt2_fallback(self):
        """API-based fallback using Hugging Face inference API"""
        model_name = "distilgpt2"
        
        # For API-based approach, we don't need to check if model is downloaded locally
        print(f"🔧 Configuring API access for fallback model: {model_name}")
        
        # Store model name for API calls
        self.fallback_model_name = model_name
        
        print(f"✅ Configured API access for fallback model: {model_name}")
        
        # Mark as "downloaded" to prevent repeated configuration
        if not self._is_model_downloaded("distilgpt2_fallback"):
            self._mark_model_as_downloaded("distilgpt2_fallback")
    
    def get_embeddings(self, texts):
        """Get embeddings for text(s) using Hugging Face API"""
        try:
            # Check if API key is available
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                print("❌ Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable.")
                # Return zero vectors as fallback
                import numpy as np
                if isinstance(texts, str):
                    return np.zeros(384)  # MiniLM-L6-v2 embedding dimension
                else:
                    return [np.zeros(384) for _ in texts]
            
            # Use API-based embedding generation
            API_URL = f"https://router.huggingface.co/hf-inference/models/{self.embedding_model_name}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Process texts in batches to avoid API limits
            embeddings = []
            for text in texts:
                payload = {
                    "inputs": text
                }
                
                response = requests.post(API_URL, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract embedding from response
                    if isinstance(result, list) and len(result) > 0:
                        embedding = result[0]  # Assuming the API returns the embedding directly
                        embeddings.append(embedding)
                    else:
                        # Fallback to zero vector
                        import numpy as np
                        embeddings.append(np.zeros(384))
                else:
                    # Fallback to zero vector on API error
                    import numpy as np
                    embeddings.append(np.zeros(384))
            
            # Return single embedding if input was single text, otherwise return list
            if len(embeddings) == 1:
                return embeddings[0]
            else:
                return embeddings
                
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            # Return zero vectors as fallback
            import numpy as np
            if isinstance(texts, str):
                return np.zeros(384)
            else:
                return [np.zeros(384) for _ in texts]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using Hugging Face API"""
        try:
            # Check if API key is available
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                print("❌ Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable.")
                # Return default sentiment scores
                return {"NEUTRAL": 1.0}
            
            # Use API-based sentiment analysis
            API_URL = f"https://router.huggingface.co/hf-inference/models/{self.sentiment_model_name}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Truncate text to prevent API limits
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            payload = {
                "inputs": text
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Process API response
                if isinstance(result, list) and len(result) > 0:
                    # Handle models that return all scores
                    if isinstance(result[0], list):
                        scores = {item['label']: item['score'] for item in result[0]}
                        return scores
                    else:
                        # Handle models that return single prediction
                        return {result[0]['label']: result[0]['score']}
                else:
                    # Return default sentiment scores
                    return {"NEUTRAL": 1.0}
            else:
                # Return default sentiment scores on API error
                return {"NEUTRAL": 1.0}
                
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return default sentiment scores
            return {"NEUTRAL": 1.0}
    
    def generate_text(self, prompt, max_length=100):
        """Generate text using Hugging Face API instead of local Phi-3 Mini"""
        try:
            # Check if API key is available
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                return "❌ Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable."
            
            # Use API-based generation
            print("🔄 Generating response with Phi-3 Mini (API inference)...")
            
            # Format prompt for Phi-3 Mini
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            # Prepare API request with the correct endpoint
            API_URL = f"https://router.huggingface.co/hf-inference/models/{self.language_model_name}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # API payload
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": min(max_length, 200),  # Limit for API
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "do_sample": True
                }
            }
            
            # Make API request
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    
                    # Post-process Phi-3 response
                    if "<|assistant|>" in generated_text:
                        response_text = generated_text.split("<|assistant|>")[-1].strip()
                    else:
                        response_text = generated_text
                    
                    # Clean up special tokens
                    response_text = response_text.replace("<|end|>", "").strip()
                    
                    # Fallback if empty
                    if not response_text:
                        response_text = "I understand your question. Based on my analysis, I recommend focusing on key areas that align with your career goals."
                    
                    return response_text
                else:
                    return "❌ Unexpected API response format."
            else:
                # Try to extract error message from response
                try:
                    error_msg = response.json().get("error", "Unknown API error")
                except Exception:
                    # If JSON parsing fails, use the raw response text or status code
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}" if response.text else f"HTTP {response.status_code}"
                return f"❌ API Error: {error_msg}"
                
        except Exception as e:
            print(f"Error in text generation: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try asking about specific topics like interview preparation or resume advice."
    
    def get_model(self, model_name):
        """Get a specific model"""
        return self.models.get(model_name)
    
    def get_tokenizer(self, tokenizer_name):
        """Get a specific tokenizer"""
        return self.tokenizers.get(tokenizer_name)
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Also try to garbage collect
            import gc
            gc.collect()
            print("🧹 Model cache cleared and memory freed")
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        print("🗑️ Unloading models to free memory...")
        self.models.clear()
        self.tokenizers.clear()
        self.clear_cache()
        print("✅ Models unloaded")