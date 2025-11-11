"""
Model Manager - Centralized model loading and caching
Handles initialization of all AI models used in the application
"""

import os
import torch
import json
import time
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
        """Load sentence transformer for embeddings"""
        from sentence_transformers import SentenceTransformer  # Import locally to avoid any scoping issues
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Check if model is already downloaded
        model_path = self.models_dir / "sentence_transformers" / f"models--{model_name.replace('/', '--')}"
        if self._is_model_downloaded("embedding_model") and self._check_if_model_downloaded(model_path):
            print("✅ Embedding model already downloaded, using cached version")
        else:
            print("📥 Downloading embedding model for the first time...")
        
        try:
            # Adjust settings based on memory mode
            if hasattr(self, 'low_memory_mode') and self.low_memory_mode:
                print("🔧 Using low-memory configuration for embedding model")
                self.models['embeddings'] = SentenceTransformer(
                    model_name,
                    cache_folder=str(self.models_dir / "sentence_transformers"),
                    device='cpu'  # Force CPU to reduce memory usage
                )
            else:
                # Handle meta tensor issue by using proper initialization
                try:
                    # First try loading with explicit device
                    self.models['embeddings'] = SentenceTransformer(
                        model_name,
                        cache_folder=str(self.models_dir / "sentence_transformers"),
                        device=self.device
                    )
                except Exception as device_e:
                    print(f"⚠️ Device-specific loading failed: {device_e}")
                    # If that fails, try loading on CPU first then moving
                    try:
                        model = SentenceTransformer(
                            model_name,
                            cache_folder=str(self.models_dir / "sentence_transformers"),
                            device='cpu'
                        )
                        # Only move to GPU if available and not in low memory mode
                        if self.device == "cuda" and torch.cuda.is_available() and not (hasattr(self, 'low_memory_mode') and self.low_memory_mode):
                            model = model.to(self.device)
                        self.models['embeddings'] = model
                    except Exception as cpu_e:
                        print(f"⚠️ CPU fallback also failed: {cpu_e}")
                        # Last resort: try minimal configuration
                        self.models['embeddings'] = SentenceTransformer(
                            model_name,
                            cache_folder=str(self.models_dir / "sentence_transformers"),
                            device='cpu'
                        )
            print(f"✅ Loaded embedding model: {model_name}")
            # Mark as downloaded if this is the first time
            if not self._is_model_downloaded("embedding_model"):
                self._mark_model_as_downloaded("embedding_model")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            try:
                # Try fallback with different settings
                print("🔄 Trying fallback embedding model configuration...")
                self.models['embeddings'] = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    cache_folder=str(self.models_dir / "sentence_transformers"),
                    device='cpu'  # Force CPU to reduce memory usage
                )
                print("✅ Loaded fallback embedding model with CPU-only mode")
                # Mark as downloaded if this is the first time
                if not self._is_model_downloaded("embedding_model"):
                    self._mark_model_as_downloaded("embedding_model")
            except Exception as fallback_e:
                print(f"❌ Failed to load fallback embedding model: {fallback_e}")
                # Use a simpler model that requires less memory
                try:
                    print("🔄 Trying ultra-lightweight embedding model...")
                    # Use a smaller model that requires less memory
                    self.models['embeddings'] = SentenceTransformer('all-MiniLM-L6-v2')
                    print("✅ Loaded ultra-lightweight embedding model")
                    # Mark as downloaded if this is the first time
                    if not self._is_model_downloaded("embedding_model"):
                        self._mark_model_as_downloaded("embedding_model")
                except Exception as ultra_fallback_e:
                    print(f"❌ Failed to load ultra-lightweight embedding model: {ultra_fallback_e}")
                    self.models['embeddings'] = None
                    print("⚠️ Embedding model not available. Some features may be limited.")
    
    def load_sentiment_model(self):
        """Load sentiment analysis model"""
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Check if model is already downloaded
        model_path = self.models_dir / "transformers" / f"models--{model_name.replace('/', '--')}"
        if self._is_model_downloaded("sentiment_model") and self._check_if_model_downloaded(model_path):
            print("✅ Sentiment model already downloaded, using cached version")
        else:
            print("📥 Downloading sentiment model for the first time...")
        
        try:
            # Suppress the expected warning about unused weights for RoBERTa models
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
                warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
                
                # Load model and tokenizer separately to avoid pipeline type issues
                sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(self.models_dir / "transformers")
                )
                sentiment_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.models_dir / "transformers")
                )
                
                # Create pipeline manually to avoid type errors
                self.models['sentiment'] = TextClassificationPipeline(
                    model=sentiment_model,
                    tokenizer=sentiment_tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                # Set return_all_scores attribute manually
                self.models['sentiment'].return_all_scores = True
                
            print(f"✅ Loaded sentiment model: {model_name}")
            # Mark as downloaded if this is the first time
            if not self._is_model_downloaded("sentiment_model"):
                self._mark_model_as_downloaded("sentiment_model")
        except Exception as e:
            print(f"❌ Failed to load sentiment model: {e}")
            # Fallback to simpler model
            try:
                # Load fallback model and tokenizer separately
                fallback_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                
                # Check if fallback model is already downloaded
                fallback_model_path = self.models_dir / "transformers" / f"models--{fallback_model_name.replace('/', '--')}"
                if self._is_model_downloaded("sentiment_model_fallback") and self._check_if_model_downloaded(fallback_model_path):
                    print("✅ Fallback sentiment model already downloaded, using cached version")
                else:
                    print("📥 Downloading fallback sentiment model for the first time...")
                
                fallback_model = AutoModelForSequenceClassification.from_pretrained(
                    fallback_model_name,
                    cache_dir=str(self.models_dir / "transformers")
                )
                fallback_tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model_name,
                    cache_dir=str(self.models_dir / "transformers")
                )
                
                # Create pipeline manually to avoid type errors
                self.models['sentiment'] = TextClassificationPipeline(
                    model=fallback_model,
                    tokenizer=fallback_tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                print("✅ Loaded fallback sentiment model")
                # Mark as downloaded if this is the first time
                if not self._is_model_downloaded("sentiment_model_fallback"):
                    self._mark_model_as_downloaded("sentiment_model_fallback")
            except Exception as fallback_e:
                print(f"❌ Failed to load fallback sentiment model: {fallback_e}")
                self.models['sentiment'] = None
    
    def load_ner_model(self):
        """Load spaCy NER model with proper error handling"""
        try:
            import spacy
            # Try to load the pre-trained model
            self.models['ner'] = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy NER model: en_core_web_sm")
        except OSError as ose:
            print("⚠️ spaCy model 'en_core_web_sm' not found.")
            print("Please install with: python -m spacy download en_core_web_sm")
            # Create a fallback blank model
            try:
                import spacy
                self.models['ner'] = spacy.blank("en")
                print("✅ Created fallback blank spaCy model")
            except Exception as fallback_error:
                print(f"❌ Failed to create fallback spaCy model: {fallback_error}")
                self.models['ner'] = None
        except ImportError as ie:
            print(f"❌ spaCy not installed: {ie}")
            print("Please install with: pip install spacy")
            self.models['ner'] = None
        except Exception as e:
            print(f"❌ Unexpected error loading spaCy NER model: {e}")
            self.models['ner'] = None
    
    def load_language_model(self):
        """Load Phi-3 Mini as a replacement for distilgpt2"""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        # Check if model is already downloaded
        model_path = self.models_dir / "transformers" / f"models--{model_name.replace('/', '--')}"
        if self._is_model_downloaded("language_model") and self._check_if_model_downloaded(model_path):
            print("✅ Language model already downloaded, using cached version")
        else:
            print("📥 Downloading language model for the first time...")
        
        try:
            # Use Phi-3 Mini instead of distilgpt2
            print(f"📦 Loading language model (replacing distilgpt2): {model_name}")
            
            # CPU-optimized settings (no GPU required)
            # Fix: Create model_kwargs as a dictionary with string values
            model_kwargs = {}
            model_kwargs["cache_dir"] = str(self.models_dir / "transformers")
            model_kwargs["torch_dtype"] = "float32"  # Store as string
            model_kwargs["low_cpu_mem_usage"] = True
            model_kwargs["device_map"] = "cpu"
            
            # Load tokenizer with Phi-3 specific settings
            self.tokenizers['lm'] = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.models_dir / "transformers"),
                padding_side="left",
                trust_remote_code=True  # Required for Phi-3
            )
            
            # Set pad token (needed for batch processing)
            if self.tokenizers['lm'].pad_token is None:
                self.tokenizers['lm'].pad_token = self.tokenizers['lm'].eos_token
            
            # Load model (no API key needed)
            # Fix: Pass parameters individually instead of using model_kwargs
            self.models['lm'] = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.models_dir / "transformers"),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            print(f"✅ Loaded language model: {model_name} (replaces distilgpt2)")
            # Mark as downloaded if this is the first time
            if not self._is_model_downloaded("language_model"):
                self._mark_model_as_downloaded("language_model")
        except Exception as e:
            print(f"⚠️ Failed to load Phi-3 Mini: {e}")
            print("🔄 Falling back to distilgpt2...")
            # Fallback to original distilgpt2
            self._load_distilgpt2_fallback()
    
    def _load_distilgpt2_fallback(self):
        """Original distilgpt2 loading as fallback"""
        model_name = "distilgpt2"
        
        # Check if model is already downloaded
        model_path = self.models_dir / "transformers" / f"models--{model_name.replace('/', '--')}"
        if self._is_model_downloaded("distilgpt2_fallback") and self._check_if_model_downloaded(model_path):
            print("✅ DistilGPT2 model already downloaded, using cached version")
        else:
            print("📥 Downloading DistilGPT2 model for the first time...")
        
        try:
            # Fix: Pass parameters individually instead of using model_kwargs dictionary
            cache_dir = str(self.models_dir / "transformers")
            
            if hasattr(self, 'low_memory_mode') and self.low_memory_mode:
                torch_dtype = torch.float32
                low_cpu_mem_usage = True
            else:
                torch_dtype = None
                low_cpu_mem_usage = False
            
            self.tokenizers['lm'] = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            # Fix: Pass parameters individually
            if hasattr(self, 'low_memory_mode') and self.low_memory_mode:
                self.models['lm'] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage
                )
            else:
                self.models['lm'] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            print(f"✅ Loaded fallback language model: {model_name}")
            # Mark as downloaded if this is the first time
            if not self._is_model_downloaded("distilgpt2_fallback"):
                self._mark_model_as_downloaded("distilgpt2_fallback")
        except Exception as e:
            print(f"⚠️ Failed to load distilgpt2: {e}")
            print("Text generation will use simple responses")
            self.models['lm'] = None
            self.tokenizers['lm'] = None
    
    def get_embeddings(self, texts):
        """Get embeddings for text(s)"""
        # Check if model is available
        if not self.models.get('embeddings'):
            print("⚠️ Embedding model not available, returning zero vectors")
            import numpy as np
            if isinstance(texts, str):
                return np.zeros(384)  # MiniLM-L6-v2 embedding dimension
            else:
                return [np.zeros(384) for _ in texts]
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Process in smaller batches to reduce memory usage
            if len(texts) > 10:
                # Process in chunks of 10
                embeddings = []
                for i in range(0, len(texts), 10):
                    chunk = texts[i:i+10]
                    chunk_embeddings = self.models['embeddings'].encode(chunk, batch_size=5, show_progress_bar=False)
                    embeddings.extend(chunk_embeddings)
                return embeddings
            else:
                return self.models['embeddings'].encode(texts, batch_size=min(5, len(texts)), show_progress_bar=False)
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            # Return zero vectors as fallback
            import numpy as np
            if isinstance(texts, str):
                return np.zeros(384)
            else:
                return [np.zeros(384) for _ in texts]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        try:
            # Truncate text to prevent tensor size issues
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.models['sentiment'](text)
            if isinstance(results[0], list):
                # Handle models that return all scores
                scores = {item['label']: item['score'] for item in results[0]}
                return scores
            else:
                # Handle models that return single prediction
                return {results[0]['label']: results[0]['score']}
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return default sentiment scores
            return {"NEUTRAL": 1.0}
    
    def extract_entities(self, text):
        """Extract named entities from text with error handling"""
        if not self.models.get('ner'):
            print("⚠️ NER model not available")
            return []
        
        try:
            doc = self.models['ner'](text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            return entities
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
    
    def generate_text(self, prompt, max_length=100):
        """Generate text using Phi-3 Mini (same interface as distilgpt2)"""
        try:
            if self.models.get('lm') and self.tokenizers.get('lm'):
                # Check if we're using Phi-3 Mini
                if (hasattr(self.models['lm'].config, 'model_type') and 
                    'phi' in self.models['lm'].config.model_type.lower()):
                    # Phi-3 Mini specific prompt formatting
                    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
                else:
                    # Fallback to distilgpt2 format
                    formatted_prompt = prompt
                
                # Tokenize with attention mask
                inputs = self.tokenizers['lm'](
                    formatted_prompt, 
                    return_tensors="pt", 
                    max_length=384, 
                    truncation=True,
                    padding=True
                )
                
                # CPU-optimized generation settings
                generation_kwargs = {
                    "max_new_tokens": min(max_length, 40),  # Balanced for CPU
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                }
                
                # Move to CPU
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                print("🔄 Generating response with Phi-3 Mini (local inference)...")
                
                # Generate text
                with torch.no_grad():
                    outputs = self.models['lm'].generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs
                    )
                
                # Decode response
                response = self.tokenizers['lm'].decode(outputs[0], skip_special_tokens=True)
                
                # Post-process Phi-3 response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                
                # Clean up special tokens
                response = response.replace("<|end|>", "").strip()
                
                # Fallback if empty
                if not response:
                    response = "I understand your question. Based on my analysis, I recommend focusing on key areas that align with your career goals."
                
                return response
            else:
                # Original fallback response
                return "I understand your question. While I can't generate a detailed response right now, I can help you with interview preparation, resume tips, and career guidance."
                
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