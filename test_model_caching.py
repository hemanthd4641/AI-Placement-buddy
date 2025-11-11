"""
Test script to verify model caching functionality
"""

import time
from utils.model_manager import ModelManager

def test_model_caching():
    """Test that models are properly cached and not reloaded unnecessarily"""
    print("🧪 Testing Model Caching Functionality")
    print("=" * 50)
    
    # First initialization - should download/load models
    print("🔄 First initialization (may take time if downloading)...")
    start_time = time.time()
    model_manager1 = ModelManager()
    first_init_time = time.time() - start_time
    print(f"✅ First initialization completed in {first_init_time:.2f} seconds")
    
    # Second initialization - should use cached models
    print("\n🔄 Second initialization (should be fast)...")
    start_time = time.time()
    model_manager2 = ModelManager()
    second_init_time = time.time() - start_time
    print(f"✅ Second initialization completed in {second_init_time:.2f} seconds")
    
    # Verify they're the same instance (singleton pattern)
    print(f"\n🔍 Singleton check: {model_manager1 is model_manager2}")
    
    # Test model functionality
    print("\n🧪 Testing model functionality...")
    
    # Test embedding model
    try:
        embeddings = model_manager1.get_embeddings(["This is a test sentence"])
        print(f"✅ Embedding model working - generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Error with embedding model: {e}")
    
    # Test sentiment model
    try:
        sentiment = model_manager1.analyze_sentiment("This is a great test!")
        print(f"✅ Sentiment model working - result: {sentiment}")
    except Exception as e:
        print(f"❌ Error with sentiment model: {e}")
    
    # Test text generation (if available)
    try:
        if model_manager1.models.get('lm'):
            response = model_manager1.generate_text("Hello, how are you?")
            print(f"✅ Language model working - response length: {len(response)} characters")
        else:
            print("⚠️ Language model not available")
    except Exception as e:
        print(f"❌ Error with language model: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Model caching test completed!")
    
    if second_init_time < first_init_time * 0.5:
        print("✅ Caching is working effectively - second initialization was significantly faster")
    else:
        print("⚠️ Caching may not be working optimally - second initialization was not much faster")

if __name__ == "__main__":
    test_model_caching()