#!/usr/bin/env python3
"""
Test script to check LM Studio connection and list available models
"""

import requests
import json
import sys
from typing import List, Tuple

# Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234"
MODELS_ENDPOINT = f"{LM_STUDIO_BASE_URL}/v1/models"

def test_connection() -> bool:
    """Test if LM Studio is running"""
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def get_all_models() -> List[dict]:
    """Get all models from LM Studio"""
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        return []
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return []

def categorize_models(models: List[dict]) -> Tuple[List[str], List[str], List[str]]:
    """Categorize models into chat, embedding, and other"""
    chat_models = []
    embedding_models = []
    other_models = []
    
    for model in models:
        model_id = model.get('id', '')
        
        if any(x in model_id.lower() for x in ['embed', 'embedding', 'tokenizer', '-mlp', '-text-']):
            embedding_models.append(model_id)
        elif any(x in model_id.lower() for x in ['chat', 'instruct', 'aiproxy', 'gpt', 'llama', 'mistral', 'neural', 'qwen']):
            chat_models.append(model_id)
        else:
            other_models.append(model_id)
    
    return chat_models, embedding_models, other_models

def main():
    print("=" * 70)
    print("🤖 LM Studio Connection Test")
    print("=" * 70)
    print(f"\n📍 Testing connection to: {LM_STUDIO_BASE_URL}\n")
    
    # Test connection
    if not test_connection():
        print("❌ LM Studio is NOT running or not accessible")
        print("\n📋 Quick Fix:")
        print("1. Open LM Studio application")
        print("2. Go to 'Local Server' tab")
        print("3. Click 'Start Server'")
        print("4. Wait for: 'Server is listening on http://0.0.0.0:1234'")
        print("5. Run this script again")
        sys.exit(1)
    
    print("✅ LM Studio is RUNNING\n")
    
    # Get all models
    models = get_all_models()
    
    if not models:
        print("❌ No models found in LM Studio")
        print("\n📋 Please load a model:")
        print("1. Go to 'Discover' tab in LM Studio")
        print("2. Search and download a model (e.g., Mistral 7B)")
        print("3. Go back to 'Local Server' tab")
        print("4. Select the model and click 'Start Server'")
        sys.exit(1)
    
    # Categorize models
    chat_models, embedding_models, other_models = categorize_models(models)
    
    print(f"📊 Total Models: {len(models)}\n")
    
    # Display chat models
    if chat_models:
        print(f"✅ CHAT MODELS ({len(chat_models)}):")
        for model in chat_models:
            print(f"   ✓ {model}")
        print()
    else:
        print("⚠️  NO CHAT MODELS FOUND\n")
    
    # Display embedding models
    if embedding_models:
        print(f"📚 EMBEDDING MODELS ({len(embedding_models)}):")
        for model in embedding_models:
            print(f"   • {model}")
        print()
    
    # Display other models
    if other_models:
        print(f"❓ OTHER MODELS ({len(other_models)}):")
        for model in other_models:
            print(f"   ? {model}")
        print()
    
    # Recommendations
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if not chat_models:
        print("""
❌ You need a chat/instruction model!

Recommended models for MPASI menu generation:
1. Mistral 7B Instruct (4GB) - ⭐ BEST - Fast & accurate
2. Neural Chat 7B (4GB) - Good quality
3. Llama 2 7B Chat (4GB) - Reliable
4. Phi 2 (1.5GB) - Fastest (CPU friendly)

To load a model:
1. Click "Discover" in LM Studio
2. Search for one of the models above
3. Click download
4. Go to "Local Server" tab
5. Select the model and click "Start Server"
6. Wait for "Server is listening..." message
""")
    else:
        print(f"""
✅ You have {len(chat_models)} chat model(s) ready!

To use with Flask app:
1. Make sure "Start Server" button in LM Studio is active
2. Run: python app.py
3. Open: http://localhost:5000
4. Select your model from the dropdown
5. Generate a menu!

Currently available models:
{chr(10).join(f'  • {m}' for m in chat_models)}
""")
    
    # Test a model if available
    if chat_models:
        print("\n" + "=" * 70)
        print("🧪 TESTING FIRST MODEL")
        print("=" * 70)
        
        test_model = chat_models[0]
        print(f"\n📤 Sending test request to: {test_model}\n")
        
        try:
            response = requests.post(
                f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
                json={
                    "model": test_model,
                    "messages": [
                        {"role": "user", "content": "Hello! What is 2+2?"}
                    ],
                    "max_tokens": 50
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
                print(f"✅ Model responded successfully!")
                print(f"📝 Response: {reply[:100]}...")
            else:
                print(f"❌ API error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"⏱️  Request timeout - model may be loading or under heavy load")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
