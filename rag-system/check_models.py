#!/usr/bin/env python3
"""
Check available Gemini models
Run this to see what models you have access to
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

print("Configuring Gemini API...")
genai.configure(api_key=api_key)

print("\n📋 Available Models:")
print("=" * 70)

try:
    models = genai.list_models()
    
    available_models = []
    for model in models:
        # Check if model supports generateContent
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            print(f"✓ {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Input Tokens: {model.input_token_limit:,}")
            print()
    
    if not available_models:
        print("❌ No models available that support generateContent")
    else:
        print("\n" + "=" * 70)
        print(f"Total models available: {len(available_models)}")
        print("\n✅ Use one of these model names in your code:")
        for model in available_models:
            model_short = model.replace('models/', '')
            print(f"  - {model_short}")
            
except Exception as e:
    print(f"❌ Error listing models: {e}")
    print("\nMake sure your GEMINI_API_KEY is valid")
