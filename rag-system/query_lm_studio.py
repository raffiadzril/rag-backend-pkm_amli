import os
import json
import time
import requests
from dotenv import load_dotenv
# Import local embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ============================================================================
# INITIALIZE CHROMADB
# ============================================================================

db_path = './chroma_db'
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Initializing local embedding model ({embedding_model_name})...")
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

# Load ChromaDB (Assume successful indexing via store.py)
try:
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    print(f"[SUCCESS] ChromaDB loaded successfully with {vectordb._collection.count()} documents")
except Exception as e:
    print(f"[ERROR] Error loading ChromaDB from {db_path}: {e}")
    exit()

# ============================================================================
# INITIALIZE LM STUDIO API
# ============================================================================

# LM Studio API Configuration
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
LM_STUDIO_API_ENDPOINT = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
LM_STUDIO_MODELS_ENDPOINT = f"{LM_STUDIO_BASE_URL}/v1/models"

lm_studio_ready = False
available_models = []

def check_lm_studio_connection():
    """Check if LM Studio is running and get available chat models"""
    global lm_studio_ready, available_models
    try:
        response = requests.get(LM_STUDIO_MODELS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            all_models = models_data.get('data', [])
            
            # Filter for chat models (exclude embedding models)
            # Chat models are typically the main/loaded model
            chat_models = []
            for model in all_models:
                model_id = model.get('id', '')
                # Exclude embedding models and tokenizers
                if not any(x in model_id.lower() for x in ['embed', 'embedding', 'tokenizer', '-mlp', '-text-']):
                    chat_models.append(model_id)
            
            # If no chat models found, try using the first model from /v1/models
            if not chat_models and all_models:
                print(f"[WARNING] No traditional chat models found. Detected models: {[m.get('id') for m in all_models]}")
                print(f"   Note: Make sure a chat/instruction model is loaded in LM Studio")
                # Use first model anyway (it might still work)
                chat_models = [model.get('id') for model in all_models if model.get('id')]
            
            available_models = chat_models
            
            if available_models:
                lm_studio_ready = True
                print(f"[SUCCESS] LM Studio connected successfully")
                print(f"  Available chat models: {', '.join(available_models)}")
                return True
            else:
                print(f"[WARNING] LM Studio API responds but no chat models loaded")
                print(f"   Please load a chat model in LM Studio (e.g., Mistral, Llama, etc.)")
                return False
        else:
            print(f"[WARNING] LM Studio API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Cannot connect to LM Studio at {LM_STUDIO_BASE_URL}")
        print(f"   Make sure LM Studio is running and listening at {LM_STUDIO_BASE_URL}")
        return False
    except Exception as e:
        print(f"[WARNING] Error checking LM Studio connection: {e}")
        return False

# Initial connection check
check_lm_studio_connection()

# ============================================================================
# CHROMADB RAG SERVICE CLASS FOR LM STUDIO
# ============================================================================

class ChromaRAGServiceLMStudio:
    def __init__(self, vectordb, base_url=LM_STUDIO_BASE_URL):
        self.vectordb = vectordb
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"
        self.models_endpoint = f"{base_url}/v1/models"

    def get_available_models(self):
        """Fetch available chat models from LM Studio"""
        try:
            response = requests.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                all_models = models_data.get('data', [])
                
                # Filter for chat models (exclude embedding models)
                chat_models = []
                for model in all_models:
                    model_id = model.get('id', '')
                    # Exclude embedding models and tokenizers
                    if not any(x in model_id.lower() for x in ['embed', 'embedding', 'tokenizer', '-mlp', '-text-']):
                        chat_models.append(model_id)
                
                return chat_models if chat_models else [model.get('id') for model in all_models if model.get('id')]
            return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def search_relevant_docs(self, query: str, top_k: int = 10) -> list:
        """Search documents from ChromaDB"""
        try:
            results = self.vectordb.similarity_search(query, k=top_k)
            relevant_docs = [doc.page_content for doc in results]
            print(f"[INFO] Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
        except Exception as e:
            print(f"[ERROR] Error searching documents: {e}")
            return []

    def generate_menu_plan_with_chroma(self, user_input: dict, model_name: str = None) -> dict:
        """Generate menu plan using LM Studio"""

        if not lm_studio_ready:
            return {"status": "error", "message": "LM Studio API not available. Make sure LM Studio is running."}

        if model_name is None:
            if available_models:
                model_name = available_models[0]
            else:
                return {"status": "error", "message": "No models available in LM Studio"}

        # STEP 1: Retrieve Rules/AKG from ChromaDB (Reduced top_k to fit context)
        age_months = user_input.get('age_months', 6)
        allergies = user_input.get('allergies', [])

        rules_query = f"Aturan MPASI dan AKG angka kecukupan gizi untuk usia {age_months} bulan"
        konteks_aturan = self.search_relevant_docs(rules_query, top_k=5) # Reduced from 15 to 5

        # === STEP 2: Prepare Prompt (Shortened) ===
        formatted_aturan = "\n---\n".join(konteks_aturan) # Reduced separator length
        allergies_text = f" Alergi: {', '.join(allergies)}." if allergies else ""

        # Build more concise prompt with TKPI codes
        json_example = '''{
"breakfast": {"time":"06:00-07:00","menu_name":"..","ingredients":[{"nama":"..","kode_tkpi":"AR001","jumlah":"100g"}],"portion":"..","instructions":"..","nutrition":{"energy_kcal":0,"protein_g":0.0,"carbs_g":0.0,"fat_g":0.0}},
"morning_snack": {...}, "lunch": {...}, "afternoon_snack": {...}, "dinner": {...},
"daily_summary": {"total_energy_kcal":0,"total_protein_g":0.0,"total_carbs_g":0.0,"total_fat_g":0.0,"akg_compliance":".."},
"notes":[".."], "recommendations":[".."]
}'''

        prompt = f"""Perencana MPASI. Buat menu 1 hari untuk bayi {age_months} bulan.{allergies_text}

Aturan: {formatted_aturan}

Format JSON: {json_example}

Instruksi:
1. WAJIB gunakan HANYA bahan dengan KODE_TKPI valid dari file TKPI-2020.json.
2. Format ingredients: [{{"nama":"exact nama dari TKPI","kode_tkpi":"KODE exact","jumlah":"gram/ml"}}].
3. Menu original, sesuai usia {age_months} bulan.
4. Nutrisi pasti (ambil dari TKPI, bukan estimasi).
5. Hindari alergen: {', '.join(allergies) if allergies else 'tidak ada'}.
6. Hanya JSON valid, tanpa markdown.

Output JSON:"""

        print("STEP 3: Generating menu plan with LM Studio...")
        try:
            # CRITICAL CHECK: Verify LM Studio is still connected
            if not lm_studio_ready:
                return {"status": "error", "message": "LM Studio connection lost. Please check if LM Studio is still running."}

            print(f"[INFO] Sending prompt to LM Studio API ({len(prompt)} characters)...")
            print(f"[INFO] Prompt preview (first 300 chars):\n{prompt[:300]}...\n")

            # Call LM Studio API
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": False
            }

            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                error_msg = f"LM Studio API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"
                raise Exception(error_msg)

            response_data = response.json()

            # --- Process Valid Response ---
            print("  [INFO] Received response from LM Studio")
            menu_data_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not menu_data_text:
                raise ValueError("Empty response from LM Studio")

            # DEBUG: Print raw response
            print(f"\nDEBUG - Raw LLM Response ({len(menu_data_text)} characters):")
            print("  " + "="*70)
            print(f"  {menu_data_text[:800]}")
            if len(menu_data_text) > 800:
                print(f"  ... (truncated, total length: {len(menu_data_text)} chars)")
            print("  " + "="*70 + "\n")

            if menu_data_text.strip().startswith("```"):
                # Clean markdown code blocks
                print(f"Cleaning markdown code blocks...")
                menu_data_text = menu_data_text.strip().strip("```json").strip("```").strip()
                print(f"  [INFO] Cleaned response ({len(menu_data_text)} characters after cleaning)")

            print(f"Parsing JSON response...")
            menu_data = json.loads(menu_data_text)
            print(f"  [INFO] Successfully parsed JSON with keys: {list(menu_data.keys())}")

            return {
                "status": "success",
                "data": menu_data,
                "rag_info": {
                    "documents_retrieved": len(konteks_aturan),
                    "retrieval_source": "ChromaDB (MPASI rules)",
                    "generation_model": f"LM Studio - {model_name}",
                    "generation_engine": "LM Studio",
                    "baby_age": age_months
                }
            }
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error parsing JSON response from LM Studio: {e}")
            print(f"\n  [INFO] Raw response text ({len(menu_data_text)} characters):")
            print("  " + "="*70)
            print(f"  {menu_data_text[:1000]}")
            if len(menu_data_text) > 1000:
                print(f"  ... (truncated, total length: {len(menu_data_text)} chars)")
            print("  " + "="*70)
            return {"status": "error", "message": f"Error parsing LM Studio response as JSON: {str(e)}", "raw_response": menu_data_text}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": f"Cannot connect to LM Studio at {self.base_url}. Make sure LM Studio is running."}
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "LM Studio API request timed out. The model may be taking too long to respond."}
        except Exception as e:
            print(f"[ERROR] Error during LM Studio API call: {e}")
            return {"status": "error", "message": f"Error generating menu plan: {str(e)}"}

# ============================================================================
# INITIALIZE RAG SERVICE SINGLETON
# ============================================================================
_rag_service_instance = None

def get_chroma_rag_service_lm_studio(base_url=LM_STUDIO_BASE_URL) -> ChromaRAGServiceLMStudio:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = ChromaRAGServiceLMStudio(vectordb, base_url)
        print("\n✓ ChromaDB RAG Service (LM Studio) initialized successfully")
    return _rag_service_instance

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ChromaDB RAG Service with LM Studio Ready for MPASI Menu Planning")
    print("="*70)

    rag_service = get_chroma_rag_service_lm_studio()

    # Get available models
    models = rag_service.get_available_models()
    if models:
        print(f"\n[INFO] Available Models:")
        for model in models:
            print(f"  - {model}")

        print("\n[INFO] Example: Generate Menu Plan for 8-month-old with LM Studio")
        print("-" * 70)
        user_input_example = {
            "age_months": 8,
            "weight_kg": 8,
            "height_cm": 70,
            "allergies": ["kacang"],
            "residence": "Bekasi"
        }

        # Use first available model
        menu_plan_result = rag_service.generate_menu_plan_with_chroma(user_input_example, models[0])

        print(f"\n--- Result ---")
        print(f"Status: {menu_plan_result.get('status')}")
        if menu_plan_result.get('status') == 'success':
            print(f"[SUCCESS] Menu plan generated!")
            print(json.dumps(menu_plan_result.get('data', {}), indent=2, ensure_ascii=False))
        else:
            print(f"[ERROR] Error: {menu_plan_result.get('message')}")
    else:
        print("\n[WARNING] No models available in LM Studio or LM Studio is not running.")
        print("Make sure to:")
        print("1. Download and install LM Studio from https://lmstudio.ai  ")
        print("2. Load a model in LM Studio")
        print("3. Start the API server (usually on port 1234)")
