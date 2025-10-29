import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Import traceback for better error logging if needed
# import traceback 

load_dotenv()

# ============================================================================
# INITIALIZE CHROMADB AND GEMINI
# ============================================================================

db_path = './chroma_db'
# Make sure this is the same model used in store.py
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2" 
print(f"Initializing local embedding model ({embedding_model_name})...")
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

# Load ChromaDB
print("Loading ChromaDB...")
try:
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    print(f"✓ ChromaDB loaded successfully with {vectordb._collection.count()} documents")
except Exception as e:
    print(f"✗ Error loading ChromaDB from {db_path}: {e}")
    print("  Ensure 'store.py' was run successfully and the database exists.")
    exit()

# Initialize Gemini API
api_key = os.getenv("GEMINI_API_KEY") # Ensure .env key is GEMINI_API_KEY
gemini_model = None
# --- CORRECTED MODEL NAME ---
# Use a valid and available model name, e.g., 'gemini-1.5-flash'
# 'gemini-2.5-flash' is likely not a valid name as of late 2025.
# If using paid tier, consider 'gemini-1.5-pro' for better results.
gemini_model_name = 'gemini-2.5-flash' 
# -----------------------------
if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(gemini_model_name)
        print(f"✓ Gemini API initialized successfully with {gemini_model_name} model")
    except Exception as e:
        print(f"⚠ Error initializing Gemini API: {e}")
        gemini_model = None # Ensure model is None if init fails
else:
    print("⚠ GEMINI_API_KEY not found in .env - query generation will not be available")

# ============================================================================
# UPLOAD TKPI FILE TO GEMINI (using File API)
# ============================================================================

tkpi_file_ref = None # Global variable to store the file reference

def upload_tkpi_to_gemini():
    """Upload TKPI-2020.json file to Gemini File API"""
    global tkpi_file_ref
    # Check if already uploaded in this session and ACTIVE
    if tkpi_file_ref and tkpi_file_ref.state == 2:  # State 2 is ACTIVE
         print("✓ TKPI file already uploaded and active.")
         return True
        
    try:
        # Construct path relative to this script file
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        tkpi_file_path = os.path.abspath(os.path.join(script_dir, "../dataset/TKPI-2020.json"))

        if not os.path.exists(tkpi_file_path):
            print(f"⚠ TKPI file not found at {tkpi_file_path}")
            return False

        print(f"📤 Uploading {tkpi_file_path} to Gemini File API...")
        # Use genai.upload_file directly
        tkpi_file_ref = genai.upload_file(path=tkpi_file_path) 
        print(f"✓ TKPI file uploaded. Name: {tkpi_file_ref.name}, State: {tkpi_file_ref.state}")
        
        # Wait for file to be processed and become ACTIVE (state = 2)
        print("  ⏳ Waiting for file to be processed...")
        max_retries = 30
        retry_count = 0
        while tkpi_file_ref.state != 2 and retry_count < max_retries:
            retry_count += 1
            time.sleep(2)
            tkpi_file_ref = genai.get_file(tkpi_file_ref.name)
            print(f"  Status: {tkpi_file_ref.state} (attempt {retry_count}/{max_retries})")
        
        if tkpi_file_ref.state == 2:
            print("✓ TKPI file is now ACTIVE and ready for use.")
        else:
            print(f"⚠ TKPI file processing timeout. State: {tkpi_file_ref.state}")
        
        return True
    except Exception as e:
        print(f"⚠ Error uploading TKPI file: {str(e)}")
        return False
    
# Upload TKPI file at initialization if Gemini is configured
if gemini_model:
    upload_tkpi_to_gemini()
else:
    print("⚠ Gemini API not configured - File upload skipped.")


# ============================================================================
# CHROMADB RAG SERVICE CLASS
# ============================================================================

class ChromaRAGService:
    """RAG Service using ChromaDB for retrieval and Gemini API for generation"""
    
    def __init__(self, vectorstore, gemini_model_instance=None):
        self.vectordb = vectorstore
        self.gemini_model = gemini_model_instance # Use the initialized model instance
        # Set default k lower, can be overridden in search
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 10}) 
    
    def search_relevant_docs(self, query: str, top_k: int = 10) -> list:
        """Search ChromaDB for relevant document contents"""
        try:
            # Use vectordb.similarity_search directly
            results = self.vectordb.similarity_search(query, k=top_k)
            relevant_docs = [doc.page_content for doc in results]
            print(f"✓ Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
        except Exception as e:
            print(f"✗ Error searching documents: {e}")
            return []
    
    def search_with_scores(self, query: str, top_k: int = 10) -> list:
        """Search ChromaDB for documents with similarity scores"""
        try:
            results = self.vectordb.similarity_search_with_score(query, k=top_k)
            print(f"✓ Retrieved {len(results)} documents with scores")
            # Results are list of (Document, score) tuples
            return [(doc.page_content, score) for doc, score in results] 
        except Exception as e:
            print(f"✗ Error searching documents with scores: {e}")
            return []
    
    def generate_menu_plan_with_chroma(self, user_input: dict) -> dict:
        """Generate meal plan using ChromaDB rules + Gemini File API TKPI"""
        
        if not self.gemini_model:
            return {"status": "error", "message": "Gemini API not configured."}
            
        global tkpi_file_ref # Access the global file reference
        if not tkpi_file_ref:
            print("  ⚠ TKPI file reference not available. Attempting upload...")
            if not upload_tkpi_to_gemini():
                 return {"status": "error", "message": "Failed to get TKPI file reference."}
        elif tkpi_file_ref.state != 'ACTIVE':
             # Optional: Handle non-active state if needed (e.g., wait or error)
             print(f"  ⚠ TKPI file state is {tkpi_file_ref.state}, generation might fail.")


        age_months = user_input.get('age_months', 6)
        # ... (get other user_input variables: weight_kg, etc.) ...
        allergies = user_input.get('allergies', [])
        
        # === STEP 1: Retrieve Rules ===
        print(f"\n📚 STEP 1: Retrieving MPASI rules and AKG for age {age_months} months...")
        rules_query = f"Aturan MPASI dan AKG angka kecukupan gizi untuk usia {age_months} bulan tekstur porsi frekuensi"
        print(f"  Query: '{rules_query}'")
        konteks_aturan = self.search_relevant_docs(rules_query, top_k=15) # Increase k for rules
        if not konteks_aturan:
             print("  ⚠ No specific rules found, trying broader query...")
             konteks_aturan = self.search_relevant_docs("Aturan MPASI AKG bayi", top_k=10)
        print(f"  ✓ Retrieved {len(konteks_aturan)} documents for rules/AKG.")
        
        # === STEP 2: Prepare Prompt (No separate ingredient retrieval needed) ===
        print(f"\n📝 STEP 2: Composing final prompt...")
        
        formatted_aturan = "\n\n---\n".join(konteks_aturan) # Simpler formatting
        
        allergies_text = f"\n- PENTING: Bayi alergi terhadap: {', '.join(allergies)}. Hindari bahan ini." if allergies else "\n- Tidak ada alergi yang dilaporkan."
        
        # --- CORRECTED PROMPT ---
        # Removed "BAGIAN B" text reference. 
        # Instruct LLM to use the ATTACHED FILE for TKPI data.
        prompt = f"""Kamu adalah AI perencana menu MPASI bayi yang SANGAT TELITI dan KREATIF.

INFORMASI BAYI:
- Usia: {age_months} bulan
- Berat Badan: {user_input.get('weight_kg', 'N/A')} kg
- Tinggi Badan: {user_input.get('height_cm', 'N/A')} cm
- Tempat Tinggal: {user_input.get('residence', 'Indonesia')}{allergies_text}

==============================================
KONTEKS ATURAN MPASI DAN AKG (WAJIB DIIKUTI):
==============================================
{formatted_aturan}

==============================================
DATA BAHAN MAKANAN (TKPI-2020):
==============================================
Gunakan data lengkap dari FILE JSON TKPI-2020 YANG DILAMPIRKAN (`TKPI-2020.json`) sebagai satu-satunya sumber informasi bahan makanan dan nilai gizinya.

==============================================
TUGAS ANDA: BUAT RENCANA MENU MPASI ORIGINAL UNTUK 1 HARI
==============================================

LANGKAH-LANGKAH WAJIB:
1. ANALISIS ATURAN: Pahami kebutuhan AKG (Energi, Protein, dll.), tekstur, porsi, dan frekuensi makan untuk usia {age_months} bulan dari KONTEKS ATURAN di atas.
2. PILIH BAHAN DARI FILE TKPI: CARI dan PILIH bahan makanan HANYA dari FILE `TKPI-2020.json` yang dilampirkan. Pilih bahan yang sesuai kebutuhan gizi dan aturan tekstur. WAJIB sertakan KODE TKPI (misal: "AR001") dan jumlah (gram) untuk setiap bahan. Hindari alergen {', '.join(allergies) if allergies else 'tidak ada'}.
3. BUAT MENU ORIGINAL: Buat kombinasi menu yang KREATIF dan bervariasi untuk sarapan, snack pagi, makan siang, snack sore, dan makan malam. JANGAN gunakan contoh menu yang mungkin ada di konteks aturan.
4. HITUNG NUTRISI PER MENU & HARIAN: Untuk setiap menu, HITUNG MANUAL total Energi (Kal), Protein (g), Lemak (g), Karbohidrat (g) berdasarkan nilai gizi dari FILE `TKPI-2020.json`. Tuliskan HANYA ANGKA HASIL perhitungan (misal: 150, bukan 100+50). Hitung juga total harian.
5. VALIDASI & FORMAT: Pastikan output adalah JSON VALID sesuai format contoh, semua bahan dari file TKPI, semua aturan diikuti, dan tidak ada informasi dari luar konteks atau file yang digunakan.

LARANGAN KETAT:
❌ JANGAN gunakan bahan APAPUN yang tidak ada di file `TKPI-2020.json` yang dilampirkan.
❌ JANGAN mengarang nilai gizi.
❌ JANGAN menyalin contoh menu.
❌ JANGAN menulis rumus di bagian nutrisi JSON.

FORMAT RESPONSE (JSON VALID - CONTOH):
{{
  "breakfast": {{ ... (sesuai contoh sebelumnya, pastikan menu_name unik) ... }},
  "morning_snack": {{ ... (sesuai contoh sebelumnya) ... }},
  "lunch": {{ ... (sesuai contoh sebelumnya) ... }},
  "afternoon_snack": {{ ... (sesuai contoh sebelumnya) ... }},
  "dinner": {{ ... (sesuai contoh sebelumnya) ... }},
  "daily_summary": {{
    "total_energy_kcal": ..., 
    "total_protein_g": ...,
    // ... (total lemak, karbo)
    "akg_requirement": "Ringkasan AKG dari konteks aturan untuk {age_months} bulan",
    "akg_compliance": "Evaluasi singkat apakah total harian memenuhi AKG"
  }},
  "notes": [
    "✓ Verifikasi: Semua bahan diambil dari file TKPI-2020.",
    "✓ Verifikasi: Aturan tekstur dan porsi dari konteks aturan diikuti.",
    "✓ Verifikasi: Nutrisi dihitung berdasarkan data file TKPI-2020."
  ]
  // ... (recommendations dan data_sources_verification bisa ditambahkan jika perlu) ...
}}

PENTING: OUTPUT HANYA JSON VALID, tidak ada teks pembuka atau penutup.
"""
        # -----------------------------

        # === STEP 3: Generate Content ===
        print("🔄 STEP 3: Generating menu plan with Gemini API...")
        try:
            # --- CORRECTED API CALL ---
            response = self.gemini_model.generate_content(
                # Pass the text prompt AND the uploaded file reference object
                contents=[prompt, tkpi_file_ref], 
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3 # Lower temperature for more deterministic results
                )
            )
            # ---------------------------

            # Check for empty response or blocks
            if not response.candidates:
                 generation_error_msg = f"Content generation failed or was blocked."
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     generation_error_msg += f" Reason: {response.prompt_feedback}"
                 raise ValueError(generation_error_msg)

            # Process valid JSON response
            menu_data_text = response.text
            # Clean potential markdown code block fences if present
            if menu_data_text.strip().startswith("```json"):
                menu_data_text = menu_data_text.strip()[7:-3].strip()
            elif menu_data_text.strip().startswith("```"):
                 menu_data_text = menu_data_text.strip()[3:-3].strip()

            menu_data = json.loads(menu_data_text) 
            
            # Add basic status and data structure for consistency
            return {
                "status": "success",
                "data": menu_data,
                "user_info": user_input, # Echo back user input
                "rag_info": {
                    "documents_retrieved_rules": len(konteks_aturan),
                    "rules_query": rules_query,
                    "ingredients_source": f"Uploaded File: {tkpi_file_ref.name} ({tkpi_file_ref.uri})",
                    "retrieval_method": "Rules (ChromaDB) + Ingredients (Gemini File API)"
                }
            }
        except json.JSONDecodeError as e:
            raw_response_text = response.text if 'response' in locals() else "Response object not available"
            print(f"✗ Error parsing JSON response from Gemini: {e}")
            print(f"--- Raw Gemini Response Start ---\n{raw_response_text}\n--- Raw Gemini Response End ---")
            return {"status": "error", "message": f"Invalid JSON response from LLM: {e}", "raw_response": raw_response_text}
        except Exception as e:
            print(f"✗ Error during Gemini API call or processing: {e}")
            # traceback.print_exc() # Uncomment for detailed error
            return {"status": "error", "message": f"Error generating menu plan: {str(e)}"}

# ============================================================================
# INITIALIZE RAG SERVICE SINGLETON & EXAMPLE USAGE
# ============================================================================
_rag_service_instance = None

def get_chroma_rag_service() -> ChromaRAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = ChromaRAGService(vectordb, gemini_model)
        print("\n✓ ChromaDB RAG Service initialized successfully")
    return _rag_service_instance

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ChromaDB RAG Service Ready for MPASI Menu Planning")
    print("="*70)
    
    rag_service = get_chroma_rag_service()
    
    # --- Example Menu Plan Generation ---
    if gemini_model and tkpi_file_ref: # Check if both API and file are ready
        print("\n🍽️ Example: Generate Menu Plan for 8-month-old")
        print("-" * 70)
        user_input_example = {
            "age_months": 8,
            "weight_kg": 8,
            "height_cm": 70,
            "allergies": ["kacang"], # Example allergy
            "residence": "Bekasi"
        }
        menu_plan_result = rag_service.generate_menu_plan_with_chroma(user_input_example)
        
        print(f"\n--- Result ---")
        print(f"Status: {menu_plan_result.get('status')}")
        if menu_plan_result.get('status') == 'success':
            print(f"✓ Menu plan generated!")
            # Pretty print the JSON data
            print(json.dumps(menu_plan_result.get('data', {}), indent=2, ensure_ascii=False)) 
            # Print RAG info
            print("\n--- RAG Info ---")
            print(json.dumps(menu_plan_result.get('rag_info', {}), indent=2))
        else:
            print(f"✗ Error: {menu_plan_result.get('message')}")
            if menu_plan_result.get('raw_response'):
                 print("\n--- Raw LLM Response (if available) ---")
                 print(menu_plan_result['raw_response'])
                 
    elif not gemini_model:
        print("\n⚠️ Skipping menu plan generation - Gemini API not configured.")
    elif not tkpi_file_ref:
         print("\n⚠️ Skipping menu plan generation - TKPI file upload failed or reference missing.")

    # --- Keep script running for interactive queries if needed ---
    # print("\n--- Interactive Query Mode (Type 'exit' to quit) ---")
    # while True:
    #    try:
    #        user_query = input("\nAsk about MPASI rules or ingredients: ")
    #        if user_query.lower() == 'exit':
    #            break
    #        docs = rag_service.search_relevant_docs(user_query, top_k=5)
    #        print("\nRelevant Information:")
    #        for i, doc in enumerate(docs, 1):
    #             print(f"{i}. {doc}\n---")
    #    except KeyboardInterrupt:
    #        break
    # print("\nExiting interactive mode.")