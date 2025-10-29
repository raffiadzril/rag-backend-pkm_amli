import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
# Import local embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import traceback # Added for detailed error traceback

load_dotenv()

# ============================================================================
# INITIALIZE CHROMADB AND GEMINI
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

# Initialize Gemini API
api_key = os.getenv("GOOGLE_API_KEY") # Use GOOGLE_API_KEY as primary name
gemini_model = None
# Use gemini-2.5-pro for stable performance
gemini_model_name = 'gemini-2.5-flash'

if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(gemini_model_name)
        print(f"[SUCCESS] Gemini API initialized successfully with {gemini_model_name} model")
    except Exception as e:
        print(f"[WARNING] Error initializing Gemini API: {e}. Check API Key format.")
        gemini_model = None
else:
    print("[WARNING] API Key not found - query generation will not be available")

# ============================================================================
# UPLOAD TKPI FILE LOGIC (CRITICAL FOR GROUNDING)
# ============================================================================

tkpi_file_ref = None # Global variable to store the file reference

def find_existing_tkpi_file():
    """Check Google API for an existing file named TKPI_COMPACT.txt"""
    tkpi_filename = "TKPI_COMPACT.txt" # Extract filename
    print(f"[INFO] Checking Google API for existing file: {tkpi_filename}")
    try:
        # List all uploaded files
        uploaded_files = genai.list_files()
        for file_info in uploaded_files:
            # Check if the display name matches the target filename
            if file_info.display_name == tkpi_filename:
                print(f"  [SUCCESS] Found existing file on API: {file_info.name} (State: {file_info.state}, URI: {file_info.uri})")
                return file_info # Return the file info object if found
        print(f"  - No existing file named '{tkpi_filename}' found on the API.")
        return None
    except Exception as e:
        print(f"  [ERROR] Error listing files from API: {e}")
        return None

def upload_tkpi_to_gemini():
    """Upload TKPI_COMPACT.txt file to Gemini File API and waits for state 2 (assumed ACTIVE).
    Checks first if a file reference already exists and is in state 2.
    Also checks the Google API for an existing file with the same name."""
    global tkpi_file_ref # <--- MOVED HERE: Declare global at the very beginning of the function
    # Assuming state 2 is the functional equivalent of ACTIVE
    TARGET_STATE = 2 # Or the specific state code Gemini reports when ready

    # Check if already uploaded and in the target state (FIRST check inside the function)
    if tkpi_file_ref and tkpi_file_ref.state == TARGET_STATE:
        print("[SUCCESS] TKPI file already uploaded (in variable) and ready (state 2) - skipping upload.")
        return True # Indicate success, no need to re-upload

    # Check the Google API for an existing file with the name TKPI_COMPACT.txt
    existing_file_info = find_existing_tkpi_file()
    if existing_file_info:
        # If found, check its state
        if existing_file_info.state == TARGET_STATE:
            print(f"  [SUCCESS] Using existing API file {existing_file_info.name} (state {TARGET_STATE}).")
            # Update the global variable to point to this existing file reference
            tkpi_file_ref = existing_file_info
            return True
        else:
            print(f"  [WARNING] Existing API file {existing_file_info.name} is not in target state {TARGET_STATE}. Current state: {existing_file_info.state}")
            # Optionally, you could try to get the latest state or wait, but for now, proceed to upload a new one.
            # Or, you could return False or handle this differently based on requirements.

    # If not found or not ready, proceed with upload
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tkpi_file_path = os.path.abspath(os.path.join(script_dir, "../dataset/TKPI_COMPACT.txt"))

        if not os.path.exists(tkpi_file_path):
            print(f"[WARNING] TKPI file not found at {tkpi_file_path}")
            return False

        print(f"[INFO] Uploading {tkpi_file_path} to Gemini File API...")
        tkpi_file_ref = genai.upload_file(path=tkpi_file_path) # This assigns to the global variable
        print(f"[SUCCESS] TKPI file uploaded. Name: {tkpi_file_ref.name}, State: {tkpi_file_ref.state}")

        # Wait for file to be processed and reach the target state (e.g., 2)
        print("  [WAIT] Waiting for file to be processed (waiting for state 2)...")
        max_retries = 60 # Increased from 30 to allow up to 5 minutes (60 * 5s)
        retry_count = 0

        while tkpi_file_ref.state != TARGET_STATE and retry_count < max_retries:
            retry_count += 1
            time.sleep(5) # Increased sleep time to 5 seconds
            tkpi_file_ref = genai.get_file(tkpi_file_ref.name) # This accesses the global variable
            # Use state name for printing
            print(f"  Status: {tkpi_file_ref.state} (attempt {retry_count}/{max_retries})")

        if tkpi_file_ref.state == TARGET_STATE:
            print(f"[SUCCESS] TKPI file is now ready (state 2) and can be used.")
        else:
            print(f"[WARNING] TKPI file processing timeout. Current State: {tkpi_file_ref.state}. Expected: {TARGET_STATE}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Error uploading TKPI file: {str(e)}")
        # Reset tkpi_file_ref on error
        tkpi_file_ref = None # This modifies the global variable
        return False

# Initial upload attempt
if gemini_model:
    upload_tkpi_to_gemini()
else:
    print("[ERROR] Gemini API not configured - File upload skipped.")


# ============================================================================
# CHROMADB RAG SERVICE CLASS (Implemented)
# ============================================================================

class ChromaRAGService:
    def __init__(self, vectordb, gemini_model):
        self.vectordb = vectordb
        self.gemini_model = gemini_model # Store the model instance

    def search_relevant_docs(self, query: str, top_k: int = 10) -> list:
        # Simplified similarity_search (using vectordb directly)
        try:
            results = self.vectordb.similarity_search(query, k=top_k)
            relevant_docs = [doc.page_content for doc in results]
            print(f"[SUCCESS] Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
        except Exception as e:
            print(f"[ERROR] Error searching documents: {e}")
            return []


    def generate_menu_plan_with_chroma(self, user_input: dict) -> dict:

        if not self.gemini_model:
            return {"status": "error", "message": "Gemini API not configured."}

        print(f"[DEBUG] USING MODEL FOR GENERATION: {self.gemini_model.model_name}")

        # Extract and validate parameters
        umur_bulan = user_input.get('umur_bulan', user_input.get('age_months', 6))  # Support both formats
        berat_badan = user_input.get('berat_badan', user_input.get('weight_kg', 7))
        tinggi_badan = user_input.get('tinggi_badan', user_input.get('height_cm', 65))
        jenis_kelamin = user_input.get('jenis_kelamin', 'laki-laki')  # Default to 'laki-laki' if not provided
        tempat_tinggal = user_input.get('tempat_tinggal', user_input.get('residence', 'Indonesia'))
        alergi = user_input.get('alergi', user_input.get('allergies', []))

        # Validate age range for MPASI (6-24 months)
        if umur_bulan < 6 or umur_bulan > 24:
            return {
                "status": "error", 
                "message": f"MPASI hanya dianjurkan untuk bayi usia 6-24 bulan. Usia yang dimasukkan: {umur_bulan} bulan."
            }

        global tkpi_file_ref

        # DEBUG: Print TKPI file reference status BEFORE the check
        print(f"\n[DEBUG] TKPI File Status BEFORE CHECK:")
        print(f"  tkpi_file_ref is None: {tkpi_file_ref is None}")
        if tkpi_file_ref:
            print(f"  tkpi_file_ref.name: {tkpi_file_ref.name}")
            print(f"  tkpi_file_ref.state: {tkpi_file_ref.state}")
            print(f"  tkpi_file_ref.uri: {tkpi_file_ref.uri if hasattr(tkpi_file_ref, 'uri') else 'N/A'}")
        else:
            print(f"  [WARNING] tkpi_file_ref is None!")
        print()

        # CRITICAL CHECK FOR FILE STATUS (NOW CHECKING FOR STATE 2)
        # This check will call upload_tkpi_to_gemini() ONLY if tkpi_file_ref is None or state is not 2
        if not tkpi_file_ref or tkpi_file_ref.state != 2: # Check specifically for state 2
             print("  [ERROR] TKPI file NOT in state 2 (or ref is None). Calling upload function...")
             if not upload_tkpi_to_gemini(): # This function now checks internally too
                  return {"status": "error", "message": "Failed to get TKPI file in state 2."}
        else:
            print("  [SUCCESS] TKPI file is already in state 2. No re-upload needed for this generation step.")

        

        # STEP 1: Retrieve Rules/AGK from ChromaDB (Context for generation)
        age_months = umur_bulan
        allergies = alergi

        # Include baby's information in the context query for more targeted information retrieval
        context_query_parts = [
            f"Aturan MPASI dan AKG angka kecukupan gizi untuk usia {age_months} bulan tekstur porsi frekuensi",
            f"MPASI untuk bayi {jenis_kelamin}",
            f"MPASI untuk bayi dengan berat badan {berat_badan} kg",
            f"MPASI untuk bayi dengan tinggi badan {tinggi_badan} cm"
        ]
        if tempat_tinggal:
            context_query_parts.append(f"MPASI lokal di {tempat_tinggal}")
        
        context_query = " ".join(context_query_parts)
        
        konteks_aturan = self.search_relevant_docs(context_query, top_k=10) # Retrieve rules/AGK

                # === STEP 2: Prepare Prompt (REVERTED TO ORIGINAL STRING-BASED INGREDIENTS) ===
        formatted_aturan = "\n\n---\n".join(konteks_aturan)
        allergies_text = f"\n- PENTING: Bayi alergi terhadap: {', '.join(allergies)}. Hindari bahan ini." if allergies else "\n- Tidak ada alergi yang dilaporkan."
        
        # --- JSON EXAMPLE ORIGINAL (string-based ingredients, NO inline objects) ---
        json_example_original = '''{
  "breakfast": {
    "time": "06:00-07:00",
    "menu_name": "nama menu original (BUKAN template)",
    "ingredients": [
      "Beras putih (AR001, 50g)",
      "Ayam dada (AY001, 30g)",
      "Wortel (SA002, 20g)"
    ],
    "portion": "150 ml / 120g",
    "instructions": [
      "Masak beras sampai lunak",
      "Rebus ayam potong kecil",
      "Campur dengan wortel yang sudah dihaluskan"
    ],
    "nutrition": {
      "energy_kcal": 145,
      "protein_g": 6.2,
      "carbs_g": 20.5,
      "fat_g": 2.8
    }
  },
  "morning_snack": {
    "time": "09:00-10:00",
    "menu_name": "nama menu original",
    "ingredients": [
      "Pisang matang (BH001, 50g)",
      "ASI/Formula (JR002, 100 ml)"
    ],
    "portion": "100 ml + 50g",
    "instructions": [
      "Haluskan pisang",
      "Campur dengan ASI/formula"
    ],
    "nutrition": {
      "energy_kcal": 65,
      "protein_g": 1.5,
      "carbs_g": 14.2,
      "fat_g": 0.3
    }
  },
  "daily_summary": {
    "total_energy_kcal": 470,
    "total_protein_g": 21.0,
    "total_carbs_g": 71.7,
    "total_fat_g": 8.0,
    "akg_requirement": "AKG energi {age_months}mo: 400-500 kcal, Protein: 10-13g (dari KONTEKS ATURAN)",
    "akg_compliance": "Evaluasi kepatuhan MKM/TID dan AKG harian"
  },
  "recommendation": "Pastikan variasi bahan makanan setiap hari untuk memenuhi kebutuhan mikronutrien. Jika asupan energi mendekati batas bawah AKG, pertimbangkan penambahan porsi lemak sehat seperti alpukat atau minyak sayur dalam batas aman."
}'''
        
        # Build prompt using the ORIGINAL format (string-based ingredients)
        prompt = f"""Kamu adalah AI perencana menu MPASI bayi yang SANGAT TELITI dan KREATIF.

INFORMASI BAYI:
- Usia: {age_months} bulan
- Berat Badan: {berat_badan} kg
- Tinggi Badan: {tinggi_badan} cm
- Jenis Kelamin: {jenis_kelamin}
- Tempat Tinggal: {tempat_tinggal}{allergies_text}

==============================================
KONTEKS ATURAN MPASI DAN AKG (WAJIB DIIKUTI DARI CHROMA DB)
==============================================
{formatted_aturan}

==============================================
DATA BAHAN MAKANAN (TKPI-COMPACT LINES):
==============================================
[!] FILE TKPI_COMPACT.txt telah DILAMPIRKAN. Gunakan data dari file ini sebagai SATU-SATUNYA sumber informasi bahan makanan (name, code, kcal, prot_g, fat_g, carb_g, iron_mg, bdd_percent).

==============================================
TUGAS ANDA: BUAT RENCANA MENU MPASI ORIGINAL UNTUK 1 HARI
==============================================

LANGKAH-LANGKAH WAJIB:
1. ANALISIS ATURAN: 
- Pastikan menu memenuhi syarat **ADEKUAT** dan **TEPAT WAKTU**.
- WAJIB memenuhi kriteria **MINIMUM KERAGAMAN MAKANAN (MKM)** dan **KONSUMSI TELUR, IKAN, DAGING (TID)**.
- WAJIB membatasi **GULA/GARAM** sesuai aturan dari KONTEKS ATURAN.

2. PILIH BAHAN DARI FILE TKPI: 
- GUNAKAN HANYA bahan yang ada di FILE `TKPI_COMPACT.txt` yang dilampirkan.
- WAJIB sertakan **Nama Bahan**, **KODE TKPI**, dan **Jumlah (gram/ml)** untuk setiap bahan dalam format *string* seperti contoh: `"Nama Bahan (KODE, jumlah)"`.
- Hindari bahan alergen: {', '.join(allergies) if allergies else 'tidak ada'}.

3. BUAT MENU ORIGINAL:
- Buat kombinasi menu yang **KREATIF** (BUKAN menyalin template).
- Pastikan **Tekstur** dan **Porsi** sesuai usia {age_months} bulan (dari KONTEKS ATURAN).

4. HITUNG NUTRISI:
- HITUNG MANUAL total nutrisi (kcal, prot_g, fat_g, carb_g) untuk setiap *meal* dan *total harian* berdasarkan nilai gizi dan **BDD (%)** dari FILE TKPI.
- Pastikan total harian **MEMENUHI AKG** dari KONTEKS ATURAN.

5. VALIDASI & FORMAT:
- Output HARUS JSON VALID sesuai format contoh di bawah.
- Semua nilai nutrisi harus **ANGKA** (number) tanpa rumus.

LARANGAN KETAT:
❌ JANGAN gunakan bahan APAPUN yang tidak ada di FILE `TKPI_COMPACT.txt`.
❌ JANGAN mengarang nilai gizi.
❌ JANGAN gunakan aturan yang tidak ada di KONTEKS ATURAN.
❌ JANGAN tulis rumus dalam nilai nutrisi JSON.
❌ JANGAN gunakan format objek untuk ingredients — gunakan STRING seperti contoh.

FORMAT RESPONSE (JSON VALID - COPY STRUKTUR INI PERSIS):
{json_example_original}
"""

        print("[INFO] STEP 3: Generating menu plan with Gemini API using ChromaDB context and TKPI file (state 2)...")
        try:
            # CORRECT API CALL: Send the prompt with the file reference
            # The LLM will use the prompt for rules and the attached file (TKPI) for ingredient lookup.
            print(f"  [INFO] Sending prompt to Gemini API ({len(prompt)} characters)...")
            print(f"  [INFO] Prompt preview (first 300 chars):\n{prompt[:300]}...\n")
            print(f"  [INFO] TKPI File: {tkpi_file_ref.name} (State: {tkpi_file_ref.state})")
            
            # DEBUG: Print the structure being sent to the API
            print(f"  [DEBUG] Contents being sent to API: [{type(prompt)}, {type(tkpi_file_ref)}]")
            print(f"  [DEBUG] Prompt type: {type(prompt)}, File ref type: {type(tkpi_file_ref)}")
            
            print("  [INFO] Calling Gemini API generate_content... (with JSON mode)")
            # Attempt the API call with JSON mode
            response = self.gemini_model.generate_content(
                [prompt, tkpi_file_ref],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                )
            )
            print("  [SUCCESS] Gemini API call (JSON mode) completed successfully.")

            # --- Error Checking for Response ---
            print("  [INFO] Checking response candidates...")
            if not response.candidates:
                 print("  [ERROR] Response has no candidates!")
                 generation_error_msg = f"Content generation failed or was blocked."
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     generation_error_msg += f" Reason: {response.prompt_feedback}"
                 raise ValueError(generation_error_msg)
            else:
                print("  ✅ Response has candidates.")

            # --- Process Valid Response ---
            print("  [SUCCESS] Received response from Gemini API")
            menu_data_text = response.text
            
            # DEBUG: Print raw response
            print("  [DEBUG] Raw LLM Response ({len(menu_data_text)} characters):")
            print("  " + "="*70)
            print(f"  {menu_data_text[:800]}")
            if len(menu_data_text) > 800:
                print(f"  ... (truncated, total length: {len(menu_data_text)} chars)")
            print("  " + "="*70 + "\n")
            
            if menu_data_text.strip().startswith("```"):
                # Simplified cleaning logic for Python compatibility
                print("  [INFO] Cleaning markdown code blocks...")
                menu_data_text = menu_data_text.strip().strip("```json").strip("```").strip()
                print(f"  [SUCCESS] Cleaned response ({len(menu_data_text)} characters after cleaning)")

            print(f"  [INFO] Parsing JSON response...")
            menu_data = json.loads(menu_data_text)
            print(f"  [SUCCESS] Successfully parsed JSON with keys: {list(menu_data.keys())}")

            return {
                "status": "success",
                "data": menu_data,
                "rag_info": {
                    "documents_retrieved": len(konteks_aturan),
                    "retrieval_source": "ChromaDB (MPASI rules) + Gemini File API (TKPI data)",
                    "generation_model": "gemini-2.5-flash",
                    "baby_age": age_months
                },
                "debug_info": {
                    "prompt": prompt,
                    "prompt_length": len(prompt),
                    "search_query": context_query
                }
            }
        except Exception as e:
            # Log the specific error and traceback here
            print(f"\n[ERROR] ERROR OCCURRED DURING GEMINI API CALL OR PROCESSING:")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error Message: {str(e)}")
            print(f"  Full Traceback:\n{traceback.format_exc()}")
            print(f"  Attempting to get raw response for debugging...")
            
            # Attempt to get raw response for debugging if available
            raw_response_text = getattr(response, 'text', 'N/A') if 'response' in locals() else 'Response object not available'
            print(f"  Last response received (or N/A): {raw_response_text[:200] if raw_response_text != 'N/A' else 'N/A'}")
            
            # --- FIX: Use safe string formatting to avoid 'Invalid format specifier' ---
            # Get the original error string
            original_error_str = str(e)
            # Construct the message using .format() which is safer for untrusted strings
            error_message = "Error during Gemini API call or processing: {}".format(original_error_str)
            # Return error message to the frontend
            return {"status": "error", "message": error_message, "raw_response": raw_response_text}


# ============================================================================
# INITIALIZE RAG SERVICE SINGLETON & EXAMPLE USAGE
# ============================================================================
_rag_service_instance = None

def get_chroma_rag_service() -> ChromaRAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = ChromaRAGService(vectordb, gemini_model)
        print("\n[SUCCESS] ChromaDB RAG Service initialized successfully")
    return _rag_service_instance

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ChromaDB RAG Service Ready for MPASI Menu Planning")
    print("="*70)

    rag_service = get_chroma_rag_service()

    # --- Example Menu Plan Generation ---
    if gemini_model and tkpi_file_ref: # Check if both API and file are ready
        print("\n[INFO] Example: Generate Menu Plan for 8-month-old")
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
            print(f"[SUCCESS] Menu plan generated!")
            # Pretty print the JSON data
            print(json.dumps(menu_plan_result.get('data', {}), indent=2, ensure_ascii=False))
            # Print RAG info (if included)
            # print("\n--- RAG Info ---")
            # print(json.dumps(menu_plan_result.get('rag_info', {}), indent=2))
        else:
            print(f"✗ Error: {menu_plan_result.get('message')}")
            if menu_plan_result.get('raw_response'):
                 print("\n--- Raw LLM Response (if available) ---")
                 print(menu_plan_result['raw_response'])

    elif not gemini_model:
        print("\n[ERROR] Skipping menu plan generation - Gemini API not configured.")
    elif not tkpi_file_ref:
         print("\n[ERROR] Skipping menu plan generation - TKPI file upload failed or reference missing.")