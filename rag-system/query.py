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
        age_months = user_input.get('age_months', 6)
        allergies = user_input.get('allergies', [])

        # Example of rules retrieval (you can optimize this)
        rules_query = f"Aturan MPASI dan AKG angka kecukupan gizi untuk usia {age_months} bulan tekstur porsi frekuensi"
        konteks_aturan = self.search_relevant_docs(rules_query, top_k=10) # Retrieve rules/AGK

        # === STEP 2: Prepare Prompt ===
        formatted_aturan = "\n\n---\n".join(konteks_aturan)
        allergies_text = f"\n- PENTING: Bayi alergi terhadap: {', '.join(allergies)}. Hindari bahan ini." if allergies else "\n- Tidak ada alergi yang dilaporkan."

        # Build comprehensive prompt with full JSON format instructions
        json_example = '''{
  "breakfast": {
    "time": "06:00-07:00",
    "menu_name": "nama menu original yang unik",
    "ingredients": [
      {{"nama": "bahan 1", "kode_tkpi": "AR001", "jumlah": "100g"}},
      {{"nama": "bahan 2", "kode_tkpi": "AR002", "jumlah": "50g"}},
      {{"nama": "bahan 3", "kode_tkpi": "AR003", "jumlah": "75g"}}
    ],
    "portion": "jumlah porsi total",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 150,
      "protein_g": 6.5,
      "carbs_g": 18.0,
      "fat_g": 3.2
    }}
  },
  "morning_snack": {{
    "time": "09:00-10:00",
    "menu_name": "nama menu snack",
    "ingredients": [
      {{"nama": "bahan 1", "kode_tkpi": "AR001", "jumlah": "50g"}},
      {{"nama": "bahan 2", "kode_tkpi": "AR004", "jumlah": "40g"}}
    ],
    "portion": "jumlah porsi total",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 80,
      "protein_g": 2.0,
      "carbs_g": 12.0,
      "fat_g": 1.5
    }}
  }},
  "lunch": {{
    "time": "12:00-13:00",
    "menu_name": "nama menu makan siang",
    "ingredients": [
      {{"nama": "bahan 1", "kode_tkpi": "AR001", "jumlah": "100g"}},
      {{"nama": "bahan 2", "kode_tkpi": "AR002", "jumlah": "60g"}},
      {{"nama": "bahan 3", "kode_tkpi": "AR005", "jumlah": "80g"}}
    ],
    "portion": "jumlah porsi total",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 160,
      "protein_g": 7.5,
      "carbs_g": 20.0,
      "fat_g": 3.5
    }}
  }},
  "afternoon_snack": {{
    "time": "15:00-16:00",
    "menu_name": "nama menu snack sore",
    "ingredients": [
      {{"nama": "bahan 1", "kode_tkpi": "AR006", "jumlah": "60g"}}
    ],
    "portion": "jumlah porsi total",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 60,
      "protein_g": 1.0,
      "carbs_g": 10.0,
      "fat_g": 0.5
    }}
  }},
  "dinner": {{
    "time": "18:00-19:00",
    "menu_name": "nama menu makan malam",
    "ingredients": [
      {{"nama": "bahan 1", "kode_tkpi": "AR001", "jumlah": "90g"}},
      {{"nama": "bahan 2", "kode_tkpi": "AR003", "jumlah": "70g"}},
      {{"nama": "bahan 3", "kode_tkpi": "AR007", "jumlah": "60g"}}
    ],
    "portion": "jumlah porsi total",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 140,
      "protein_g": 6.0,
      "carbs_g": 18.0,
      "fat_g": 2.8
    }}
  }},
  "daily_summary": {{
    "total_energy_kcal": 590,
    "total_protein_g": 23.0,
    "total_carbs_g": 78.0,
    "total_fat_g": 11.5,
    "akg_requirement": "Kebutuhan AKG untuk bayi",
    "akg_compliance": "Evaluasi apakah total harian memenuhi AKG"
  }},
  "notes": [
    "[SUCCESS] Catatan 1",
    "[SUCCESS] Catatan 2"
  ],
  "recommendations": [
    "Rekomendasi 1",
    "Rekomendasi 2"
  ]
}}'''

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
TUGAS ANDA: BUAT RENCANA MENU MPASI ORIGINAL UNTUK 1 HARI
==============================================

[!] PERHATIAN PENTING: FILE TKPI_COMPACT.txt TELAH DILAMPIRKAN. WAJIB GUNAKAN HANYA DATA DARI FILE INI!

WAJIB OUTPUT SEBAGAI JSON YANG VALID! Tidak boleh ada text lain selain JSON.

FORMAT JSON YANG HARUS DIOUTPUT (COPY STRUKTUR INI PERSIS):
{json_example}

INSTRUKSI WAJIB:
1. [!] GUNAKAN HANYA BAHAN DARI FILE TKPI_COMPACT.txt YANG DILAMPIRKAN - JANGAN MENGGUNAKAN BAHAN LAIN
2. [!] SETIAP BAHAN HARUS MEMILIKI KODE TKPI (KODE) YANG VALID DARI FILE
3. [!] AMBIL NILAI NUTRISI HANYA DARI FILE TKPI_COMPACT.txt - JANGAN ESTIMASI ATAU MENGARANG
4. [!] FORMAT INGREDIENTS: [{{"nama":"nama bahan exact dari TKPI","kode_tkpi":"KODE exact","jumlah":"gram/ml"}}]
5. [+] Buat menu ORIGINAL dan UNIK dengan kombinasi bahan dari TKPI
6. [+] Gunakan bahan yang sesuai untuk usia {age_months} bulan
7. [+] Hitung nutrisi dengan ANGKA PASTI dari file TKPI
8. [+] Semua nilai nutrisi harus ANGKA (number), bukan string
9. [+] Hindari bahan alergen: {', '.join(allergies) if allergies else 'tidak ada'}
10. [+] WAJIB OUTPUT HANYA JSON VALID, tidak ada text pembuka/penutup
11. [+] Jangan gunakan markdown code blocks (```)
12. [+] Setiap field HARUS ada, tidak boleh ada field yang kosong

[!] PERINGATAN PENTING:
- SEMUA KODE_TKPI HARUS SAMA DENGAN KODE DI FILE TKPI_COMPACT.txt (contoh: AR001, AR002, etc)
- Jika ada bahan yang tidak ditemukan di file TKPI_COMPACT.txt, JANGAN gunakan bahan itu
- Jika tidak yakin nilai nutrisi atau kode, cek di file TKPI_COMPACT.txt
- Semua keputusan menu HARUS berdasarkan data di file TKPI_COMPACT.txt yang dilampirkan"""

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
