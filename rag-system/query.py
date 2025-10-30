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

    def search_relevant_docs(self, query: str, top_k: int = 10, umur_bulan=None, jenis_kelamin=None, berat_badan=None, tinggi_badan=None, tempat_tinggal=None) -> list:
        """
        Enhanced search with metadata filtering and query transformation
        """
        try:
            # 1. Transform the user query to multiple focused sub-queries
            sub_queries = self.transform_query(query, umur_bulan)
            
            all_results = []
            
            # 2. For each sub-query, perform metadata-filtered search
            for sub_query in sub_queries:
                # Base filter - if age provided, filter by relevant age range
                filter_dict = {}
                if umur_bulan is not None:
                    # Find documents relevant to the baby's age
                    filter_dict = {
                        "$and": [
                            {"usia_mulai_bulan": {"$lte": umur_bulan}},
                            {"usia_selesai_bulan": {"$gte": umur_bulan}}
                        ]
                    }
                
                # Perform similarity search with metadata filter
                if filter_dict:
                    # Use similarity_search_with_filter if available, otherwise use metadata filtering
                    try:
                        results = self.vectordb.similarity_search(
                            sub_query, 
                            k=top_k,
                            filter=filter_dict
                        )
                    except:
                        # Fallback: get results and filter manually
                        all_docs = self.vectordb.similarity_search(sub_query, k=top_k*2)
                        results = []
                        for doc in all_docs:
                            if self._doc_matches_filter(doc, umur_bulan):
                                results.append(doc)
                                if len(results) >= top_k:
                                    break
                else:
                    # No age filter, just regular search
                    results = self.vectordb.similarity_search(sub_query, k=top_k)
                
                all_results.extend(results)
            
            # Remove duplicates while preserving order
            seen_content = set()
            unique_results = []
            for result in all_results:
                content = result.page_content
                if content not in seen_content:
                    seen_content.add(content)
                    unique_results.append(result)
            
            relevant_docs = [doc.page_content for doc in unique_results[:top_k]]
            print(f"[SUCCESS] Retrieved {len(relevant_docs)} relevant documents for transformed queries: {sub_queries}")
            return relevant_docs
        except Exception as e:
            print(f"[ERROR] Error searching documents: {e}")
            # Fallback to original search method
            try:
                results = self.vectordb.similarity_search(query, k=top_k)
                relevant_docs = [doc.page_content for doc in results]
                print(f"[WARNING] Fallback search used. Retrieved {len(relevant_docs)} relevant documents")
                return relevant_docs
            except Exception as fallback_e:
                print(f"[ERROR] Fallback search also failed: {fallback_e}")
                return []

    def _doc_matches_filter(self, doc, umur_bulan):
        """Helper method to manually check if a document matches the age filter"""
        if umur_bulan is None:
            return True
        metadata = doc.metadata
        usia_mulai = metadata.get('usia_mulai_bulan', 0)
        usia_selesai = metadata.get('usia_selesai_bulan', 100)  # default high value
        return usia_mulai <= umur_bulan <= usia_selesai

    def transform_query(self, original_query, umur_bulan=None):
        """
        Transform the user query into focused sub-queries for better retrieval
        """
        # Define age-specific sub-queries based on the baby's age
        age_context = f"untuk usia {umur_bulan} bulan" if umur_bulan else ""
        
        # Generate focused sub-queries for different aspects of MPASI
        sub_queries = [
            f"Aturan MPASI dan AKG angka kecukupan gizi {age_context}",
            f"Angka Kecukupan Gizi (AKG) untuk bayi {age_context}",
            f"Aturan porsi tekstur frekuensi MPASI {age_context}",
            f"Prinsip dasar dan syarat pemberian MPASI",
            f"Makanan yang dianjurkan atau dilarang untuk MPASI {age_context}",
            f"Tekstur dan konsistensi MPASI {age_context}",
            f"Frekuensi pemberian MPASI {age_context}",
            f"Jumlah porsi MPASI {age_context}",
        ]
        
        # Filter out empty sub-queries and return
        return [q for q in sub_queries if q.strip()]


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
        
        # Pass baby information to the enhanced search function for hybrid search
        konteks_aturan = self.search_relevant_docs(
            context_query, 
            top_k=10,
            umur_bulan=age_months,
            jenis_kelamin=jenis_kelamin,
            berat_badan=berat_badan,
            tinggi_badan=tinggi_badan,
            tempat_tinggal=tempat_tinggal
        ) # Retrieve rules/AGK with metadata filtering

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

4. KALKULASI NUTRISI:
- Sistem akan MENGHITUNG MANUAL nutrisi setelah menerima menu dari Anda berdasarkan data dari FILE TKPI.
- ANDA TIDAK PERLU MENGHITUNG nilai nutrisi secara akurat, hanya menyediakan informasi bahan dan jumlahnya.
- Fokus pada kreativitas dan kepatuhan terhadap aturan MPASI.

5. VALIDASI & FORMAT:
- Output HARUS JSON VALID sesuai format contoh di bawah.
- Nilai nutrisi dapat diisi sementara, karena akan dihitung ulang secara manual oleh sistem.

LARANGAN KETAT:
❌ JANGAN gunakan bahan APAPUN yang tidak ada di FILE `TKPI_COMPACT.txt`.
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

            # Calculate nutrition based on ingredients if available in the menu data
            calculated_menu_data = self.calculate_nutrition_for_menu(menu_data)
            
            return {
                "status": "success",
                "data": calculated_menu_data,
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

    def _doc_matches_filter(self, doc, umur_bulan):
        """Helper method to manually check if a document matches the age filter"""
        if umur_bulan is None:
            return True
        metadata = doc.metadata
        usia_mulai = metadata.get('usia_mulai_bulan', 0)
        usia_selesai = metadata.get('usia_selesai_bulan', 100)  # default high value
        return usia_mulai <= umur_bulan <= usia_selesai

    def calculate_nutrition_for_menu(self, menu_data):
        """
        Calculate nutrition based on ingredients using TKPI-2020.json data
        """
        try:
            # Load TKPI data
            tkpi_data = self.load_tkpi_data()
            
            # Process each meal in the menu
            for meal_key in ['breakfast', 'morning_snack', 'lunch', 'afternoon_snack', 'dinner']:
                if meal_key in menu_data:
                    meal = menu_data[meal_key]
                    if 'ingredients' in meal:
                        # Calculate nutrition for this meal
                        meal_nutrition = self.calculate_meal_nutrition(meal['ingredients'], tkpi_data)
                        # Update the nutrition in the meal
                        if 'nutrition' in meal:
                            meal['nutrition'].update(meal_nutrition)
                        else:
                            meal['nutrition'] = meal_nutrition

            # Calculate daily summary if it doesn't exist or needs updating
            if 'daily_summary' not in menu_data:
                menu_data['daily_summary'] = {}
            
            daily_nutrition = self.calculate_daily_nutrition(menu_data, ['breakfast', 'morning_snack', 'lunch', 'afternoon_snack', 'dinner'])
            menu_data['daily_summary'].update(daily_nutrition)
            
            return menu_data
        except Exception as e:
            print(f"[ERROR] Error calculating nutrition: {str(e)}")
            # Return original menu data if calculation fails
            return menu_data

    def load_tkpi_data(self):
        """
        Load the TKPI-2020.json file and parse it into a dictionary
        Format: {code: {name, ENERGI (Kal), PROTEIN (g), LEMAK (g), KH (g), ...}}
        """
        import os
        import json
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tkpi_file_path = os.path.abspath(os.path.join(script_dir, "../dataset/TKPI-2020.json"))
        
        tkpi_data = {}
        
        if os.path.exists(tkpi_file_path):
            with open(tkpi_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for item in data:
                        code = item.get('KODE')
                        if code:
                            tkpi_data[code] = item
                except json.JSONDecodeError:
                    print(f"[ERROR] Could not parse TKPI-2020.json file")
        else:
            print(f"[ERROR] TKPI-2020.json not found at {tkpi_file_path}")
        
        return tkpi_data

    def calculate_meal_nutrition(self, ingredients, tkpi_data):
        """
        Calculate total nutrition for a meal based on its ingredients
        Ingredients format: ["Name (CODE, amount)", ...]
        """
        total_energy = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        
        for ingredient_str in ingredients:
            # Parse ingredient string: "Name (CODE, amount)"
            # Expected format: "Beras putih (AR001, 50g)" or "Ayam dada (AY001, 30g)"
            import re
            # Look for pattern: (CODE, amount)
            match = re.search(r'\(([^,]+),\s*([0-9.]+)\s*([a-zA-Z]*)\)', ingredient_str)
            if match:
                code = match.group(1).strip()
                amount = float(match.group(2))
                unit = match.group(3).strip().lower()
                
                # Convert amount to grams if needed (assume ml = g for simplicity)
                if unit == 'ml':
                    amount = amount  # 1 ml ~ 1 g for food
                elif unit == 'g' or unit == '':  # Empty unit means grams
                    amount = amount
                else:
                    # If other units, assume it's in grams
                    amount = amount
                
                # Look up in TKPI data
                if code in tkpi_data:
                    item = tkpi_data[code]
                    
                    # Get nutritional values per 100g from TKPI-2020.json format
                    kcal_per_100g = float(item.get('ENERGI (Kal)', 0))
                    prot_per_100g = float(item.get('PROTEIN (g)', 0))
                    fat_per_100g = float(item.get('LEMAK (g)', 0))
                    carb_per_100g = float(item.get('KH (g)', 0))
                    
                    # Calculate nutrition based on the amount used
                    # Formula: (value per 100g) * (amount in grams) / 100
                    total_energy += (kcal_per_100g * amount) / 100
                    total_protein += (prot_per_100g * amount) / 100
                    total_fat += (fat_per_100g * amount) / 100
                    total_carbs += (carb_per_100g * amount) / 100
                else:
                    print(f"[WARNING] Ingredient code '{code}' not found in TKPI data")
            else:
                print(f"[WARNING] Could not parse ingredient: {ingredient_str}")
        
        return {
            "energy_kcal": round(total_energy, 1),
            "protein_g": round(total_protein, 1),
            "carbs_g": round(total_carbs, 1),
            "fat_g": round(total_fat, 1)
        }

    def calculate_daily_nutrition(self, menu_data, meal_keys):
        """
        Calculate total daily nutrition by summing up all meals
        """
        total_energy = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        
        for meal_key in meal_keys:
            if meal_key in menu_data and 'nutrition' in menu_data[meal_key]:
                nutrition = menu_data[meal_key]['nutrition']
                total_energy += nutrition.get('energy_kcal', 0)
                total_protein += nutrition.get('protein_g', 0)
                total_carbs += nutrition.get('carbs_g', 0)
                total_fat += nutrition.get('fat_g', 0)
        
        return {
            "total_energy_kcal": round(total_energy, 1),
            "total_protein_g": round(total_protein, 1),
            "total_carbs_g": round(total_carbs, 1),
            "total_fat_g": round(total_fat, 1)
        }


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