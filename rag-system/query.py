import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
# Import local embedding model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import traceback # Added for detailed error traceback
# Use CrossEncoder for reranking (required)
from sentence_transformers import CrossEncoder

load_dotenv()

# ============================================================================
# INITIALIZE CHROMADB AND GEMINI
# ============================================================================

db_path = './chroma_db'
# Use BGE embeddings for improved semantic density
embedding_model_name = "BAAI/bge-large-en-v1.5"
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
        self.gemini_model = gemini_model  # Store the model instance
        # Initialize cross-encoder reranker (required for reranking)
        # NOTE: This will raise ImportError if sentence_transformers is not installed
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print('[INFO] Cross-encoder reranker initialized')

    def search_relevant_docs(self, query: str, top_k: int = 10, umur_bulan=None, jenis_kelamin=None, berat_badan=None, tinggi_badan=None, tempat_tinggal=None) -> list:
        """
        Enhanced search with metadata filtering and query transformation
        """
        try:
            # 1. Transform the user query to multiple focused sub-queries
            # Retrieval should be focused on age only. Weight, height and location
            # will NOT be included in retrieval queries and are only used in the LLM prompt.
            sub_queries = self.transform_query_age_only(umur_bulan)
            
            all_results = []
            
            # 2. For each sub-query, perform similarity search WITHOUT relying on Chroma metadata filters.
            #    Some indexes may have no metadata (user requested metadata removal). To be robust we always
            #    retrieve a larger candidate set and then apply age-based filtering in Python.
            for sub_query in sub_queries:
                # Retrieve a larger candidate pool (no filter) so reranker has enough context
                try:
                    # Pull a larger set to allow post-filtering and reranking
                    candidates = self.vectordb.similarity_search(sub_query, k=max(top_k * 5, top_k * 2))
                except Exception:
                    # As a last resort, try a smaller fetch
                    candidates = self.vectordb.similarity_search(sub_query, k=top_k)

                # If age filtering requested, apply it in-Python using document metadata (which may be empty)
                if umur_bulan is not None:
                    results = [doc for doc in candidates if self._doc_matches_filter(doc, umur_bulan)]
                else:
                    results = candidates

                # Trim to top_k per sub-query to limit volume
                all_results.extend(results[: top_k])
            
            # Record which sub-queries we used for retrieval (for debugging)
            try:
                self.last_retrieval_queries = sub_queries
                self.last_retrieval_query = sub_queries[0] if sub_queries else None
            except Exception:
                self.last_retrieval_queries = []
                self.last_retrieval_query = None

            # Remove duplicates while preserving order
            seen_content = set()
            unique_results = []
            for result in all_results:
                content = result.page_content
                if content not in seen_content:
                    seen_content.add(content)
                    unique_results.append(result)
            
            # Dense retrieval: use Chroma similarity search results
            dense_texts = [doc.page_content for doc in unique_results]

            # Select a candidate pool (dense_top * multiplier) for reranking
            candidate_pool = dense_texts[: max(top_k * 5, len(dense_texts))]

            # Rerank with cross-encoder (pairwise scoring)
            pairs = [[query, t] for t in candidate_pool]
            scores = self.reranker.predict(pairs)
            scored = list(zip(candidate_pool, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            final_texts = [t for t, s in scored]

            relevant_docs = final_texts[:top_k]
            print(f"[SUCCESS] Retrieved {len(relevant_docs)} relevant documents for transformed queries: {sub_queries}")
            return relevant_docs
        except Exception as e:
            print(f"[ERROR] Error searching documents: {e}")
            # Fallback to original search method
            try:
                # Record fallback query
                self.last_retrieval_queries = [query]
                self.last_retrieval_query = query
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


    def transform_query_age_only(self, umur_bulan=None):
        """
        Produce a single, focused age-based query for retrieval.
        This ensures retrieval prioritizes age-relevant rules (portion, texture, frequency).
        """
        age_context = f"untuk usia {umur_bulan} bulan" if umur_bulan else ""
        # Return multiple focused queries: one explicitly for AKG (angka kecukupan gizi)
        # and one for practical MPASI rules (porsi, tekstur, frekuensi). Both are age-scoped.
        q_akg = f"Angka Kecukupan Gizi (AKG) untuk bayi {age_context}"
        q_rules = f"Aturan MPASI untuk bayi {age_context} - porsi, tekstur, frekuensi, prinsip pemberian"
        return [q_akg, q_rules]


    def generate_menu_plan_with_chroma(self, user_input: dict) -> dict:

        if not self.gemini_model:
            return {"status": "error", "message": "Gemini API not configured."}

        print(f"[DEBUG] USING MODEL FOR GENERATION: {self.gemini_model.model_name}")

        # Extract and validate parameters
        # Do NOT use hardcoded defaults here — require the caller to provide values.
        # Accept both Indonesian and English keys, but if a required value is missing, return an error.
        umur_bulan = None
        if 'umur_bulan' in user_input:
            umur_bulan = user_input['umur_bulan']
        elif 'age_months' in user_input:
            umur_bulan = user_input['age_months']

        berat_badan = None
        if 'berat_badan' in user_input:
            berat_badan = user_input['berat_badan']
        elif 'weight_kg' in user_input:
            berat_badan = user_input['weight_kg']

        tinggi_badan = None
        if 'tinggi_badan' in user_input:
            tinggi_badan = user_input['tinggi_badan']
        elif 'height_cm' in user_input:
            tinggi_badan = user_input['height_cm']

        # Optional fields (no hard defaults). Use provided values when present.
        jenis_kelamin = user_input.get('jenis_kelamin') if 'jenis_kelamin' in user_input else user_input.get('gender')
        tempat_tinggal = user_input.get('tempat_tinggal') if 'tempat_tinggal' in user_input else user_input.get('residence')
        alergi = user_input.get('alergi') if 'alergi' in user_input else user_input.get('allergies')

        # Validate required numeric inputs and convert types. Fail fast with clear messages.
        missing = []
        if umur_bulan is None:
            missing.append("umur_bulan / age_months")
        if berat_badan is None:
            missing.append("berat_badan / weight_kg")
        if tinggi_badan is None:
            missing.append("tinggi_badan / height_cm")
        if missing:
            return {"status": "error", "message": f"Missing required fields: {', '.join(missing)}"}

        # Convert types and report conversion errors
        try:
            umur_bulan = int(umur_bulan)
        except Exception:
            return {"status": "error", "message": "Invalid umur_bulan / age_months: must be an integer"}

        try:
            berat_badan = float(berat_badan)
        except Exception:
            return {"status": "error", "message": "Invalid berat_badan / weight_kg: must be a number"}

        try:
            tinggi_badan = float(tinggi_badan)
        except Exception:
            return {"status": "error", "message": "Invalid tinggi_badan / height_cm: must be a number"}

        # Normalize remaining optional fields and standardize the input keys to Indonesian names
        if jenis_kelamin is None:
            jenis_kelamin = None

        # Normalize allergies to a list
        if alergi is None:
            alergi = []
        elif isinstance(alergi, str):
            # allow comma separated string
            alergi = [a.strip() for a in alergi.split(',') if a.strip()]
        elif not isinstance(alergi, list):
            # coerce other iterable types to list
            try:
                alergi = list(alergi)
            except Exception:
                alergi = []

        # Build a standardized input dictionary using Indonesian keys
        normalized_input = {
            'umur_bulan': umur_bulan,
            'berat_badan': berat_badan,
            'tinggi_badan': tinggi_badan,
            'jenis_kelamin': jenis_kelamin,
            'tempat_tinggal': tempat_tinggal,
            'alergi': alergi
        }

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
            top_k=7,
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
        prompt = f"""Kamu adalah AI perencana menu MPASI bayi yang SANGAT TELITI, KREATIF, dan TAAT ATURAN GIZI NASIONAL.

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
[!] FILE TKPI_COMPACT.txt telah DILAMPIRKAN. Gunakan data dari file ini sebagai SATU-SATUNYA sumber informasi bahan makanan 
(name, code, kcal, prot_g, fat_g, carb_g, iron_mg, bdd_percent).

==============================================
TUGAS ANDA: BUAT RENCANA MENU MPASI ORIGINAL UNTUK 1 HARI
==============================================

LANGKAH-LANGKAH WAJIB:

1. ANALISIS ATURAN:
- Pastikan menu memenuhi syarat **ADEKUAT**, **TEPAT WAKTU**, **MKM (Minimum Keragaman Makanan)**, dan **TID (Telur, Ikan, Daging)**.
- Batasi **GULA/GARAM** sesuai pedoman dari KONTEKS ATURAN.

2. SESUAIKAN DENGAN TEMPAT TINGGAL:
- Gunakan lokasi `{tempat_tinggal}` untuk menyesuaikan bahan makanan utama.
    - Jika *pegunungan*, utamakan bahan lokal untuk bahan yang mudah ditemukan di daerah pegunungan jangan sampai ada ikan air laut
    - Jika *pesisir pantai*, utamakan bahan laut seperti ikan segar, udang, rumput laut, dan buah tropis.
- Kamu boleh menggunakan bahan umum jika bahan khas daerah tidak mencukupi kebutuhan gizi, tetapi **prioritaskan bahan lokal lebih dulu.**

3. PILIH BAHAN DARI FILE TKPI:
- GUNAKAN HANYA bahan yang ada di FILE `TKPI_COMPACT.txt`.
- WAJIB sertakan **Nama Bahan**, **KODE TKPI**, dan **Jumlah (gram/ml ATAU takaran rumah tangga)**.
- Gunakan satuan praktis seperti:
  - `sdm` (sendok makan)
  - `sdt` (sendok teh)
  - `butir`, `iris`, `potong`, `lembar`, `gelas`, atau `ml`
  - Contoh format: `"Ikan kembung (TKPI123, 1 ekor sedang)"` atau `"Minyak kelapa (TKPI045, 1 sdt)"`.
- **PENTING: Format string ingredient HARUS konsisten. JANGAN gunakan titik (.) untuk memisahkan kata. Gunakan spasi normal.**
  - ✅ BENAR: `"Ayam hati segar (FR007, 25g)"`
  - ❌ SALAH: `"Ayam. hati. segar (FR007, 25g)"`
  - ✅ BENAR: `"Santan (KP003, 40 ml)"`
  - ❌ SALAH: `"Santan (dengan air) (KP003, 40 ml)"` (jangan tambah parenthesis atau deskripsi dalam kode TKPI)
- Hindari bahan alergen: {', '.join(allergies) if allergies else 'tidak ada'}.

4. BUAT MENU ORIGINAL SESUAI ATURAN:
- Menu harus **KREATIF** (bukan hasil salinan template).
- Analisis konteks dari ATURAN untuk memastikan kepatuhan terhadap:
    - **NUTRISI:** 
        - Targetkan total kalori harian agar **TIDAK KURANG** dari nilai “Kebutuhan Jumlah Energi dari MP-ASI” 
          yang ditemukan di konteks aturan (misalnya: 200 kkal, 300 kkal, dst).
        - Menu boleh sedikit melebihi target (hingga ±10%), tetapi **TIDAK BOLEH KURANG** dari nilai tersebut.
    - **TEKSTUR & PORSI:** Sesuaikan dengan usia {age_months} bulan 
      (misalnya "bubur kental", "dicincang halus", "125 ml per kali makan").
    - **FREKUENSI:** Patuhi aturan frekuensi dari konteks:
        - `Utama` = (`breakfast`, `lunch`, `dinner`)
        - `Selingan` = (`morning_snack`, `afternoon_snack`)
        - Contoh: jika aturan adalah "Utama: 2-3x" dan "Selingan: 1-2x", 
          kamu boleh memilih 3 utama dan 1 selingan agar total energi mendekati kebutuhan harian.

5. KALKULASI & VALIDASI INTERNAL:
- Kamu tidak perlu menghitung nutrisi secara numerik, 
  tapi WAJIB memastikan bahwa secara *perkiraan bahan* menu tidak menghasilkan total energi di bawah kebutuhan harian.
- Jika menu tampak kurang kalori, tambahkan bahan sumber energi sehat (misalnya karbohidrat, lemak sehat, santan, atau minyak kelapa)
  agar mendekati atau sedikit melebihi kebutuhan.

6. FORMAT OUTPUT (WAJIB JSON VALID):
- Output HARUS dalam format JSON VALID sesuai contoh.
- Nutrisi hanya diisi sementara (placeholder), karena sistem akan menghitung ulang berdasarkan file TKPI.
- Pastikan struktur JSON identik dengan contoh berikut:

{json_example_original}

==============================================
❗ LARANGAN KETAT:
❌ JANGAN gunakan bahan apapun yang tidak ada di FILE `TKPI_COMPACT.txt`.
❌ JANGAN gunakan format objek untuk ingredients — gunakan STRING seperti contoh.
❌ JANGAN hasilkan total energi yang kurang dari kebutuhan harian yang tercantum di KONTEKS ATURAN.
❌ JANGAN abaikan konteks tempat tinggal dalam memilih bahan makanan.

📋 CATATAN PENTING:
Sebelum memberikan hasil akhir, lakukan pemeriksaan internal:
- Apakah total energi >= kebutuhan minimal dari “Kebutuhan Jumlah Energi dari MP-ASI”? 
  Jika belum, tambahkan bahan yang sesuai (misal: karbohidrat, minyak, kacang, daging).
- Apakah bahan-bahan sesuai konteks tempat tinggal `{tempat_tinggal}`?
- Pastikan kombinasi bahan tetap sesuai prinsip MPASI dan tidak melanggar pembatasan gula/garam.

Hasil akhir harus berupa JSON valid berisi menu 1 hari penuh yang MEMENUHI atau SEDIKIT MELEBIHI kebutuhan kalori dan 
MENGGUNAKAN BAHAN LOKAL yang sesuai dengan daerah tempat tinggal bayi.
"""

        # Extra instruction to avoid list-wrapping or markdown fences in model output
        prompt += (
            "\n\nPENTING: KEMBALIKAN HANYA SATU OBJEK JSON VALID (TIDAK DALAM BENTUK ARRAY). "
            "JANGAN sertakan teks penjelasan, markdown fences (```), atau output tambahan di luar JSON. "
            "Respons harus berupa *satu* objek JSON persis seperti format contoh di atas."
        )

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

            # Normalize response: some LLMs return a single-object inside a list (e.g. [ { ... } ]).
            # Handle these cases robustly so downstream code can assume a dict-like menu_data.
            if isinstance(menu_data, list):
                if len(menu_data) == 0:
                    raise ValueError("LLM returned an empty list instead of a menu object.")
                first = menu_data[0]
                if isinstance(first, dict):
                    print("  [INFO] Response is a list; using the first object as the menu JSON.")
                    menu_data = first
                else:
                    # Wrap non-dict list responses into an envelope so downstream code won't crash
                    print("  [WARNING] Response is a list of non-dict items; wrapping into {'menu_list': [...] }.")
                    menu_data = {"menu_list": menu_data}

            if not isinstance(menu_data, dict):
                raise ValueError(f"Unexpected JSON type for menu_data: {type(menu_data)}")

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
                    "search_query": context_query,
                    "retrieval_queries": getattr(self, 'last_retrieval_queries', []),
                    "retrieval_query_used": getattr(self, 'last_retrieval_query', None),
                    "normalized_input": normalized_input
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
            # Get the original error stringx
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
        Handles flexible LLM output: dots instead of spaces, descriptive quantities, parenthetical text in codes, etc.
        """
        total_energy = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        
        for ingredient_str in ingredients:
            # Parse ingredient string with flexible regex to handle LLM quirks
            # Expected format: "Beras putih (AR001, 50g)" or "Ayam. hati. segar (FR007, 25g)"
            # Also handles: "Santan (dengan air) (KP003, 40 ml)" where code may have extra text
            import re
            
            # Step 1: Find all parenthetical groups with potential TKPI codes
            # Look for pattern like (CODE, ...) where CODE is 2 letters + 3+ digits
            paren_groups = re.findall(r'\(([^)]+)\)', ingredient_str)
            
            code = None
            amount_desc = None
            
            # Find the group that contains a TKPI code (2 letters + 3+ digits)
            for group in paren_groups:
                if re.search(r'[A-Z]{2}\d{3,}', group):
                    # This group likely contains the code
                    parts = group.split(',', 1)  # Split on first comma
                    if len(parts) == 2:
                        code_part = parts[0].strip()
                        amount_desc = parts[1].strip()
                        
                        # Extract just the code (2 letters + 3+ digits)
                        code_match = re.search(r'([A-Z]{2}\d{3,})', code_part)
                        if code_match:
                            code = code_match.group(1)
                            break
            
            if not code or not amount_desc:
                print(f"[WARNING] Could not parse ingredient: {ingredient_str}")
                continue
            
            # Step 2: Extract numeric amount from the description
            # Handle cases like: "3 sdm atau 45g", "40 ml", "1/2 buah", etc.
            # Prefer the last number (often the gram measurement) or the first if only one
            amount_matches = list(re.finditer(r'(\d+(?:\.\d+)?)', amount_desc))
            if not amount_matches:
                print(f"[WARNING] Could not find numeric amount in: {amount_desc}")
                continue
            
            try:
                # Use the last number found (often more specific, like 45g instead of 3 sdm)
                amount_str = amount_matches[-1].group(1)
                amount = float(amount_str)
            except (ValueError, IndexError):
                print(f"[WARNING] Could not parse amount from: {amount_desc}")
                continue
            
            # Step 3: Extract unit (g, ml, sdt, sdm, etc.) - look after the last number
            unit_match = re.search(r'\d+(?:\.\d+)?\s*([a-zA-Z]*)', amount_desc)
            unit = unit_match.group(1).strip().lower() if unit_match else ''
            
            # Step 4: Convert amount to grams if needed (assume ml = g for simplicity)
            if unit == 'ml':
                amount = amount  # 1 ml ~ 1 g for food
            elif unit == 'g' or unit == '':  # Empty unit means grams
                amount = amount
            else:
                # If other units, assume it's in grams
                amount = amount
            
            # Step 5: Look up in TKPI data
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