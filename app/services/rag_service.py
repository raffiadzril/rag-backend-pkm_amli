import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class RAGService:
    def __init__(self, dataset_dir="dataset"):
        """Initialize RAG system dengan Gemini API"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.dataset_dir = Path(dataset_dir)
        self.data = []
        self.text_chunks = []
        
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """Load semua file JSON dan Markdown dari folder dataset"""
        if not self.dataset_dir.exists():
            print(f"Folder {self.dataset_dir} tidak ditemukan")
            return
        
        # Load JSON files
        json_files = list(self.dataset_dir.glob("*.json"))
        print(f"Menemukan {len(json_files)} file JSON")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.data.extend(data)
                    else:
                        self.data.append(data)
                print(f"✓ Loaded JSON: {json_file.name} ({len(data)} items)")
            except Exception as e:
                print(f"✗ Error loading {json_file.name}: {e}")
        
        # Load Markdown files
        md_files = list(self.dataset_dir.glob("*.md"))
        print(f"Menemukan {len(md_files)} file Markdown")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split markdown into sections based on headers
                    sections = self._parse_markdown(content, md_file.name)
                    self.data.extend(sections)
                print(f"✓ Loaded MD: {md_file.name} ({len(sections)} sections)")
            except Exception as e:
                print(f"✗ Error loading {md_file.name}: {e}")
        
        print(f"\nTotal data loaded: {len(self.data)} items")
        self.create_embeddings()
    
    def _parse_markdown(self, content: str, filename: str):
        """Parse markdown content into structured sections"""
        sections = []
        lines = content.split('\n')
        current_section = {"source_file": filename, "content": ""}
        current_header = ""
        
        for line in lines:
            line = line.strip()
            
            # Detect headers (# Header)
            if line.startswith('#'):
                # Save previous section if it has content
                if current_section["content"].strip():
                    current_section["header"] = current_header
                    sections.append(current_section)
                
                # Start new section
                current_header = line.lstrip('#').strip()
                current_section = {
                    "source_file": filename,
                    "header": current_header,
                    "content": line + "\n"
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add last section
        if current_section["content"].strip():
            current_section["header"] = current_header
            sections.append(current_section)
        
        return sections
    
    def create_embeddings(self):
        """Buat embeddings untuk semua data"""
        print("\nMembuat embeddings...")
        self.text_chunks = []
        
        for item in self.data:
            text = self.item_to_text(item)
            self.text_chunks.append(text)
        
        print(f"✓ Siap melayani {len(self.text_chunks)} dokumen")
    
    def item_to_text(self, item):
        """Convert data item menjadi text untuk RAG"""
        # Handle markdown content
        if "content" in item and "source_file" in item:
            header = item.get("header", "")
            content = item.get("content", "")
            source = item.get("source_file", "")
            return f"[{source}] {header}\n{content}".strip()
        
        # Handle JSON data
        text_parts = []
        for key, value in item.items():
            if value is not None and value != "" and key != "":
                text_parts.append(f"{key}: {value}")
        return " | ".join(text_parts)
    
    def search_relevant_docs(self, query, top_k=5):
        """Cari dokumen yang relevan dengan query"""
        query_lower = query.lower()
        query_words = query_lower.split()
        scores = []
        
        for i, text in enumerate(self.text_chunks):
            text_lower = text.lower()
            score = 0
            
            if query_lower in text_lower:
                score += 100
            
            for query_word in query_words:
                if len(query_word) < 3:
                    continue
                    
                if query_word in text_lower.split():
                    score += 10
                elif query_word in text_lower:
                    score += 5
                else:
                    for text_word in text_lower.split():
                        if query_word in text_word or text_word in query_word:
                            score += 3
                            break
            
            matching_words = sum(1 for word in query_words if word in text_lower)
            score += matching_words * 2
            
            scores.append((score, i))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        if not top_indices and query_words:
            for i, text in enumerate(self.text_chunks):
                text_lower = text.lower()
                for query_word in query_words:
                    if len(query_word) >= 3 and query_word in text_lower:
                        top_indices.append(i)
                        if len(top_indices) >= top_k:
                            break
                if len(top_indices) >= top_k:
                    break
        
        relevant_docs = [self.text_chunks[i] for i in top_indices]
        return relevant_docs
    
    def generate_menu_plan(self, user_input: dict):
        """Generate meal plan berdasarkan input user"""
        
        age_months = user_input.get('age_months', 6)
        weight_kg = user_input.get('weight_kg', 7)
        height_cm = user_input.get('height_cm', 65)
        allergies = user_input.get('allergies', [])
        residence = user_input.get('residence', 'Indonesia')
        
        # Query yang lebih comprehensive untuk RAG
        query_parts = [
            f"MPASI bayi {age_months} bulan",
            f"usia {age_months} bulan",
            "angka kecukupan gizi",
            "AKG",
            "aturan MPASI",
            "menu makanan bayi",
            "tekstur makanan",
            "porsi makan",
            "frekuensi makan",
        ]
        
        if allergies:
            query_parts.append(f"alergi {', '.join(allergies)}")
        
        query = " ".join(query_parts)
        
        # Cari dokumen relevan dengan jumlah lebih banyak untuk konteks maksimal
        relevant_docs = self.search_relevant_docs(query, top_k=20)
        
        if not relevant_docs:
            relevant_docs = self.search_relevant_docs("MPASI menu bayi", top_k=20)
        
        # Format context dengan lebih terstruktur
        context = "\n\n".join([f"=== REFERENSI {i+1} ===\n{doc}" for i, doc in enumerate(relevant_docs)])
        
        allergies_text = f"\n- PENTING: WAJIB hindari semua bahan yang mengandung {', '.join(allergies)}" if allergies else ""
        
        prompt = f"""Kamu adalah sistem AI yang HANYA menggunakan data yang diberikan untuk membuat rencana menu MPASI.

INFORMASI BAYI:
- Usia: {age_months} bulan
- Berat Badan: {weight_kg} kg
- Tinggi Badan: {height_cm} cm
- Tempat Tinggal: {residence}{allergies_text}

DATA REFERENSI GIZI DAN ATURAN MPASI:
{context}

ATURAN STRICT (WAJIB DIIKUTI):
1. HANYA gunakan bahan makanan yang ADA di data TKPI-2020 di atas
2. HANYA gunakan aturan MPASI yang ADA di data referensi di atas
3. HANYA gunakan nilai AKG yang ADA di data referensi di atas
4. Kandungan gizi HARUS dihitung berdasarkan data TKPI-2020, TIDAK boleh perkiraan
5. Tekstur, porsi, dan frekuensi HARUS sesuai dengan data referensi untuk usia {age_months} bulan
6. JANGAN tambahkan informasi dari pengetahuan umum jika tidak ada di data referensi
7. Jika data tidak mencukupi, tambahkan di "notes" bahwa informasi terbatas

LARANGAN:
❌ DILARANG menggunakan bahan makanan yang tidak ada di TKPI-2020
❌ DILARANG membuat angka gizi tanpa acuan dari TKPI-2020
❌ DILARANG menambahkan aturan yang tidak ada di data referensi
❌ DILARANG menggunakan pengetahuan di luar data yang diberikan

TUGAS:
Buatkan rencana menu MPASI untuk 1 hari BERDASARKAN DATA REFERENSI SAJA:
1. Cari nilai AKG untuk usia {age_months} bulan dari data referensi
2. Cari aturan MPASI (tekstur, porsi, frekuensi) untuk usia {age_months} bulan dari data referensi
3. Pilih bahan makanan HANYA dari TKPI-2020
4. Hitung kandungan gizi menggunakan data TKPI-2020
5. Pastikan total gizi memenuhi AKG dari data referensi

PENTING: Response HARUS dalam format JSON yang valid seperti ini:
{{
  "breakfast": {{
    "time": "06:00-07:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1 (kode TKPI jika ada)", "bahan 2", "bahan 3"],
    "portion": "porsi dalam ml atau gram",
    "instructions": "cara membuat singkat",
    "nutrition": {{
      "energy_kcal": 100,
      "protein_g": 5,
      "carbs_g": 15,
      "fat_g": 3
    }}
  }},
  "morning_snack": {{
    "time": "09:00-10:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1", "bahan 2"],
    "portion": "porsi",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 80,
      "protein_g": 2,
      "carbs_g": 10,
      "fat_g": 3
    }}
  }},
  "lunch": {{
    "time": "12:00-13:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1", "bahan 2", "bahan 3"],
    "portion": "porsi",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 120,
      "protein_g": 8,
      "carbs_g": 18,
      "fat_g": 4
    }}
  }},
  "afternoon_snack": {{
    "time": "15:00-16:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1"],
    "portion": "porsi",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 50,
      "protein_g": 1,
      "carbs_g": 10,
      "fat_g": 1
    }}
  }},
  "dinner": {{
    "time": "18:00-19:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1", "bahan 2", "bahan 3"],
    "portion": "porsi",
    "instructions": "cara membuat",
    "nutrition": {{
      "energy_kcal": 130,
      "protein_g": 10,
      "carbs_g": 15,
      "fat_g": 5
    }}
  }},
  "daily_summary": {{
    "total_energy_kcal": 480,
    "total_protein_g": 26,
    "total_carbs_g": 68,
    "total_fat_g": 16,
    "akg_compliance": "Memenuhi/Kurang/Melebihi AKG berdasarkan data referensi",
    "akg_reference": "AKG untuk usia {age_months} bulan dari data: [nilai dari data]"
  }},
  "notes": [
    "Semua bahan menggunakan TKPI-2020: [sebutkan kode TKPI]",
    "Tekstur sesuai usia {age_months} bulan dari data referensi",
    "Jika ada keterbatasan data, sebutkan di sini"
  ],
  "recommendations": [
    "Rekomendasi HANYA berdasarkan data referensi",
    "Variasi bahan dari TKPI-2020 untuk hari berikutnya"
  ],
  "data_sources_used": [
    "TKPI-2020: [kode-kode bahan yang digunakan]",
    "AKG: [kelompok usia yang digunakan]",
    "Aturan MPASI: [aturan spesifik yang diikuti]"
  ]
}}

PENTING: 
- Jika bahan tidak ada di TKPI-2020, JANGAN gunakan
- Jika aturan tidak ada di data referensi, JANGAN buat sendiri
- Sebutkan kode TKPI untuk setiap bahan (contoh: AR001 untuk beras)
- Cantumkan sumber data yang digunakan di "data_sources_used"
- SEMUA NILAI NUTRISI HARUS ANGKA BULAT atau DESIMAL, BUKAN RUMUS MATEMATIKA
- Contoh BENAR: "energy_kcal": 150
- Contoh SALAH: "energy_kcal": 134 + 50 (ini akan error!)
- Hitung semua nilai terlebih dahulu, lalu masukkan hasilnya sebagai angka

RESPONSE HANYA JSON VALID, TIDAK ADA TEXT LAIN!"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3,  # Lebih rendah untuk lebih strict mengikuti data
                )
            )
            
            # Parse JSON response
            menu_data = json.loads(response.text)
            
            return {
                "status": "success",
                "data": menu_data,
                "user_info": {
                    "age_months": age_months,
                    "weight_kg": weight_kg,
                    "height_cm": height_cm,
                    "residence": residence,
                    "allergies": allergies
                },
                "rag_info": {
                    "documents_retrieved": len(relevant_docs),
                    "query_used": query
                }
            }
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Error parsing JSON response: {str(e)}",
                "raw_response": response.text if 'response' in locals() else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating menu plan: {str(e)}"
            }


_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(dataset_dir="dataset")
    return _rag_service_instance
