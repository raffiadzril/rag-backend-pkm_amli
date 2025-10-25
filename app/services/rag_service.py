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
        """Load semua file JSON dari folder dataset"""
        if not self.dataset_dir.exists():
            print(f"Folder {self.dataset_dir} tidak ditemukan")
            return
        
        json_files = list(self.dataset_dir.glob("*.json"))
        print(f"Menemukan {len(json_files)} file dataset")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.data.extend(data)
                    else:
                        self.data.append(data)
                print(f"✓ Loaded: {json_file.name} ({len(data)} items)")
            except Exception as e:
                print(f"✗ Error loading {json_file.name}: {e}")
        
        print(f"\nTotal data loaded: {len(self.data)} items")
        self.create_embeddings()
    
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
        
        query_parts = [
            f"MPASI bayi {age_months} bulan",
            f"berat badan {weight_kg} kg",
            "menu makanan bayi",
            "angka kecukupan gizi",
            "aturan MPASI"
        ]
        
        if allergies:
            query_parts.append(f"tidak boleh {', '.join(allergies)}")
        
        query = " ".join(query_parts)
        
        relevant_docs = self.search_relevant_docs(query, top_k=10)
        
        if not relevant_docs:
            relevant_docs = self.search_relevant_docs("MPASI menu bayi", top_k=10)
        
        context = "\n\n".join([f"Data {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
        
        allergies_text = f"\n- PENTING: Hindari bahan yang mengandung {', '.join(allergies)}" if allergies else ""
        
        prompt = f"""Kamu adalah ahli gizi dan dokter anak yang berpengalaman dalam MPASI (Makanan Pendamping ASI).

INFORMASI BAYI:
- Usia: {age_months} bulan
- Berat Badan: {weight_kg} kg
- Tinggi Badan: {height_cm} cm
- Tempat Tinggal: {residence}{allergies_text}

DATA REFERENSI GIZI DAN ATURAN MPASI:
{context}

TUGAS:
Buatkan rencana menu MPASI untuk 1 hari yang LENGKAP dan SESUAI dengan:
1. Usia bayi dan kebutuhan gizi berdasarkan data AKG (Angka Kecukupan Gizi)
2. Aturan MPASI yang benar
3. Tekstur makanan yang sesuai usia
4. Variasi nutrisi seimbang
5. Hindari alergi yang disebutkan

PENTING: Response HARUS dalam format JSON yang valid seperti ini:
{{
  "breakfast": {{
    "time": "06:00-07:00",
    "menu_name": "nama menu",
    "ingredients": ["bahan 1", "bahan 2", "bahan 3"],
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
  "notes": [
    "Catatan keamanan 1",
    "Catatan keamanan 2",
    "Tips nutrisi"
  ],
  "recommendations": [
    "Rekomendasi 1",
    "Rekomendasi 2"
  ]
}}

RESPONSE HANYA JSON, TIDAK ADA TEXT LAIN!"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
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
