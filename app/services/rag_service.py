import os
import json
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Load env
load_dotenv()


class RAGService:
    """Lightweight RAG service adapted for FastAPI backend.

    It loads JSON datasets from `dataset/` and performs simple
    fuzzy search + calls Gemini to generate menu plans.
    """

    def __init__(self, dataset_dir: str = "dataset"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

        genai.configure(api_key=self.api_key)
        # choose a model name available in your account
        self.model = genai.GenerativeModel("gemini-2.1")

        self.dataset_dir = Path(dataset_dir)
        self.data = []
        self.text_chunks = []
        self._load_all_datasets()

    def _load_all_datasets(self):
        if not self.dataset_dir.exists():
            return
        json_files = list(self.dataset_dir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        self.data.extend(d)
                    else:
                        self.data.append(d)
            except Exception:
                continue

        for item in self.data:
            self.text_chunks.append(self._item_to_text(item))

    def _item_to_text(self, item: dict) -> str:
        parts = []
        for k, v in item.items():
            if v is None or v == "":
                continue
            if isinstance(v, (list, dict)):
                continue
            parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def _search(self, query: str, top_k: int = 5) -> List[str]:
        q = query.lower()
        qwords = [w for w in q.split() if len(w) > 2]
        scores = []
        for i, t in enumerate(self.text_chunks):
            tl = t.lower()
            score = 0
            if q in tl:
                score += 100
            for w in qwords:
                if w in tl.split():
                    score += 10
                elif w in tl:
                    score += 5
            matching = sum(1 for w in qwords if w in tl)
            score += matching * 2
            scores.append((score, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        indices = [i for s, i in scores[:top_k] if s > 0]
        return [self.text_chunks[i] for i in indices]

    def query(self, question: str, top_k: int = 5) -> str:
        docs = self._search(question, top_k=top_k)
        if not docs:
            return "Maaf, tidak ada data relevan." 
        context = "\n\n".join(f"Dokumen {i+1}:\n{d}" for i, d in enumerate(docs))
        prompt = f"""Anda adalah asisten nutrisi MPASI.
Berdasarkan data referensi berikut:
{context}
Jawab pertanyaan berikut dengan jelas dan susun rencana menu 1 hari (pagi, siang, malam, 1-2 selingan) sesuai batasan usia/berat/tinggi/alergi.

PERTANYAAN: {question}

Berikan hasil dalam format yang mudah di-parse: daftar item per waktu makan dan singkat alasan/notes jika perlu."""

        try:
            resp = self.model.generate_content(prompt)
            return resp.text
        except Exception as e:
            return f"Error generating: {e}"

    def generate_menu(self,
                      age_months: int,
                      weight_kg: Optional[float] = None,
                      height_cm: Optional[float] = None,
                      residence: Optional[str] = None,
                      allergies: Optional[List[str]] = None) -> str:
        """Build a question from parameters and call query()"""
        allergy_text = ", ".join(allergies) if allergies else "tidak ada"
        parts = [f"Usia: {age_months} bulan", f"Alergi: {allergy_text}"]
        if weight_kg is not None:
            parts.append(f"Berat badan: {weight_kg} kg")
        if height_cm is not None:
            parts.append(f"Tinggi: {height_cm} cm")
        if residence:
            parts.append(f"Tempat tinggal: {residence}")

        question = "Buat rencana menu MPASI 1 hari berdasarkan " + "; ".join(parts)
        return self.query(question, top_k=5)


# create a default singleton for app to reuse
_default_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    global _default_service
    if _default_service is None:
        _default_service = RAGService(dataset_dir="dataset")
    return _default_service
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables
load_dotenv()


class RAGService:
    """Service untuk RAG system dengan Gemini API"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern untuk RAG service"""
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize RAG system dengan Gemini API"""
        if self._initialized:
            return
            
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.dataset_dir = Path("dataset")
        self.data = []
        self.text_chunks = []
        
        self.load_all_datasets()
        self._initialized = True
    
    def load_all_datasets(self):
        """Load semua file JSON dari folder dataset"""
        if not self.dataset_dir.exists():
            print(f"Folder {self.dataset_dir} tidak ditemukan")
            return
        
        json_files = list(self.dataset_dir.glob("*.json"))
        print(f"📂 Menemukan {len(json_files)} file dataset")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.data.extend(data)
                    else:
                        self.data.append(data)
                print(f"   ✓ {json_file.name}: {len(data)} items")
            except Exception as e:
                print(f"   ✗ Error loading {json_file.name}: {e}")
        
        print(f"📊 Total data loaded: {len(self.data)} items\n")
        self.create_text_chunks()
    
    def create_text_chunks(self):
        """Buat text chunks untuk semua data"""
        self.text_chunks = []
        for item in self.data:
            text = self.item_to_text(item)
            self.text_chunks.append(text)
    
    def item_to_text(self, item: Dict[str, Any]) -> str:
        """Convert data item menjadi text untuk RAG"""
        text_parts = []
        for key, value in item.items():
            if value is not None and value != "" and key != "":
                text_parts.append(f"{key}: {value}")
        return " | ".join(text_parts)
    
    def search_relevant_docs(self, query: str, top_k: int = 5) -> List[str]:
        """Cari dokumen yang relevan dengan query - improved fuzzy matching"""
        query_lower = query.lower()
        query_words = query_lower.split()
        scores = []
        
        for i, text in enumerate(self.text_chunks):
            text_lower = text.lower()
            score = 0
            
            # 1. Exact phrase match (highest score)
            if query_lower in text_lower:
                score += 100
            
            # 2. Partial word matching (flexible)
            for query_word in query_words:
                if len(query_word) < 3:  # Skip kata pendek
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
            
            # 3. Bonus jika banyak kata yang match
            matching_words = sum(1 for word in query_words if word in text_lower)
            score += matching_words * 2
            
            scores.append((score, i))
        
        # Sort dan ambil top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        # Fallback: cari minimal 1 kata match
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
    
    def query(self, question: str, top_k: int = 5, custom_prompt: str = None) -> str:
        """Query RAG system dengan pertanyaan"""
        # Cari dokumen relevan
        relevant_docs = self.search_relevant_docs(question, top_k)
        
        if not relevant_docs:
            return "Maaf, tidak ada data yang relevan ditemukan untuk pertanyaan Anda."
        
        # Buat context dari dokumen relevan
        context = "\n\n".join([f"Dokumen {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
        
        # Gunakan custom prompt jika ada, atau default prompt
        if custom_prompt:
            prompt = custom_prompt.format(context=context, question=question)
        else:
            prompt = f"""Berdasarkan data berikut, jawab pertanyaan dengan detail dan akurat:

DATA REFERENSI:
{context}

PERTANYAAN: {question}

Jawab dengan informasi dari data di atas. Jika ada nilai numerik, sebutkan dengan jelas."""
        
        # Generate response dengan Gemini
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error saat generate response: {e}"
    
    def get_nutrition_requirements(self, umur_bulan: int) -> Dict[str, Any]:
        """Ambil kebutuhan gizi berdasarkan umur"""
        query = f"kebutuhan gizi anak umur {umur_bulan} bulan"
        relevant_docs = self.search_relevant_docs(query, top_k=3)
        
        if not relevant_docs:
            # Default values jika tidak ada data
            return {
                "energi": 725,
                "protein": 11,
                "lemak": 25,
                "karbohidrat": 82,
                "kalsium": 270
            }
        
        context = "\n".join(relevant_docs)
        prompt = f"""Berdasarkan data berikut, berikan kebutuhan gizi harian untuk anak umur {umur_bulan} bulan.

DATA:
{context}

Berikan jawaban dalam format JSON dengan key: energi, protein, lemak, karbohidrat, kalsium.
Hanya berikan JSON tanpa penjelasan tambahan."""

        try:
            response = self.model.generate_content(prompt)
            # Parse JSON dari response
            import re
            json_match = re.search(r'\{[^}]+\}', response.text)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "energi": 725,
                    "protein": 11,
                    "lemak": 25,
                    "karbohidrat": 82,
                    "kalsium": 270
                }
        except Exception as e:
            print(f"Error getting nutrition requirements: {e}")
            return {
                "energi": 725,
                "protein": 11,
                "lemak": 25,
                "karbohidrat": 82,
                "kalsium": 270
            }


# Global instance
rag_service = RAGService()
