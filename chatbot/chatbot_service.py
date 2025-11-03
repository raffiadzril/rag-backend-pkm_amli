"""
Chatbot Service untuk GATA MPASI Assistant
Menggunakan RAG (Retrieval Augmented Generation) untuk menjawab pertanyaan tentang MPASI dan gizi anak
"""

import os
import json
import gc
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class ChatbotService:
    def __init__(self):
        """Initialize Chatbot that uses only the project's Chroma RAG service for retrieval.

        This class no longer attempts to load a local `dataset/` folder. It requires the
        `get_chroma_rag_service()` factory from the project's `query` module to be available
        (i.e. `rag-system` must be on sys.path and its Chroma DB accessible). If the RAG
        service cannot be imported or initialized, initialization will fail fast with an
        exception so the deployment clearly indicates the missing dependency.
        """
        # Add proper cleanup handling
        import gc
        import torch
        import atexit
        
        # Register cleanup on Python exit
        atexit.register(self.cleanup)
        # Prefer the project-wide name GOOGLE_API_KEY but accept GEMINI_API_KEY for backward compatibility
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY atau GEMINI_API_KEY tidak ditemukan di .env (set GOOGLE_API_KEY preferred)")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Require and use the project's Chroma RAG service exclusively
        try:
            from query import get_chroma_rag_service
        except Exception as e:
            raise RuntimeError(f"Chatbot requires rag-system on sys.path but 'query.get_chroma_rag_service' could not be imported: {e}")

        try:
            self.rag_service = get_chroma_rag_service()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma RAG service: {e}")

        # Confirm service appears usable
        try:
            # some rag services expose a doc count or similar attribute; use search with empty query to sanity check
            docs = self.rag_service.search_relevant_docs("test", top_k=1)
            print(f"✓ Chatbot: connected to Chroma RAG service (sample retrieval returned {len(docs)} docs)")
        except Exception as e:
            raise RuntimeError(f"Chroma RAG service initialized but sample retrieval failed: {e}")
    
    def search_relevant_context(self, query: str, top_k: int = 10):
        """Return relevant context using the Chroma RAG service exclusively.

        This method assumes the ChatbotService was initialized successfully and
        `self.rag_service` is available.
        """
        try:
            docs = self.rag_service.search_relevant_docs(query, top_k=top_k)
            print(f"✓ Chatbot: Retrieved {len(docs)} docs from Chroma RAG for query")
            return docs
        except Exception as e:
            # Fail-fast: if retrieval fails at runtime, surface an error so the
            # deployment operator notices. Returning an empty list could hide issues.
            raise RuntimeError(f"Chroma RAG retrieval error: {e}")

    # The old in-memory dataset loading and keyword search were intentionally
    # removed. This ChatbotService relies solely on the project's Chroma RAG
    # service for retrieval.
    
    def generate_response(self, user_message: str, conversation_history: list = None):
        """Generate response menggunakan RAG dan Gemini"""
        
        # Search relevant context dengan jumlah lebih banyak
        relevant_docs = self.search_relevant_context(user_message, top_k=10)
        
        # Format context dengan lebih lengkap
        context = "\n\n".join([
            f"=== REFERENSI {i+1} ===\n{str(doc)[:800]}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Build conversation history
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\nRIWAYAT PERCAKAPAN:\n"
            for msg in conversation_history[-8:]:  # Last 8 messages untuk konteks lebih baik
                role = "User" if msg.get("sender") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('text', '')}\n"
        
        # Create enhanced prompt
        prompt = f"""Kamu adalah Asisten Gizi GATA (Gizi Anak Terpadu Aman), seorang ahli gizi profesional yang ramah, informatif, dan membantu orangtua Indonesia dalam memberikan MPASI (Makanan Pendamping ASI) yang tepat untuk bayi mereka.

KONTEKS PENGETAHUAN (DATA REFERENSI LENGKAP):
{context}

{history_text}

PERTANYAAN USER:
{user_message}

PANDUAN MENJAWAB (IKUTI DENGAN TELITI):

1. **GUNAKAN DATA REFERENSI SECARA MAKSIMAL**:
   - Baca SEMUA referensi yang diberikan dengan teliti
   - Berikan jawaban yang LENGKAP dan DETAIL dari data referensi
   - Gabungkan informasi dari berbagai referensi jika relevan
   - Sebutkan angka spesifik (porsi, kalori, usia) jika ada di data

2. **FORMAT JAWABAN YANG RAPI**:
   - Mulai dengan salam singkat (Halo! atau Hai Bunda! dll)
   - Jawab langsung pertanyaan dengan jelas
   - Gunakan **paragraf terpisah** untuk topik berbeda (pisahkan dengan baris kosong)
   - Gunakan **poin-poin bernomor** (1., 2., 3.) atau **bullet points** (•) untuk list
   - Gunakan **bold** untuk highlight kata penting (gunakan **kata**)
   - Akhiri dengan tips praktis atau closing yang ramah

3. **STRUKTUR MARKDOWN YANG BENAR**:
   - Untuk judul section: Gunakan **JUDUL SECTION:**
   - Untuk list bernomor: 
     1. Item pertama
     2. Item kedua
   - Untuk bullet points:
     • Item pertama
     • Item kedua
   - Untuk spacing: Gunakan baris kosong antar paragraf/section
   - Contoh format baik:
     ```
     Halo Bunda! 👋
     
     **4 Prinsip MPASI yang Penting:**
     
     1. **Tepat Waktu** - Mulai di usia 6 bulan
     2. **Bergizi Seimbang** - Lengkap karbohidrat, protein, lemak
     3. **Aman & Higienis** - Bahan segar, pengolahan bersih
     4. **Cara Pemberian Benar** - Responsive feeding
     
     💡 **Tips:** Perhatikan tanda lapar dan kenyang bayi ya!
     ```

4. **EMOJI DAN VISUAL**:
   - Gunakan emoji yang relevan untuk membuat menarik: 
     🍽️ (menu/makanan) 🥗 (sayuran) 🍚 (karbohidrat) 🥩 (protein)
     👶 (bayi) ✅ (ya/benar) ❌ (tidak/hindari) ⚠️ (peringatan)
     📌 (poin penting) 💡 (tips) 🕐 (waktu/jadwal) 👋 (salam)
   - Jangan berlebihan, 3-5 emoji per jawaban sudah cukup
   - Tempatkan emoji di awal kalimat penting atau section header

5. **KONTEN SPESIFIK**:
   - **Untuk pertanyaan menu**: 
     Format: Nama menu → Bahan (list) → Cara membuat (singkat) → Porsi
     Contoh:
     ```
     **Menu Bubur Ayam Wortel:**
     • Bahan: Beras 20g, daging ayam 20g, wortel 20g
     • Cara: Masak beras hingga bubur, tambah ayam cincang dan wortel parut
     • Porsi: 125ml (1/2 mangkok kecil)
     ```
   
   - **Untuk pertanyaan aturan**: 
     Gunakan numbered list dengan penjelasan singkat per poin
   
   - **Untuk pertanyaan usia**: 
     Format: Usia → Tekstur → Porsi → Frekuensi (rapi dalam poin)
   
   - **Untuk pertanyaan gizi**: 
     Sebutkan angka dengan jelas dan bold angka penting
     Contoh: Bayi 8 bulan butuh **650 kkal** per hari
   
   - **Untuk pertanyaan alergi**: 
     Format: Bahan alergi → Alternatif pengganti (list) → Nilai gizi setara

6. **PANJANG JAWABAN**:
   - Pertanyaan sederhana: 3-5 kalimat dalam 1-2 paragraf
   - Pertanyaan kompleks: 6-10 kalimat dalam 3-4 paragraf dengan poin-poin
   - Selalu pisahkan paragraf dengan baris kosong
   - Jangan ragu memberikan detail jika pertanyaan kompleks

7. **KEAMANAN DAN AKURASI**:
   - HANYA gunakan informasi dari data referensi yang diberikan
   - Jika data tidak lengkap, katakan: "Berdasarkan data yang saya miliki..."
   - Selalu prioritaskan keamanan bayi
   - Jika perlu konsultasi medis, sarankan dengan jelas

8. **JIKA TIDAK ADA INFO DI DATA REFERENSI**:
   - Katakan dengan jujur: "Maaf, informasi spesifik tentang [topik] tidak tersedia dalam database saya."
   - Berikan informasi umum yang tersedia jika ada
   - Sarankan konsultasi dokter anak/ahli gizi
   - Jangan mengarang informasi medis

LARANGAN MUTLAK:
❌ JANGAN membuat informasi medis/gizi yang tidak ada di data referensi
❌ JANGAN memberikan diagnosis penyakit
❌ JANGAN merekomendasikan obat atau suplemen tanpa data referensi
❌ JANGAN menggunakan bahasa yang membuat orangtua panik atau cemas
❌ JANGAN memberikan jawaban singkat jika pertanyaan kompleks
❌ JANGAN abaikan data referensi yang tersedia

CONTOH FORMAT JAWABAN YANG RAPI:

**Contoh 1 - Pertanyaan Prinsip:**
```
Pertanyaan: "Apa saja prinsip MPASI?"

Halo Bunda! 👋

Ada **4 Prinsip MPASI** yang sangat penting untuk pertumbuhan optimal bayi:

1. **Tepat Waktu** ⏰
   Mulai MPASI saat bayi berusia 6 bulan, tidak terlalu cepat atau lambat.

2. **Bergizi Seimbang** 🥗
   Menu harus lengkap: karbohidrat, protein hewani & nabati, lemak, vitamin, dan mineral.

3. **Aman & Higienis** ✅
   Gunakan bahan segar, pengolahan bersih, dan tekstur sesuai usia bayi.

4. **Cara Pemberian Benar** 👶
   Responsive feeding - perhatikan tanda lapar/kenyang, beri dengan sabar dan penuh kasih sayang.

💡 **Tips Praktis:** Mulai dengan tekstur lumat halus di usia 6 bulan, lalu bertahap ke tekstur lebih kasar sesuai kemampuan bayi!
```

**Contoh 2 - Pertanyaan Menu:**
```
Pertanyaan: "Menu apa yang bagus untuk bayi 8 bulan?"

Hai! 👋

Untuk bayi 8 bulan, teksturnya sudah bisa **lembut dan sedikit kasar**. Berikut contoh menu:

**Menu Bubur Ayam Sayur:**
• Beras putih 30g
• Daging ayam giling 25g  
• Wortel parut 20g
• Bayam cincang 10g
• Minyak kelapa 1 sdt

**Cara Membuat:**
Masak beras jadi bubur, tambahkan ayam dan sayuran, masak hingga matang. Tambah minyak sebelum disajikan.

**Porsi:** 125-150ml (sekitar 2/3 mangkok kecil) per sekali makan.

💡 **Variasi:** Bisa ganti ayam dengan ikan atau daging sapi, dan variasikan sayuran!
```

PENTING: Selalu pisahkan paragraf dengan baris kosong, gunakan bold untuk kata penting, dan gunakan emoji secukupnya!

Sekarang jawab pertanyaan user dengan FORMAT RAPI, LENGKAP, dan INFORMATIF sesuai panduan di atas:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=800,  # Lebih banyak token untuk jawaban lengkap
                    top_p=0.95,
                    top_k=40,
                )
            )
            
            return {
                "status": "success",
                "response": response.text,
                "sources_used": len(relevant_docs),
                "has_context": len(relevant_docs) > 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "response": f"Maaf, terjadi kesalahan teknis. Silakan coba lagi. Error: {str(e)}",
                "sources_used": 0,
                "has_context": False
            }


    def cleanup(self):
        """Clean up resources and memory"""
        try:
            print("Cleaning up ChatbotService resources...")
            # Clear Gemini model
            if hasattr(self, 'model'):
                self.model = None
            
            # Clean up RAG service
            if hasattr(self, 'rag_service'):
                if hasattr(self.rag_service, 'embeddings'):
                    if hasattr(self.rag_service.embeddings, 'client'):
                        self.rag_service.embeddings.client = None
                if hasattr(self.rag_service, 'vectordb'):
                    self.rag_service.vectordb = None
                self.rag_service = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✓ ChatbotService cleanup completed")
        except Exception as e:
            print(f"✗ Error during ChatbotService cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted"""
        self.cleanup()


# Singleton instance
_chatbot_service = None

def get_chatbot_service():
    """Get or create chatbot service singleton"""
    global _chatbot_service
    if _chatbot_service is None:
        # Docker / production: ChatbotService no longer accepts a dataset_dir.
        # It requires the project's Chroma RAG service to be available on sys.path.
        _chatbot_service = ChatbotService()
    return _chatbot_service
