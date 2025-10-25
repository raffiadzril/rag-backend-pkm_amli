import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

# Load environment variables
load_dotenv()

class GeminiRAG:
    def __init__(self, dataset_dir="dataset"):
        """Initialize RAG system dengan Gemini API"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.dataset_dir = Path(dataset_dir)
        self.data = []
        self.embeddings = []
        
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
            # Convert item menjadi text yang readable
            text = self.item_to_text(item)
            self.text_chunks.append(text)
        
        # Gunakan embedding sederhana (keyword-based) untuk efisiensi
        # Anda bisa ganti dengan Gemini embedding API jika perlu
        print(f"✓ Siap melayani {len(self.text_chunks)} dokumen")
    
    def item_to_text(self, item):
        """Convert data item menjadi text untuk RAG"""
        text_parts = []
        for key, value in item.items():
            if value is not None and value != "" and key != "":
                text_parts.append(f"{key}: {value}")
        return " | ".join(text_parts)
    
    def search_relevant_docs(self, query, top_k=5):
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
                if len(query_word) < 3:  # Skip kata pendek seperti "di", "ke"
                    continue
                    
                # Cek exact word match
                if query_word in text_lower.split():
                    score += 10
                # Cek partial match (substring)
                elif query_word in text_lower:
                    score += 5
                # Cek fuzzy match (untuk typo atau variasi)
                else:
                    for text_word in text_lower.split():
                        # Jika kata query ada dalam kata text (e.g., "merah" in "merah-merah")
                        if query_word in text_word or text_word in query_word:
                            score += 3
                            break
            
            # 3. Bonus jika banyak kata yang match
            matching_words = sum(1 for word in query_words if word in text_lower)
            score += matching_words * 2
            
            scores.append((score, i))
        
        # Sort dan ambil top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Ambil dokumen dengan score > 0
        top_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        # Jika tidak ada hasil, coba pencarian lebih lenient
        if not top_indices and query_words:
            # Cari dokumen yang mengandung minimal 1 kata dari query
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
    
    def query(self, question, top_k=5):
        """Query RAG system dengan pertanyaan"""
        print(f"\n🔍 Mencari dokumen relevan untuk: {question}")
        
        # Cari dokumen relevan
        relevant_docs = self.search_relevant_docs(question, top_k)
        
        if not relevant_docs:
            return "Maaf, tidak ada data yang relevan ditemukan untuk pertanyaan Anda."
        
        print(f"✓ Ditemukan {len(relevant_docs)} dokumen relevan")
        
        # Buat context dari dokumen relevan
        context = "\n\n".join([f"Dokumen {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
        
        # Buat prompt untuk Gemini
        prompt = f"""Berdasarkan data berikut, jawab pertanyaan dengan detail dan akurat:

DATA REFERENSI:
{context}

PERTANYAAN: {question}

Jawab dengan informasi dari data di atas. Jika ada nilai numerik, sebutkan dengan jelas. Jika data tidak mencukupi, katakan itu."""
        
        print("🤖 Generating jawaban dari Gemini...")
        
        # Generate response dengan Gemini
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error saat generate response: {e}"
    
    def chat(self):
        """Mode interaktif chat"""
        print("\n" + "="*60)
        print("🤖 GEMINI RAG SYSTEM")
        print("="*60)
        print(f"Dataset loaded: {len(self.data)} items")
        print("Ketik 'exit' atau 'quit' untuk keluar\n")
        
        while True:
            try:
                question = input("❓ Tanya: ").strip()
                if question.lower() in ['exit', 'quit', 'keluar']:
                    print("Terima kasih! 👋")
                    break
                
                if not question:
                    continue
                
                answer = self.query(question)
                print(f"\n💡 Jawaban:\n{answer}\n")
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\nTerima kasih! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Inisialisasi RAG system
    rag = GeminiRAG(dataset_dir="dataset")
    
    # Jalankan chat interaktif
    rag.chat()
