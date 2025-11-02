"""
Chatbot Service untuk GATA MPASI Assistant
Menggunakan RAG (Retrieval Augmented Generation) untuk menjawab pertanyaan tentang MPASI dan gizi anak
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class ChatbotService:
    def __init__(self, dataset_dir="../dataset"):
        """Initialize Chatbot dengan Gemini API dan RAG system"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.dataset_dir = Path(dataset_dir)
        self.knowledge_base = []
        
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load knowledge base dari dataset dengan parsing yang lebih baik"""
        if not self.dataset_dir.exists():
            print(f"⚠️ Dataset folder tidak ditemukan: {self.dataset_dir}")
            return
        
        print("📚 Loading knowledge base...")
        
        # Load JSON files
        json_files = list(self.dataset_dir.glob("*.json"))
        print(f"📄 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Add metadata untuk tracking
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item['_source_file'] = json_file.name
                        self.knowledge_base.extend(data)
                        print(f"  ✓ {json_file.name}: {len(data)} items")
                    else:
                        if isinstance(data, dict):
                            data['_source_file'] = json_file.name
                        self.knowledge_base.append(data)
                        print(f"  ✓ {json_file.name}: 1 item")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")
        
        # Load Markdown files - parse into sections untuk retrieval lebih baik
        md_files = list(self.dataset_dir.glob("*.md"))
        print(f"📝 Found {len(md_files)} Markdown files")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Parse markdown into sections berdasarkan header
                    sections = self._parse_markdown_sections(content, md_file.name)
                    
                    if sections:
                        self.knowledge_base.extend(sections)
                        print(f"  ✓ {md_file.name}: {len(sections)} sections")
                    else:
                        # Jika tidak ada section, load sebagai satu dokumen utuh
                        self.knowledge_base.append({
                            "source": md_file.name,
                            "content": content,
                            "type": "markdown",
                            "_source_file": md_file.name
                        })
                        print(f"  ✓ {md_file.name}: 1 document")
            except Exception as e:
                print(f"  ✗ Error loading {md_file.name}: {e}")
        
        # Load TXT files jika ada
        txt_files = list(self.dataset_dir.glob("*.txt"))
        if txt_files:
            print(f"📄 Found {len(txt_files)} TXT files")
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.knowledge_base.append({
                            "source": txt_file.name,
                            "content": content,
                            "type": "text",
                            "_source_file": txt_file.name
                        })
                    print(f"  ✓ {txt_file.name}")
                except Exception as e:
                    print(f"  ✗ Error loading {txt_file.name}: {e}")
        
        print(f"\n✅ Total knowledge base loaded: {len(self.knowledge_base)} documents")
        print(f"📊 Ready to answer questions!\n")
    
    def _parse_markdown_sections(self, content: str, filename: str):
        """Parse markdown content menjadi sections berdasarkan headers"""
        sections = []
        lines = content.split('\n')
        current_section = {
            "source": filename,
            "type": "markdown",
            "_source_file": filename,
            "header": "",
            "content": ""
        }
        
        for line in lines:
            # Detect headers (# Header atau ## Header, etc)
            if line.strip().startswith('#'):
                # Save previous section jika ada content
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                current_section = {
                    "source": filename,
                    "type": "markdown",
                    "_source_file": filename,
                    "header": header_text,
                    "header_level": header_level,
                    "content": line + "\n"
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def search_relevant_context(self, query: str, top_k: int = 10):
        """Cari konteks yang relevan dari knowledge base dengan scoring yang lebih pintar"""
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]  # Filter kata pendek
        
        # Keywords untuk topik spesifik
        topic_keywords = {
            'mpasi': ['mpasi', 'makanan pendamping asi', 'makanan bayi'],
            'aturan': ['aturan', 'prinsip', 'panduan', 'syarat'],
            'menu': ['menu', 'resep', 'makanan', 'bahan'],
            'gizi': ['gizi', 'nutrisi', 'kalori', 'protein', 'karbohidrat', 'lemak', 'vitamin'],
            'akg': ['akg', 'angka kecukupan gizi', 'kebutuhan gizi'],
            'tekstur': ['tekstur', 'lumat', 'lembut', 'kental', 'cincang'],
            'usia': ['bulan', 'usia', 'umur', 'tahap'],
            'alergi': ['alergi', 'intoleransi', 'pantangan', 'hindari'],
            'porsi': ['porsi', 'takaran', 'jumlah', 'banyak'],
            'frekuensi': ['frekuensi', 'jadwal', 'kali', 'sehari'],
        }
        
        # Deteksi topik dari query
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_topics.append(topic)
        
        scored_docs = []
        for doc in self.knowledge_base:
            score = 0
            doc_text = str(doc).lower()
            
            # Exact phrase match - skor tertinggi
            if query_lower in doc_text:
                score += 200
            
            # Topic relevance - boost jika dokumen cocok dengan topik yang terdeteksi
            for topic in detected_topics:
                topic_words = topic_keywords[topic]
                matches = sum(1 for kw in topic_words if kw in doc_text)
                if matches > 0:
                    score += matches * 15
            
            # Individual word matching dengan konteks
            for word in query_words:
                # Exact word match
                if f" {word} " in f" {doc_text} ":
                    score += 20
                # Partial word match
                elif word in doc_text:
                    score += 10
                # Fuzzy match (kata mirip)
                else:
                    for doc_word in doc_text.split():
                        if word in doc_word or doc_word in word:
                            if len(word) > 3 and len(doc_word) > 3:
                                score += 5
                                break
            
            # Boost untuk dokumen yang memiliki banyak keyword match
            matching_words = sum(1 for word in query_words if word in doc_text)
            if matching_words > 0:
                score += matching_words * 8
                # Bonus jika hampir semua kata ada
                if matching_words >= len(query_words) * 0.7:
                    score += 50
            
            # Boost untuk markdown content (biasanya lebih terstruktur)
            if isinstance(doc, dict) and doc.get("type") == "markdown":
                score = int(score * 1.2)
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top K
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Fallback: jika skor tertinggi terlalu rendah, coba cari lagi dengan topik umum
        if not scored_docs or scored_docs[0][0] < 30:
            print(f"⚠️ Low confidence results, adding general MPASI docs")
            for doc in self.knowledge_base:
                doc_text = str(doc).lower()
                if any(kw in doc_text for kw in ['mpasi', 'bayi', 'anak', 'gizi']):
                    # Cek apakah sudah ada
                    if doc not in [d for _, d in scored_docs]:
                        scored_docs.append((10, doc))
        
        return [doc for score, doc in scored_docs[:top_k]]
    
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


# Singleton instance
_chatbot_service = None

def get_chatbot_service():
    """Get or create chatbot service singleton"""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService(dataset_dir="../dataset")
    return _chatbot_service
