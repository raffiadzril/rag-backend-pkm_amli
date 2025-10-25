# Gemini RAG System

Sistem RAG (Retrieval-Augmented Generation) menggunakan Google Gemini API untuk menjawab pertanyaan berdasarkan dataset lokal.

## Setup

1. **Aktifkan virtual environment:**
   ```bash
   source gemini-rag/bin/activate
   ```

2. **Install dependencies (sudah terinstall):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurasi API Key:**
   Edit file `.env` jika perlu mengganti API key

## Cara Menggunakan

### Mode Interaktif (Chat)
```bash
python rag_system.py
```

### Mode Programmatic
```bash
python example_usage.py
```

Atau gunakan di script Anda:
```python
from rag_system import GeminiRAG

rag = GeminiRAG(dataset_dir="dataset")
jawaban = rag.query("Apa kandungan protein dalam beras?")
print(jawaban)
```

## Menambah Dataset Baru

Cukup tambahkan file JSON baru ke folder `dataset/`:
- Format bisa array of objects atau single object
- Sistem otomatis load semua file `.json` di folder dataset
- Tidak perlu modifikasi kode

## Struktur Project

```
gemini-rag/
├── dataset/           # Folder dataset (tambah file JSON di sini)
├── rag_system.py      # Main RAG system
├── example_usage.py   # Contoh penggunaan
├── .env              # API key configuration
└── requirements.txt   # Dependencies
```

## Tips

- Sistem menggunakan keyword-based search yang efisien
- Bisa handle multiple dataset files
- Scalable untuk dataset besar
- API key disimpan aman di .env
