from flask import Flask, render_template, request, jsonify
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Import RAG service from existing query.py
from query import get_chroma_rag_service

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize RAG service (uses existing ChromaDB)
try:
    rag_service = get_chroma_rag_service()
    RAG_READY = True
    print("✓ Connected to existing ChromaDB and RAG service")
except Exception as e:
    print(f"✗ Error initializing RAG service: {e}")
    RAG_READY = False


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', rag_ready=RAG_READY)


@app.route('/api/search', methods=['POST'])
def search():
    """Search API endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        docs = rag_service.search_relevant_docs(query, top_k=top_k)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'top_k': top_k,
            'results_count': len(docs),
            'results': docs
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search-with-scores', methods=['POST'])
def search_with_scores():
    """Search with similarity scores API endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        results = rag_service.search_with_scores(query, top_k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'similarity_score': float(score)
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'top_k': top_k,
            'results_count': len(formatted_results),
            'results': formatted_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generate-menu', methods=['POST'])
def generate_menu():
    """Generate menu plan API endpoint"""
    try:
        data = request.get_json()
        
        user_input = {
            'age_months': data.get('age_months', 6),
            'weight_kg': data.get('weight_kg', 7),
            'height_cm': data.get('height_cm', 65),
            'allergies': data.get('allergies', []),
            'residence': data.get('residence', 'Indonesia')
        }
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        menu_plan = rag_service.generate_menu_plan_with_chroma(user_input)
        
        return jsonify(menu_plan)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/status')
def status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'rag_service': 'ready' if RAG_READY else 'unavailable'
    })


@app.route('/api/debug-prompt', methods=['POST'])
def debug_prompt():
    """DEBUG: Show the full prompt that would be sent to Gemini API (Two-Step Retrieval)"""
    try:
        data = request.get_json()
        
        user_input = {
            'age_months': data.get('age_months', 6),
            'weight_kg': data.get('weight_kg', 7),
            'height_cm': data.get('height_cm', 65),
            'allergies': data.get('allergies', []),
            'residence': data.get('residence', 'Indonesia')
        }
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        # Build the prompt exactly as generate_menu_plan_with_chroma would
        age_months = user_input['age_months']
        weight_kg = user_input['weight_kg']
        height_cm = user_input['height_cm']
        allergies = user_input['allergies']
        residence = user_input['residence']
        
        # STEP 1: Retrieve rules and AKG
        rules_query_parts = [
            f"MPASI usia {age_months} bulan",
            "angka kecukupan gizi AKG",
            "aturan pemberian MPASI",
            "tekstur makanan",
            "porsi makan",
            "frekuensi makan bayi",
        ]
        rules_query = " ".join(rules_query_parts)
        
        konteks_aturan = rag_service.search_relevant_docs(rules_query, top_k=15)
        if not konteks_aturan:
            konteks_aturan = rag_service.search_relevant_docs("AKG MPASI aturan", top_k=15)
        
        # STEP 2: Load TKPI data for context
        print(f"  Loading TKPI data...")
        
        tkpi_file = os.path.join(os.path.dirname(__file__), "../dataset/TKPI-2020.json")
        
        try:
            with open(tkpi_file, 'r', encoding='utf-8') as f:
                tkpi_data = json.load(f)
            print(f"  ✓ Loaded {len(tkpi_data)} items from TKPI dataset")
            
            # Create compact TKPI reference for the prompt
            tkpi_summary = "DAFTAR BAHAN MAKANAN TKPI-2020:\n\n"
            for i, item in enumerate(tkpi_data[:200]):  # Limit to first 200 for token efficiency
                kode = item.get('KODE', 'N/A')
                nama = item.get('NAMA BAHAN', 'N/A')
                energi = item.get('ENERGI (Kal)', '0')
                protein = item.get('PROTEIN (g)', '0')
                lemak = item.get('LEMAK (g)', '0')
                kh = item.get('KH (g)', '0')
                tkpi_summary += f"{kode} | {nama} | E:{energi}K P:{protein}g L:{lemak}g KH:{kh}g\n"
            
            konteks_bahan = [tkpi_summary]
            print(f"  ✓ Formatted 200 most common TKPI items")
            
        except Exception as e:
            print(f"  ❌ Error loading TKPI: {str(e)}")
            return jsonify({"error": f"Failed to load TKPI data: {str(e)}"}), 400
        
        # Format contexts
        formatted_aturan = "\n\n".join([
            f"=== ATURAN/AKG {i+1} ===\n{doc}" 
            for i, doc in enumerate(konteks_aturan)
        ])
        
        formatted_bahan = "\n\n".join([
            f"=== BAHAN MAKANAN TERSEDIA {i+1} ===\n{doc}" 
            for i, doc in enumerate(konteks_bahan)
        ])
        
        allergies_text = f"\n- PENTING: WAJIB hindari semua bahan yang mengandung {', '.join(allergies)}" if allergies else ""
        
        # Build final prompt
        prompt = f"""Kamu adalah sistem AI yang membuat rencana menu MPASI ORIGINAL untuk bayi.

INFORMASI BAYI:
- Usia: {age_months} bulan
- Berat Badan: {weight_kg} kg
- Tinggi Badan: {height_cm} cm
- Tempat Tinggal: {residence}{allergies_text}

==============================================
BAGIAN A: ATURAN MPASI DAN AKG (WAJIB IKUTI)
==============================================
{formatted_aturan}

==============================================
BAGIAN B: BAHAN MAKANAN YANG TERSEDIA (TKPI)
==============================================
{formatted_bahan}

==============================================
INSTRUKSI PEMBUATAN MENU (LANGKAH-LANGKAH):
==============================================

1. ANALISIS KEBUTUHAN GIZI:
   - Dari BAGIAN A, identifikasi AKG untuk usia {age_months} bulan
   - Dari BAGIAN A, identifikasi aturan tekstur dan porsi
   - Catat kebutuhan energi, protein, lemak, karbohidrat untuk hari ini

2. PILIH BAHAN DARI TKPI:
   - GUNAKAN HANYA bahan dari BAGIAN B (data TKPI-2020)
   - Pilih bahan dengan melihat kecocokan nutrisinya dengan kebutuhan
   - Sertakan KODE TKPI jika tersedia (contoh: AR001)
   - Hindari bahan alergen: {', '.join(allergies) if allergies else 'tidak ada'}

3. BUAT MENU YANG ORIGINAL:
   - Kombinasikan bahan-bahan TKPI dengan cara yang UNIK
   - Jangan hanya copy-paste kombinasi yang sudah ada
   - Variasikan tekstur sesuai usia {age_months} bulan (dari BAGIAN A)
   - Setiap makanan harus menggunakan bahan dari BAGIAN B

4. HITUNG NUTRISI:
   - Cari nilai gizi setiap bahan dari BAGIAN B
   - HITUNG MANUAL total nutrisi untuk setiap meal
   - Pastikan total harian MEMENUHI AKG dari BAGIAN A
   - Tulis ANGKA AKHIR, bukan rumus (contoh: 145 bukan 50+95)

5. VALIDASI:
   - Semua bahan harus dari BAGIAN B (TKPI)
   - Semua aturan harus dari BAGIAN A (MPASI rules)
   - Tidak boleh menggunakan pengetahuan di luar kedua bagian ini

LARANGAN KETAT:
❌ JANGAN gunakan bahan yang tidak ada di BAGIAN B
❌ JANGAN buat kombinasi menu dari template/hafalan
❌ JANGAN gunakan aturan yang tidak ada di BAGIAN A
❌ JANGAN tulis rumus dalam nilai nutrisi JSON
❌ JANGAN tambahkan informasi dari di luar BAGIAN A dan B

[JSON FORMAT EXAMPLE - lihat di query.py untuk full format]

PENTING: 
- HANYA gunakan data dari BAGIAN A (aturan) dan BAGIAN B (bahan)
- JANGAN tambah informasi dari luar
- TULIS HANYA JSON VALID, TIDAK ADA TEXT LAIN!"""
        
        return jsonify({
            'status': 'success',
            'debug_info': {
                'step_1_rules_query': rules_query,
                'step_2_source': 'Full TKPI Dataset',
                'step_1_documents_retrieved': len(konteks_aturan),
                'step_2_documents_retrieved': len(konteks_bahan),
                'total_documents': len(konteks_aturan) + len(konteks_bahan),
                'baby_info': user_input
            },
            'full_prompt': prompt,
            'prompt_length_chars': len(prompt),
            'context_aturan_summary': [
                {
                    'doc_index': i,
                    'length': len(doc),
                    'preview': doc[:200] + '...' if len(doc) > 200 else doc
                }
                for i, doc in enumerate(konteks_aturan)
            ],
            'context_bahan_summary': [
                {
                    'doc_index': i,
                    'length': len(doc),
                    'preview': doc[:200] + '...' if len(doc) > 200 else doc
                }
                for i, doc in enumerate(konteks_bahan)
            ]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🍽️  RAG MPASI Menu Generator - Flask Web UI")
    print("="*70)
    print(f"ChromaDB: {'✓ Connected' if RAG_READY else '✗ Not available'}")
    print(f"Gemini API: {'✓ Ready' if RAG_READY else '✗ Not configured'}")
    print("\n📍 Server: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
