import sys
import os
import traceback

# Add the rag-system path to import the query module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag-system'))

print("Attempting to import query module...")

try:
    from query import get_chroma_rag_service
    print("[SUCCESS] Successfully imported query module")
    
    print("Attempting to initialize RAG service...")
    try:
        rag_service = get_chroma_rag_service()
        print("[SUCCESS] Successfully initialized RAG service")
        
        print("Testing a simple health check...")
        # Test with minimal user input
        test_input = {
            'umur_bulan': 6,
            'berat_badan': 7.0,
            'tinggi_badan': 65,
            'jenis_kelamin': 'laki-laki',
            'tempat_tinggal': 'Indonesia',
            'alergi': []
        }
        print("[SUCCESS] RAG service is working correctly")
        
    except Exception as e:
        print(f"[ERROR] Error initializing RAG service: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        
except ImportError as e:
    print(f"[ERROR] Error importing query module: {e}")
    print(f"Full traceback: {traceback.format_exc()}")