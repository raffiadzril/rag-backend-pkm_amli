"""
Contoh penggunaan RAG system secara programmatic
"""
from rag_system import GeminiRAG

# Inisialisasi RAG
rag = GeminiRAG(dataset_dir="dataset")

# Contoh query tunggal
print("="*60)
print("CONTOH 1: Query tunggal")
print("="*60)
answer = rag.query("Berapa Energi dalam 100 gram nasi beras merah?")
print(f"\nJawaban:\n{answer}\n")

# Contoh query lain
print("\n" + "="*60)
print("CONTOH 2: Query berbeda")
print("="*60)
answer = rag.query("Berapa kandungan kalsium dalam beras giling?")
print(f"\nJawaban:\n{answer}\n")

# Untuk mode chat interaktif, uncomment baris berikut:
# rag.chat()
