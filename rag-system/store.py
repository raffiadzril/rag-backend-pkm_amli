import os
from dotenv import load_dotenv
import json
import re

# 1. IMPORT THE LOCAL HUGGINGFACE MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

data_path = '../dataset'
db_path = './chroma_db'

# Files to exclude from automatic JSON ingestion (case-insensitive)
# We exclude files that are processed separately so they are not ingested twice.
EXCLUDE_JSON_FILES = {'akg_merged.json', 'aturan-mpasi.json', 'TKPI-2020.json'}

# 2. INITIALIZE THE LOCAL MODEL
# This model (all-MiniLM-L6-v2) is small, fast, and excellent for RAG.
# It runs entirely on your computer (no API key needed).
print("Initializing local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("successfully initialized embeddings model.")

if os.path.exists(db_path):
    print(f"Database folder '{db_path}' already exists.")
    print("To re-index, please delete this folder first and re-run.")
    exit()

def process_akg_file():
    """Process the AKG file with metadata for hybrid search"""
    akg_file = os.path.join(data_path, 'akg_merged.json')
    if not os.path.exists(akg_file):
        return []
    
    print(f"Processing AKG file: {akg_file}")
    with open(akg_file, 'r', encoding='utf-8') as f:
        akg_data = json.load(f)
    
    chunks = []
    for entry in akg_data:
        # Extract age range from "Kelompok Umur" field
        age_range = str(entry.get("Kelompok Umur", "")).lower()
        
        # Extract age range like "0 - 5 Bulan" or "6 - 11 Bulan"
        age_match = re.search(r'(\d+)\s*[-–—–]\s*(\d+)\s*(bulan|bln|months)', age_range)
        if age_match:
            age_start = int(age_match.group(1))
            age_end = int(age_match.group(2))
        else:
            # Try to find single age values like "24 - 59 bulan"
            age_match = re.search(r'^(\d+)\s*[-–—–]\s*(\d+)\s*', age_range)
            if age_match:
                age_start = int(age_match.group(1))
                age_end = int(age_match.group(2))
            else:
                # If no clear age range found, set defaults
                age_start = age_end = 0
        
        # Create document with metadata
        content = json.dumps(entry, ensure_ascii=False, indent=2)
        doc = Document(
            page_content=content,
            metadata={
                "tipe_dokumen": "akg",
                "usia_mulai_bulan": age_start,
                "usia_selesai_bulan": age_end,
                "topik": "angka_kecukupan_gizi",
                "original_file": "akg_merged.json"
            }
        )
        chunks.append(doc)
    
    return chunks

def process_aturan_file():
    """Process the aturan-mpasi.json file with metadata for hybrid search"""
    aturan_file = os.path.join(data_path, 'aturan-mpasi.json')
    if not os.path.exists(aturan_file):
        return []
    
    print(f"Processing Aturan MPASI file: {aturan_file}")
    with open(aturan_file, 'r', encoding='utf-8') as f:
        aturan_data = json.load(f)
    
    chunks = []
    for entry in aturan_data:
        # Extract age range from "Usia" field
        age_info = str(entry.get("Usia", "")).lower()
        
        # Extract age range like "6-8 bulan"
        age_match = re.search(r'(\d+)\s*[-–—–]\s*(\d+)\s*(bulan|bln|months)', age_info)
        if age_match:
            age_start = int(age_match.group(1))
            age_end = int(age_match.group(2))
        else:
            # Try to match "Jika tidak mendapat ASI (6-23 bulan)" format
            alt_match = re.search(r'\((\d+)\s*[-–—–]\s*(\d+)\s*(bulan|bln|months)\)', age_info)
            if alt_match:
                age_start = int(alt_match.group(1))
                age_end = int(alt_match.group(2))
            else:
                # If no clear age found, set defaults
                age_start = age_end = 0
        
        # Create document with metadata
        content = json.dumps(entry, ensure_ascii=False, indent=2)
        doc = Document(
            page_content=content,
            metadata={
                "tipe_dokumen": "aturan_porsi",
                "usia_mulai_bulan": age_start,
                "usia_selesai_bulan": age_end,
                "topik": "aturan_porsi_frekuensi_tekstur",
                "original_file": "aturan-mpasi.json"
            }
        )
        chunks.append(doc)
    
    return chunks

def process_markdown_files():
    """Process markdown files with appropriate metadata"""
    markdown_docs = []
    
    # Define md files with their expected metadata
    md_files_metadata = {
        '2-prinsip-mpasi.md': {"tipe_dokumen": "prinsip_dasar", "topik": "prinsip_dasar"},
        '4-syarat-mpasi.md': {"tipe_dokumen": "prinsip_dasar", "topik": "4_syarat_mpasi"},
        'makanan-dilarang.md': {"tipe_dokumen": "batasan_mpasi", "topik": "makanan_dilarang"},
        'masalah-pemberian-mpasi.md': {"tipe_dokumen": "masalah", "topik": "masalah_pemberian"},
        'mpasi-yang-baik.md': {"tipe_dokumen": "rekomendasi", "topik": "mpasi_yang_baik"},
        'penyiapan-pemberian-mpasi.md': {"tipe_dokumen": "cara_pemberian", "topik": "penyiapan_pemberian"}
    }
    
    for md_file, metadata in md_files_metadata.items():
        file_path = os.path.join(data_path, md_file)
        if os.path.exists(file_path):
            print(f"Processing markdown file: {file_path}")
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            
            # Add metadata to each document
            for doc in docs:
                doc.metadata.update({
                    "tipe_dokumen": metadata["tipe_dokumen"],
                    "topik": metadata["topik"],
                    "original_file": md_file
                })
                # Set default age range for general information
                if "usia_mulai_bulan" not in doc.metadata:
                    doc.metadata["usia_mulai_bulan"] = 0
                if "usia_selesai_bulan" not in doc.metadata:
                    doc.metadata["usia_selesai_bulan"] = 24  # Up to 24 months
                    
                markdown_docs.append(doc)
    
    return markdown_docs

def process_other_json_files():
    """Process other JSON files that aren't AKG or Aturan"""
    other_json_docs = []
    
    # Process all JSON files except akg_merged.json and aturan-mpasi.json
    # Exclude core files (AKG, Aturan, TKPI) and compare in lower-case to avoid case mismatches
    exclude_lower = {fn.lower() for fn in EXCLUDE_JSON_FILES}
    for file in os.listdir(data_path):
        file_lower = file.lower()
        if file_lower.endswith('.json') and file_lower not in exclude_lower:
            json_file_path = os.path.join(data_path, file)
            print(f"Processing other JSON file: {json_file_path}")
            
            # Load with JSONLoader
            loader = JSONLoader(
                file_path=json_file_path,
                jq_schema='.[] | tostring',
                text_content=True
            )
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "tipe_dokumen": "lainnya",
                    "topik": "lainnya",
                    "original_file": file,
                    "usia_mulai_bulan": 0,  # General info
                    "usia_selesai_bulan": 24  # Up to 24 months
                })
                other_json_docs.append(doc)
    
    return other_json_docs

# Process all files with enhanced metadata
print("Processing AKG data with metadata...")
akg_docs = process_akg_file()

print("Processing Aturan MPASI data with metadata...")
aturan_docs = process_aturan_file()

print("Processing markdown files with metadata...")
md_docs = process_markdown_files()

print("Processing other JSON files with metadata...")
other_json_docs = process_other_json_files()

# Combine all documents
all_docs = akg_docs + aturan_docs + md_docs + other_json_docs
print(f"Total documents with enhanced metadata: {len(all_docs)}")

print("Adding additional content-based metadata...")
# Add content-based metadata for better retrieval
for doc in all_docs:
    content = doc.page_content.lower()
    
    # Initialize kategori if not present
    if "kategori" not in doc.metadata:
        doc.metadata["kategori"] = ""
    
    # Add more specific metadata based on content
    categories = []
    if "energi" in content or "kcal" in content or "kkal" in content:
        categories.append("energi")
    if "protein" in content:
        categories.append("protein")
    if "tekstur" in content:
        categories.append("tekstur")
    if "frekuensi" in content or "kali" in content:
        categories.append("frekuensi")
    if "porsi" in content:
        categories.append("porsi")
    if "variasi" in content:
        categories.append("variasi")
    
    if categories:
        doc.metadata["kategori"] = doc.metadata["kategori"] + "," + ",".join(categories)
        # Remove leading comma if it exists
        if doc.metadata["kategori"].startswith(","):
            doc.metadata["kategori"] = doc.metadata["kategori"][1:]

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
doc_chunks = text_splitter.split_documents(all_docs)
print(f"Total chunks created: {len(doc_chunks)}")

# 3. CREATE THE VECTOR STORE (No batching needed)
# This will run locally and process all chunks at once.
print("Creating Chroma vector store... (This may take a moment)")
vectordb = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embeddings,
    persist_directory=db_path
)

print("Vector store successfully created and persisted to disk.")
print("Setup complete. Vector store is ready for use with hybrid search capabilities.")