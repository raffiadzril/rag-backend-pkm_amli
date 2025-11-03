import os
import gc
from dotenv import load_dotenv
import json
import re
import torch

# 1. IMPORT THE LOCAL HUGGINGFACE MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

load_dotenv()

# Memory optimization: configure ChromaDB settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True
)

data_path = '../dataset'
db_path = './chroma_db'

# Memory optimization: clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")

# Exact file names to exclude from ingestion (case-insensitive).
# We intentionally exclude TKPI dataset files (they are processed/uploaded separately)
# so they are NOT ingested into Chroma. Use exact filename matching (case-insensitive).
EXCLUDE_FILE_NAMES = {"TKPI-2020.json", "TKPI_COMPACT.txt", "akg_merged.json", "akg_merged_with_descriptions.json"}

# 2. INITIALIZE THE LOCAL MODEL
# Use BGE-small for reduced memory footprint (~500MB vs ~1.3GB for large)
print("Initializing local embedding model (BAAI/bge-small-en-v1.5)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        'device': 'cpu',  # Force CPU to control memory usage
        'trust_remote_code': True
    },
    encode_kwargs={
        'normalize_embeddings': True,  # Improves quality while keeping memory usage same
        'batch_size': 32  # Smaller batch size for less memory
    }
)
print("successfully initialized embeddings model.")

if os.path.exists(db_path):
    print(f"Database folder '{db_path}' already exists.")
    print("To re-index, please delete this folder first and re-run.")
    exit()


def collect_documents_simple():
    """Collect all dataset files and convert them to Document objects with no metadata.

    Behavior:
    - Skip JSON files listed in EXCLUDE_JSON_FILES (case-insensitive).
    - For .json files: load as JSON and convert each top-level entry to a string document.
    - For .md and other text files: read file content as a single document.
    - Return a list of Documents with empty metadata.
    """
    docs = []
    # Prepare a lower-cased set of exact filenames to skip
    exclude_names = {s.lower() for s in EXCLUDE_FILE_NAMES}

    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path, fname)
        if os.path.isdir(fpath):
            continue

        lower = fname.lower()
        try:
            # Global exclusion check (apply to any file type).
            # Use literal filename matching (case-insensitive) rather than substring matching.
            if lower in exclude_names:
                print(f"Skipping excluded dataset file (literal match): {fname}")
                continue
            if lower.endswith('.json'):
                print(f"Loading JSON file: {fname}")
                with open(fpath, 'r', encoding='utf-8') as f:
                    try:
                        arr = json.load(f)
                        # If it's a list, convert each item to a document
                        if isinstance(arr, list):
                            for item in arr:
                                content = json.dumps(item, ensure_ascii=False)
                                docs.append(Document(page_content=content, metadata={}))
                        else:
                            # Single JSON object -> stringify
                            content = json.dumps(arr, ensure_ascii=False)
                            docs.append(Document(page_content=content, metadata={}))
                    except Exception as e:
                        # Fallback: read raw text
                        print(f"  Warning: failed to parse JSON {fname}: {e}. Reading raw content.")
                        with open(fpath, 'r', encoding='utf-8') as fr:
                            docs.append(Document(page_content=fr.read(), metadata={}))

            elif lower.endswith('.md') or lower.endswith('.txt'):
                print(f"Loading text/markdown file: {fname}")
                with open(fpath, 'r', encoding='utf-8') as f:
                    docs.append(Document(page_content=f.read(), metadata={}))
            else:
                # Generic fallback for other file types: read as text
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        docs.append(Document(page_content=f.read(), metadata={}))
                except Exception:
                    # Binary or unreadable file -> skip
                    print(f"Skipping file (unreadable as text): {fname}")
        except Exception as e:
            print(f"Error processing file {fname}: {e}")

    return docs


print("Collecting documents (simple mode, no metadata)...")
collected_docs = collect_documents_simple()
print(f"Total documents collected: {len(collected_docs)}")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
doc_chunks = text_splitter.split_documents(collected_docs)
print(f"Total chunks created: {len(doc_chunks)}")

# Memory optimization: process in smaller batches
print("Creating Chroma vector store... (This may take a moment)")
BATCH_SIZE = 100  # Process 100 documents at a time

for i in range(0, len(doc_chunks), BATCH_SIZE):
    batch = doc_chunks[i:i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1} of {len(doc_chunks)//BATCH_SIZE + 1}")
    
    if i == 0:
        # First batch: create new vectordb
        vectordb = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=db_path,
            client_settings=CHROMA_SETTINGS
        )
    else:
        # Subsequent batches: add to existing
        vectordb.add_documents(batch)
    
    # Memory optimization: force garbage collection after each batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Ensure everything is written to disk
vectordb.persist()

print("Vector store successfully created and persisted to disk.")
print("Setup complete. Vector store is ready (no metadata stored).")

# Final cleanup
del vectordb
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()