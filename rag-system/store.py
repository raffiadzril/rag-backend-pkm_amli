import os
from dotenv import load_dotenv
import json 

# 1. IMPORT THE LOCAL HUGGINGFACE MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import DirectoryLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

data_path = '../dataset'
db_path = './chroma_db'

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

print(f"loading json and markdown files from {data_path}")

# --- 1. Markdown Loader ---
md_loader = DirectoryLoader(
    data_path,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True,
    recursive=True
)

# --- 2. JSON Loader ---
json_loader = DirectoryLoader(
    data_path,
    glob="**/*.json",
    loader_cls=JSONLoader,
    loader_kwargs={'jq_schema': '.[] | tostring', 'text_content': True},
    show_progress=True,
    recursive=True
)

print("loading documents...")
md_docs = md_loader.load()
json_docs = json_loader.load()
print(f"successfully loaded {len(md_docs)} markdown and {len(json_docs)} json documents.")

print("merging documents...")
all_docs = md_docs + json_docs
print(f"total documents loaded: {len(all_docs)}")

print("splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
doc_chunks = text_splitter.split_documents(all_docs)
print(f"total chunks created: {len(doc_chunks)}")

# 3. CREATE THE VECTOR STORE (No batching needed)
# This will run locally and process all chunks at once.
print("Creating Chroma vector store... (This may take a moment)")
vectordb = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embeddings,
    persist_directory=db_path
)

print("vector store successfully created and persisted to disk.")
print("Setup complete. Vector store is ready for use.")