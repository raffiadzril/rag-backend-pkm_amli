from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add the rag-system path to import the query module
# The repo layout has `rag-system` as a sibling of `fastapi_app`, so resolve the parent
import pathlib
rag_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag-system'))
if not os.path.isdir(rag_system_path):
    # Fallback: try repo root rag-system (if running from repo root)
    rag_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'rag-system'))
    
sys.path.insert(0, rag_system_path)
print(f"[DEBUG] Added rag-system to sys.path: {rag_system_path}")

# Import only the Gemini RAG service
try:
    from query import get_chroma_rag_service
    print("✓ Successfully imported query module")
    GEMINI_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"✗ Error importing Gemini RAG service: {e}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    get_chroma_rag_service = None
    GEMINI_IMPORT_SUCCESS = False

app = FastAPI(
    title="MPASI Menu Generator API",
    description="API for generating MPASI (Pendamping ASI) menus for babies",
    version="1.0.0"
)

# Allow all CORS (development / permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG services
GEMINI_READY = False

if GEMINI_IMPORT_SUCCESS:
    try:
        print("Attempting to initialize Gemini RAG service...")
        rag_service_gemini = get_chroma_rag_service()
        GEMINI_READY = True
        print("✓ Successfully connected to ChromaDB and initialized Gemini API service")
    except Exception as e:
        print(f"✗ Error initializing Gemini RAG service: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        GEMINI_READY = False
else:
    print("✗ Gemini RAG service not available due to import error")

RAG_READY = GEMINI_READY
print(f"RAG Ready Status: {RAG_READY} (GEMINI_READY: {GEMINI_READY}, GEMINI_IMPORT_SUCCESS: {GEMINI_IMPORT_SUCCESS})")

# Pydantic models for request/response
class MenuGenerationRequest(BaseModel):
    umur_bulan: int
    berat_badan: float
    tinggi_badan: int
    jenis_kelamin: str = "laki-laki"
    tempat_tinggal: str = "Indonesia"
    alergi: List[str] = []
    model_type: str = "gemini"  # 'gemini' or 'lm_studio'
    model_name: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "MPASI Menu Generator API", "status": "running"}

@app.get("/api/status")
def get_status():
    """Check API status"""
    return {
        "status": "online",
        "services": {
            "chromadb": "ready",
            "gemini": "ready" if GEMINI_READY else "unavailable"
        }
    }

@app.get("/api/models")
def get_models():
    """Get available models"""
    models = []
    
    if GEMINI_READY:
        models.append({
            "id": "gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "provider": "Google Gemini API",
            "available": True
        })
    
    return {
        "status": "success",
        "models": models,
        "total": len(models)
    }

@app.post("/api/generate-menu")
def generate_menu(request: MenuGenerationRequest):
    """Generate MPASI menu plan"""
    try:
        if not RAG_READY:
            detail_msg = f"RAG service not available. Status - GEMINI_READY: {GEMINI_READY}, GEMINI_IMPORT_SUCCESS: {GEMINI_IMPORT_SUCCESS}"
            print(f"✗ RAG service not available: {detail_msg}")
            raise HTTPException(status_code=503, detail=detail_msg)
        
        # Prepare user input in the format expected by the RAG service
        user_input = {
            'umur_bulan': request.umur_bulan,
            'berat_badan': request.berat_badan,
            'tinggi_badan': request.tinggi_badan,
            'jenis_kelamin': request.jenis_kelamin,
            'tempat_tinggal': request.tempat_tinggal,
            'alergi': request.alergi
        }
        
        # Use Gemini service only
        if not GEMINI_READY:
            raise HTTPException(status_code=503, detail="Gemini API service not available.")
        
        menu_plan = rag_service_gemini.generate_menu_plan_with_chroma(user_input)
        
        if menu_plan.get('status') == 'error':
            raise HTTPException(status_code=400, detail=menu_plan.get('message', 'Unknown error occurred'))
        
        return menu_plan
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating menu plan: {str(e)}")

@app.post("/api/debug-prompt")
def debug_prompt(request: MenuGenerationRequest):
    """Debug endpoint to return the prompt that would be sent to the AI"""
    try:
        # For now, this is a placeholder - in a real implementation, you'd have a method
        # to generate the prompt without calling the LLM
        return {
            "status": "success",
            "search_query": f"Aturan MPASI dan AKG angka kecukupan gizi untuk usia {request.umur_bulan} bulan",
            "documents_retrieved": 0,
            "prompt_length": 0,
            "full_prompt": "Prompt generation not implemented for debug endpoint in FastAPI"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating debug prompt: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)