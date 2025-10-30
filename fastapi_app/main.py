from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add the rag-system path to import the query module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag-system'))

from query import get_chroma_rag_service
from query_lm_studio import get_chroma_rag_service_lm_studio, available_models as lm_studio_models, lm_studio_ready

app = FastAPI(
    title="MPASI Menu Generator API",
    description="API for generating MPASI (Pendamping ASI) menus for babies",
    version="1.0.0"
)

# Initialize RAG services
try:
    rag_service_gemini = get_chroma_rag_service()
    GEMINI_READY = True
    print("✓ Connected to ChromaDB and Gemini API service")
except Exception as e:
    print(f"✗ Error initializing Gemini RAG service: {e}")
    GEMINI_READY = False

try:
    rag_service_lm_studio = get_chroma_rag_service_lm_studio()
    LM_STUDIO_READY = lm_studio_ready
    print(f"✓ Connected to ChromaDB and LM Studio service (Ready: {LM_STUDIO_READY})")
except Exception as e:
    print(f"✗ Error initializing LM Studio RAG service: {e}")
    LM_STUDIO_READY = False

RAG_READY = GEMINI_READY or LM_STUDIO_READY

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
            "gemini": "ready" if GEMINI_READY else "unavailable",
            "lm_studio": "ready" if LM_STUDIO_READY else "unavailable"
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
    
    if LM_STUDIO_READY:
        lm_models = rag_service_lm_studio.get_available_models()
        for model in lm_models:
            models.append({
                "id": model,
                "name": model,
                "provider": "LM Studio (Local)",
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
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        # Prepare user input in the format expected by the RAG service
        user_input = {
            'umur_bulan': request.umur_bulan,
            'berat_badan': request.berat_badan,
            'tinggi_badan': request.tinggi_badan,
            'jenis_kelamin': request.jenis_kelamin,
            'tempat_tinggal': request.tempat_tinggal,
            'alergi': request.alergi
        }
        
        # Determine which service to use
        if request.model_type == 'lm_studio':
            if not LM_STUDIO_READY:
                raise HTTPException(status_code=503, detail="LM Studio service not available. Make sure LM Studio is running.")
            
            if request.model_name is None and available_models:
                model_name = available_models[0] if available_models else None
            else:
                model_name = request.model_name
                
            menu_plan = rag_service_lm_studio.generate_menu_plan_with_chroma(user_input, model_name)
        else:  # Default to Gemini
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