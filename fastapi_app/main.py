from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# Add chatbot path
chatbot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chatbot'))
sys.path.insert(0, chatbot_path)
print(f"[DEBUG] Added chatbot to sys.path: {chatbot_path}")

# Memory management and cleanup
import gc
import torch

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

# Import chatbot service
try:
    from chatbot_service import get_chatbot_service
    print("✓ Successfully imported chatbot_service module")
    CHATBOT_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"✗ Error importing chatbot service: {e}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    get_chatbot_service = None
    CHATBOT_IMPORT_SUCCESS = False

app = FastAPI(
    title="GATA MPASI Backend API",
    description="API untuk Menu Generation dan Chatbot MPASI",
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

# Initialize Chatbot service
CHATBOT_READY = False

if CHATBOT_IMPORT_SUCCESS:
    try:
        print("Attempting to initialize Chatbot service...")
        chatbot_service = get_chatbot_service()
        CHATBOT_READY = True
        print("✓ Successfully initialized Chatbot service")
    except Exception as e:
        print(f"✗ Error initializing Chatbot service: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        CHATBOT_READY = False
else:
    print("✗ Chatbot service not available due to import error")

# Pydantic models for Menu Generation
class MenuGenerationRequest(BaseModel):
    umur_bulan: int
    berat_badan: float
    tinggi_badan: int
    jenis_kelamin: str = "laki-laki"
    tempat_tinggal: str = "Indonesia"
    alergi: List[str] = []
    model_type: str = "gemini"  # 'gemini' or 'lm_studio'
    model_name: Optional[str] = None


# Pydantic models for Chatbot
class ChatMessage(BaseModel):
    sender: str = Field(..., description="user or bot")
    text: str = Field(..., description="Message content")
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Conversation history"
    )


class ChatResponse(BaseModel):
    response: str
    status: str
    sources_used: int
    has_context: bool

@app.get("/")
def read_root():
    """Root endpoint dengan info API"""
    return {
        "service": "GATA MPASI Backend API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "menu_generation": {
                "generate": "/api/generate-menu (POST)",
                "status": "/api/status (GET)",
                "models": "/api/models (GET)",
                "debug": "/api/debug-prompt (POST)"
            },
            "chatbot": {
                "chat": "/api/chatbot/chat (POST)",
                "status": "/api/chatbot/status (GET)",
                "test": "/api/chatbot/test (GET)"
            },
            "docs": "/docs (Swagger UI)"
        },
        "services": {
            "menu_generation": "ready" if GEMINI_READY else "unavailable",
            "chatbot": "ready" if CHATBOT_READY else "unavailable"
        }
    }

@app.get("/api/status")
def get_status():
    """Check API status"""
    return {
        "status": "online",
        "services": {
            "chromadb": "ready",
            "gemini": "ready" if GEMINI_READY else "unavailable",
            "chatbot": "ready" if CHATBOT_READY else "unavailable"
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


# ========================================
# CHATBOT ENDPOINTS
# ========================================

@app.post("/api/chatbot/chat", response_model=dict)
async def chatbot_chat(request: ChatRequest = Body(...)):
    """
    Chat dengan asisten gizi MPASI
    
    Request body:
    {
        "message": "Kapan harus mulai MPASI?",
        "conversation_history": [
            {"sender": "user", "text": "halo"},
            {"sender": "bot", "text": "halo juga"}
        ]
    }
    
    Response:
    {
        "status": "success",
        "response": "Jawaban dari bot...",
        "sources_used": 3,
        "has_context": true
    }
    """
    try:
        if not CHATBOT_READY:
            raise HTTPException(
                status_code=503, 
                detail="Chatbot service not available"
            )
        
        chatbot = get_chatbot_service()
        
        # Convert conversation history to dict
        history = [
            {"sender": msg.sender, "text": msg.text}
            for msg in request.conversation_history
        ] if request.conversation_history else []
        
        result = chatbot.generate_response(
            user_message=request.message,
            conversation_history=history
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error in chatbot: {str(e)}"
        )


@app.get("/api/chatbot/status")
async def chatbot_status():
    """Check chatbot service status"""
    try:
        if not CHATBOT_READY:
            return {
                "status": "unavailable",
                "service": "GATA Chatbot Service",
                "ready": False,
                "error": "Chatbot service not initialized"
            }
        
        chatbot = get_chatbot_service()
        # Test retrieval to confirm service is working
        test_docs = chatbot.search_relevant_context("test", top_k=1)
        
        return {
            "status": "ok",
            "service": "GATA Chatbot Service",
            "rag_service": "connected",
            "ready": True
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "GATA Chatbot Service",
            "error": str(e),
            "ready": False
        }


@app.get("/api/chatbot/test")
async def chatbot_test():
    """Test endpoint untuk memastikan chatbot berfungsi"""
    try:
        if not CHATBOT_READY:
            return {
                "status": "error",
                "message": "Chatbot service not available"
            }
        
        chatbot = get_chatbot_service()
        
        test_message = "Kapan bayi boleh mulai MPASI?"
        result = chatbot.generate_response(test_message)
        
        return {
            "test_message": test_message,
            "result": result,
            "timestamp": "test"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    global rag_service_gemini, chatbot_service
    print("\nShutting down FastAPI application...")
    
    # Clean up RAG service
    if GEMINI_READY and rag_service_gemini:
        try:
            print("Cleaning up RAG service...")
            # Clear embeddings from memory
            if hasattr(rag_service_gemini, 'embeddings'):
                if hasattr(rag_service_gemini.embeddings, 'client'):
                    rag_service_gemini.embeddings.client = None
            
            # Clean up vector store
            if hasattr(rag_service_gemini, 'vectordb'):
                rag_service_gemini.vectordb = None
            
            rag_service_gemini = None
            print("✓ RAG service cleanup completed")
        except Exception as e:
            print(f"✗ Error during RAG service cleanup: {e}")
    
    # Clean up chatbot service
    if CHATBOT_READY and chatbot_service:
        try:
            print("Cleaning up Chatbot service...")
            chatbot_service.model = None
            chatbot_service.rag_service = None
            chatbot_service = None
            print("✓ Chatbot service cleanup completed")
        except Exception as e:
            print(f"✗ Error during Chatbot service cleanup: {e}")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Memory cleanup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)