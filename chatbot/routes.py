"""
Chatbot API Routes
Endpoint terpisah untuk chatbot agar tidak mengganggu menu generation
"""

from typing import List, Optional
from fastapi import APIRouter, Body
from pydantic import BaseModel
from chatbot.chatbot_service import get_chatbot_service

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


class ChatMessage(BaseModel):
    sender: str  # "user" or "bot"
    text: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str
    status: str
    sources_used: int
    has_context: bool


@router.post("/chat", response_model=dict)
async def chat(request: ChatRequest = Body(...)):
    """
    Endpoint untuk chatbot conversation
    
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


@router.get("/status")
async def chatbot_status():
    """Check chatbot service status"""
    try:
        chatbot = get_chatbot_service()
        return {
            "status": "ok",
            "service": "GATA Chatbot Service",
            "knowledge_base_size": len(chatbot.knowledge_base),
            "ready": True
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "GATA Chatbot Service",
            "error": str(e),
            "ready": False
        }


@router.get("/test")
async def test_chatbot():
    """Test endpoint untuk memastikan chatbot berfungsi"""
    chatbot = get_chatbot_service()
    
    test_message = "Kapan bayi boleh mulai MPASI?"
    result = chatbot.generate_response(test_message)
    
    return {
        "test_message": test_message,
        "result": result,
        "timestamp": "test"
    }
