from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import menu

# Import chatbot routes (folder terpisah, tidak mengganggu yang lain)
try:
    from chatbot import routes as chatbot_routes
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False
    print("⚠️ Chatbot module not available")

app = FastAPI(
    title="GATA MPASI Backend API",
    description="Backend API untuk Menu Planner dan Chatbot"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router untuk menu generation (existing)
app.include_router(menu.router)

# Router untuk chatbot (new, terpisah)
if CHATBOT_AVAILABLE:
    app.include_router(chatbot_routes.router)
    print("✅ Chatbot routes loaded")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    """Root endpoint dengan info API"""
    return {
        "service": "GATA MPASI Backend API",
        "version": "1.0.0",
        "endpoints": {
            "menu": "/api/generate-menu (POST) - Generate menu MPASI",
            "menu_status": "/api/status (GET) - Status menu service",
            "chatbot": "/api/chatbot/chat (POST) - Chat dengan asisten gizi" if CHATBOT_AVAILABLE else "Not available",
            "chatbot_status": "/api/chatbot/status (GET) - Status chatbot" if CHATBOT_AVAILABLE else "Not available",
            "docs": "/docs - Swagger UI documentation"
        },
        "chatbot_enabled": CHATBOT_AVAILABLE
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
