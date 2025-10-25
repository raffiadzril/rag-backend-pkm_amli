from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import menu_routes

# Create FastAPI app
app = FastAPI(
    title="MPASI Menu Planner API",
    description="API untuk generate menu MPASI menggunakan RAG dengan Gemini AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan domain frontend yang spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(menu_routes.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MPASI Menu Planner API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
