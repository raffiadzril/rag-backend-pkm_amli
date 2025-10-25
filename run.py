"""
Script untuk menjalankan server FastAPI
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload saat ada perubahan code
        log_level="info"
    )
