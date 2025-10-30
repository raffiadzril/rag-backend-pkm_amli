@echo off
echo Starting FastAPI MPASI Menu Generator...
cd /d "C:\Users\Putra\Documents\Project\Lomba\PKM-AMLI\PKM-KI\rag-backend-pkm_amli\fastapi_app"
conda activate gata
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000