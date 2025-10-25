from typing import List, Optional
from fastapi import APIRouter, Query
from app.services.rag_service import get_rag_service
from app.models.schemas import MenuPlanResponse

router = APIRouter()


@router.get("/api/menu-plan", response_model=dict)
def menu_plan(
    age_months: int = Query(..., description="Usia bayi dalam bulan"),
    weight_kg: Optional[float] = Query(None, description="Berat badan dalam kg"),
    height_cm: Optional[float] = Query(None, description="Tinggi dalam cm"),
    residence: Optional[str] = Query(None, description="Kota/daerah tempat tinggal"),
    allergies: Optional[str] = Query(None, description="Daftar alergi, pisahkan dengan koma"),
):
    """Generate a 1-day MPASI menu plan using RAG LLM.

    This endpoint accepts parameters as query so you can test in Swagger UI.
    """
    rag = get_rag_service()
    allergy_list = [a.strip() for a in allergies.split(",")] if allergies else []
    result_text = rag.generate_menu(
        age_months=age_months,
        weight_kg=weight_kg,
        height_cm=height_cm,
        residence=residence,
        allergies=allergy_list,
    )

    return {"result": result_text}
