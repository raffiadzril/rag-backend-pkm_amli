from fastapi import APIRouter, HTTPException
from app.models.menu_models import MenuPlanRequest, MenuPlanResponse
from app.services.menu_service import menu_planner_service

router = APIRouter(prefix="/api", tags=["Menu Planning"])


@router.post("/menu-plan", response_model=MenuPlanResponse)
async def generate_menu_plan(request: MenuPlanRequest):
    """
    Generate menu plan MPASI berdasarkan profil anak
    
    **Input:**
    - user_profile: Profil anak (umur, BB, TB, jenis kelamin, alergi, dll)
    - jumlah_hari: Jumlah hari menu yang diinginkan (1-7 hari)
    - preferensi_tambahan: Preferensi atau catatan tambahan (optional)
    
    **Output:**
    - Menu plan lengkap untuk pagi, selingan pagi, siang, selingan sore, dan malam
    - Kebutuhan gizi harian
    - Rekomendasi umum dari AI
    """
    try:
        response = menu_planner_service.generate_menu_plan(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating menu plan: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MPASI Menu Planner API",
        "version": "1.0.0"
    }
