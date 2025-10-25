from typing import Dict, List, Optional
from pydantic import BaseModel


class MenuPlan(BaseModel):
    breakfast: List[str]
    lunch: List[str]
    dinner: List[str]
    snacks: List[str]


class MenuPlanResponse(BaseModel):
    menu: MenuPlan
    notes: Optional[str]
