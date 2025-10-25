from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Gender(str, Enum):
    MALE = "laki-laki"
    FEMALE = "perempuan"


class ActivityLevel(str, Enum):
    LOW = "rendah"
    MODERATE = "sedang"
    HIGH = "tinggi"


class UserProfile(BaseModel):
    """Model untuk profil anak MPASI"""
    umur_bulan: int = Field(..., ge=6, le=24, description="Umur anak dalam bulan (6-24 bulan)")
    berat_badan: float = Field(..., gt=0, description="Berat badan dalam kg")
    tinggi_badan: float = Field(..., gt=0, description="Tinggi badan dalam cm")
    jenis_kelamin: Gender = Field(..., description="Jenis kelamin anak")
    tempat_tinggal: str = Field(..., description="Kota/daerah tempat tinggal")
    alergi: Optional[List[str]] = Field(default=[], description="Daftar alergi makanan")
    aktivitas: ActivityLevel = Field(default=ActivityLevel.MODERATE, description="Tingkat aktivitas")
    
    class Config:
        json_schema_extra = {
            "example": {
                "umur_bulan": 8,
                "berat_badan": 8.5,
                "tinggi_badan": 70.0,
                "jenis_kelamin": "laki-laki",
                "tempat_tinggal": "Jakarta",
                "alergi": ["kacang", "telur"],
                "aktivitas": "sedang"
            }
        }


class MenuPlanRequest(BaseModel):
    """Request untuk generate menu plan"""
    user_profile: UserProfile
    jumlah_hari: int = Field(default=1, ge=1, le=7, description="Jumlah hari menu yang diinginkan")
    preferensi_tambahan: Optional[str] = Field(default=None, description="Preferensi atau catatan tambahan")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_profile": {
                    "umur_bulan": 8,
                    "berat_badan": 8.5,
                    "tinggi_badan": 70.0,
                    "jenis_kelamin": "laki-laki",
                    "tempat_tinggal": "Jakarta",
                    "alergi": ["kacang"],
                    "aktivitas": "sedang"
                },
                "jumlah_hari": 1,
                "preferensi_tambahan": "Lebih suka menu dengan sayuran hijau"
            }
        }


class Meal(BaseModel):
    """Model untuk satu waktu makan"""
    waktu: str = Field(..., description="Waktu makan (Pagi/Siang/Malam/Selingan)")
    menu: str = Field(..., description="Nama menu makanan")
    bahan: List[str] = Field(..., description="Daftar bahan yang digunakan")
    porsi: str = Field(..., description="Ukuran porsi")
    kalori: Optional[float] = Field(default=None, description="Estimasi kalori")
    protein: Optional[float] = Field(default=None, description="Estimasi protein (g)")
    cara_membuat: Optional[str] = Field(default=None, description="Cara membuat singkat")


class MealPlan(BaseModel):
    """Model untuk rencana menu satu hari"""
    hari: int = Field(..., description="Hari ke-")
    pagi: Meal
    selingan_pagi: Meal
    siang: Meal
    selingan_sore: Meal
    malam: Meal
    total_kalori: Optional[float] = Field(default=None, description="Total kalori per hari")
    catatan: Optional[str] = Field(default=None, description="Catatan khusus untuk hari ini")


class MenuPlanResponse(BaseModel):
    """Response untuk menu plan"""
    status: str = Field(default="success", description="Status response")
    user_profile: UserProfile
    kebutuhan_gizi: dict = Field(..., description="Kebutuhan gizi harian")
    menu_plan: List[MealPlan] = Field(..., description="Rencana menu")
    rekomendasi_umum: str = Field(..., description="Rekomendasi umum dari AI")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "user_profile": {
                    "umur_bulan": 8,
                    "berat_badan": 8.5,
                    "tinggi_badan": 70.0,
                    "jenis_kelamin": "laki-laki",
                    "tempat_tinggal": "Jakarta",
                    "alergi": ["kacang"],
                    "aktivitas": "sedang"
                },
                "kebutuhan_gizi": {
                    "energi": 725,
                    "protein": 11,
                    "lemak": 25,
                    "kalsium": 270
                },
                "menu_plan": [],
                "rekomendasi_umum": "Menu disesuaikan dengan kebutuhan gizi anak usia 8 bulan"
            }
        }
