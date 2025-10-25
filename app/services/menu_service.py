from app.models.menu_models import (
    UserProfile, 
    MenuPlanRequest, 
    MenuPlanResponse,
    MealPlan,
    Meal
)
from app.services.rag_service import rag_service
from typing import List
import json
import re


class MenuPlannerService:
    """Service untuk generate menu plan MPASI menggunakan RAG"""
    
    def __init__(self):
        self.rag = rag_service
    
    def generate_menu_plan(self, request: MenuPlanRequest) -> MenuPlanResponse:
        """Generate menu plan berdasarkan user profile"""
        profile = request.user_profile
        
        # 1. Get kebutuhan gizi
        kebutuhan_gizi = self.rag.get_nutrition_requirements(profile.umur_bulan)
        
        # 2. Generate menu untuk setiap hari
        menu_plans = []
        for hari in range(1, request.jumlah_hari + 1):
            meal_plan = self._generate_daily_menu(profile, hari, kebutuhan_gizi, request.preferensi_tambahan)
            menu_plans.append(meal_plan)
        
        # 3. Generate rekomendasi umum
        rekomendasi = self._generate_general_recommendation(profile, kebutuhan_gizi)
        
        return MenuPlanResponse(
            status="success",
            user_profile=profile,
            kebutuhan_gizi=kebutuhan_gizi,
            menu_plan=menu_plans,
            rekomendasi_umum=rekomendasi
        )
    
    def _generate_daily_menu(
        self, 
        profile: UserProfile, 
        hari: int,
        kebutuhan_gizi: dict,
        preferensi: str = None
    ) -> MealPlan:
        """Generate menu untuk satu hari"""
        
        # Buat prompt untuk generate menu
        alergi_str = ", ".join(profile.alergi) if profile.alergi else "tidak ada"
        preferensi_str = f"\nPreferensi tambahan: {preferensi}" if preferensi else ""
        
        prompt = f"""Kamu adalah ahli gizi anak. Buatkan menu MPASI untuk 1 hari dengan detail berikut:

PROFIL ANAK:
- Umur: {profile.umur_bulan} bulan
- Berat badan: {profile.berat_badan} kg
- Tinggi badan: {profile.tinggi_badan} cm
- Jenis kelamin: {profile.jenis_kelamin}
- Tempat tinggal: {profile.tempat_tinggal}
- Alergi: {alergi_str}
- Aktivitas: {profile.aktivitas}{preferensi_str}

KEBUTUHAN GIZI HARIAN:
- Energi: {kebutuhan_gizi.get('energi', 725)} kkal
- Protein: {kebutuhan_gizi.get('protein', 11)} g
- Lemak: {kebutuhan_gizi.get('lemak', 25)} g
- Kalsium: {kebutuhan_gizi.get('kalsium', 270)} mg

INSTRUKSI:
Buatkan menu untuk 5 waktu makan (pagi, selingan pagi, siang, selingan sore, malam) yang:
1. Sesuai dengan usia dan kebutuhan gizi anak
2. Menghindari bahan yang menyebabkan alergi
3. Mudah dibuat dan menggunakan bahan lokal Indonesia
4. Tekstur sesuai usia (untuk {profile.umur_bulan} bulan)

Berikan dalam format JSON dengan struktur berikut:
{{
  "pagi": {{
    "waktu": "Pagi (08:00)",
    "menu": "nama menu",
    "bahan": ["bahan1", "bahan2"],
    "porsi": "jumlah porsi",
    "kalori": estimasi kalori,
    "protein": estimasi protein,
    "cara_membuat": "langkah singkat"
  }},
  "selingan_pagi": {{ ... }},
  "siang": {{ ... }},
  "selingan_sore": {{ ... }},
  "malam": {{ ... }},
  "total_kalori": total kalori per hari,
  "catatan": "catatan khusus jika ada"
}}

Berikan HANYA JSON tanpa penjelasan tambahan."""

        try:
            # Query RAG untuk mendapatkan data makanan yang relevan
            food_context = self.rag.search_relevant_docs(
                f"makanan mpasi bayi {profile.umur_bulan} bulan beras sayur protein", 
                top_k=10
            )
            
            context_str = "\n".join(food_context[:5]) if food_context else ""
            
            full_prompt = f"""DATA KOMPOSISI PANGAN UNTUK REFERENSI:
{context_str}

{prompt}"""
            
            # Generate dengan Gemini
            response = self.rag.model.generate_content(full_prompt)
            
            # Parse JSON dari response
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                menu_data = json.loads(json_match.group())
                
                # Convert ke MealPlan object
                return MealPlan(
                    hari=hari,
                    pagi=Meal(**menu_data.get('pagi', {})),
                    selingan_pagi=Meal(**menu_data.get('selingan_pagi', {})),
                    siang=Meal(**menu_data.get('siang', {})),
                    selingan_sore=Meal(**menu_data.get('selingan_sore', {})),
                    malam=Meal(**menu_data.get('malam', {})),
                    total_kalori=menu_data.get('total_kalori'),
                    catatan=menu_data.get('catatan')
                )
            else:
                # Fallback menu jika gagal parse
                return self._get_fallback_menu(hari, profile)
                
        except Exception as e:
            print(f"Error generating menu: {e}")
            return self._get_fallback_menu(hari, profile)
    
    def _get_fallback_menu(self, hari: int, profile: UserProfile) -> MealPlan:
        """Fallback menu jika generation gagal"""
        return MealPlan(
            hari=hari,
            pagi=Meal(
                waktu="Pagi (08:00)",
                menu="Bubur Beras Putih",
                bahan=["Beras putih", "Air"],
                porsi="50 ml",
                kalori=145,
                protein=2.8,
                cara_membuat="Masak beras dengan air hingga lembut, blender halus"
            ),
            selingan_pagi=Meal(
                waktu="Selingan Pagi (10:00)",
                menu="Puree Pisang",
                bahan=["Pisang ambon"],
                porsi="50 gram",
                kalori=45,
                protein=0.5,
                cara_membuat="Haluskan pisang matang dengan garpu"
            ),
            siang=Meal(
                waktu="Siang (12:00)",
                menu="Bubur Ayam Wortel",
                bahan=["Beras putih", "Ayam fillet", "Wortel"],
                porsi="80 ml",
                kalori=180,
                protein=8.5,
                cara_membuat="Masak beras dengan ayam dan wortel, blender halus"
            ),
            selingan_sore=Meal(
                waktu="Selingan Sore (15:00)",
                menu="Puree Alpukat",
                bahan=["Alpukat"],
                porsi="50 gram",
                kalori=80,
                protein=1.0,
                cara_membuat="Kerok alpukat matang hingga halus"
            ),
            malam=Meal(
                waktu="Malam (18:00)",
                menu="Bubur Ikan Bayam",
                bahan=["Beras putih", "Ikan salmon", "Bayam"],
                porsi="80 ml",
                kalori=170,
                protein=9.0,
                cara_membuat="Masak beras dengan ikan dan bayam, blender halus"
            ),
            total_kalori=620,
            catatan="Menu dasar MPASI yang aman untuk bayi"
        )
    
    def _generate_general_recommendation(self, profile: UserProfile, kebutuhan_gizi: dict) -> str:
        """Generate rekomendasi umum berdasarkan profil"""
        prompt = f"""Berikan rekomendasi umum untuk pemberian MPASI kepada anak dengan profil:
- Umur: {profile.umur_bulan} bulan
- Berat badan: {profile.berat_badan} kg
- Tinggi badan: {profile.tinggi_badan} cm

Kebutuhan gizi harian: {kebutuhan_gizi}

Berikan rekomendasi dalam 3-4 poin singkat tentang:
1. Tips pemberian MPASI untuk usia ini
2. Hal yang perlu diperhatikan
3. Perkembangan yang diharapkan

Jawab dengan singkat dan praktis."""

        try:
            response = self.rag.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Berikan MPASI sesuai usia anak dengan tekstur yang tepat. Perhatikan tanda alergi dan konsultasikan dengan dokter anak secara berkala."


# Global instance
menu_planner_service = MenuPlannerService()
