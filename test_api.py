"""
Test script untuk backend API
"""
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("="*60)
print("TEST 1: Health Check")
print("="*60)
response = requests.get(f"{BASE_URL}/api/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 2: Generate menu plan
print("="*60)
print("TEST 2: Generate Menu Plan")
print("="*60)

request_data = {
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

print("Request:")
print(json.dumps(request_data, indent=2))
print("\nSending request to /api/menu-plan...")

response = requests.post(
    f"{BASE_URL}/api/menu-plan",
    json=request_data,
    headers={"Content-Type": "application/json"}
)

print(f"\nStatus: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("\n✅ SUCCESS! Menu Plan Generated:")
    print(f"\nKebutuhan Gizi: {json.dumps(data['kebutuhan_gizi'], indent=2)}")
    print(f"\nJumlah hari menu: {len(data['menu_plan'])}")
    
    for i, day in enumerate(data['menu_plan'], 1):
        print(f"\n--- HARI {day['hari']} ---")
        print(f"Pagi: {day['pagi']['menu']}")
        print(f"Selingan Pagi: {day['selingan_pagi']['menu']}")
        print(f"Siang: {day['siang']['menu']}")
        print(f"Selingan Sore: {day['selingan_sore']['menu']}")
        print(f"Malam: {day['malam']['menu']}")
        print(f"Total Kalori: {day['total_kalori']} kkal")
    
    print(f"\n📋 Rekomendasi Umum:")
    print(data['rekomendasi_umum'])
else:
    print(f"❌ ERROR: {response.text}")

print("\n" + "="*60)
