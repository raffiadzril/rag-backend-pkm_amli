import json
import os
from typing import Any, Dict

# --- KUNCI GIZI PENTING DAN ALIAS UNTUK KOMPRESI ---
KEY_MAPPING = {
    "KODE": "code",
    "NAMA BAHAN": "name",
    "BDD (%)": "bdd_percent",
    "ENERGI (Kal)": "kcal",
    "PROTEIN (g)": "prot_g",
    "LEMAK (g)": "fat_g",
    "KH (g)": "carb_g",
    "BESI (mg)": "iron_mg",
    "KALSIUM (mg)": "calc_mg",
    "VIT_C (mg)": "vitc_mg",
    "SUMBER": "source",
    "Jenis": "type",
    "Sumber": "category"
}

def clean_and_normalize_value(value: Any) -> Any:
    """Membersihkan dan menormalkan nilai gizi."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip().replace('"', '').replace("'", '').replace(',', '.')
        try:
            float_value = float(cleaned)
            if float_value == 0.0:
                return None
            # Kembalikan sebagai string yang sudah dibersihkan
            return cleaned 
        except ValueError:
            return cleaned
    return str(value)


def run_preprocessing():
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    input_file_path = os.path.abspath(os.path.join(script_dir, "../dataset/TKPI-2020.json"))
    # Output sebagai file teks untuk token-efficiency terbaik
    output_file_path = os.path.abspath(os.path.join(script_dir, "../dataset/TKPI_COMPACT.txt"))
    
    if not os.path.exists(input_file_path):
        print(f"❌ Error: File TKPI-2020.json tidak ditemukan di {input_file_path}")
        return

    print(f"Membaca data dari {input_file_path}...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data_tkpi = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Gagal membaca JSON. Detail: {e}")
        return

    print(f"Memproses {len(data_tkpi)} entri TKPI...")
    compact_lines = []

    for item in data_tkpi:
        new_item = {}
        has_essential_data = False
        
        for original_key, alias in KEY_MAPPING.items():
            value = item.get(original_key)
            cleaned_value = clean_and_normalize_value(value)
            
            if cleaned_value is None:
                continue

            new_item[alias] = cleaned_value
            
            if alias in ["kcal", "prot_g", "fat_g", "carb_g", "iron_mg"]:
                has_essential_data = True
        
        if 'code' in new_item and 'name' in new_item and has_essential_data:
             # Konversi objek ringkas ke JSON string minimal
             line = json.dumps(new_item, ensure_ascii=False, separators=(',', ':'))
             compact_lines.append(line)


    # Simpan data yang sudah diringkas sebagai Array of Strings (dipisahkan newline)
    print(f"Menyimpan {len(compact_lines)} entri ringkas ke {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # PENTING: Menulis setiap objek JSON sebagai baris terpisah
        f.write('\n'.join(compact_lines))

    print("✅ Preprocessing selesai.")
    print(f"   File ringkas (MAX TOKEN EFFICIENCY) disimpan di: {output_file_path}")
    print(f"   Ini adalah format paling efisien untuk menghindari error 500.")

if __name__ == "__main__":
    run_preprocessing()