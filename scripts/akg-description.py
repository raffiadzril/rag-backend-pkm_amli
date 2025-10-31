#!/usr/bin/env python3
"""Create human-readable descriptions for AKG entries to improve retrieval.

Reads: dataset/akg_merged.json
Writes:
 - dataset/akg_merged_with_descriptions.json (augmented entries)
 - dataset/akg_descriptions.txt (one-line description per entry)

The description field summarizes key macro/micronutrients and the target age group.
"""
import json
import os
from typing import Any, Dict


def safe_get(d: Dict[str, Any], keys, default=None):
    """Try multiple possible keys and return the first present non-empty value."""
    if isinstance(keys, (list, tuple)):
        for k in keys:
            v = d.get(k)
            if v is not None and v != "":
                return v
        return default
    return d.get(keys, default)


def make_description(item: Dict[str, Any]) -> str:
    kelompok = safe_get(item, ["Kelompok Umur", "kelompok umur", "Kelompok umur"], "Unknown age group")
    berat = safe_get(item, ["Berat Badan (kg)", "Berat Badan (kg)", "Berat Badan"], None)
    tinggi = safe_get(item, ["Tinggi Badan (cm)", "Tinggi Badan"], None)
    energi = safe_get(item, ["Energi (kkal)", "Energi (kkal)", "Energi"], None)
    protein = safe_get(item, ["Protein (g)", "Protein"], None)
    lemak = safe_get(item, ["Lemak Total (g)", "Lemak Total"], None)
    karbo = safe_get(item, ["Karbohidrat (g)", "Karbohidrat"], None)
    kalsium = safe_get(item, ["Kalsium (mg)", "Kalsium"], None)
    besi = safe_get(item, ["Besi (mg)", "Besi"], None)
    vit_a = safe_get(item, ["Vit A (RE)", "Vit A"], None)
    vit_d = safe_get(item, ["Vit D (mcg)", "Vit D"], None)

    parts = [f"Kelompok umur: {kelompok}"]
    if berat:
        parts.append(f"Berat ~{berat} kg")
    if tinggi:
        parts.append(f"Tinggi ~{tinggi} cm")

    macro = []
    if energi:
        macro.append(f"Energi: {energi} kcal")
    if protein:
        macro.append(f"Protein: {protein} g")
    if lemak:
        macro.append(f"Lemak: {lemak} g")
    if karbo:
        macro.append(f"Karbohidrat: {karbo} g")
    if macro:
        parts.append("; ".join(macro))

    micro = []
    if kalsium:
        micro.append(f"Kalsium: {kalsium} mg")
    if besi:
        micro.append(f"Besi: {besi} mg")
    if vit_a:
        micro.append(f"VitA: {vit_a}")
    if vit_d:
        micro.append(f"VitD: {vit_d} mcg")
    if micro:
        parts.append("Micron: " + ", ".join(micro))

    # Short natural language guidance that improves semantic search
    guidance = (
        "Target AKG untuk kelompok usia ini. Gunakan informasi ini saat memilih porsi, tekstur, dan komposisi makronutrien."
    )
    parts.append(guidance)

    return ". ".join(parts)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "dataset", "akg_merged.json")
    out_json = os.path.join(script_dir, "..", "dataset", "akg_merged_with_descriptions.json")
    out_txt = os.path.join(script_dir, "..", "dataset", "akg_descriptions.txt")

    if not os.path.exists(dataset_path):
        print(f"Input file not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented = []
    lines = []
    for idx, item in enumerate(data):
        desc = make_description(item)
        new_item = dict(item)
        new_item['description'] = desc
        augmented.append(new_item)
        # One-line description for ingestion/search (shorter)
        lines.append(f"{idx+1}. {desc}")

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"Wrote augmented JSON: {out_json} ({len(augmented)} entries)")
    print(f"Wrote descriptions text: {out_txt}")


if __name__ == '__main__':
    main()
