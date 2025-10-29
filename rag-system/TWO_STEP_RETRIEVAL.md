# Two-Step Retrieval Architecture

## Overview
The RAG system has been upgraded to use a **two-step retrieval process** for better accuracy and control:

1. **STEP 1: Retrieve MPASI Rules & AKG Requirements**
   - Searches for age-specific MPASI rules
   - Retrieves AKG (Angka Kecukupan Gizi) nutritional requirements
   - Extracts texture, portion, and feeding frequency guidelines
   - Saved as `konteks_aturan`

2. **STEP 2: Retrieve Available TKPI Ingredients**
   - Searches for food ingredients from TKPI-2020 database
   - Retrieves protein sources (hewani & nabati)
   - Retrieves carbohydrate sources
   - Retrieves vegetables, fruits, and healthy fats
   - Saved as `konteks_bahan`

3. **STEP 3: Generate Menu Using Both Contexts**
   - LLM receives clear separation between rules and ingredients
   - LLM must use ONLY data from `konteks_aturan` for rules
   - LLM must use ONLY ingredients from `konteks_bahan`
   - Ensures menu generation is grounded in actual dataset

## File Changes

### `rag-system/query.py`
**Modified:** `generate_menu_plan_with_chroma()` method

**Changes:**
- Replaced single query with two targeted queries
- **Query 1 (Rules):** `"MPASI usia X bulan angka kecukupan gizi AKG aturan pemberian MPASI tekstur makanan porsi makan frekuensi makan bayi"`
- **Query 2 (Ingredients):** `"TKPI bahan makanan makanan bayi sehat protein hewani protein nabati karbohidrat sehat lemak esensial sayuran hijau buah-buahan"`

**Retrieval:**
```python
# STEP 1: Get rules (top_k=15)
konteks_aturan = self.search_relevant_docs(rules_query, top_k=15)

# STEP 2: Get ingredients (top_k=20)
konteks_bahan = self.search_relevant_docs(ingredients_query, top_k=20)
```

**Prompt Structure:**
```
INFORMASI BAYI
↓
BAGIAN A: ATURAN MPASI DAN AKG (WAJIB IKUTI)
├─ Formatted konteks_aturan (15 documents)
↓
BAGIAN B: BAHAN MAKANAN YANG TERSEDIA (TKPI)
├─ Formatted konteks_bahan (20 documents)
↓
INSTRUKSI PEMBUATAN MENU (5-step instructions)
├─ 1. Analisis kebutuhan gizi dari BAGIAN A
├─ 2. Pilih bahan dari BAGIAN B
├─ 3. Buat menu ORIGINAL (kombinasi unik)
├─ 4. Hitung nutrisi manual
└─ 5. Validasi hasil
```

**Return Value:**t
```python
{
    "rag_info": {
        "documents_retrieved_rules": len(konteks_aturan),      # ~15 docs
        "documents_retrieved_ingredients": len(konteks_bahan),  # ~20 docs
        "total_documents_retrieved": len(konteks_aturan) + len(konteks_bahan),
        "rules_query": rules_query,
        "ingredients_query": ingredients_query,
        "retrieval_method": "Two-Step ChromaDB (Rules + Ingredients)"
    }
}
```

### `rag-system/app.py`
**Modified:** `/api/debug-prompt` endpoint

**Changes:**
- Updated to replicate the two-step retrieval process
- Returns separate summaries for rules and ingredients contexts
- Provides visibility into what each step retrieves

**Response Structure:**
```python
{
    "debug_info": {
        "step_1_rules_query": "...",
        "step_2_ingredients_query": "...",
        "step_1_documents_retrieved": 15,
        "step_2_documents_retrieved": 20,
        "total_documents": 35,
        "baby_info": {...}
    },
    "full_prompt": "...",  # Complete prompt with both sections
    "prompt_length_chars": 8432,
    "context_aturan_summary": [...],  # Rules context preview
    "context_bahan_summary": [...]    # Ingredients context preview
}
```

## Workflow

### For Menu Generation (`/api/generate-menu`):

```
User Input (age, weight, height, allergies)
    ↓
STEP 1: Search ChromaDB for rules
    └─ Query: "MPASI usia X bulan AKG aturan..."
    └─ Return: 15 documents about rules
    └─ Save as: konteks_aturan
    ↓
STEP 2: Search ChromaDB for ingredients
    └─ Query: "TKPI bahan makanan protein hewani..."
    └─ Return: 20 documents about ingredients
    └─ Save as: konteks_bahan
    ↓
STEP 3: Build prompt with both sections
    ├─ BAGIAN A: konteks_aturan (rules)
    ├─ BAGIAN B: konteks_bahan (ingredients)
    └─ Instructions telling LLM to use only these two sections
    ↓
STEP 4: Send to Gemini API
    └─ Gemini receives clear instructions to use ONLY provided data
    └─ Generates menu JSON
    ↓
Return menu with rag_info showing both retrieval stats
```

### For Debug (`/api/debug-prompt`):

```
User Input
    ↓
Same STEP 1-3 as above
    ↓
Return full prompt + retrieval stats WITHOUT calling Gemini
    └─ Allows inspection before menu generation
```

## Benefits

1. **Clear Separation of Concerns**
   - Rules vs. Ingredients are explicitly separated
   - LLM knows exactly where to look for each type of information

2. **Better Grounding**
   - Reduces hallucination by having separate contexts
   - LLM must explicitly reference rules and ingredients

3. **Improved Control**
   - Can tune `top_k` separately for rules vs. ingredients
   - Can adjust queries independently

4. **Transparency**
   - Debug endpoint shows exactly what data is used
   - Clear visibility into retrieval quality

5. **Scalability**
   - Easy to add more retrieval steps if needed
   - Can separately improve rules or ingredients retrieval

## Testing

### In Web UI:
1. Open http://localhost:5000
2. Go to "Generate Menu" tab
3. Enter baby info
4. Click "Generate Menu Plan"
5. Inspect "🐛 Debug: Prompt Sent to Gemini" section
   - See Step 1 Rules Query
   - See Step 2 Ingredients Query
   - See both document counts
   - See full prompt before menu generation

### With Debug Endpoint:
```bash
curl -X POST http://localhost:5000/api/debug-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "age_months": 6,
    "weight_kg": 7,
    "height_cm": 65,
    "allergies": [],
    "residence": "Indonesia"
  }'
```

Response shows:
- `step_1_rules_query`: Rules retrieval query
- `step_1_documents_retrieved`: Number of rules docs
- `step_2_ingredients_query`: Ingredients retrieval query
- `step_2_documents_retrieved`: Number of ingredient docs
- `context_aturan_summary`: Preview of rules docs
- `context_bahan_summary`: Preview of ingredient docs
- `full_prompt`: Complete prompt sent to Gemini

## Notes

- Retrieval is done from the **same ChromaDB** (`./chroma_db`)
- Different queries help surface different document types
- Total 35 documents retrieved per menu generation (15 rules + 20 ingredients)
- Prompt size remains ~8000-10000 characters depending on document content
- LLM explicitly instructed not to use knowledge outside the two provided sections
