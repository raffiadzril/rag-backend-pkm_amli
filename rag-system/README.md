# RAG System - Flask Web UI

The Flask web application is now integrated within the `rag-system` folder for a cleaner, more organized structure.

## 📁 Folder Structure

```
rag-system/
├── app.py                    # Flask application
├── store.py                  # ChromaDB indexing
├── query.py                  # RAG service
├── chroma_db/                # Vector database (created after first run)
├── templates/
│   └── index.html           # Web UI template
└── static/
    ├── css/
    │   └── style.css        # Styling
    └── js/
        └── script.js        # Client-side logic
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```powershell
pip install flask google-generativeai langchain sentence-transformers chromadb python-dotenv
```

### Step 2: Create ChromaDB (one-time setup)

From the project root:

```powershell
cd rag-system
python store.py
```

Wait for completion. You'll see:
```
✓ ChromaDB loaded successfully with [number] documents
```

### Step 3: Start Flask Server

Still in the `rag-system` folder, run:

```powershell
python app.py
```

You should see:
```
======================================================================
Flask RAG Web Interface
======================================================================
RAG Service: ✓ Ready
Starting Flask server on http://localhost:5000
======================================================================
```

### Step 4: Open in Browser

Navigate to: **http://localhost:5000**

---

## 🎯 Using the Web Interface

### 1. **Search Tab** 🔍
- Enter a query (e.g., "MPASI usia 6 bulan")
- Select number of results (3-20)
- Click "Search" to find relevant documents

### 2. **Search with Scores Tab** 📊
- Same as search, but shows similarity percentages
- Helps assess result relevance

### 3. **Generate Menu Tab** 🍽️
- Fill in baby information:
  - **Age** (4-36 months)
  - **Weight** (kg)
  - **Height** (cm)
  - **Residence** (optional)
  - **Allergies** (optional, comma-separated)
- Click "Generate Menu Plan"
- Get complete daily menu with nutrition info!

---

## 📊 Generated Menu Includes

✅ **5 Meals Per Day**
- Breakfast
- Morning Snack
- Lunch
- Afternoon Snack
- Dinner

✅ **Per Meal Details**
- Time & name
- Portion size
- Ingredients list
- Preparation instructions
- Nutrition values (Energy, Protein, Carbs, Fat)

✅ **Daily Summary**
- Total nutrition values
- AKG compliance status
- Notes & recommendations

---

## 🔧 API Endpoints

If you want to use the API directly:

### Check Status
```bash
curl http://localhost:5000/api/status
```

### Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "MPASI 6 bulan", "top_k": 5}'
```

### Search with Scores
```bash
curl -X POST http://localhost:5000/api/search-with-scores \
  -H "Content-Type: application/json" \
  -d '{"query": "AKG gizi", "top_k": 5}'
```

### Generate Menu
```bash
curl -X POST http://localhost:5000/api/generate-menu \
  -H "Content-Type: application/json" \
  -d '{
    "age_months": 6,
    "weight_kg": 7,
    "height_cm": 65,
    "allergies": ["egg"],
    "residence": "Indonesia"
  }'
```

---

## ⚙️ Configuration

### Change Port
Edit `app.py` (line ~94):
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Change Model
Edit `query.py` (line ~25):
```python
gemini_model = genai.GenerativeModel('gemini-2.0-flash')  # Change model
```

---

## 🐛 Troubleshooting

### Issue: "RAG Service Unavailable"
**Solution:**
- Run `python store.py` first
- Check `.env` file has `GEMINI_API_KEY`
- Verify `chroma_db` folder exists

### Issue: "Port 5000 already in use"
**Solution:**
- Change port in `app.py`
- Or: `netstat -ano | findstr :5000` to find process, then kill it

### Issue: "ModuleNotFoundError"
**Solution:**
- Make sure you're in `rag-system` folder when running
- Run `pip install` commands again

### Issue: Slow first load
**Solution:**
- First load downloads the embedding model (~50MB)
- Subsequent loads are faster
- Be patient on first run!

---

## 📦 Dependencies

- **Flask** - Web framework
- **google-generativeai** - Gemini API
- **langchain** - LLM framework
- **sentence-transformers** - Embeddings
- **chromadb** - Vector database

---

## 🎨 Customization

### Change Styling
Edit `static/css/style.css`

### Change Layout
Edit `templates/index.html`

### Add Features
Edit `app.py` (add endpoints) and `static/js/script.js` (add frontend)

---

## 📝 Notes

- ChromaDB is stored locally in `chroma_db/` folder
- Delete `chroma_db/` to re-index the dataset
- The embedding model is cached after first download
- Menu generation requires GEMINI_API_KEY

---

## ✨ Features

✅ Modern responsive UI  
✅ Real-time status indicator  
✅ Tab-based navigation  
✅ Document search & retrieval  
✅ Similarity scoring  
✅ AI-powered menu generation  
✅ Mobile-friendly design  
✅ Error handling & loading states  

---

**Enjoy using the RAG MPASI Menu Generator!** 🎉
