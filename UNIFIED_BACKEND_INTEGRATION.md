# CHATBOT INTEGRATION - Unified Backend

## Tanggal: November 3, 2025
**Status**: ✅ **SELESAI**

## Perubahan Besar

Semua fungsi chatbot telah **dipindahkan dari `app/main.py` ke `fastapi_app/main.py`** untuk menyatukan semua layanan backend dalam satu file.

### Alasan Konsolidasi
- ✅ **Deployment lebih sederhana** - Docker hanya run satu file (`fastapi_app/main.py`)
- ✅ **Struktur lebih jelas** - Semua endpoints dalam satu tempat
- ✅ **Maintenance lebih mudah** - Tidak perlu manage dua FastAPI app terpisah
- ✅ **Resource sharing** - Menu generation dan chatbot bisa share utilities

---

## File yang Diubah

### 1. `fastapi_app/main.py` ⭐ (File Utama)

**Imports Ditambahkan:**
```python
from fastapi import FastAPI, HTTPException, Body  # Added Body
from pydantic import BaseModel, Field  # Added Field

# Chatbot path
chatbot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chatbot'))
sys.path.insert(0, chatbot_path)
```

**Import Chatbot Service:**
```python
try:
    from chatbot_service import get_chatbot_service
    print("✓ Successfully imported chatbot_service module")
    CHATBOT_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"✗ Error importing chatbot service: {e}")
    CHATBOT_IMPORT_SUCCESS = False
```

**Initialize Chatbot:**
```python
CHATBOT_READY = False

if CHATBOT_IMPORT_SUCCESS:
    try:
        print("Attempting to initialize Chatbot service...")
        chatbot_service = get_chatbot_service()
        CHATBOT_READY = True
        print("✓ Successfully initialized Chatbot service")
    except Exception as e:
        print(f"✗ Error initializing Chatbot service: {e}")
        CHATBOT_READY = False
```

**Models Ditambahkan:**
```python
class ChatMessage(BaseModel):
    sender: str = Field(..., description="user or bot")
    text: str = Field(..., description="Message content")
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Conversation history"
    )

class ChatResponse(BaseModel):
    response: str
    status: str
    sources_used: int
    has_context: bool
```

**Endpoints Chatbot:**
- ✅ `POST /api/chatbot/chat` - Chat dengan AI
- ✅ `GET /api/chatbot/status` - Status chatbot service
- ✅ `GET /api/chatbot/test` - Test chatbot functionality

### 2. `fastapi_app/Dockerfile`

**Ditambahkan:**
```dockerfile
# Copy chatbot service (for integrated chatbot in main.py)
COPY chatbot/ ./chatbot/
```

### 3. `fastapi_app/start_server.bat` (Baru)

Script untuk start development server dengan mudah:
```batch
cd fastapi_app
start_server.bat
```

---

## Struktur Backend Final

```
fastapi_app/main.py (UNIFIED BACKEND)
├── Menu Generation (Existing)
│   ├── POST /api/generate-menu
│   ├── GET /api/status
│   ├── GET /api/models
│   └── POST /api/debug-prompt
│
└── Chatbot (NEW - Integrated)
    ├── POST /api/chatbot/chat
    ├── GET /api/chatbot/status
    └── GET /api/chatbot/test
```

### Dependencies
```
fastapi_app/main.py
├── imports: rag-system/query.py (menu generation)
├── imports: chatbot/chatbot_service.py (chatbot)
└── uses: dataset/ (knowledge base)
```

---

## Testing

### 1. Local Development

**Start Server:**
```bash
cd fastapi_app
python -m uvicorn main:app --reload --port 8000

# Atau pakai batch script:
start_server.bat
```

**Test Menu Generation:**
```bash
curl -X POST http://localhost:8000/api/generate-menu \
  -H "Content-Type: application/json" \
  -d "{\"umur_bulan\":8,\"berat_badan\":8.5,\"tinggi_badan\":70,\"jenis_kelamin\":\"laki-laki\",\"alergi\":[]}"
```

**Test Chatbot:**
```bash
# Status check
curl http://localhost:8000/api/chatbot/status

# Chat
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Kapan bayi boleh mulai MPASI?\",\"conversation_history\":[]}"

# Test endpoint
curl http://localhost:8000/api/chatbot/test
```

**Swagger UI:**
- Open: http://localhost:8000/docs
- Test all endpoints interactively

### 2. Docker Build & Run

**Build Image:**
```bash
cd rag-backend-pkm_amli
docker build -f fastapi_app/Dockerfile -t gata-backend .
```

**Run Container:**
```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  gata-backend
```

**Test Docker Container:**
```bash
# Check all services
curl http://localhost:8000/

# Menu generation
curl -X POST http://localhost:8000/api/generate-menu \
  -H "Content-Type: application/json" \
  -d "{\"umur_bulan\":8,\"berat_badan\":8.5,\"tinggi_badan\":70}"

# Chatbot
curl http://localhost:8000/api/chatbot/status
```

### 3. Azure Deployment

**Deploy:**
```bash
cd fastapi_app

# Method 1: Az CLI
az webapp up --name giziasisten --runtime "PYTHON:3.11"

# Method 2: Docker to Azure Container Registry
docker build -f Dockerfile -t gata-backend ..
docker tag gata-backend giziasisten.azurecr.io/backend:latest
docker push giziasisten.azurecr.io/backend:latest
```

**Environment Variables (Azure App Service):**
```
GEMINI_API_KEY=AIzaSyC8YKUXHr3VeefPKpXFbXnXkov7JYoiErc
ENABLE_RAG=true
CHATBOT_MODEL=gemini-2.0-flash
MENU_MODEL=gemini-1.5-pro-002
```

**Test Azure:**
```bash
# Check status
curl https://giziasisten.azurewebsites.net/

# Menu generation
curl -X POST https://giziasisten.azurewebsites.net/api/generate-menu \
  -H "Content-Type: application/json" \
  -d "{\"umur_bulan\":8,\"berat_badan\":8.5,\"tinggi_badan\":70}"

# Chatbot
curl https://giziasisten.azurewebsites.net/api/chatbot/status
curl -X POST https://giziasisten.azurewebsites.net/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Apa itu MPASI?\",\"conversation_history\":[]}"
```

---

## API Endpoints Reference

### Root Info
```
GET /
Response: API info, available endpoints, service status
```

### Menu Generation
```
POST /api/generate-menu
Body: {
  "umur_bulan": 8,
  "berat_badan": 8.5,
  "tinggi_badan": 70,
  "jenis_kelamin": "laki-laki",
  "alergi": []
}
Response: Menu plan for 7 days
```

### Chatbot
```
POST /api/chatbot/chat
Body: {
  "message": "Kapan bayi boleh mulai MPASI?",
  "conversation_history": [
    {"sender": "user", "text": "halo"},
    {"sender": "bot", "text": "halo juga"}
  ]
}
Response: {
  "status": "success",
  "response": "Bayi boleh mulai MPASI di usia 6 bulan...",
  "sources_used": 3,
  "has_context": true
}
```

```
GET /api/chatbot/status
Response: {
  "status": "ok",
  "service": "GATA Chatbot Service",
  "knowledge_base_size": 150,
  "ready": true
}
```

```
GET /api/chatbot/test
Response: Test message and response from chatbot
```

---

## Frontend Configuration

**No Changes Required!** Frontend sudah dikonfigurasi untuk Azure:

```typescript
// Frontend/.env.local
NEXT_PUBLIC_API_URL=https://giziasisten.azurewebsites.net
NEXT_PUBLIC_MENU_API_URL=https://giziasisten.azurewebsites.net
NEXT_PUBLIC_CHATBOT_API_URL=https://giziasisten.azurewebsites.net
```

Frontend akan otomatis connect ke unified backend di Azure.

---

## File yang Tidak Diperlukan Lagi

### `app/main.py` ❌
- **Status**: Tidak digunakan lagi untuk deployment
- **Alasan**: Semua fungsi sudah ada di `fastapi_app/main.py`
- **Action**: Bisa dihapus atau diarchive

### `chatbot/routes.py` ❌
- **Status**: Tidak digunakan lagi
- **Alasan**: Endpoints sudah langsung di `fastapi_app/main.py`
- **Action**: Bisa dihapus atau diarchive

---

## Keuntungan Konsolidasi

### Before (2 FastAPI Apps):
```
app/main.py          → Menu + Chatbot router import
chatbot/routes.py    → Chatbot endpoints
fastapi_app/main.py  → Menu generation only

Problem:
- Docker harus pilih mana yang di-run
- Dua server terpisah
- Kompleks untuk deployment
```

### After (1 Unified App):
```
fastapi_app/main.py  → Menu + Chatbot (all-in-one)

Benefits:
✅ Satu server untuk semua
✅ Docker langsung run main.py
✅ Deployment sederhana
✅ Maintenance mudah
✅ Resource sharing
```

---

## Troubleshooting

### Import Error: chatbot_service
```bash
# Pastikan PYTHONPATH correct
export PYTHONPATH=/app  # atau sesuai struktur folder
```

### Chatbot Status: unavailable
```bash
# Check:
1. Dataset folder ada di ../dataset/
2. GEMINI_API_KEY set di environment
3. File-file markdown/json ada di dataset/
```

### ChromaDB Error
```bash
# Check:
1. chroma_db folder copied di Dockerfile
2. Path ke rag-system/query.py correct
```

---

## Next Steps

1. ✅ **Test local**: `cd fastapi_app && python -m uvicorn main:app --reload`
2. ✅ **Test Docker**: `docker build -f fastapi_app/Dockerfile -t gata-backend .`
3. ✅ **Deploy Azure**: `az webapp up --name giziasisten`
4. ✅ **Test frontend**: Pastikan popup chatbot dan menu generation work
5. 📝 **Monitor logs**: Check Azure App Service logs untuk errors

---

## Support

Jika ada masalah:
1. Check logs: `docker logs <container_id>` atau Azure logs
2. Test endpoint: `/api/chatbot/status` dan `/api/status`
3. Verify environment variables set correct
4. Check Swagger UI: `/docs` untuk test manual

Happy coding! 🚀
