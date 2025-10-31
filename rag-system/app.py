from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG services from both query.py and query_lm_studio.py
from query import get_chroma_rag_service
from query_lm_studio import get_chroma_rag_service_lm_studio, available_models as lm_studio_models, lm_studio_ready

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Globals for services (initialized later)
rag_service_gemini = None
rag_service_lm_studio = None
GEMINI_READY = False
LM_STUDIO_READY = False
RAG_READY = False


def init_services():
    """Initialize RAG services. This function is safe to call only in the reloader child
    process to avoid double-initialization when Flask debug reloader is enabled."""
    global rag_service_gemini, rag_service_lm_studio, GEMINI_READY, LM_STUDIO_READY, RAG_READY

    # Initialize Gemini-backed RAG service
    try:
        rag_service_gemini = get_chroma_rag_service()
        GEMINI_READY = True
        print("[INFO] Connected to ChromaDB and Gemini API service")
    except Exception as e:
        print(f"[ERROR] Error initializing Gemini RAG service: {e}")
        GEMINI_READY = False

    # Initialize LM Studio RAG service (if available)
    try:
        rag_service_lm_studio = get_chroma_rag_service_lm_studio()
        LM_STUDIO_READY = lm_studio_ready
        print(f"[INFO] Connected to ChromaDB and LM Studio service (Ready: {LM_STUDIO_READY})")
    except Exception as e:
        print(f"[ERROR] Error initializing LM Studio RAG service: {e}")
        LM_STUDIO_READY = False

    # Determine default service availability
    RAG_READY = GEMINI_READY or LM_STUDIO_READY

    return


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', rag_ready=RAG_READY)


@app.route('/api/search', methods=['POST'])
def search():
    """Search API endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        # Use Gemini service if available, otherwise use LM Studio
        if GEMINI_READY:
            docs = rag_service_gemini.search_relevant_docs(query, top_k=top_k)
        else:
            docs = rag_service_lm_studio.search_relevant_docs(query, top_k=top_k)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'top_k': top_k,
            'results_count': len(docs),
            'results': docs
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search-with-scores', methods=['POST'])
def search_with_scores():
    """Search with similarity scores API endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        # Use Gemini service if available, otherwise use LM Studio
        if GEMINI_READY:
            # Gemini service doesn't have search_with_scores, so we'll use Chroma directly
            results = rag_service_gemini.vectordb.similarity_search_with_score(query, k=top_k)
        else:
            results = rag_service_lm_studio.vectordb.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'similarity_score': float(score)
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'top_k': top_k,
            'results_count': len(formatted_results),
            'results': formatted_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generate-menu', methods=['POST'])
def generate_menu():
    """Generate menu plan API endpoint"""
    try:
        data = request.get_json() or {}

        # Use the incoming form data directly. Do NOT substitute hardcoded defaults here.
        # We pass through whatever the frontend submitted (Indonesian or English keys).
        # Remove control keys (model selection) from the user_input payload.
        user_input = {k: v for k, v in data.items() if k not in ('model_type', 'model_name')}

        model_type = data.get('model_type', 'gemini')  # 'gemini' or 'lm_studio'
        model_name = data.get('model_name', None)  # Specific model for LM Studio
        
        if not RAG_READY:
            return jsonify({'status': 'error', 'message': 'RAG service not available'}), 503
        
        if model_type == 'lm_studio':
            if not LM_STUDIO_READY:
                return jsonify({'status': 'error', 'message': 'LM Studio service not available. Make sure LM Studio is running.'}), 503
            menu_plan = rag_service_lm_studio.generate_menu_plan_with_chroma(user_input, model_name)
        else:  # Default to Gemini
            if not GEMINI_READY:
                return jsonify({'status': 'error', 'message': 'Gemini API service not available.'}), 503
            menu_plan = rag_service_gemini.generate_menu_plan_with_chroma(user_input)
        
        return jsonify(menu_plan)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/status')
def status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'services': {
            'chromadb': 'ready',
            'gemini': 'ready' if GEMINI_READY else 'unavailable',
            'lm_studio': 'ready' if LM_STUDIO_READY else 'unavailable'
        }
    })


@app.route('/api/models')
def get_models():
    """Get available models from LM Studio"""
    try:
        models = []
        
        if GEMINI_READY:
            models.append({
                'id': 'gemini-2.5-flash',
                'name': 'Gemini 2.5 Flash',
                'provider': 'Google Gemini API',
                'available': True
            })
        
        if LM_STUDIO_READY:
            lm_models = rag_service_lm_studio.get_available_models()
            for model in lm_models:
                models.append({
                    'id': model,
                    'name': model,
                    'provider': 'LM Studio (Local)',
                    'available': True
                })
        
        return jsonify({
            'status': 'success',
            'models': models,
            'total': len(models)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("RAG MPASI Menu Generator - Flask Web UI (Dual Model Support)")
    print("="*70)
    print(f"ChromaDB:       {'[AVAILABLE] Connected' if RAG_READY else 'Not available'}")
    print(f"Gemini API:     {'[AVAILABLE] Ready' if GEMINI_READY else 'Not configured'}")
    print(f"LM Studio:      {'[AVAILABLE] Ready' if LM_STUDIO_READY else 'Not running'}")
    if LM_STUDIO_READY and lm_studio_models:
        print(f"  Available models: {', '.join(lm_studio_models)}")
    print("\nServer: http://localhost:5000")
    print("="*70 + "\n")
    # When debug=True Flask runs a reloader which spawns a monitoring parent and
    # a child worker process. Heavy initialization should run only in the child.
    # The WERKZEUG_RUN_MAIN env var is set to 'true' in the child process.
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        # Initialize services in the serving process
        init_services()

    # Run the app. Disable the reloader if you prefer to avoid the two-process behavior
    app.run(debug=True, host='0.0.0.0', port=5000)
