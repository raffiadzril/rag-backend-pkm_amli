// Global model tracking
let selectedModel = {
    type: 'gemini',
    name: 'gemini-2.5-flash'
};

// Check API status on page load
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupTabNavigation();
    loadAvailableModels();
});

// Setup tab navigation
function setupTabNavigation() {
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and tabs
            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked button and corresponding tab
            this.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });
}

// Check API status
function checkStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            const statusText = document.getElementById('status-text');
            const statusIndicator = document.querySelector('.status-indicator');
            
            // Check if any service is available
            const allReady = (data.services?.chromadb === 'ready' || true) && 
                            (data.services?.gemini === 'ready' || data.services?.lm_studio === 'ready');
            
            if (allReady) {
                statusText.textContent = 'System Ready';
                statusIndicator.classList.add('ready');
            } else {
                statusText.textContent = 'System Unavailable';
                statusIndicator.classList.remove('ready');
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            document.getElementById('status-text').textContent = 'Connection Error';
        });
}

// Load available models from server
function loadAvailableModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.models.length > 0) {
                const modelSelect = document.getElementById('model-select');
                const modelInfo = document.getElementById('model-info');
                
                // Clear existing options
                modelSelect.innerHTML = '';
                
                // Add new options
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = JSON.stringify({
                        type: model.provider.includes('Gemini') ? 'gemini' : 'lm_studio',
                        name: model.id
                    });
                    option.textContent = `${model.name} (${model.provider})`;
                    modelSelect.appendChild(option);
                });
                
                // Set default to first available model
                if (data.models.length > 0) {
                    modelSelect.selectedIndex = 0;
                    updateModelSelection();
                    modelInfo.textContent = `${data.models.length} model(s) available`;
                }
            } else {
                document.getElementById('model-info').textContent = 'No models available';
            }
        })
        .catch(error => {
            console.error('Error loading models:', error);
            document.getElementById('model-info').textContent = 'Error loading models';
        });
}

// Update model selection when dropdown changes
function updateModelSelection() {
    const modelSelect = document.getElementById('model-select');
    try {
        selectedModel = JSON.parse(modelSelect.value);
        console.log('Selected model:', selectedModel);
    } catch (error) {
        console.error('Error parsing model selection:', error);
    }
}

// Perform search
function performSearch() {
    const query = document.getElementById('search-query').value.trim();
    const topK = parseInt(document.getElementById('search-top-k').value);
    
    if (!query) {
        showError('search-results', 'Please enter a query');
        return;
    }

    showLoading('search-loading', true);

    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            top_k: topK
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading('search-loading', false);
        
        if (data.status === 'success') {
            displaySearchResults(data);
        } else {
            showError('search-results', data.message);
        }
    })
    .catch(error => {
        showLoading('search-loading', false);
        showError('search-results', `Error: ${error.message}`);
    });
}

// Perform search with scores
function performSearchWithScores() {
    const query = document.getElementById('scores-query').value.trim();
    const topK = parseInt(document.getElementById('scores-top-k').value);
    
    if (!query) {
        showError('scores-results', 'Please enter a query');
        return;
    }

    showLoading('scores-loading', true);

    fetch('/api/search-with-scores', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            top_k: topK
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading('scores-loading', false);
        
        if (data.status === 'success') {
            displayScoresResults(data);
        } else {
            showError('scores-results', data.message);
        }
    })
    .catch(error => {
        showLoading('scores-loading', false);
        showError('scores-results', `Error: ${error.message}`);
    });
}

// Generate menu plan
function generateMenu(event) {
    event.preventDefault();
    
    const ageMonths = parseInt(document.getElementById('age-months').value);
    const weightKg = parseFloat(document.getElementById('weight-kg').value);
    const heightCm = parseInt(document.getElementById('height-cm').value);
    const residence = document.getElementById('residence').value.trim();
    const allergiesStr = document.getElementById('allergies').value.trim();
    
    const allergies = allergiesStr 
        ? allergiesStr.split(',').map(a => a.trim()).filter(a => a)
        : [];

    showLoading('menu-loading', true);

    // Fetch the actual menu generation
    fetch('/api/generate-menu', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            age_months: ageMonths,
            weight_kg: weightKg,
            height_cm: heightCm,
            residence: residence,
            allergies: allergies,
            model_type: selectedModel.type,
            model_name: selectedModel.name
        })
    })
    .then(response => response.json())
    .then(menuData => {
        showLoading('menu-loading', false);
        
        // Display menu plan
        if (menuData.status === 'success') {
            displayMenuPlan(menuData);
        } else {
            showError('menu-results', menuData.message);
        }
    })
    .catch(error => {
        showLoading('menu-loading', false);
        showError('menu-results', `Error: ${error.message}`);
    });
}

// Display search results
function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<p class="error">No results found</p>';
        resultsDiv.classList.remove('hidden');
        return;
    }

    let html = `<h3>Search Results (${data.results_count} found)</h3>`;
    
    data.results.forEach((result, index) => {
        html += `
            <div class="result-item">
                <div class="result-header">
                    <strong>Result ${index + 1}</strong>
                </div>
                <div class="result-content">
                    <p>${escapeHtml(result.substring(0, 500))}</p>
                </div>
            </div>
        `;
    });

    resultsDiv.innerHTML = html;
    resultsDiv.classList.remove('hidden');
}

// Display search with scores results
function displayScoresResults(data) {
    const resultsDiv = document.getElementById('scores-results');
    
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<p class="error">No results found</p>';
        resultsDiv.classList.remove('hidden');
        return;
    }

    let html = `<h3>Search Results with Similarity Scores (${data.results_count} found)</h3>`;
    
    data.results.forEach((result, index) => {
        const percentage = (result.similarity_score * 100).toFixed(1);
        html += `
            <div class="result-item">
                <div class="result-header">
                    <strong>Result ${index + 1}</strong>
                    <span class="result-score">${percentage}% Match</span>
                </div>
                <div class="result-content">
                    <p>${escapeHtml(result.content.substring(0, 500))}</p>
                </div>
            </div>
        `;
    });

    resultsDiv.innerHTML = html;
    resultsDiv.classList.remove('hidden');
}



// Display menu plan
// Display menu plan
function displayMenuPlan(data) {
    const resultsDiv = document.getElementById('menu-results');

    // Check if the data is an error response
    if (data.status === 'error') {
        // Display the error message directly
        showError('menu-results', data.message);
        return;
    }

    // If it's not an error, proceed with displaying the menu plan
    if (!data.data) {
        showError('menu-results', 'Invalid menu plan data');
        return;
    }

    const menuData = data.data;
    const userInfo = data.user_info || {
        age_months: 'N/A',
        weight_kg: 'N/A',
        height_cm: 'N/A',
        residence: 'N/A',
        allergies: []
    };
    const ragInfo = data.rag_info || {
        documents_retrieved: 0
    };

    // Helper functions for safe string/number handling
    function safeString(val, defaultVal = 'N/A') {
        if (val === null || val === undefined) {
            return defaultVal;
        }
        if (typeof val === 'object') {
            return JSON.stringify(val);
        }
        return String(val);
    }

    function safeNumber(val, defaultVal = 0) {
        const num = Number(val);
        return isNaN(num) ? defaultVal : num;
    }

    let html = `
        <div class="success">
            <strong>✓ Menu plan generated successfully!</strong>
            <br>Documents retrieved: ${safeNumber(ragInfo.documents_retrieved, 0)}
        </div>

        <h3>Baby Information</h3>
        <div class="result-item">
            <div class="result-content">
                <p><strong>Age:</strong> ${safeNumber(userInfo.age_months, 'N/A')} months</p>
                <p><strong>Weight:</strong> ${safeNumber(userInfo.weight_kg, 'N/A')} kg</p>
                <p><strong>Height:</strong> ${safeNumber(userInfo.height_cm, 'N/A')} cm</p>
                <p><strong>Residence:</strong> ${safeString(userInfo.residence, 'N/A')}</p>
                ${(Array.isArray(userInfo.allergies) && userInfo.allergies.length > 0) ? `<p><strong>Allergies:</strong> ${escapeHtml(userInfo.allergies.join(', '))}</p>` : ''}
            </div>
        </div>

        <h3 style="margin-top: 25px;">Daily Menu Plan</h3>
    `;

    // Display each meal
    const meals = ['breakfast', 'morning_snack', 'lunch', 'afternoon_snack', 'dinner'];
    meals.forEach(mealKey => {
        if (menuData[mealKey]) {
            const meal = menuData[mealKey];
            // Use safeString for text fields and safeNumber for numeric fields
            const time = safeString(meal.time, 'N/A');
            const menuName = safeString(meal.menu_name, mealKey);
            const portion = safeString(meal.portion, 'N/A');
            const instructions = safeString(meal.instructions, 'N/A');

            const energyKcal = safeNumber(meal.nutrition?.energy_kcal, 0);
            const proteinG = safeNumber(meal.nutrition?.protein_g, 0);
            const carbsG = safeNumber(meal.nutrition?.carbs_g, 0);
            const fatG = safeNumber(meal.nutrition?.fat_g, 0);

            const ingredients = Array.isArray(meal.ingredients) ? meal.ingredients : [];

            html += `
                <div class="meal-section">
                    <div class="meal-time">${time}</div>
                    <div class="meal-name">${escapeHtml(menuName)}</div>

                    <div class="meal-info">
                        <div class="info-item">
                            <div class="info-label">Portion</div>
                            <div class="info-value">${escapeHtml(portion)}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Energy</div>
                            <div class="info-value">${energyKcal} kcal</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Protein</div>
                            <div class="info-value">${proteinG}g</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Carbs</div>
                            <div class="info-value">${carbsG}g</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Fat</div>
                            <div class="info-value">${fatG}g</div>
                        </div>
                    </div>

                    <div style="margin-top: 15px;">
                        <strong>Ingredients:</strong>
                        <div style="margin: 10px 0 0 0;">
                            ${ingredients.map(ing => {
                                // Handle both old format (string) and new format (object with TKPI code)
                                if (typeof ing === 'string') {
                                    return `<div class="ingredient-item">
                                        <div class="ingredient-name">${escapeHtml(ing)}</div>
                                    </div>`;
                                } else if (typeof ing === 'object' && ing !== null) {
                                    // Use safeString for object properties
                                    const name = safeString(ing.nama, 'N/A');
                                    const kode = safeString(ing.kode_tkpi, 'N/A');
                                    const jumlah = safeString(ing.jumlah, 'N/A');
                                    return `<div class="ingredient-item">
                                        <div class="ingredient-name">${escapeHtml(name)}</div>
                                        <div class="ingredient-meta">
                                            <span class="ingredient-label">KODE:</span>
                                            <span class="ingredient-code">${escapeHtml(kode)}</span>
                                            <span class="ingredient-label">JUMLAH:</span>
                                            <span class="ingredient-quantity">${escapeHtml(jumlah)}</span>
                                        </div>
                                    </div>`;
                                } else {
                                    // If ingredient is neither string nor object, stringify it
                                    return `<div class="ingredient-item">
                                        <div class="ingredient-name">${escapeHtml(safeString(ing))}</div>
                                    </div>`;
                                }
                            }).join('')}
                        </div>
                    </div>

                    <div style="margin-top: 15px;">
                        <strong>Instructions:</strong>
                        <p style="color: #333; margin-top: 8px;">${escapeHtml(instructions)}</p>
                    </div>
                </div>
            `;
        }
    });

    // Display daily summary
    if (menuData.daily_summary) {
        const summary = menuData.daily_summary;
        const totalEnergy = safeNumber(summary.total_energy_kcal, 0);
        const totalProtein = safeNumber(summary.total_protein_g, 0);
        const totalCarbs = safeNumber(summary.total_carbs_g, 0);
        const totalFat = safeNumber(summary.total_fat_g, 0);
        const akgCompliance = safeString(summary.akg_compliance, 'N/A');
        const akgReference = safeString(summary.akg_reference, 'N/A'); // Assuming this field might exist

        html += `
            <div class="daily-summary">
                <h3>📊 Daily Nutrition Summary</h3>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-item-label">Total Energy</div>
                        <div class="summary-item-value">${totalEnergy} kcal</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Protein</div>
                        <div class="summary-item-value">${totalProtein}g</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Carbs</div>
                        <div class="summary-item-value">${totalCarbs}g</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Fat</div>
                        <div class="summary-item-value">${totalFat}g</div>
                    </div>
                </div>
                <p style="margin-top: 15px; opacity: 0.95;">
                    <strong>AKG Compliance:</strong> ${escapeHtml(akgCompliance)}<br>
                    <strong>Reference:</strong> ${escapeHtml(akgReference)}
                </p>
            </div>
        `;
    }

    // Display notes and recommendations
    if (Array.isArray(menuData.notes) && menuData.notes.length > 0) {
        html += `
            <div style="margin-top: 20px;">
                <strong>📝 Notes:</strong>
                <ul style="margin: 10px 0 0 20px; color: #333;">
                    ${menuData.notes.map(note => `<li>${escapeHtml(safeString(note))}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (Array.isArray(menuData.recommendations) && menuData.recommendations.length > 0) {
        html += `
            <div style="margin-top: 20px;">
                <strong>💡 Recommendations:</strong>
                <ul style="margin: 10px 0 0 20px; color: #333;">
                    ${menuData.recommendations.map(rec => `<li>${escapeHtml(safeString(rec))}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    resultsDiv.innerHTML = html;
    resultsDiv.classList.remove('hidden');
}

// Helper functions
function showLoading(elementId, show) {
    const element = document.getElementById(elementId);
    if (show) {
        element.classList.add('active');
    } else {
        element.classList.remove('active');
    }
}

function showError(elementId, message) {
    console.log("showError called with message:", message, "Type:", typeof message); // Add this line
    const element = document.getElementById(elementId);
    element.innerHTML = `<p class="error"><strong>Error:</strong> ${escapeHtml(message)}</p>`;
    element.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
