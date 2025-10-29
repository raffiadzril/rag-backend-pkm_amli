// Check API status on page load
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupTabNavigation();
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
            
            if (data.rag_service === 'ready') {
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

    // First fetch the debug prompt to show what will be sent
    const debugPromise = fetch('/api/debug-prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            age_months: ageMonths,
            weight_kg: weightKg,
            height_cm: heightCm,
            residence: residence,
            allergies: allergies
        })
    }).then(response => response.json());

    // Then fetch the actual menu generation
    const menuPromise = fetch('/api/generate-menu', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            age_months: ageMonths,
            weight_kg: weightKg,
            height_cm: heightCm,
            residence: residence,
            allergies: allergies
        })
    }).then(response => response.json());

    // Wait for both promises
    Promise.all([debugPromise, menuPromise])
        .then(([debugData, menuData]) => {
            showLoading('menu-loading', false);
            
            // Display debug info
            if (debugData.status === 'success') {
                displayDebugPrompt(debugData);
            }
            
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

// Display debug prompt information
function displayDebugPrompt(data) {
    const debugPanel = document.getElementById('menu-debug');
    const resultsDiv = document.getElementById('menu-results');
    
    if (data.debug_info) {
        document.getElementById('debug-search-query').textContent = data.debug_info.search_query;
        document.getElementById('debug-docs-count').textContent = data.debug_info.documents_retrieved;
    }
    
    if (data.full_prompt) {
        document.getElementById('debug-prompt-text').textContent = data.full_prompt;
    }
    
    if (data.prompt_length_chars) {
        document.getElementById('debug-prompt-length').textContent = data.prompt_length_chars.toLocaleString();
    }
    
    debugPanel.classList.remove('hidden');
}

// Display menu plan
function displayMenuPlan(data) {
    const resultsDiv = document.getElementById('menu-results');
    
    if (!data.data) {
        showError('menu-results', 'Invalid menu plan data');
        return;
    }

    const menuData = data.data;
    const userInfo = data.user_info;
    const ragInfo = data.rag_info;

    let html = `
        <div class="success">
            <strong>✓ Menu plan generated successfully!</strong>
            <br>Documents retrieved: ${ragInfo.documents_retrieved}
        </div>
        
        <h3>Baby Information</h3>
        <div class="result-item">
            <div class="result-content">
                <p><strong>Age:</strong> ${userInfo.age_months} months</p>
                <p><strong>Weight:</strong> ${userInfo.weight_kg} kg</p>
                <p><strong>Height:</strong> ${userInfo.height_cm} cm</p>
                <p><strong>Residence:</strong> ${userInfo.residence}</p>
                ${userInfo.allergies.length > 0 ? `<p><strong>Allergies:</strong> ${userInfo.allergies.join(', ')}</p>` : ''}
            </div>
        </div>

        <h3 style="margin-top: 25px;">Daily Menu Plan</h3>
    `;

    // Display each meal
    const meals = ['breakfast', 'morning_snack', 'lunch', 'afternoon_snack', 'dinner'];
    meals.forEach(mealKey => {
        if (menuData[mealKey]) {
            const meal = menuData[mealKey];
            html += `
                <div class="meal-section">
                    <div class="meal-time">${meal.time || 'N/A'}</div>
                    <div class="meal-name">${escapeHtml(meal.menu_name || mealKey)}</div>
                    
                    <div class="meal-info">
                        <div class="info-item">
                            <div class="info-label">Portion</div>
                            <div class="info-value">${escapeHtml(meal.portion || 'N/A')}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Energy</div>
                            <div class="info-value">${meal.nutrition?.energy_kcal || 0} kcal</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Protein</div>
                            <div class="info-value">${meal.nutrition?.protein_g || 0}g</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Carbs</div>
                            <div class="info-value">${meal.nutrition?.carbs_g || 0}g</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Fat</div>
                            <div class="info-value">${meal.nutrition?.fat_g || 0}g</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Ingredients:</strong>
                        <ul style="margin: 10px 0 0 20px; color: #333;">
                            ${(meal.ingredients || []).map(ing => `<li>${escapeHtml(ing)}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Instructions:</strong>
                        <p style="color: #333; margin-top: 8px;">${escapeHtml(meal.instructions || 'N/A')}</p>
                    </div>
                </div>
            `;
        }
    });

    // Display daily summary
    if (menuData.daily_summary) {
        const summary = menuData.daily_summary;
        html += `
            <div class="daily-summary">
                <h3>📊 Daily Nutrition Summary</h3>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-item-label">Total Energy</div>
                        <div class="summary-item-value">${summary.total_energy_kcal || 0} kcal</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Protein</div>
                        <div class="summary-item-value">${summary.total_protein_g || 0}g</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Carbs</div>
                        <div class="summary-item-value">${summary.total_carbs_g || 0}g</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Total Fat</div>
                        <div class="summary-item-value">${summary.total_fat_g || 0}g</div>
                    </div>
                </div>
                <p style="margin-top: 15px; opacity: 0.95;">
                    <strong>AKG Compliance:</strong> ${escapeHtml(summary.akg_compliance || 'N/A')}<br>
                    <strong>Reference:</strong> ${escapeHtml(summary.akg_reference || 'N/A')}
                </p>
            </div>
        `;
    }

    // Display notes and recommendations
    if (menuData.notes && menuData.notes.length > 0) {
        html += `
            <div style="margin-top: 20px;">
                <strong>📝 Notes:</strong>
                <ul style="margin: 10px 0 0 20px; color: #333;">
                    ${menuData.notes.map(note => `<li>${escapeHtml(note)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (menuData.recommendations && menuData.recommendations.length > 0) {
        html += `
            <div style="margin-top: 20px;">
                <strong>💡 Recommendations:</strong>
                <ul style="margin: 10px 0 0 20px; color: #333;">
                    ${menuData.recommendations.map(rec => `<li>${escapeHtml(rec)}</li>`).join('')}
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
    const element = document.getElementById(elementId);
    element.innerHTML = `<p class="error"><strong>Error:</strong> ${escapeHtml(message)}</p>`;
    element.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
