/**
 * Human Disease Diagnosis System - Full-Stack Client Application
 * End-to-end frontend controller handling ML inference, interactive maps,
 * Chart.js analytics, chat assistant, and auth state.
 */

// Application Global State
const state = {
    token: localStorage.getItem('medicare_jwt') || null,
    user: JSON.parse(localStorage.getItem('medicare_user') || 'null'),
    symptoms: [],
    symptomCategories: [],
    selectedSymptoms: new Set(),
    activeCategory: 'All',
    searchTerm: '',
    lastDiagnosisResult: null,
    charts: {},
    map: null,
    mapMarkers: []
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    initAuthUI();
    initNavigation();
    await loadSymptomsData();
    initSymptomChecker();
    initSpecializedForms();
    initChatAssistant();
    initHospitalLocator();
    initKnowledgeBase();
    initAdminObservatory();
});

// ==========================================================================
// Toast Notifications
// ==========================================================================
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    const bgMap = {
        success: 'linear-gradient(135deg, #10b981, #059669)',
        error: 'linear-gradient(135deg, #f43f5e, #e11d48)',
        info: 'linear-gradient(135deg, #06b6d4, #6366f1)',
        warning: 'linear-gradient(135deg, #f59e0b, #d97706)'
    };
    toast.style.cssText = `background:${bgMap[type] || bgMap.info};color:#fff;padding:12px 20px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,0.3);font-size:0.9rem;font-weight:600;display:flex;align-items:center;gap:10px;animation:slideIn 0.25s ease;`;
    toast.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-circle-check' : type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i> <span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ==========================================================================
// Navigation & Tab Switching
// ==========================================================================
function initNavigation() {
    const navBtns = document.querySelectorAll('.nav-tab-btn');
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-tab-btn').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === `tab-${tabId}`);
    });

    // Lazy load or refresh tab contents
    if (tabId === 'history') {
        loadHistoryAndStats();
    } else if (tabId === 'locator') {
        setTimeout(() => {
            if (state.map) state.map.invalidateSize();
        }, 200);
    } else if (tabId === 'admin') {
        loadAdminMetrics();
    }
}

// ==========================================================================
// Authentication & User State
// ==========================================================================
function initAuthUI() {
    updateUserBadge();

    const authModal = document.getElementById('auth-modal');
    const openAuthBtn = document.getElementById('open-auth-btn');
    const closeAuthBtn = document.getElementById('close-auth-btn');
    const authTabLogin = document.getElementById('auth-tab-login');
    const authTabRegister = document.getElementById('auth-tab-register');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    if (openAuthBtn) openAuthBtn.addEventListener('click', () => openModal('auth-modal'));
    if (closeAuthBtn) closeAuthBtn.addEventListener('click', () => closeModal('auth-modal'));

    if (authTabLogin && authTabRegister) {
        authTabLogin.addEventListener('click', () => {
            authTabLogin.classList.add('active');
            authTabRegister.classList.remove('active');
            loginForm.style.display = 'block';
            registerForm.style.display = 'none';
        });
        authTabRegister.addEventListener('click', () => {
            authTabRegister.classList.add('active');
            authTabLogin.classList.remove('active');
            registerForm.style.display = 'block';
            loginForm.style.display = 'none';
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Login failed');
                handleAuthSuccess(data);
            } catch (err) {
                showToast(err.message, 'error');
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = document.getElementById('reg-name').value;
            const email = document.getElementById('reg-email').value;
            const password = document.getElementById('reg-password').value;
            const role = document.getElementById('reg-role').value;
            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, email, password, role })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Registration failed');
                handleAuthSuccess(data);
            } catch (err) {
                showToast(err.message, 'error');
            }
        });
    }
}

async function quickDemoLogin(role) {
    try {
        const res = await fetch('/api/auth/demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Demo login failed');
        handleAuthSuccess(data);
    } catch (err) {
        showToast(err.message, 'error');
    }
}

function handleAuthSuccess(data) {
    state.token = data.access_token;
    state.user = data.user;
    localStorage.setItem('medicare_jwt', data.access_token);
    localStorage.setItem('medicare_user', JSON.stringify(data.user));
    updateUserBadge();
    closeModal('auth-modal');
    showToast(`Welcome, ${data.user.name} (${data.user.role.toUpperCase()})`, 'success');
    if (document.getElementById('tab-history').classList.contains('active')) {
        loadHistoryAndStats();
    }
}

function logout() {
    state.token = null;
    state.user = null;
    localStorage.removeItem('medicare_jwt');
    localStorage.removeItem('medicare_user');
    updateUserBadge();
    showToast('Logged out successfully', 'info');
}

function updateUserBadge() {
    const container = document.getElementById('user-auth-badge-container');
    if (!container) return;

    if (state.user) {
        container.innerHTML = `
            <div class="user-profile-badge" title="Click to view options" onclick="toggleUserDropdown(event)">
                <div class="user-avatar">${state.user.name.charAt(0).toUpperCase()}</div>
                <div style="font-size:0.85rem; font-weight:600;">
                    ${state.user.name.split(' ')[0]} 
                    <span style="font-size:0.7rem; color:var(--accent-cyan); text-transform:uppercase;">[${state.user.role}]</span>
                </div>
                <i class="fa-solid fa-chevron-down" style="font-size:0.7rem; color:var(--text-muted);"></i>
            </div>
            <div id="user-dropdown-menu" style="display:none; position:absolute; top:55px; right:20px; background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:10px; min-width:180px; box-shadow:0 10px 25px rgba(0,0,0,0.5); z-index:110;">
                <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid var(--border-color);">
                    ${state.user.email}
                </div>
                <button class="btn-secondary" style="width:100%; text-align:left; padding:6px 10px; margin-bottom:6px; font-size:0.82rem;" onclick="switchTab('history'); hideUserDropdown();">
                    <i class="fa-solid fa-clock-rotate-left"></i> My Records
                </button>
                <button class="btn-danger" style="width:100%; text-align:left; padding:6px 10px; font-size:0.82rem;" onclick="logout(); hideUserDropdown();">
                    <i class="fa-solid fa-arrow-right-from-bracket"></i> Sign Out
                </button>
            </div>
        `;
    } else {
        container.innerHTML = `
            <button class="btn-secondary" id="open-auth-btn" onclick="openModal('auth-modal')" style="font-size:0.85rem; padding:0.45rem 1rem;">
                <i class="fa-solid fa-user-lock"></i> Sign In / Register
            </button>
        `;
    }
}

function toggleUserDropdown(e) {
    e.stopPropagation();
    const menu = document.getElementById('user-dropdown-menu');
    if (menu) {
        menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    }
}
function hideUserDropdown() {
    const menu = document.getElementById('user-dropdown-menu');
    if (menu) menu.style.display = 'none';
}
document.addEventListener('click', hideUserDropdown);

// ==========================================================================
// Symptom-Based Multi-Class Predictor
// ==========================================================================
async function loadSymptomsData() {
    try {
        const res = await fetch('/api/symptoms');
        const data = await res.json();
        state.symptoms = data.symptoms;
        state.symptomCategories = ['All', ...data.categories];
        renderCategoryFilters();
        renderSymptomsCloud();
    } catch (err) {
        console.error('Error loading symptoms:', err);
    }
}

function renderCategoryFilters() {
    const container = document.getElementById('category-chips-container');
    if (!container) return;

    container.innerHTML = state.symptomCategories.map(cat => `
        <button class="filter-chip ${state.activeCategory === cat ? 'active' : ''}" onclick="filterCategory('${cat}')">
            ${cat}
        </button>
    `).join('');
}

function filterCategory(category) {
    state.activeCategory = category;
    renderCategoryFilters();
    renderSymptomsCloud();
}

function renderSymptomsCloud() {
    const container = document.getElementById('symptoms-cloud');
    if (!container) return;

    const filtered = state.symptoms.filter(sym => {
        const matchesCat = state.activeCategory === 'All' || sym.category === state.activeCategory;
        const matchesSearch = !state.searchTerm ||
            sym.display_name.toLowerCase().includes(state.searchTerm.toLowerCase()) ||
            sym.name.toLowerCase().includes(state.searchTerm.toLowerCase());
        return matchesCat && matchesSearch;
    });

    if (filtered.length === 0) {
        container.innerHTML = `<div style="padding:15px; color:var(--text-muted); font-size:0.85rem;">No symptoms found matching "${state.searchTerm}".</div>`;
        return;
    }

    container.innerHTML = filtered.map(sym => {
        const isSelected = state.selectedSymptoms.has(sym.name);
        const sevClass = sym.weight <= 3 ? 'sev-low' : sym.weight <= 5 ? 'sev-med' : 'sev-high';
        return `
            <div class="symptom-tag-chip ${isSelected ? 'selected' : ''}" onclick="toggleSymptom('${sym.name}')">
                <span>${sym.display_name}</span>
                <span class="severity-pill ${sevClass}">L${sym.weight}</span>
                <i class="fa-solid ${isSelected ? 'fa-check' : 'fa-plus'}" style="font-size:0.7rem;"></i>
            </div>
        `;
    }).join('');
}

function toggleSymptom(symptomName) {
    if (state.selectedSymptoms.has(symptomName)) {
        state.selectedSymptoms.delete(symptomName);
    } else {
        state.selectedSymptoms.add(symptomName);
    }
    renderSymptomsCloud();
    renderSelectedSymptomsTray();
}

function renderSelectedSymptomsTray() {
    const container = document.getElementById('selected-symptoms-list');
    const countBadge = document.getElementById('selected-count-badge');
    const predictBtn = document.getElementById('run-symptom-predict-btn');

    if (countBadge) countBadge.textContent = `${state.selectedSymptoms.size} selected`;
    if (predictBtn) predictBtn.disabled = state.selectedSymptoms.size === 0;

    if (!container) return;

    if (state.selectedSymptoms.size === 0) {
        container.innerHTML = `<span style="color:var(--text-muted); font-size:0.85rem; font-style:italic;">No symptoms added yet. Click from the list above or search.</span>`;
        return;
    }

    const chips = Array.from(state.selectedSymptoms).map(name => {
        const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `
            <div class="active-chip">
                <span>${displayName}</span>
                <i class="fa-solid fa-xmark remove-btn" onclick="toggleSymptom('${name}')"></i>
            </div>
        `;
    }).join('');

    container.innerHTML = chips;
}

function clearAllSymptoms() {
    state.selectedSymptoms.clear();
    renderSymptomsCloud();
    renderSelectedSymptomsTray();
    showToast('Cleared selected symptoms', 'info');
}

function applySymptomPreset(presetName) {
    state.selectedSymptoms.clear();
    const presets = {
        pneumonia: ['chills', 'cough', 'high_fever', 'breathlessness', 'sweating', 'chest_pain', 'phlegm'],
        dengue: ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'high_fever', 'headache', 'muscle_pain'],
        gerd: ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'],
        migraine: ['headache', 'blurred_and_distorted_vision', 'excessive_hunger', 'stiff_neck', 'irritability'],
        typhoid: ['chills', 'vomiting', 'high_fever', 'headache', 'constipation', 'abdominal_pain', 'diarrhoea'],
        allergy: ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
        diabetes: ['fatigue', 'weight_loss', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'excessive_hunger', 'polyuria']
    };

    const target = presets[presetName] || [];
    target.forEach(s => state.selectedSymptoms.add(s));
    renderSymptomsCloud();
    renderSelectedSymptomsTray();
    showToast(`Loaded "${presetName.toUpperCase()}" clinical scenario preset`, 'success');
}

function initSymptomChecker() {
    const searchInput = document.getElementById('symptom-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            state.searchTerm = e.target.value;
            renderSymptomsCloud();
        });
    }

    const predictBtn = document.getElementById('run-symptom-predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', runSymptomDiagnosis);
    }
}

async function runSymptomDiagnosis() {
    if (state.selectedSymptoms.size === 0) {
        showToast('Please select at least one symptom to evaluate.', 'warning');
        return;
    }

    const predictBtn = document.getElementById('run-symptom-predict-btn');
    const originalText = predictBtn.innerHTML;
    predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Clinical Patterns...';
    predictBtn.disabled = true;

    try {
        const payload = {
            symptoms: Array.from(state.selectedSymptoms),
            notes: document.getElementById('symptom-patient-notes')?.value || null
        };

        const headers = { 'Content-Type': 'application/json' };
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

        const res = await fetch('/api/predict/symptoms', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        state.lastDiagnosisResult = data;
        renderDiagnosisResult(data);
        showToast(`Diagnosis generated: ${data.primary_disease} (${data.confidence}%)`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        predictBtn.innerHTML = originalText;
        predictBtn.disabled = false;
    }
}

function renderDiagnosisResult(res) {
    const container = document.getElementById('diagnosis-results-card');
    if (!container) return;

    const primary = res.primary_details || {};
    const triageClass = res.severity_level === 'Critical' ? 'triage-critical' :
        res.severity_level === 'High' ? 'triage-high' :
            res.severity_level === 'Moderate' ? 'triage-moderate' : 'triage-mild';

    const emergencyBanner = res.emergency_flags && res.emergency_flags.length > 0 ? `
        <div class="triage-alert-box triage-critical" style="margin-bottom:15px;">
            <i class="fa-solid fa-triangle-exclamation" style="font-size:1.4rem;"></i>
            <div>
                <strong>RED-FLAG EMERGENCY WARNING</strong><br>
                ${res.emergency_flags.map(e => `• ${e.symptom}: ${e.warning}`).join('<br>')}
                <div style="margin-top:6px; font-weight:700;">Please call emergency medical services immediately!</div>
            </div>
        </div>
    ` : '';

    const topKList = (res.top_predictions || []).map((pred, i) => `
        <div class="topk-item">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-weight:700; color:${i === 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'};">#${i + 1}</span>
                <span style="font-weight:600;">${pred.disease}</span>
                <span class="badge-tag" style="font-size:0.65rem;">${pred.category}</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width:${Math.min(100, pred.confidence)}%;"></div>
                </div>
                <span style="font-weight:700; font-size:0.85rem; color:var(--accent-cyan);">${pred.confidence}%</span>
            </div>
        </div>
    `).join('');

    container.innerHTML = `
        <div class="glass-card-header">
            <div>
                <h3 class="card-title"><i class="fa-solid fa-stethoscope" style="color:var(--accent-cyan);"></i> Diagnostic Assessment</h3>
                <p class="card-subtitle">AI & Multi-Class Machine Learning Clinical Decision Support</p>
            </div>
            <button class="btn-secondary" style="font-size:0.8rem;" onclick="openPrintableSummaryModal()">
                <i class="fa-solid fa-file-pdf"></i> Medical Summary
            </button>
        </div>

        ${emergencyBanner}

        <div class="primary-result-banner">
            <div>
                <span class="badge-tag">${primary.category || 'General Medicine'}</span>
                <div class="disease-name-highlight">${res.primary_disease}</div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:4px;">
                    Matched ${res.symptom_count} symptom markers | Severity Score: ${res.severity_score} pts
                </div>
            </div>
            <div class="confidence-meter">
                <div class="conf-value">${res.confidence}%</div>
                <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; font-weight:600;">Confidence</div>
            </div>
        </div>

        <div class="triage-alert-box ${triageClass}">
            <i class="fa-solid fa-bell"></i>
            <div>
                <strong>Triage Status: ${res.severity_level}</strong> — ${res.triage_urgency}
            </div>
        </div>

        <h4 style="font-size:0.95rem; font-weight:700; margin:15px 0 8px 0; color:var(--text-secondary);">
            Differential Diagnoses (Ranked Top-k):
        </h4>
        <div class="topk-ranking-list">${topKList}</div>

        <!-- Disease Info Sub-Tabs -->
        <div class="info-subtabs">
            <button class="info-subtab-btn active" onclick="switchInfoSubtab('overview')"><i class="fa-solid fa-circle-info"></i> Overview</button>
            <button class="info-subtab-btn" onclick="switchInfoSubtab('precautions')"><i class="fa-solid fa-shield-halved"></i> Precautions</button>
            <button class="info-subtab-btn" onclick="switchInfoSubtab('medications')"><i class="fa-solid fa-pills"></i> Medications</button>
            <button class="info-subtab-btn" onclick="switchInfoSubtab('lifestyle')"><i class="fa-solid fa-apple-whole"></i> Diet & Lifestyle</button>
        </div>

        <div id="subtab-overview" class="info-subtab-content" style="display:block;">
            <p>${primary.description || 'Comprehensive clinical overview available upon medical consultation.'}</p>
        </div>

        <div id="subtab-precautions" class="info-subtab-content" style="display:none;">
            <ul class="bullet-list">
                ${(primary.precautions || []).map(p => `<li>${p}</li>`).join('') || '<li>Maintain standard clinical hygiene and observe symptoms.</li>'}
            </ul>
        </div>

        <div id="subtab-medications" class="info-subtab-content" style="display:none;">
            <p style="font-size:0.85rem; color:var(--accent-amber); margin-bottom:8px;">
                <i class="fa-solid fa-triangle-exclamation"></i> Prescription Notice: Medications must be verified and dispensed under licensed physician supervision.
            </p>
            <ul class="bullet-list">
                ${(primary.medications || []).map(m => `<li>${m}</li>`).join('')}
            </ul>
        </div>

        <div id="subtab-lifestyle" class="info-subtab-content" style="display:none;">
            <div style="margin-bottom:10px;">
                <strong style="color:var(--accent-cyan);"><i class="fa-solid fa-utensils"></i> Recommended Nutrition:</strong>
                <ul class="bullet-list" style="margin-top:4px;">
                    ${(primary.diet || []).map(d => `<li>${d}</li>`).join('')}
                </ul>
            </div>
            <div>
                <strong style="color:var(--accent-emerald);"><i class="fa-solid fa-person-walking"></i> Activity Guidance:</strong>
                <ul class="bullet-list" style="margin-top:4px;">
                    ${(primary.workout || []).map(w => `<li>${w}</li>`).join('')}
                </ul>
            </div>
        </div>

        <div style="display:flex; gap:10px; margin-top:20px;">
            <button class="btn-secondary" style="flex:1;" onclick="switchTab('locator'); filterHospitalsByDepartment('${primary.category || ''}');">
                <i class="fa-solid fa-location-dot"></i> Find Specialists
            </button>
            <button class="btn-secondary" style="flex:1;" onclick="askAssistantAboutCurrentDiagnosis('${res.primary_disease}')">
                <i class="fa-solid fa-comments"></i> Ask AI Assistant
            </button>
        </div>
    `;
}

function switchInfoSubtab(subtabName) {
    document.querySelectorAll('.info-subtab-btn').forEach(b => {
        b.classList.toggle('active', b.textContent.toLowerCase().includes(subtabName));
    });
    ['overview', 'precautions', 'medications', 'lifestyle'].forEach(tab => {
        const el = document.getElementById(`subtab-${tab}`);
        if (el) el.style.display = tab === subtabName ? 'block' : 'none';
    });
}

function askAssistantAboutCurrentDiagnosis(diseaseName) {
    switchTab('chat');
    const input = document.getElementById('chat-input-text');
    if (input) {
        input.value = `Can you explain more about ${diseaseName} and what next steps I should take?`;
        sendChatMessage();
    }
}

// ==========================================================================
// Specialized Diagnostic Labs (Diabetes, Heart Disease, Parkinson's)
// ==========================================================================
function initSpecializedForms() {
    // Specialized Subtabs
    document.querySelectorAll('.spec-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const specId = btn.getAttribute('data-spec');
            document.querySelectorAll('.spec-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ['diabetes', 'heart', 'parkinsons'].forEach(s => {
                const formEl = document.getElementById(`spec-form-${s}`);
                if (formEl) formEl.style.display = s === specId ? 'block' : 'none';
            });
        });
    });

    // Diabetes Form Submit
    const diabForm = document.getElementById('form-diabetes-prediction');
    if (diabForm) {
        diabForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                Glucose: parseFloat(document.getElementById('diab-glucose').value),
                BMI: parseFloat(document.getElementById('diab-bmi').value),
                Age: parseFloat(document.getElementById('diab-age').value),
                BloodPressure: parseFloat(document.getElementById('diab-bp').value),
                Insulin: parseFloat(document.getElementById('diab-insulin').value),
                Pregnancies: parseFloat(document.getElementById('diab-pregnancies').value),
                SkinThickness: parseFloat(document.getElementById('diab-skinthickness').value),
                DiabetesPedigreeFunction: parseFloat(document.getElementById('diab-dpf').value)
            };
            await executeSpecializedPrediction('/api/predict/diabetes', payload, 'diabetes');
        });
    }

    // Heart Disease Form Submit
    const heartForm = document.getElementById('form-heart-prediction');
    if (heartForm) {
        heartForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                age: parseFloat(document.getElementById('heart-age').value),
                sex: parseFloat(document.getElementById('heart-sex').value),
                cp: parseFloat(document.getElementById('heart-cp').value),
                trestbps: parseFloat(document.getElementById('heart-trestbps').value),
                chol: parseFloat(document.getElementById('heart-chol').value),
                fbs: parseFloat(document.getElementById('heart-fbs').value),
                restecg: parseFloat(document.getElementById('heart-restecg').value),
                thalach: parseFloat(document.getElementById('heart-thalach').value),
                exang: parseFloat(document.getElementById('heart-exang').value),
                oldpeak: parseFloat(document.getElementById('heart-oldpeak').value),
                slope: parseFloat(document.getElementById('heart-slope').value),
                ca: parseFloat(document.getElementById('heart-ca').value),
                thal: parseFloat(document.getElementById('heart-thal').value)
            };
            await executeSpecializedPrediction('/api/predict/heart', payload, 'heart');
        });
    }

    // Parkinson's Form Submit
    const parkForm = document.getElementById('form-parkinsons-prediction');
    if (parkForm) {
        parkForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                "MDVP:Fo(Hz)": parseFloat(document.getElementById('park-fo').value),
                "MDVP:Fhi(Hz)": parseFloat(document.getElementById('park-fhi').value),
                "MDVP:Flo(Hz)": parseFloat(document.getElementById('park-flo').value),
                "MDVP:Jitter(%)": parseFloat(document.getElementById('park-jitter').value),
                "MDVP:Shimmer": parseFloat(document.getElementById('park-shimmer').value),
                "HNR": parseFloat(document.getElementById('park-hnr').value),
                "RPDE": parseFloat(document.getElementById('park-rpde').value),
                "DFA": parseFloat(document.getElementById('park-dfa').value),
                "spread1": parseFloat(document.getElementById('park-spread1').value),
                "spread2": parseFloat(document.getElementById('park-spread2').value),
                "PPE": parseFloat(document.getElementById('park-ppe').value)
            };
            await executeSpecializedPrediction('/api/predict/parkinsons', payload, 'parkinsons');
        });
    }
}

async function executeSpecializedPrediction(endpoint, payload, type) {
    const resBox = document.getElementById('specialized-result-box');
    resBox.innerHTML = `<div style="text-align:center; padding:30px;"><i class="fa-solid fa-spinner fa-spin fa-2x" style="color:var(--accent-cyan);"></i><p style="margin-top:10px;">Running diagnostic inference...</p></div>`;

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        renderSpecializedResult(data, type);
        showToast(`${type.toUpperCase()} assessment: ${data.result} (${data.probability}%)`, 'success');
    } catch (err) {
        resBox.innerHTML = `<div style="padding:20px; color:#f87171;">Error: ${err.message}</div>`;
        showToast(err.message, 'error');
    }
}

function renderSpecializedResult(data, type) {
    const resBox = document.getElementById('specialized-result-box');
    const isHigh = data.probability >= 50;
    const badgeColor = isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)';

    resBox.innerHTML = `
        <div class="glass-card" style="border-color:${badgeColor}; animation:fadeIn 0.3s ease;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <div>
                    <span class="badge-tag" style="background:${badgeColor}22; color:${badgeColor}; border-color:${badgeColor}44;">
                        ${type.toUpperCase()} CLINICAL REPORT
                    </span>
                    <h3 style="font-size:1.4rem; font-weight:800; margin-top:4px;">${data.result}</h3>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.8rem; font-weight:800; color:${badgeColor};">${data.probability}%</div>
                    <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">RISK PROBABILITY</div>
                </div>
            </div>

            <div class="triage-alert-box" style="background:${badgeColor}15; border:1px solid ${badgeColor}44; color:#fff; margin-bottom:15px;">
                <i class="fa-solid fa-notes-medical" style="color:${badgeColor}; font-size:1.2rem;"></i>
                <div>
                    <strong>Risk Tier:</strong> ${data.risk_tier}
                </div>
            </div>

            <h4 style="font-size:0.92rem; font-weight:700; color:var(--text-secondary); margin-bottom:8px;">
                Clinical Guidance & Next Steps:
            </h4>
            <ul class="bullet-list">
                ${(data.recommendations || []).map(r => `<li>${r}</li>`).join('')}
            </ul>

            <div style="margin-top:15px; padding-top:15px; border-top:1px solid var(--border-color); display:flex; gap:10px;">
                <button class="btn-secondary" style="flex:1;" onclick="switchTab('locator')">
                    <i class="fa-solid fa-user-doctor"></i> Consult Specialist
                </button>
            </div>
        </div>
    `;
}

// Specialized Presets
function applySpecializedPreset(testType, presetName) {
    if (testType === 'diabetes') {
        if (presetName === 'healthy') {
            document.getElementById('diab-glucose').value = 95;
            document.getElementById('diab-bmi').value = 22.4;
            document.getElementById('diab-age').value = 28;
            document.getElementById('diab-bp').value = 68;
            document.getElementById('diab-insulin').value = 55;
            document.getElementById('diab-pregnancies').value = 0;
            document.getElementById('diab-skinthickness').value = 18;
            document.getElementById('diab-dpf').value = 0.25;
        } else if (presetName === 'prediabetic') {
            document.getElementById('diab-glucose').value = 135;
            document.getElementById('diab-bmi').value = 28.5;
            document.getElementById('diab-age').value = 52;
            document.getElementById('diab-bp').value = 78;
            document.getElementById('diab-insulin').value = 110;
            document.getElementById('diab-pregnancies').value = 2;
            document.getElementById('diab-skinthickness').value = 26;
            document.getElementById('diab-dpf').value = 0.52;
        } else if (presetName === 'highrisk') {
            document.getElementById('diab-glucose').value = 185;
            document.getElementById('diab-bmi').value = 36.8;
            document.getElementById('diab-age').value = 58;
            document.getElementById('diab-bp').value = 88;
            document.getElementById('diab-insulin').value = 240;
            document.getElementById('diab-pregnancies').value = 5;
            document.getElementById('diab-skinthickness').value = 38;
            document.getElementById('diab-dpf').value = 0.88;
        }
    } else if (testType === 'heart') {
        if (presetName === 'normal') {
            document.getElementById('heart-age').value = 42;
            document.getElementById('heart-sex').value = 1;
            document.getElementById('heart-cp').value = 2;
            document.getElementById('heart-trestbps').value = 118;
            document.getElementById('heart-chol').value = 185;
            document.getElementById('heart-fbs').value = 0;
            document.getElementById('heart-restecg').value = 0;
            document.getElementById('heart-thalach').value = 168;
            document.getElementById('heart-exang').value = 0;
            document.getElementById('heart-oldpeak').value = 0.2;
            document.getElementById('heart-slope').value = 2;
            document.getElementById('heart-ca').value = 0;
            document.getElementById('heart-thal').value = 2;
        } else if (presetName === 'highrisk') {
            document.getElementById('heart-age').value = 63;
            document.getElementById('heart-sex').value = 1;
            document.getElementById('heart-cp').value = 0;
            document.getElementById('heart-trestbps').value = 158;
            document.getElementById('heart-chol').value = 295;
            document.getElementById('heart-fbs').value = 1;
            document.getElementById('heart-restecg').value = 1;
            document.getElementById('heart-thalach').value = 115;
            document.getElementById('heart-exang').value = 1;
            document.getElementById('heart-oldpeak').value = 2.8;
            document.getElementById('heart-slope').value = 0;
            document.getElementById('heart-ca').value = 2;
            document.getElementById('heart-thal').value = 3;
        }
    } else if (testType === 'parkinsons') {
        if (presetName === 'healthy') {
            document.getElementById('park-fo').value = 197.0;
            document.getElementById('park-fhi').value = 206.0;
            document.getElementById('park-flo').value = 192.0;
            document.getElementById('park-jitter').value = 0.0028;
            document.getElementById('park-shimmer').value = 0.015;
            document.getElementById('park-hnr').value = 26.5;
            document.getElementById('park-rpde').value = 0.35;
            document.getElementById('park-dfa').value = 0.62;
            document.getElementById('park-spread1').value = -6.8;
            document.getElementById('park-spread2').value = 0.12;
            document.getElementById('park-ppe').value = 0.11;
        } else if (presetName === 'instability') {
            document.getElementById('park-fo').value = 116.0;
            document.getElementById('park-fhi').value = 137.0;
            document.getElementById('park-flo').value = 86.0;
            document.getElementById('park-jitter').value = 0.018;
            document.getElementById('park-shimmer').value = 0.075;
            document.getElementById('park-hnr').value = 12.4;
            document.getElementById('park-rpde').value = 0.64;
            document.getElementById('park-dfa').value = 0.81;
            document.getElementById('park-spread1').value = -3.8;
            document.getElementById('park-spread2').value = 0.38;
            document.getElementById('park-ppe').value = 0.42;
        }
    }
    showToast(`Loaded ${presetName.toUpperCase()} sample profile for ${testType.toUpperCase()}`, 'info');
}

// ==========================================================================
// AI Symptom Assistant Chatbot
// ==========================================================================
const chatHistory = [];

function initChatAssistant() {
    const input = document.getElementById('chat-input-text');
    const sendBtn = document.getElementById('chat-send-btn');

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    if (sendBtn) {
        sendBtn.addEventListener('click', sendChatMessage);
    }
}

function sendPromptText(text) {
    const input = document.getElementById('chat-input-text');
    if (input) {
        input.value = text;
        sendChatMessage();
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input-text');
    const messagesBox = document.getElementById('chat-messages-box');
    const text = input.value.trim();
    if (!text) return;

    // Append user message
    input.value = '';
    appendMessageToUI('user', text);
    chatHistory.push({ role: 'user', content: text });

    // Show typing indicator
    const typingId = 'typing-indicator-' + Date.now();
    const typingNode = document.createElement('div');
    typingNode.id = typingId;
    typingNode.className = 'chat-message assistant';
    typingNode.innerHTML = `
        <div class="chat-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="chat-bubble"><i class="fa-solid fa-ellipsis fa-fade"></i> AI Clinical Assistant is analyzing...</div>
    `;
    messagesBox.appendChild(typingNode);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: chatHistory.slice(-6)
            })
        });

        const data = await res.json();
        document.getElementById(typingId)?.remove();

        chatHistory.push({ role: 'assistant', content: data.reply });
        appendAssistantResponseToUI(data);
    } catch (err) {
        document.getElementById(typingId)?.remove();
        appendMessageToUI('assistant', `⚠️ Sorry, I encountered an issue: ${err.message}`);
    }
}

function appendMessageToUI(role, content) {
    const messagesBox = document.getElementById('chat-messages-box');
    const msg = document.createElement('div');
    msg.className = `chat-message ${role}`;
    msg.innerHTML = `
        <div class="chat-avatar"><i class="fa-solid ${role === 'assistant' ? 'fa-robot' : 'fa-user'}"></i></div>
        <div class="chat-bubble">${formatMarkdownText(content)}</div>
    `;
    messagesBox.appendChild(msg);
    messagesBox.scrollTop = messagesBox.scrollHeight;
}

function appendAssistantResponseToUI(data) {
    const messagesBox = document.getElementById('chat-messages-box');
    const msg = document.createElement('div');
    msg.className = 'chat-message assistant';

    let quickActionHtml = '';
    if (data.quick_prediction && data.quick_prediction.primary_disease) {
        const qp = data.quick_prediction;
        quickActionHtml = `
            <div style="margin-top:12px; padding:10px; background:rgba(6, 182, 212, 0.15); border:1px solid var(--accent-cyan); border-radius:8px;">
                <div style="font-size:0.8rem; font-weight:700; color:var(--accent-cyan); text-transform:uppercase;">
                    ⚡ Live Predictive Triage
                </div>
                <div style="font-weight:800; font-size:1.1rem; color:#fff;">
                    ${qp.primary_disease} <span style="font-size:0.85rem; color:var(--accent-cyan);">(${qp.confidence}%)</span>
                </div>
                <button class="btn-secondary" style="margin-top:8px; width:100%; font-size:0.8rem;" onclick="loadSymptomsFromChatAndDiagnose(${JSON.stringify(data.extracted_symptoms).replace(/"/g, '&quot;')})">
                    <i class="fa-solid fa-arrow-up-right-from-square"></i> Open Full Diagnostic Panel
                </button>
            </div>
        `;
    }

    msg.innerHTML = `
        <div class="chat-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="chat-bubble">
            ${formatMarkdownText(data.reply)}
            ${quickActionHtml}
        </div>
    `;
    messagesBox.appendChild(msg);

    // Update Suggested Prompts
    const promptContainer = document.getElementById('chat-suggested-prompts');
    if (promptContainer && data.suggested_questions) {
        promptContainer.innerHTML = data.suggested_questions.map(q => `
            <div class="prompt-chip" onclick="sendPromptText('${q.replace(/'/g, "\\'")}')">
                <i class="fa-solid fa-sparkles" style="font-size:0.65rem;"></i> ${q}
            </div>
        `).join('');
    }

    messagesBox.scrollTop = messagesBox.scrollHeight;
}

function loadSymptomsFromChatAndDiagnose(symptoms) {
    state.selectedSymptoms.clear();
    symptoms.forEach(s => state.selectedSymptoms.add(s));
    renderSymptomsCloud();
    renderSelectedSymptomsTray();
    switchTab('symptoms');
    runSymptomDiagnosis();
}

function formatMarkdownText(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

// ==========================================================================
// Hospital Locator & Leaflet Map
// ==========================================================================
let allHospitals = [];

async function initHospitalLocator() {
    const mapEl = document.getElementById('map-container');
    if (!mapEl) return;

    try {
        // Initialize Leaflet Map centered on coordinates
        state.map = L.map('map-container').setView([37.7749, -122.4194], 12);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(state.map);

        await loadHospitals();
    } catch (err) {
        console.error('Map init error:', err);
    }
}

async function loadHospitals(dept = '', emergencyOnly = false, query = '') {
    try {
        let url = `/api/hospitals?user_lat=37.7749&user_lng=-122.4194`;
        if (dept) url += `&department=${encodeURIComponent(dept)}`;
        if (emergencyOnly) url += `&emergency_only=true`;
        if (query) url += `&query=${encodeURIComponent(query)}`;

        const res = await fetch(url);
        const data = await res.json();
        allHospitals = data.hospitals;
        renderHospitalsList(allHospitals);
        renderMapMarkers(allHospitals);
    } catch (err) {
        console.error('Error fetching hospitals:', err);
    }
}

function renderHospitalsList(hospitals) {
    const container = document.getElementById('hospitals-list-container');
    if (!container) return;

    if (hospitals.length === 0) {
        container.innerHTML = `<div style="padding:20px; color:var(--text-muted);">No medical facilities match the current criteria.</div>`;
        return;
    }

    container.innerHTML = hospitals.map(h => `
        <div class="hospital-card" onclick="focusHospitalLocation(${h.lat}, ${h.lng}, '${h.name.replace(/'/g, "\\'")}')">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <h4 style="font-weight:700; font-size:1rem; color:#fff;">${h.name}</h4>
                    <p style="font-size:0.8rem; color:var(--accent-cyan); margin:2px 0;">${h.department}</p>
                </div>
                <div style="display:flex; align-items:center; gap:4px; font-weight:700; color:var(--accent-amber); font-size:0.85rem;">
                    <i class="fa-solid fa-star"></i> ${h.rating}
                </div>
            </div>
            <div style="font-size:0.8rem; color:var(--text-secondary); margin:6px 0;">
                <i class="fa-solid fa-location-dot"></i> ${h.address}
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.78rem;">
                <span style="color:${h.emergency_ready ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-weight:600;">
                    <i class="fa-solid ${h.emergency_ready ? 'fa-truck-medical' : 'fa-clock'}"></i> ${h.emergency_ready ? '24/7 Emergency' : 'Standard Hours'}
                </span>
                <span style="color:var(--accent-indigo); font-weight:700;">
                    <i class="fa-solid fa-route"></i> ${h.distance_miles} miles away
                </span>
            </div>
            <div style="margin-top:8px; display:flex; gap:6px;">
                <a href="tel:${h.phone}" class="btn-secondary" style="flex:1; text-align:center; font-size:0.75rem; text-decoration:none; padding:4px 8px;" onclick="event.stopPropagation();">
                    <i class="fa-solid fa-phone"></i> ${h.phone}
                </a>
            </div>
        </div>
    `).join('');
}

function renderMapMarkers(hospitals) {
    if (!state.map) return;
    state.mapMarkers.forEach(m => state.map.removeLayer(m));
    state.mapMarkers = [];

    hospitals.forEach(h => {
        const marker = L.marker([h.lat, h.lng])
            .addTo(state.map)
            .bindPopup(`
                <div style="color:#111; font-family:sans-serif;">
                    <strong style="font-size:1rem;">${h.name}</strong><br>
                    <span style="font-size:0.8rem; color:#0284c7;">${h.department}</span><br>
                    <span style="font-size:0.8rem;">${h.address}</span><br>
                    <strong>Rating:</strong> ⭐ ${h.rating} | <strong>Dist:</strong> ${h.distance_miles} mi<br>
                    <a href="tel:${h.phone}" style="display:inline-block; margin-top:6px; color:#0284c7; font-weight:bold;">Call ${h.phone}</a>
                </div>
            `);
        state.mapMarkers.push(marker);
    });
}

function focusHospitalLocation(lat, lng, name) {
    if (state.map) {
        state.map.setView([lat, lng], 15);
        state.mapMarkers.forEach(m => {
            const pos = m.getLatLng();
            if (Math.abs(pos.lat - lat) < 0.0001 && Math.abs(pos.lng - lng) < 0.0001) {
                m.openPopup();
            }
        });
    }
}

function filterHospitalsByDepartment(dept) {
    const select = document.getElementById('hospital-dept-filter');
    if (select) select.value = dept;
    loadHospitals(dept);
}

// ==========================================================================
// Disease Knowledge Base
// ==========================================================================
async function initKnowledgeBase() {
    try {
        const res = await fetch('/api/diseases');
        const data = await res.json();
        renderKnowledgeCards(data.diseases);

        const searchInput = document.getElementById('kb-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const term = e.target.value.toLowerCase();
                const filtered = data.diseases.filter(d =>
                    d.name.toLowerCase().includes(term) ||
                    d.category.toLowerCase().includes(term) ||
                    d.description.toLowerCase().includes(term)
                );
                renderKnowledgeCards(filtered);
            });
        }
    } catch (err) {
        console.error('Error loading knowledge base:', err);
    }
}

function renderKnowledgeCards(diseases) {
    const container = document.getElementById('knowledge-cards-grid');
    if (!container) return;

    container.innerHTML = diseases.map(d => `
        <div class="glass-card" style="padding:18px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                <span class="badge-tag">${d.category}</span>
                <span style="font-size:0.75rem; color:var(--text-muted);"><i class="fa-solid fa-book-medical"></i> 4 Precautions</span>
            </div>
            <h4 style="font-size:1.15rem; font-weight:700; color:#fff; margin-bottom:6px;">${d.name}</h4>
            <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.4; margin-bottom:12px;">
                ${d.description}
            </p>
            <div style="font-size:0.8rem; color:var(--accent-cyan); font-weight:600; margin-bottom:10px;">
                <i class="fa-solid fa-pills"></i> Medications:
                <span style="color:var(--text-primary); font-weight:normal;"> ${(d.medications || []).slice(0, 2).join(', ')}</span>
            </div>
            <button class="btn-secondary" style="width:100%; font-size:0.8rem;" onclick="openDiseaseModal('${d.name.replace(/'/g, "\\'")}')">
                <i class="fa-solid fa-file-medical"></i> View Full Protocol
            </button>
        </div>
    `).join('');
}

async function openDiseaseModal(name) {
    try {
        const res = await fetch(`/api/diseases/${encodeURIComponent(name)}`);
        const d = await res.json();
        const modal = document.getElementById('disease-detail-modal');
        const content = document.getElementById('disease-modal-content');

        content.innerHTML = `
            <div class="glass-card-header">
                <div>
                    <span class="badge-tag">${d.category}</span>
                    <h2 style="font-size:1.5rem; font-weight:800; color:#fff; margin-top:4px;">${d.name}</h2>
                </div>
            </div>
            <p style="color:#cbd5e1; margin-bottom:15px;">${d.description}</p>
            
            <h4 style="color:var(--accent-cyan); font-size:0.95rem; font-weight:700; margin-bottom:6px;">
                <i class="fa-solid fa-shield-halved"></i> Key Clinical Precautions:
            </h4>
            <ul class="bullet-list" style="margin-bottom:15px;">
                ${(d.precautions || []).map(p => `<li>${p}</li>`).join('')}
            </ul>

            <h4 style="color:var(--accent-amber); font-size:0.95rem; font-weight:700; margin-bottom:6px;">
                <i class="fa-solid fa-pills"></i> Recommended Pharmacotherapy / Medications:
            </h4>
            <ul class="bullet-list" style="margin-bottom:15px;">
                ${(d.medications || []).map(m => `<li>${m}</li>`).join('')}
            </ul>

            <h4 style="color:var(--accent-emerald); font-size:0.95rem; font-weight:700; margin-bottom:6px;">
                <i class="fa-solid fa-apple-whole"></i> Dietary Guidelines:
            </h4>
            <ul class="bullet-list">
                ${(d.diet || []).map(dt => `<li>${dt}</li>`).join('')}
            </ul>
        `;

        openModal('disease-detail-modal');
    } catch (err) {
        showToast('Error opening disease profile', 'error');
    }
}

// ==========================================================================
// Patient History & Chart Analytics
// ==========================================================================
async function loadHistoryAndStats() {
    try {
        const headers = {};
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

        // Fetch records
        const resHist = await fetch('/api/history', { headers });
        const histData = await resHist.json();

        // Fetch stats
        const resStats = await fetch('/api/history/stats', { headers });
        const statsData = await resStats.json();

        renderHistoryCounters(statsData);
        renderHistoryCharts(statsData);
        renderHistoryTable(histData.records);
    } catch (err) {
        console.error('Error loading history:', err);
    }
}

function renderHistoryCounters(stats) {
    document.getElementById('stat-total-preds').textContent = stats.total_predictions || 0;
    document.getElementById('stat-symptom-checks').textContent = stats.total_symptom_checks || 0;
    document.getElementById('stat-specialized-tests').textContent = stats.total_specialized_tests || 0;
    document.getElementById('stat-registered-users').textContent = stats.total_registered_users || 0;
}

function renderHistoryCharts(stats) {
    // 1. Top Diagnosed Diseases Bar Chart
    const ctxDiseases = document.getElementById('chart-top-diseases')?.getContext('2d');
    if (ctxDiseases) {
        if (state.charts.diseases) state.charts.diseases.destroy();
        const labels = (stats.top_diseases || []).map(d => d.disease);
        const counts = (stats.top_diseases || []).map(d => d.count);

        state.charts.diseases = new Chart(ctxDiseases, {
            type: 'bar',
            data: {
                labels: labels.length ? labels : ['Fungal Infection', 'GERD', 'Asthma', 'Diabetes', 'Migraine', 'Hypertension'],
                datasets: [{
                    label: 'Frequency',
                    data: counts.length ? counts : [14, 11, 9, 8, 6, 5],
                    backgroundColor: 'rgba(6, 182, 212, 0.7)',
                    borderColor: '#06b6d4',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }

    // 2. Severity Doughnut Chart
    const ctxSev = document.getElementById('chart-severity-distribution')?.getContext('2d');
    if (ctxSev) {
        if (state.charts.severity) state.charts.severity.destroy();
        const sevData = stats.severity_distribution || { Mild: 12, Moderate: 18, High: 8, Critical: 3 };

        state.charts.severity = new Chart(ctxSev, {
            type: 'doughnut',
            data: {
                labels: Object.keys(sevData),
                datasets: [{
                    data: Object.values(sevData),
                    backgroundColor: ['#10b981', '#6366f1', '#f59e0b', '#f43f5e'],
                    borderColor: '#111827',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#cbd5e1', boxWidth: 12 } }
                }
            }
        });
    }
}

function renderHistoryTable(records) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    if (!records || records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:30px; color:var(--text-muted);">No prediction history recorded yet. Run a diagnostic check to see logs here.</td></tr>`;
        return;
    }

    tbody.innerHTML = records.map(r => {
        const isSymptom = r.record_type === 'symptom_prediction';
        const dateStr = new Date(r.created_at).toLocaleString();
        const resultLabel = isSymptom ? r.predicted_disease : `${r.type.toUpperCase()}: ${r.result}`;
        const scoreLabel = isSymptom ? `${r.confidence}% Conf` : `${r.probability}% Risk`;
        const tagColor = (r.severity_level === 'Critical' || (r.probability && r.probability >= 60)) ? 'var(--accent-rose)' :
            (r.severity_level === 'High' || (r.probability && r.probability >= 40)) ? 'var(--accent-amber)' : 'var(--accent-emerald)';

        return `
            <tr>
                <td style="font-weight:600; color:#fff;">${dateStr}</td>
                <td><span class="badge-tag">${isSymptom ? 'General Symptoms' : `Specialized ${r.type}`}</span></td>
                <td style="font-weight:700; color:#fff;">${resultLabel}</td>
                <td style="color:${tagColor}; font-weight:700;">${scoreLabel}</td>
                <td style="font-size:0.8rem; color:var(--text-secondary);">${r.user_email || 'guest'}</td>
                <td>
                    <button class="btn-danger" style="padding:3px 8px; font-size:0.75rem;" onclick="deleteRecord(${r.id}, '${isSymptom ? 'symptom' : 'specialized'}')">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

async function deleteRecord(id, type) {
    if (!confirm('Are you sure you want to delete this diagnosis record?')) return;
    try {
        const headers = {};
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

        const res = await fetch(`/api/history/${id}?record_type=${type}`, {
            method: 'DELETE',
            headers
        });
        if (!res.ok) throw new Error('Failed to delete record');
        showToast('Record deleted', 'info');
        loadHistoryAndStats();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ==========================================================================
// Admin & ML Model Observatory
// ==========================================================================
async function initAdminObservatory() {
    // Retrain Button
    const retrainBtn = document.getElementById('admin-retrain-btn');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', async () => {
            retrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retraining Pipeline...';
            retrainBtn.disabled = true;
            try {
                const res = await fetch('/api/admin/retrain?model_type=all', { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Retraining failed');
                showToast(data.message, 'success');
                loadAdminMetrics();
            } catch (err) {
                showToast(err.message, 'error');
            } finally {
                retrainBtn.innerHTML = '<i class="fa-solid fa-rotate"></i> Retrain All Models';
                retrainBtn.disabled = false;
            }
        });
    }
}

async function loadAdminMetrics() {
    try {
        const res = await fetch('/api/admin/metrics');
        const data = await res.json();
        renderAdminMetricsView(data.models);
    } catch (err) {
        console.error('Error loading admin metrics:', err);
    }
}

function renderAdminMetricsView(models) {
    const container = document.getElementById('admin-metrics-cards-grid');
    if (!container) return;

    const cards = Object.entries(models).map(([key, meta]) => {
        const title = key.replace(/_/g, ' ').toUpperCase();
        const selected = meta.selected_model || 'Ensemble Model';
        const bestStats = meta.comparison ? meta.comparison[selected] : {};

        return `
            <div class="glass-card" style="padding:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span class="badge-tag">${title}</span>
                    <span style="font-size:0.75rem; color:var(--accent-emerald); font-weight:700;">
                        <i class="fa-solid fa-circle-check"></i> Active in Production
                    </span>
                </div>
                <h4 style="font-size:1.1rem; font-weight:700; color:#fff;">${selected}</h4>
                
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:15px 0;">
                    <div style="background:rgba(15,23,42,0.6); padding:10px; border-radius:8px;">
                        <div style="font-size:0.75rem; color:var(--text-muted);">Test Accuracy</div>
                        <div style="font-size:1.2rem; font-weight:800; color:var(--accent-cyan);">
                            ${bestStats?.test_accuracy || 100}%
                        </div>
                    </div>
                    <div style="background:rgba(15,23,42,0.6); padding:10px; border-radius:8px;">
                        <div style="font-size:0.75rem; color:var(--text-muted);">F1-Score</div>
                        <div style="font-size:1.2rem; font-weight:800; color:var(--accent-emerald);">
                            ${bestStats?.f1_score || 100}%
                        </div>
                    </div>
                </div>

                <div style="font-size:0.8rem; color:var(--text-secondary);">
                    <div><strong>Algorithms Tested:</strong> ${Object.keys(meta.comparison || {}).join(', ')}</div>
                    <div><strong>Dataset Samples:</strong> ${meta.total_training_samples || meta.samples_count || 'N/A'} samples</div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = cards;
}

// ==========================================================================
// Printable Medical Summary PDF Export Modal
// ==========================================================================
function openPrintableSummaryModal() {
    const res = state.lastDiagnosisResult;
    if (!res) {
        showToast('No diagnosis generated yet to print', 'warning');
        return;
    }

    const content = document.getElementById('printable-summary-content');
    const primary = res.primary_details || {};
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    content.innerHTML = `
        <div style="text-align:center; border-bottom:2px solid #06b6d4; padding-bottom:12px; margin-bottom:16px;">
            <h2 style="font-size:1.6rem; font-weight:800; color:#fff;">MEDICARE AI DIAGNOSTIC REPORT</h2>
            <p style="font-size:0.82rem; color:#94a3b8;">Decision-Support Summary • Generated on ${dateStr}</p>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:15px; font-size:0.88rem; background:rgba(15,23,42,0.5); padding:12px; border-radius:8px;">
            <div><strong>Patient Name:</strong> ${state.user ? state.user.name : 'Outpatient (Guest)'}</div>
            <div><strong>Report ID:</strong> #MED-${res.record_id || Date.now().toString().slice(-6)}</div>
            <div><strong>Primary Diagnosis:</strong> <span style="color:#38bdf8; font-weight:700;">${res.primary_disease}</span></div>
            <div><strong>Model Confidence:</strong> ${res.confidence}%</div>
            <div><strong>Triage Level:</strong> ${res.severity_level}</div>
            <div><strong>Clinical Urgency:</strong> ${res.triage_urgency}</div>
        </div>

        <div style="margin-bottom:12px;">
            <strong style="color:#38bdf8;">Reported Symptom Markers:</strong>
            <p style="font-size:0.85rem; color:#cbd5e1; margin-top:2px;">${(res.matched_symptoms || []).map(s => s.replace(/_/g, ' ').toUpperCase()).join(', ')}</p>
        </div>

        <div style="margin-bottom:12px;">
            <strong style="color:#38bdf8;">Condition Overview:</strong>
            <p style="font-size:0.85rem; color:#cbd5e1; margin-top:2px;">${primary.description || ''}</p>
        </div>

        <div style="margin-bottom:12px;">
            <strong style="color:#34d399;">Prescribed Precautions:</strong>
            <ul style="padding-left:20px; font-size:0.85rem; color:#cbd5e1; margin-top:2px;">
                ${(primary.precautions || []).map(p => `<li>${p}</li>`).join('')}
            </ul>
        </div>

        <div style="margin-bottom:12px;">
            <strong style="color:#fbbf24;">Medication Class / Regimen:</strong>
            <ul style="padding-left:20px; font-size:0.85rem; color:#cbd5e1; margin-top:2px;">
                ${(primary.medications || []).map(m => `<li>${m}</li>`).join('')}
            </ul>
        </div>

        <div style="margin-top:20px; padding:10px; border:1px solid rgba(255,255,255,0.1); border-radius:6px; font-size:0.75rem; color:#94a3b8; text-align:center;">
            <strong>CLINICAL DISCLAIMER:</strong> This report is generated by an automated Machine Learning decision-support system intended for portfolio and educational demonstration. It does not replace clinical judgment or certified medical diagnosis.
        </div>
    `;

    openModal('printable-summary-modal');
}

function printSummary() {
    window.print();
}

// ==========================================================================
// Modal Helpers
// ==========================================================================
function openModal(modalId) {
    const el = document.getElementById(modalId);
    if (el) el.classList.add('active');
}
function closeModal(modalId) {
    const el = document.getElementById(modalId);
    if (el) el.classList.remove('active');
}
