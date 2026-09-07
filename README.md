# MediCare AI — Human Disease Diagnosis System

A full-stack, Machine Learning-powered clinical decision support platform built with **Python (FastAPI)**, **scikit-learn**, **SQLite**, and a modern **Vanilla JS/CSS3** responsive interface.

---

## 🌟 Key Features

1. **Multi-Class Symptom-Based Disease Predictor**:
   - Classifies 42 human diseases based on 132 standardized clinical symptoms.
   - Evaluates multi-algorithm ensembles (Random Forest, Naive Bayes, Decision Tree, Logistic Regression, SVM).
   - Returns top-k ranked predictions with confidence percentages and urgency levels.
   - Clinical Decision Support details: disease overview, 4-step precautions, medication classes, tailored nutrition, and exercise routines.

2. **Specialized Diagnostic Predictive Labs**:
   - **Diabetes Mellitus Predictor**: Pima Indians clinical features (Glucose, BMI, Insulin, Blood Pressure, DPF, etc.).
   - **Cardiovascular Disease Predictor**: UCI Cleveland clinical features (Resting BP, Cholesterol, ECG, Max HR, ST depression, etc.).
   - **Parkinson's Disease Voice Biomarker Analyzer**: UCI biomedical vocal features (Fundamental frequency, Jitter, Shimmer, HNR, RPDE, PPE).

3. **Intelligent Conversational Medical Assistant**:
   - Natural language symptom extraction directly from patient chat messages.
   - Identifies emergency red-flags (chest pain, stroke signs, acute respiratory distress) with instant triage notices.
   - 1-click execution of differential diagnoses directly from chat conversations.

4. **Doctor & Hospital Locator**:
   - Interactive Leaflet.js mapping with OpenStreetMap tiles.
   - Specialty and emergency 24/7 filters with real-time distance calculations.

5. **Patient Health Dashboard & Analytics**:
   - Interactive Chart.js charts: top condition frequencies, triage severity distribution, and specialized risk assessments.
   - Comprehensive consultation history logs with PDF/Printable Medical Summary generation.

6. **Admin ML Model Observatory**:
   - Real-time model evaluation metrics (accuracy, precision, recall, F1, cross-validation).
   - One-click online model retraining and hot-reloading.

---

## 🏗️ Clean Project Architecture

```
Project/
├── frontend/
│   ├── templates/
│   │   └── index.html                # Single page responsive clinical interface
│   └── static/
│       ├── css/
│       │   └── main.css              # Glassmorphism dark mode healthcare stylesheet
│       ├── js/
│       │   └── app.js                # Frontend client controller & Chart.js/Leaflet integration
│       └── images/                   # UI asset icons & imagery
│
├── backend/
│   ├── main.py                       # FastAPI entry point, static & template mounts, router aggregation
│   ├── api/                          # REST API endpoint routers
│   │   ├── admin.py                  # ML observatory & retraining (/api/admin/*)
│   │   ├── auth.py                   # User registration, JWT login & profiles (/api/auth/*)
│   │   ├── chat.py                   # NLP symptom extraction chatbot (/api/chat)
│   │   ├── diseases.py               # Disease & symptom catalogue (/api/diseases, /api/symptoms)
│   │   ├── history.py                # Consultation logs & analytics stats (/api/history/*)
│   │   ├── locator.py                # Hospital & specialist clinic finder (/api/hospitals)
│   │   └── predict.py                # Symptom & specialized risk predictors (/api/predict/*)
│   ├── database/
│   │   ├── __init__.py               # Database layer package exports
│   │   └── database.py               # SQLite connection, schema tables, and data seeders
│   └── services/
│       ├── __init__.py               # Service layer package exports
│       └── auth_service.py           # JWT security, bcrypt password hashing, role guards
│
├── machine_learning/
│   ├── __init__.py                   # ML package exports
│   ├── data_loader.py                # Clinical dataset synthesis and preprocessing pipeline
│   ├── predictor.py                  # ClinicalPredictorEngine singleton inference & lookup
│   ├── train_symptom_model.py        # 42-disease multi-class classifier training
│   ├── train_diabetes_model.py       # Specialized diabetes risk classifier training
│   ├── train_heart_model.py          # Specialized heart disease classifier training
│   └── train_parkinsons_model.py     # Specialized Parkinson's voice classifier training
│
├── ml_models/                        # Serialized .joblib model artifacts & evaluation metrics
│   ├── diabetes_metrics.json
│   ├── diabetes_model.joblib
│   ├── heart_metrics.json
│   ├── heart_model.joblib
│   ├── parkinsons_metrics.json
│   ├── parkinsons_model.joblib
│   ├── symptom_metrics.json
│   └── symptom_model.joblib
│
├── data/
│   ├── diagnosis_system.db           # SQLite database
│   ├── raw/                          # 10 clinical reference CSV datasets
│   └── processed/
│
├── tests/
│   └── test_api.py                   # Automated pytest suite (10 unit & integration tests)
│
├── main.py                           # Root launcher delegating to backend.main:app
├── requirements.txt                  # Python dependencies
└── README.md                         # Presentation & technical documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets & Train Models (Pre-trained models included)
```bash
python -m machine_learning.data_loader
python -m machine_learning.train_symptom_model
python -m machine_learning.train_diabetes_model
python -m machine_learning.train_heart_model
python -m machine_learning.train_parkinsons_model
```

### 3. Run the Server
You can start the server using either:
```bash
python main.py
```
or
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 4. Interactive API Documentation
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5. Run Automated Test Suite
```bash
pytest
```
or
```bash
python -m pytest tests/ -v
```

---

## 🧪 Demo Accounts
- **Patient**: `patient@medicare.io` / `Patient@123`
- **Doctor**: `doctor@medicare.io` / `Doctor@123`
- **Admin**: `admin@medicare.io` / `Admin@123`
*(Or click the 1-Click Demo Login buttons in the web interface)*

---

## ⚖️ Clinical Disclaimer
This system is an automated Machine Learning decision-support platform designed for educational and portfolio reference. It is not a substitute for clinical diagnosis, professional medical advice, or emergency care.
