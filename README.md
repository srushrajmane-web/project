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

## 🏗️ Project Architecture

```
diagnosis-system/
├── main.py                     # FastAPI server entry point and route aggregator
├── config.py                   # Configuration and environment variables
├── database.py                 # SQLite database initialization, schema, and seed data
├── auth.py                     # JWT token issuance, bcrypt hashing, role guards
├── requirements.txt            # Python dependencies
├── ml/
│   ├── data_loader.py          # Data generation and preprocessing pipeline
│   ├── predictor.py            # Unified inference engine and clinical metadata lookup
│   ├── train_symptom_model.py  # 42-disease multi-class classifier training
│   ├── train_diabetes_model.py # Diabetes classifier training
│   ├── train_heart_model.py    # Heart disease classifier training
│   └── train_parkinsons_model.py# Parkinson's classifier training
├── routes/
│   ├── predict.py              # Prediction endpoints (/api/predict/*)
│   ├── diseases.py             # Disease & symptom catalogue (/api/diseases, /api/symptoms)
│   ├── auth.py                 # Authentication endpoints (/api/auth/*)
│   ├── history.py              # User history & dashboard stats (/api/history/*)
│   ├── chat.py                 # AI conversational triage assistant (/api/chat)
│   ├── locator.py              # Healthcare provider locator (/api/hospitals)
│   └── admin.py                # Model metrics and retraining (/api/admin/*)
├── models/                     # Serialized .joblib model artifacts & evaluation metrics
├── data/
│   └── raw/                    # Clinical CSV datasets and metadata catalogues
├── static/
│   ├── css/main.css            # Dark slate healthcare glassmorphism stylesheet
│   └── js/app.js               # Client-side SPA controller
├── templates/
│   └── index.html              # Modern single page application layout
└── tests/
    └── test_api.py             # Automated pytest suite
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets & Train Models (Pre-trained models included)
```bash
python ml/data_loader.py
python ml/train_symptom_model.py
python ml/train_diabetes_model.py
python ml/train_heart_model.py
python ml/train_parkinsons_model.py
```

### 3. Run the Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 4. Interactive API Documentation
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5. Run Automated Test Suite
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
