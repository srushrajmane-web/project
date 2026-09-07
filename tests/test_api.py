"""
Automated Test Suite for Human Disease Diagnosis System REST API.
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.database import init_db

init_db()
client = TestClient(app)

def test_startup_and_health():
    """Verifies server health and database seeding."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["engine_status"] == "ready"

def test_symptom_prediction():
    """Tests multi-class symptom disease prediction with top-k ranking and precautions."""
    payload = {
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache"],
        "notes": "Patient reports fever spike with night chills"
    }
    res = client.post("/api/predict/symptoms", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "primary_disease" in data
    assert data["confidence"] > 0
    assert len(data["top_predictions"]) > 0
    assert "triage_urgency" in data
    assert "severity_level" in data
    assert "primary_details" in data
    assert len(data["primary_details"]["precautions"]) > 0

def test_specialized_diabetes_prediction():
    """Tests specialized Pima Indians diabetes classifier."""
    payload = {
        "Glucose": 180,
        "BMI": 35.5,
        "Age": 55,
        "BloodPressure": 85,
        "Insulin": 200,
        "Pregnancies": 4,
        "SkinThickness": 32,
        "DiabetesPedigreeFunction": 0.85
    }
    res = client.post("/api/predict/diabetes", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["type"] == "diabetes"
    assert "probability" in data
    assert len(data["recommendations"]) > 0

def test_specialized_heart_prediction():
    """Tests specialized UCI Cleveland heart disease predictor."""
    payload = {
        "age": 62,
        "sex": 1,
        "cp": 0,
        "trestbps": 150,
        "chol": 280,
        "fbs": 1,
        "restecg": 1,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 0,
        "ca": 2,
        "thal": 3
    }
    res = client.post("/api/predict/heart", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["type"] == "heart"
    assert "probability" in data
    assert len(data["recommendations"]) > 0

def test_specialized_parkinsons_prediction():
    """Tests specialized Parkinson's vocal biomarker analyzer."""
    payload = {
        "MDVP:Fo(Hz)": 116.0,
        "MDVP:Fhi(Hz)": 137.0,
        "MDVP:Flo(Hz)": 86.0,
        "MDVP:Jitter(%)": 0.018,
        "MDVP:Shimmer": 0.075,
        "HNR": 12.4,
        "RPDE": 0.64,
        "DFA": 0.81,
        "spread1": -3.8,
        "spread2": 0.38,
        "PPE": 0.42
    }
    res = client.post("/api/predict/parkinsons", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["type"] == "parkinsons"
    assert "probability" in data

def test_diseases_and_symptoms_endpoints():
    """Tests listing all 42 diseases and 132 symptoms."""
    res_dis = client.get("/api/diseases")
    assert res_dis.status_code == 200
    data_dis = res_dis.json()
    assert data_dis["total"] >= 40
    
    # Detail endpoint
    first_disease = data_dis["diseases"][0]["name"]
    res_detail = client.get(f"/api/diseases/{first_disease}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["name"] == first_disease
    assert len(detail["precautions"]) > 0

    # Symptoms endpoint
    res_sym = client.get("/api/symptoms")
    assert res_sym.status_code == 200
    data_sym = res_sym.json()
    assert data_sym["total"] >= 100
    assert len(data_sym["categories"]) >= 4

def test_chat_assistant():
    """Tests conversational AI assistant NLP symptom extraction and response."""
    payload = {
        "message": "I have high fever and severe shivering and body aches",
        "history": []
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["extracted_symptoms"]) > 0

def test_hospital_locator():
    """Tests hospital locator and specialty filtering."""
    res = client.get("/api/hospitals")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 0
    assert "distance_miles" in data["hospitals"][0]

def test_auth_and_history():
    """Tests demo login and history retrieval."""
    # Demo login as patient
    res_login = client.post("/api/auth/demo", json={"role": "patient"})
    assert res_login.status_code == 200
    auth_data = res_login.json()
    token = auth_data["access_token"]
    assert token is not None

    headers = {"Authorization": f"Bearer {token}"}
    res_me = client.get("/api/auth/me", headers=headers)
    assert res_me.status_code == 200
    assert res_me.json()["role"] == "patient"

    # History stats
    res_stats = client.get("/api/history/stats", headers=headers)
    assert res_stats.status_code == 200
    assert "total_predictions" in res_stats.json()

def test_admin_metrics():
    """Tests admin model observatory metrics."""
    res = client.get("/api/admin/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["loaded_models_count"] >= 4
    assert "symptom_model" in data["models"]
