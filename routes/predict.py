"""
Prediction Routes for General Symptoms and Specialized Clinical Diagnostics.
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, status

from ml.predictor import predictor_engine
from auth import get_current_user_optional
from database import get_db

router = APIRouter(prefix="/api/predict", tags=["Predictions"])

# Pydantic Input Schemas
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1, description="List of patient-reported symptoms")
    notes: Optional[str] = Field(None, description="Optional patient notes or duration")

class DiabetesPredictionRequest(BaseModel):
    Pregnancies: float = Field(0, ge=0, le=25)
    Glucose: float = Field(120, ge=40, le=350)
    BloodPressure: float = Field(70, ge=30, le=200)
    SkinThickness: float = Field(20, ge=0, le=100)
    Insulin: float = Field(79, ge=0, le=900)
    BMI: float = Field(25.0, ge=10.0, le=70.0)
    DiabetesPedigreeFunction: float = Field(0.47, ge=0.01, le=3.0)
    Age: float = Field(30, ge=1, le=120)

class HeartPredictionRequest(BaseModel):
    age: float = Field(50, ge=18, le=100)
    sex: float = Field(1, ge=0, le=1, description="1 = Male, 0 = Female")
    cp: float = Field(0, ge=0, le=3, description="Chest pain type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
    trestbps: float = Field(120, ge=80, le=250, description="Resting blood pressure mm Hg")
    chol: float = Field(200, ge=100, le=600, description="Serum cholesterol mg/dl")
    fbs: float = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: float = Field(0, ge=0, le=2, description="Resting ECG results: 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy")
    thalach: float = Field(150, ge=60, le=230, description="Maximum heart rate achieved")
    exang: float = Field(0, ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(0.0, ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: float = Field(1, ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: float = Field(0, ge=0, le=4, description="Number of major vessels colored by flourosopy")
    thal: float = Field(2, ge=1, le=3, description="1=Normal, 2=Fixed defect, 3=Reversible defect")

class ParkinsonsPredictionRequest(BaseModel):
    model_config = {"populate_by_name": True}
    fo: float = Field(154.0, ge=50.0, le=350.0, alias="MDVP:Fo(Hz)", description="Average vocal fundamental frequency")
    fhi: float = Field(197.0, ge=70.0, le=600.0, alias="MDVP:Fhi(Hz)", description="Maximum vocal fundamental frequency")
    flo: float = Field(116.0, ge=50.0, le=300.0, alias="MDVP:Flo(Hz)", description="Minimum vocal fundamental frequency")
    jitter_pct: float = Field(0.006, ge=0.0, le=0.1, alias="MDVP:Jitter(%)", description="MDVP jitter in percentage")
    shimmer: float = Field(0.029, ge=0.0, le=0.3, alias="MDVP:Shimmer", description="MDVP local shimmer")
    hnr: float = Field(21.8, ge=0.0, le=50.0, alias="HNR", description="Harmonics-to-Noise Ratio")
    rpde: float = Field(0.49, ge=0.0, le=1.0, alias="RPDE", description="Recurrence period density entropy")
    dfa: float = Field(0.71, ge=0.0, le=1.5, alias="DFA", description="Detrended fluctuation analysis")
    spread1: float = Field(-5.6, ge=-10.0, le=0.0, alias="spread1", description="Nonlinear measure of fundamental frequency variation")
    spread2: float = Field(0.22, ge=0.0, le=1.0, alias="spread2", description="Nonlinear frequency variation")
    ppe: float = Field(0.20, ge=0.0, le=1.0, alias="PPE", description="Pitch period entropy")

@router.post("/symptoms")
async def predict_symptoms_endpoint(
    req: SymptomPredictionRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Predicts disease from reported symptoms and logs prediction to history."""
    try:
        result = predictor_engine.predict_symptoms(req.symptoms)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        # Log to Database
        conn = get_db()
        cursor = conn.cursor()
        user_id = current_user.get("id") if current_user else None
        user_email = current_user.get("email") if current_user else "guest@medicare.io"
        
        cursor.execute("""
        INSERT INTO predictions (user_id, user_email, input_symptoms, predicted_disease, confidence, top_predictions, severity_level, triage_urgency, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_email,
            json.dumps(result["matched_symptoms"]),
            result["primary_disease"],
            result["confidence"],
            json.dumps(result["top_predictions"]),
            result["severity_level"],
            result["triage_urgency"],
            req.notes
        ))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        result["record_id"] = prediction_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/diabetes")
async def predict_diabetes_endpoint(
    req: DiabetesPredictionRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Predicts diabetes risk from clinical parameters."""
    try:
        data = req.model_dump()
        result = predictor_engine.predict_diabetes(data)
        
        # Save specialized report
        conn = get_db()
        cursor = conn.cursor()
        user_id = current_user.get("id") if current_user else None
        user_email = current_user.get("email") if current_user else "guest@medicare.io"
        
        cursor.execute("""
        INSERT INTO specialized_reports (user_id, user_email, type, input_values, result, probability, risk_score, insights)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_email,
            "diabetes",
            json.dumps(data),
            result["result"],
            result["probability"],
            result["risk_tier"],
            json.dumps(result["recommendations"])
        ))
        conn.commit()
        result["record_id"] = cursor.lastrowid
        conn.close()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diabetes prediction error: {str(e)}")

@router.post("/heart")
async def predict_heart_endpoint(
    req: HeartPredictionRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Predicts heart disease risk from cardiovascular test values."""
    try:
        data = req.model_dump()
        result = predictor_engine.predict_heart_disease(data)
        
        conn = get_db()
        cursor = conn.cursor()
        user_id = current_user.get("id") if current_user else None
        user_email = current_user.get("email") if current_user else "guest@medicare.io"
        
        cursor.execute("""
        INSERT INTO specialized_reports (user_id, user_email, type, input_values, result, probability, risk_score, insights)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_email,
            "heart",
            json.dumps(data),
            result["result"],
            result["probability"],
            result["risk_tier"],
            json.dumps(result["recommendations"])
        ))
        conn.commit()
        result["record_id"] = cursor.lastrowid
        conn.close()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heart disease prediction error: {str(e)}")

@router.post("/parkinsons")
async def predict_parkinsons_endpoint(
    req: ParkinsonsPredictionRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Predicts Parkinson's risk from acoustic voice measurements."""
    try:
        data = req.model_dump(by_alias=True)
        result = predictor_engine.predict_parkinsons(data)
        
        conn = get_db()
        cursor = conn.cursor()
        user_id = current_user.get("id") if current_user else None
        user_email = current_user.get("email") if current_user else "guest@medicare.io"
        
        cursor.execute("""
        INSERT INTO specialized_reports (user_id, user_email, type, input_values, result, probability, risk_score, insights)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_email,
            "parkinsons",
            json.dumps(data),
            result["result"],
            result["probability"],
            result["risk_tier"],
            json.dumps(result["recommendations"])
        ))
        conn.commit()
        result["record_id"] = cursor.lastrowid
        conn.close()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parkinson's prediction error: {str(e)}")
