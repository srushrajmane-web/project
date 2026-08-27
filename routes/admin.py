"""
Admin and Model Observatory Endpoints.
Provides model evaluation metrics, dataset inspection, and retraining controls.
"""

import os
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from database import get_db
from auth import get_current_user_optional, require_role
from ml.predictor import predictor_engine

router = APIRouter(prefix="/api/admin", tags=["Admin & ML Observatory"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@router.get("/metrics")
async def get_model_metrics():
    """Returns serialized performance metrics, cross-validation scores, and comparison tables for all ML models."""
    metrics_data = {
        "models": {},
        "system_status": "Operational",
        "loaded_models_count": 0
    }
    
    files = {
        "symptom_model": "symptom_metrics.json",
        "diabetes_model": "diabetes_metrics.json",
        "heart_model": "heart_metrics.json",
        "parkinsons_model": "parkinsons_metrics.json"
    }
    
    loaded = 0
    for key, filename in files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                metrics_data["models"][key] = json.load(f)
                loaded += 1
        else:
            metrics_data["models"][key] = {"status": "Model metrics file pending"}
            
    metrics_data["loaded_models_count"] = loaded
    return metrics_data

@router.post("/retrain")
async def trigger_model_retrain(
    model_type: str = "all"
):
    """
    Triggers retraining of models and hot-reloads model instances in memory.
    """
    try:
        from ml.train_symptom_model import train_symptom_models
        from ml.train_diabetes_model import train_diabetes_model
        from ml.train_heart_model import train_heart_model
        from ml.train_parkinsons_model import train_parkinsons_model
        
        results = {}
        if model_type in ["all", "symptom"]:
            results["symptom"] = train_symptom_models()
        if model_type in ["all", "diabetes"]:
            results["diabetes"] = train_diabetes_model()
        if model_type in ["all", "heart"]:
            results["heart"] = train_heart_model()
        if model_type in ["all", "parkinsons"]:
            results["parkinsons"] = train_parkinsons_model()
            
        # Reload models in predictor engine
        predictor_engine.load_all_models()
        
        return {
            "status": "success",
            "message": f"Successfully retrained and reloaded models for: {model_type}",
            "details": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")
