"""
Unified Prediction & Clinical Decision Support Engine.
Loads serialized machine learning models from ml_models/ and performs:
- Top-K Multi-class General Symptom Prediction
- Emergency Red-Flag Triage Detection
- Severity & Urgency Index Calculation
- Specialized Diabetes, Heart Disease, and Parkinson's Risk Inference
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Red-flag emergency symptoms that warrant immediate hospital triage
EMERGENCY_SYMPTOMS = {
    "weakness_of_one_body_side": "Signs of Acute Stroke / Cerebrovascular Event",
    "altered_sensorium": "Signs of Altered Mental State / Neurological Emergency",
    "coma": "Critical Comatose State",
    "chest_pain": "Severe Cardiac / Anginal Risk",
    "acute_liver_failure": "Signs of Fulminant Hepatic Failure",
    "blood_in_sputum": "Severe Pulmonary Hemorrhage Indicator",
    "stomach_bleeding": "Acute Upper Gastrointestinal Bleed",
    "breathlessness": "Acute Respiratory Distress"
}

class ClinicalPredictorEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClinicalPredictorEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.symptom_bundle = None
        self.diabetes_bundle = None
        self.heart_bundle = None
        self.parkinsons_bundle = None
        
        self.severity_map = {}
        self.disease_details = {}
        
        self.load_all_models()
        self.load_reference_data()
        self._initialized = True
        
    def load_all_models(self):
        """Loads all serialized joblib model artifacts from ml_models/."""
        symptom_path = os.path.join(MODELS_DIR, "symptom_model.joblib")
        diabetes_path = os.path.join(MODELS_DIR, "diabetes_model.joblib")
        heart_path = os.path.join(MODELS_DIR, "heart_model.joblib")
        parkinsons_path = os.path.join(MODELS_DIR, "parkinsons_model.joblib")
        
        if os.path.exists(symptom_path):
            self.symptom_bundle = joblib.load(symptom_path)
            print(f"[PredictorEngine] Loaded Symptom Model: {self.symptom_bundle.get('model_name')}")
            
        if os.path.exists(diabetes_path):
            self.diabetes_bundle = joblib.load(diabetes_path)
            print(f"[PredictorEngine] Loaded Diabetes Model: {self.diabetes_bundle.get('model_name')}")
            
        if os.path.exists(heart_path):
            self.heart_bundle = joblib.load(heart_path)
            print(f"[PredictorEngine] Loaded Heart Model: {self.heart_bundle.get('model_name')}")
            
        if os.path.exists(parkinsons_path):
            self.parkinsons_bundle = joblib.load(parkinsons_path)
            print(f"[PredictorEngine] Loaded Parkinson's Model: {self.parkinsons_bundle.get('model_name')}")

    def load_reference_data(self):
        """Loads severity dictionary and disease descriptions/precautions."""
        sev_file = os.path.join(RAW_DATA_DIR, "symptom_severity.csv")
        if os.path.exists(sev_file):
            df_sev = pd.read_csv(sev_file)
            self.severity_map = {str(r["Symptom"]).strip(): int(r["weight"]) for _, r in df_sev.iterrows()}
            
        desc_file = os.path.join(RAW_DATA_DIR, "symptom_Description.csv")
        prec_file = os.path.join(RAW_DATA_DIR, "symptom_precaution.csv")
        med_file = os.path.join(RAW_DATA_DIR, "disease_medications.csv")
        life_file = os.path.join(RAW_DATA_DIR, "disease_lifestyle.csv")
        
        if os.path.exists(desc_file):
            df_desc = pd.read_csv(desc_file)
            df_prec = pd.read_csv(prec_file) if os.path.exists(prec_file) else None
            df_med = pd.read_csv(med_file) if os.path.exists(med_file) else None
            df_life = pd.read_csv(life_file) if os.path.exists(life_file) else None
            
            for _, r in df_desc.iterrows():
                dis = str(r["Disease"]).strip()
                desc = str(r["Description"]).strip()
                
                precautions = []
                if df_prec is not None:
                    p_match = df_prec[df_prec["Disease"].str.strip() == dis]
                    if not p_match.empty:
                        p_row = p_match.iloc[0]
                        precautions = [p_row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(p_row.get(f"Precaution_{i}")) and str(p_row.get(f"Precaution_{i}")).strip()]
                        
                medications = ["Doctor-prescribed therapeutic treatment", "Supportive hydration and rest"]
                if df_med is not None:
                    m_match = df_med[df_med["Disease"].str.strip() == dis]
                    if not m_match.empty and pd.notna(m_match.iloc[0]["Medication"]):
                        medications = json.loads(m_match.iloc[0]["Medication"])
                        
                diet = ["High-nutrient whole foods", "Electrolyte fluids", "Fresh vegetables"]
                workout = ["Gentle restorative walking", "Adequate rest"]
                category = "General Medicine"
                
                if df_life is not None:
                    l_match = df_life[df_life["Disease"].str.strip() == dis]
                    if not l_match.empty:
                        if pd.notna(l_match.iloc[0]["Diet"]):
                            diet = json.loads(l_match.iloc[0]["Diet"])
                        if pd.notna(l_match.iloc[0]["Workout"]):
                            workout = json.loads(l_match.iloc[0]["Workout"])
                        if pd.notna(l_match.iloc[0]["Category"]):
                            category = str(l_match.iloc[0]["Category"])
                            
                self.disease_details[dis] = {
                    "disease": dis,
                    "category": category,
                    "description": desc,
                    "precautions": precautions,
                    "medications": medications,
                    "diet": diet,
                    "workout": workout
                }

    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Returns comprehensive info pack for a disease."""
        # Try exact or case-insensitive match
        for k, v in self.disease_details.items():
            if k.strip().lower() == disease_name.strip().lower():
                return v
        return {
            "disease": disease_name,
            "category": "General Medicine",
            "description": f"Condition characterized by clinically reported symptoms related to {disease_name}.",
            "precautions": ["Consult a certified medical professional", "Monitor symptom progression", "Maintain hydration", "Avoid self-medication"],
            "medications": ["Consult physician for diagnosis and prescription"],
            "diet": ["Nutritious balanced diet", "Adequate water intake"],
            "workout": ["Light walking", "Adequate rest"]
        }

    def predict_symptoms(self, symptoms: List[str], top_k: int = 4) -> Dict[str, Any]:
        """
        Runs multi-class classification on the provided symptoms list.
        Returns top-k predicted diseases, confidence levels, emergency alerts, and triage summary.
        """
        if not self.symptom_bundle:
            raise RuntimeError("Symptom model is not initialized or trained yet.")
            
        model = self.symptom_bundle["model"]
        encoder = self.symptom_bundle["encoder"]
        feature_names = self.symptom_bundle["feature_names"]
        
        # Clean and match symptoms
        normalized_inputs = [s.strip().lower().replace(" ", "_") for s in symptoms if s.strip()]
        
        # Check emergency flags
        emergency_flags = []
        for s in normalized_inputs:
            if s in EMERGENCY_SYMPTOMS:
                emergency_flags.append({"symptom": s.replace("_", " ").title(), "warning": EMERGENCY_SYMPTOMS[s]})
                
        # Build binary feature vector
        vector = np.zeros((1, len(feature_names)), dtype=int)
        matched_symptoms = []
        
        for s in normalized_inputs:
            if s in feature_names:
                idx = feature_names.index(s)
                vector[0, idx] = 1
                matched_symptoms.append(s)
            else:
                # Fuzzy fallback matching
                for i, fn in enumerate(feature_names):
                    if s in fn or fn in s:
                        vector[0, i] = 1
                        matched_symptoms.append(fn)
                        break
                        
        if len(matched_symptoms) == 0:
            return {
                "error": "No matching symptoms recognized from database catalogue.",
                "matched_symptoms": [],
                "top_predictions": []
            }
            
        # Calculate severity and triage index
        total_sev = sum(self.severity_map.get(s, 3) for s in matched_symptoms)
        avg_sev = total_sev / len(matched_symptoms)
        
        if len(emergency_flags) > 0:
            urgency = "CRITICAL EMERGENCY - IMMEDIATE MEDICAL ATTENTION"
            severity_level = "Critical"
        elif total_sev >= 24 or avg_sev >= 5.5:
            urgency = "High Urgency - Schedule Physician Consultation Within 24h"
            severity_level = "High"
        elif total_sev >= 12 or avg_sev >= 3.5:
            urgency = "Moderate Urgency - Monitor Symptoms and Consult Clinic"
            severity_level = "Moderate"
        else:
            urgency = "Low Urgency - Self-care and Routine Observation"
            severity_level = "Mild"

        # Model Inference
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            top_indices = np.argsort(probs)[::-1][:top_k]
            
            top_predictions = []
            for idx in top_indices:
                prob = float(probs[idx])
                if prob > 0.001:  # Filter near-zero probabilities
                    dis_name = str(encoder.inverse_transform([idx])[0])
                    details = self.get_disease_info(dis_name)
                    top_predictions.append({
                        "disease": dis_name,
                        "confidence": round(prob * 100, 2),
                        "category": details.get("category", "General Medicine"),
                        "description": details.get("description", ""),
                        "precautions": details.get("precautions", []),
                        "medications": details.get("medications", []),
                        "diet": details.get("diet", []),
                        "workout": details.get("workout", [])
                    })
        else:
            pred_idx = model.predict(vector)[0]
            dis_name = str(encoder.inverse_transform([pred_idx])[0])
            details = self.get_disease_info(dis_name)
            top_predictions = [{
                "disease": dis_name,
                "confidence": 98.5,
                "category": details.get("category", "General Medicine"),
                "description": details.get("description", ""),
                "precautions": details.get("precautions", []),
                "medications": details.get("medications", []),
                "diet": details.get("diet", []),
                "workout": details.get("workout", [])
            }]
            
        primary = top_predictions[0] if top_predictions else None
        
        return {
            "status": "success",
            "primary_disease": primary["disease"] if primary else "Inconclusive",
            "confidence": primary["confidence"] if primary else 0.0,
            "matched_symptoms": matched_symptoms,
            "symptom_count": len(matched_symptoms),
            "severity_score": total_sev,
            "average_severity": round(avg_sev, 2),
            "severity_level": severity_level,
            "triage_urgency": urgency,
            "emergency_flags": emergency_flags,
            "top_predictions": top_predictions,
            "primary_details": primary
        }

    def predict_diabetes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts diabetes risk using clinical features."""
        if not self.diabetes_bundle:
            raise RuntimeError("Diabetes model not initialized.")
            
        model = self.diabetes_bundle["model"]
        scaler = self.diabetes_bundle["scaler"]
        feature_names = self.diabetes_bundle["features"]
        
        # Extract inputs in proper order with sensible defaults
        vals = [
            float(data.get("Pregnancies", 0)),
            float(data.get("Glucose", 110)),
            float(data.get("BloodPressure", 70)),
            float(data.get("SkinThickness", 20)),
            float(data.get("Insulin", 79)),
            float(data.get("BMI", 25.0)),
            float(data.get("DiabetesPedigreeFunction", 0.35)),
            float(data.get("Age", 30))
        ]
        
        X = np.array([vals])
        X_scaled = scaler.transform(X)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            prob_diabetic = float(probs[1])
        else:
            pred = model.predict(X_scaled)[0]
            prob_diabetic = 0.85 if pred == 1 else 0.15
            
        percentage = round(prob_diabetic * 100, 2)
        
        if percentage >= 65:
            risk_tier = "High Risk (Indicative of Diabetes / Hyperglycemia)"
            result_label = "Diabetic Indicators Present"
            recommendations = [
                "Schedule a fasting plasma glucose and HbA1c blood test promptly.",
                "Consult an endocrinologist for clinical confirmation.",
                "Adopt a structured low-glycemic, high-fiber Mediterranean diet.",
                "Engage in at least 150 minutes of moderate aerobic exercise per week."
            ]
        elif percentage >= 38:
            risk_tier = "Moderate / Pre-Diabetic Risk"
            result_label = "Pre-Diabetes Borderline"
            recommendations = [
                "Annual blood glucose and HbA1c screening is recommended.",
                "Reduce intake of refined carbohydrates, sodas, and processed sugars.",
                "Target a 5-7% reduction in body weight if BMI is elevated.",
                "Increase daily walking activity to 8,000+ steps."
            ]
        else:
            risk_tier = "Low / Normal Risk"
            result_label = "Non-Diabetic Range"
            recommendations = [
                "Maintain healthy balanced eating habits and regular hydration.",
                "Continue routine physical fitness.",
                "Follow periodic routine wellness checkups."
            ]
            
        return {
            "status": "success",
            "type": "diabetes",
            "result": result_label,
            "probability": percentage,
            "risk_tier": risk_tier,
            "input_metrics": {name: val for name, val in zip(feature_names, vals)},
            "recommendations": recommendations
        }

    def predict_heart_disease(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts cardiovascular disease risk using Cleveland clinical features."""
        if not self.heart_bundle:
            raise RuntimeError("Heart model not initialized.")
            
        model = self.heart_bundle["model"]
        scaler = self.heart_bundle["scaler"]
        feature_names = self.heart_bundle["features"]
        
        vals = [
            float(data.get("age", 50)),
            float(data.get("sex", 1)),
            float(data.get("cp", 0)),
            float(data.get("trestbps", 120)),
            float(data.get("chol", 200)),
            float(data.get("fbs", 0)),
            float(data.get("restecg", 0)),
            float(data.get("thalach", 150)),
            float(data.get("exang", 0)),
            float(data.get("oldpeak", 0.0)),
            float(data.get("slope", 1)),
            float(data.get("ca", 0)),
            float(data.get("thal", 2))
        ]
        
        X = np.array([vals])
        X_scaled = scaler.transform(X)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            prob_disease = float(probs[1])
        else:
            pred = model.predict(X_scaled)[0]
            prob_disease = 0.85 if pred == 1 else 0.15
            
        percentage = round(prob_disease * 100, 2)
        
        if percentage >= 65:
            risk_tier = "High Cardiovascular Risk"
            result_label = "Elevated Heart Disease Indicators"
            recommendations = [
                "Urgent consultation with a cardiologist for ECG, Echocardiogram, and TMT.",
                "Adhere to a strict low-sodium, heart-healthy DASH or Mediterranean diet.",
                "Monitor daily resting blood pressure and lipid profile.",
                "Avoid sudden extreme physical exertion without medical clearance."
            ]
        elif percentage >= 35:
            risk_tier = "Moderate Cardiovascular Risk"
            result_label = "Borderline Cardiac Risk"
            recommendations = [
                "Schedule a routine lipid panel (Total Chol, LDL, HDL, Triglycerides).",
                "Maintain blood pressure below 120/80 mmHg through lifestyle management.",
                "Quit smoking and reduce alcoholic consumption.",
                "Engage in 30 minutes of low-impact walking daily."
            ]
        else:
            risk_tier = "Low / Healthy Cardiac Profile"
            result_label = "Normal Cardiac Range"
            recommendations = [
                "Maintain optimal cardiovascular endurance through regular physical exercise.",
                "Continue balanced dietary practices low in saturated/trans fats.",
                "Annual routine preventive cardiology checkups."
            ]
            
        return {
            "status": "success",
            "type": "heart",
            "result": result_label,
            "probability": percentage,
            "risk_tier": risk_tier,
            "input_metrics": {name: val for name, val in zip(feature_names, vals)},
            "recommendations": recommendations
        }

    def predict_parkinsons(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts Parkinson's risk using voice instability biomedical features."""
        if not self.parkinsons_bundle:
            raise RuntimeError("Parkinson's model not initialized.")
            
        model = self.parkinsons_bundle["model"]
        scaler = self.parkinsons_bundle["scaler"]
        feature_names = self.parkinsons_bundle["features"]
        
        # Fill features with user values or normative defaults
        defaults = {
            "MDVP:Fo(Hz)": 154.0, "MDVP:Fhi(Hz)": 197.0, "MDVP:Flo(Hz)": 116.0,
            "MDVP:Jitter(%)": 0.006, "MDVP:Jitter(Abs)": 0.00004, "MDVP:RAP": 0.003,
            "MDVP:PPQ": 0.0035, "Jitter:DDP": 0.009, "MDVP:Shimmer": 0.029,
            "MDVP:Shimmer(dB)": 0.28, "Shimmer:APQ3": 0.015, "Shimmer:APQ5": 0.018,
            "MDVP:APQ": 0.024, "Shimmer:DDA": 0.045, "NHR": 0.024,
            "HNR": 21.8, "RPDE": 0.49, "DFA": 0.71, "spread1": -5.6,
            "spread2": 0.22, "D2": 2.38, "PPE": 0.20
        }
        
        vals = []
        for feat in feature_names:
            v = data.get(feat, defaults.get(feat, 0.0))
            vals.append(float(v))
            
        X = np.array([vals])
        X_scaled = scaler.transform(X)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            prob_status = float(probs[1])
        else:
            pred = model.predict(X_scaled)[0]
            prob_status = 0.85 if pred == 1 else 0.15
            
        percentage = round(prob_status * 100, 2)
        
        if percentage >= 60:
            risk_tier = "Parkinsonian Biomarkers Detected"
            result_label = "Positive Voice Tremor & Instability Pattern"
            recommendations = [
                "Consult a movement disorder specialist or neurologist for clinical UPDRS examination.",
                "Comprehensive motor evaluation including gait, rest tremor, and rigidity tests.",
                "Consider speech-language pathology (Lee Silverman Voice Treatment / LSVT LOUD).",
                "Regular aerobic exercises like cycling, tai chi, and dance for motor coordination."
            ]
        elif percentage >= 35:
            risk_tier = "Borderline / Mild Vocal Instability"
            result_label = "Borderline Acoustic Biomarkers"
            recommendations = [
                "Monitor for subtle motor symptoms such as micro-graphia, reduced arm swing, or resting tremors.",
                "Repeat vocal frequency and jitter screening after 3-6 months.",
                "Maintain active physical and cognitive routines."
            ]
        else:
            risk_tier = "Normal / Healthy Biomarkers"
            result_label = "No Parkinsonian Vocal Tremors Detected"
            recommendations = [
                "Acoustic frequency and harmonic-to-noise ratios are within normative ranges.",
                "Maintain general neurological and physical wellness."
            ]
            
        return {
            "status": "success",
            "type": "parkinsons",
            "result": result_label,
            "probability": percentage,
            "risk_tier": risk_tier,
            "input_metrics": {name: val for name, val in zip(feature_names, vals)},
            "recommendations": recommendations
        }

# Global Singleton Engine Instance
predictor_engine = ClinicalPredictorEngine()
