"""
Conversational Clinical Assistant & Intelligent Symptom Triage Chatbot.
Parses natural language user complaints, extracts matching symptoms, flags emergency risks,
and guides the patient with follow-up questions and instant diagnostic routing.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter
from machine_learning.predictor import predictor_engine, EMERGENCY_SYMPTOMS

router = APIRouter(prefix="/api/chat", tags=["Chatbot Assistant"])

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(...)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = Field(default_factory=list)

# Common symptom synonym lookup mappings for natural language parsing
SYMPTOM_SYNONYMS = {
    "fever": "high_fever",
    "temperature": "high_fever",
    "pyrexia": "high_fever",
    "mild fever": "mild_fever",
    "chills": "chills",
    "shivering": "shivering",
    "sweat": "sweating",
    "sweating": "sweating",
    "night sweats": "sweating",
    "cough": "cough",
    "coughing": "cough",
    "phlegm": "phlegm",
    "sputum": "phlegm",
    "blood in phlegm": "blood_in_sputum",
    "blood in sputum": "blood_in_sputum",
    "headache": "headache",
    "head ache": "headache",
    "migraine": "headache",
    "tired": "fatigue",
    "fatigue": "fatigue",
    "exhaustion": "fatigue",
    "weakness": "muscle_weakness",
    "chest pain": "chest_pain",
    "tightness in chest": "chest_pain",
    "shortness of breath": "breathlessness",
    "breathless": "breathlessness",
    "difficulty breathing": "breathlessness",
    "nausea": "nausea",
    "queasy": "nausea",
    "vomit": "vomiting",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "stomach pain": "stomach_pain",
    "belly pain": "belly_pain",
    "abdominal pain": "abdominal_pain",
    "stomach ache": "stomach_pain",
    "acidity": "acidity",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "gas": "passage_of_gases",
    "bloating": "distention_of_abdomen",
    "loose motion": "diarrhoea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "constipation": "constipation",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "itching": "itching",
    "itchy skin": "itching",
    "itch": "itching",
    "yellow skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes",
    "dark urine": "dark_urine",
    "burning urine": "burning_micturition",
    "frequent urination": "polyuria",
    "joint pain": "joint_pain",
    "body aches": "muscle_pain",
    "muscle pain": "muscle_pain",
    "knee pain": "knee_pain",
    "back pain": "back_pain",
    "neck pain": "neck_pain",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "spinning": "spinning_movements",
    "vertigo": "spinning_movements",
    "blurred vision": "blurred_and_distorted_vision",
    "slurred speech": "slurred_speech",
    "one side weakness": "weakness_of_one_body_side",
    "paralysis": "weakness_of_one_body_side",
    "weight loss": "weight_loss",
    "weight gain": "weight_gain",
    "sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "congestion": "congestion",
    "stuffy nose": "congestion",
    "loss of smell": "loss_of_smell",
    "sore throat": "throat_irritation"
}

def extract_symptoms_from_text(text: str) -> List[str]:
    """Matches text against the known clinical symptom synonyms & vocabulary."""
    cleaned = text.lower()
    found = set()
    
    # Sort keys by descending length so multi-word tokens match first
    for syn in sorted(SYMPTOM_SYNONYMS.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(syn) + r"\b"
        if re.search(pattern, cleaned):
            found.add(SYMPTOM_SYNONYMS[syn])
            
    # Also check exact matches in known symptom list
    if predictor_engine.symptom_bundle:
        for feat in predictor_engine.symptom_bundle["feature_names"]:
            clean_feat = feat.replace("_", " ")
            if clean_feat in cleaned:
                found.add(feat)
                
    return sorted(list(found))

@router.post("")
async def conversational_assistant(req: ChatRequest):
    """Processes user dialogue, performs NLP symptom extraction, and provides clinical advice."""
    user_text = req.message.strip()
    extracted_symptoms = extract_symptoms_from_text(user_text)
    
    # Check for critical red-flag alerts
    emergency_alerts = []
    for sym in extracted_symptoms:
        if sym in EMERGENCY_SYMPTOMS:
            emergency_alerts.append(f"**{sym.replace('_', ' ').title()}**: {EMERGENCY_SYMPTOMS[sym]}")
            
    response_data = {
        "reply": "",
        "extracted_symptoms": extracted_symptoms,
        "suggested_questions": [],
        "emergency_warning": len(emergency_alerts) > 0,
        "quick_prediction": None
    }
    
    # Critical emergency response
    if len(emergency_alerts) > 0:
        alert_str = "\n- " + "\n- ".join(emergency_alerts)
        response_data["reply"] = (
            f"⚠️ **EMERGENCY CLINICAL NOTICE DETECTED**\n\n"
            f"Your message mentions symptoms that require urgent medical attention:\n{alert_str}\n\n"
            f"**Recommended Action**: Please contact emergency medical services (**911 / 112 / 108**) or proceed to the nearest emergency department immediately."
        )
        response_data["suggested_questions"] = [
            "Find nearest Emergency Hospital",
            "What should I do while waiting for help?",
            "Call local emergency dispatch"
        ]
        return response_data

    # Scenario 1: Symptoms detected (2 or more)
    if len(extracted_symptoms) >= 2:
        # Run live differential prediction
        pred_result = predictor_engine.predict_symptoms(extracted_symptoms, top_k=3)
        response_data["quick_prediction"] = pred_result
        
        top_dis = pred_result.get("primary_disease", "Health Condition")
        conf = pred_result.get("confidence", 0.0)
        sym_names = ", ".join([s.replace("_", " ").title() for s in extracted_symptoms])
        
        response_data["reply"] = (
            f"I have detected the following reported symptoms: **{sym_names}**.\n\n"
            f"Based on our machine learning diagnostic model, the primary indicative pattern points to **{top_dis}** (~{conf:.1f}% confidence).\n\n"
            f"• **Severity Level**: {pred_result.get('severity_level', 'Moderate')}\n"
            f"• **Triage Urgency**: {pred_result.get('triage_urgency')}\n\n"
            f"Would you like to review full precaution recommendations, diet advice, or view nearby specialist clinics for {top_dis}?"
        )
        response_data["suggested_questions"] = [
            f"Show precautions for {top_dis}",
            f"What medications or tests are recommended?",
            f"Find specialists near me",
            "Add more symptoms to refine diagnosis"
        ]
        return response_data
        
    # Scenario 2: 1 symptom detected -> Ask for more details
    elif len(extracted_symptoms) == 1:
        sym = extracted_symptoms[0].replace("_", " ").title()
        response_data["reply"] = (
            f"I noted you are experiencing **{sym}**. To help narrow down probable causes and provide an accurate assessment, could you share:\n\n"
            f"1. How long have you experienced this symptom?\n"
            f"2. Are you experiencing any other symptoms, such as fever, body aches, coughing, nausea, or headache?"
        )
        response_data["suggested_questions"] = [
            f"I also have a fever and body aches",
            f"I also have nausea and stomach discomfort",
            f"I also have a cough and throat irritation",
            f"It started today"
        ]
        return response_data
        
    # Scenario 3: General greeting or query
    else:
        lower = user_text.lower()
        if any(w in lower for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            response_data["reply"] = (
                "Hello! I am your **Medicare AI Clinical Assistant**. 👋\n\n"
                "I can help you analyze your symptoms, evaluate diabetes or cardiovascular risk profiles, find medical specialists, and review health guidance.\n\n"
                "How are you feeling today? You can describe any symptoms you are experiencing in plain words."
            )
            response_data["suggested_questions"] = [
                "I have high fever, chills, and headache",
                "I have stomach pain and nausea",
                "I want to check my diabetes risk",
                "I feel chest discomfort and shortness of breath"
            ]
        elif "diabetes" in lower:
            response_data["reply"] = (
                "For diabetes assessment, our **Specialized Diabetes Predictor** evaluates key biometric markers including fasting glucose, BMI, insulin, blood pressure, and age. "
                "You can test your values in the **Specialized Labs** tab or enter your latest lab readings."
            )
            response_data["suggested_questions"] = [
                "Open Diabetes Risk Calculator",
                "What are normal fasting glucose levels?",
                "What symptoms indicate high blood sugar?"
            ]
        elif "heart" in lower or "cardiac" in lower:
            response_data["reply"] = (
                "Our **Cardiovascular Risk Predictor** analyzes resting blood pressure, cholesterol levels, resting ECG, max heart rate, and chest pain indicators. "
                "If you are currently experiencing acute chest pressure radiating to the arm or jaw, seek emergency medical care immediately."
            )
            response_data["suggested_questions"] = [
                "Open Heart Disease Diagnostic Lab",
                "What is a healthy blood pressure target?",
                "What are early warning signs of heart issues?"
            ]
        else:
            response_data["reply"] = (
                f"Thank you for sharing. Could you describe specific symptoms you are noticing (for example: fever, cough, joint pain, skin rash, or dizziness)? "
                f"This will allow our ML diagnostic model to cross-reference with our database of 42 conditions and 132 symptoms."
            )
            response_data["suggested_questions"] = [
                "I have fever, cough, and fatigue",
                "I have rash and severe itching",
                "I have dizziness and headache",
                "I have burning sensation during urination"
            ]
            
    return response_data
