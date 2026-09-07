"""
Prediction History and Health Analytics Routes.
"""

import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from backend.database.database import get_db
from backend.services.auth_service import get_current_user_optional, get_current_user

router = APIRouter(prefix="/api/history", tags=["History & Analytics"])

@router.get("")
async def get_history(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    limit: int = Query(50, ge=1, le=200),
    type: Optional[str] = Query(None, description="'symptoms' or specialized 'diabetes'/'heart'/'parkinsons'")
):
    """Retrieves user prediction history and specialized diagnostic reports."""
    conn = get_db()
    cursor = conn.cursor()
    
    user_id = current_user.get("id") if current_user else None
    user_email = current_user.get("email") if current_user else "guest@medicare.io"
    user_role = current_user.get("role") if current_user else "patient"
    
    results = []
    
    # If doctor or admin, allow viewing all or user-specific records
    is_privileged = user_role in ["doctor", "admin"]
    
    # 1. Symptom Predictions History
    if type is None or type == "symptoms":
        if is_privileged:
            cursor.execute("SELECT id, user_id, user_email, input_symptoms, predicted_disease, confidence, top_predictions, severity_level, triage_urgency, notes, created_at FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT id, user_id, user_email, input_symptoms, predicted_disease, confidence, top_predictions, severity_level, triage_urgency, notes, created_at FROM predictions WHERE user_id = ? OR user_email = ? ORDER BY created_at DESC LIMIT ?", (user_id, user_email, limit))
            
        rows = cursor.fetchall()
        for r in rows:
            results.append({
                "id": r["id"],
                "record_type": "symptom_prediction",
                "user_email": r["user_email"],
                "input_symptoms": json.loads(r["input_symptoms"]),
                "predicted_disease": r["predicted_disease"],
                "confidence": r["confidence"],
                "top_predictions": json.loads(r["top_predictions"]),
                "severity_level": r["severity_level"],
                "triage_urgency": r["triage_urgency"],
                "notes": r["notes"],
                "created_at": r["created_at"]
            })
            
    # 2. Specialized Diagnostic Reports
    if type is None or type in ["diabetes", "heart", "parkinsons"]:
        spec_query = "SELECT id, user_id, user_email, type, input_values, result, probability, risk_score, insights, created_at FROM specialized_reports"
        params = []
        conds = []
        
        if not is_privileged:
            conds.append("(user_id = ? OR user_email = ?)")
            params.extend([user_id, user_email])
            
        if type and type in ["diabetes", "heart", "parkinsons"]:
            conds.append("type = ?")
            params.append(type)
            
        if conds:
            spec_query += " WHERE " + " AND ".join(conds)
            
        spec_query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(spec_query, params)
        rows_spec = cursor.fetchall()
        for r in rows_spec:
            results.append({
                "id": r["id"],
                "record_type": f"specialized_{r['type']}",
                "user_email": r["user_email"],
                "type": r["type"],
                "input_values": json.loads(r["input_values"]),
                "result": r["result"],
                "probability": r["probability"],
                "risk_score": r["risk_score"],
                "insights": json.loads(r["insights"]) if r["insights"] else [],
                "created_at": r["created_at"]
            })
            
    # Sort combined results by created_at descending
    results.sort(key=lambda x: x["created_at"], reverse=True)
    conn.close()
    
    return {"total": len(results), "records": results}

@router.get("/stats")
async def get_history_stats(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    """Computes aggregate analytics for charts and dashboard visualization."""
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Total counts
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_symptom_preds = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM specialized_reports")
    total_specialized_reports = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users")
    total_registered_users = cursor.fetchone()[0]
    
    # 2. Top Diagnosed Diseases
    cursor.execute("""
    SELECT predicted_disease, COUNT(*) as count 
    FROM predictions 
    GROUP BY predicted_disease 
    ORDER BY count DESC 
    LIMIT 6
    """)
    top_diseases = [{"disease": r["predicted_disease"], "count": r["count"]} for r in cursor.fetchall()]
    
    # 3. Severity Distribution
    cursor.execute("""
    SELECT severity_level, COUNT(*) as count 
    FROM predictions 
    GROUP BY severity_level
    """)
    severity_dist = {r["severity_level"]: r["count"] for r in cursor.fetchall()}
    
    # 4. Specialized Reports breakdown
    cursor.execute("""
    SELECT type, COUNT(*) as count, AVG(probability) as avg_prob 
    FROM specialized_reports 
    GROUP BY type
    """)
    specialized_breakdown = {
        r["type"]: {
            "count": r["count"],
            "avg_probability": round(float(r["avg_prob"] or 0), 1)
        }
        for r in cursor.fetchall()
    }
    
    conn.close()
    
    return {
        "total_predictions": total_symptom_preds + total_specialized_reports,
        "total_symptom_checks": total_symptom_preds,
        "total_specialized_tests": total_specialized_reports,
        "total_registered_users": total_registered_users,
        "top_diseases": top_diseases,
        "severity_distribution": severity_dist,
        "specialized_breakdown": specialized_breakdown
    }

@router.delete("/{record_id}")
async def delete_history_record(
    record_id: int,
    record_type: str = Query("symptom", description="'symptom' or 'specialized'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Deletes a history item."""
    conn = get_db()
    cursor = conn.cursor()
    
    table = "predictions" if record_type == "symptom" else "specialized_reports"
    
    # Verify ownership or admin role
    if current_user.get("role") in ["admin", "doctor"]:
        cursor.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
    else:
        cursor.execute(f"DELETE FROM {table} WHERE id = ? AND (user_id = ? OR user_email = ?)", (record_id, current_user.get("id"), current_user.get("email")))
        
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found or not authorized to delete.")
        
    return {"status": "success", "message": "Record deleted successfully"}
