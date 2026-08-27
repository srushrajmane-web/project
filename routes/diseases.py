"""
Routes for Disease Information Panel and Symptom Knowledge Base.
"""

import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from database import get_db

router = APIRouter(prefix="/api", tags=["Diseases & Symptoms"])

@router.get("/diseases")
async def get_all_diseases(
    category: Optional[str] = Query(None, description="Filter by disease category"),
    search: Optional[str] = Query(None, description="Search query")
):
    """Retrieves list of all 42 clinical disease entities with summary metadata."""
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT id, name, category, description, precautions, medications, diet, workout FROM diseases"
    params = []
    conditions = []
    
    if category:
        conditions.append("category = ?")
        params.append(category)
    if search:
        conditions.append("(name LIKE ? OR description LIKE ?)")
        params.append(f"%{search}%")
        params.append(f"%{search}%")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY name ASC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    diseases = []
    for r in rows:
        diseases.append({
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "description": r["description"],
            "precautions": json.loads(r["precautions"]),
            "medications": json.loads(r["medications"]),
            "diet": json.loads(r["diet"]),
            "workout": json.loads(r["workout"])
        })
    conn.close()
    return {"total": len(diseases), "diseases": diseases}

@router.get("/diseases/{name}")
async def get_disease_detail(name: str):
    """Retrieves full information panel for a specific disease."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, category, description, precautions, medications, diet, workout FROM diseases WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))",
        (name,)
    )
    r = cursor.fetchone()
    conn.close()
    
    if not r:
        raise HTTPException(status_code=404, detail=f"Disease '{name}' not found in database.")
        
    return {
        "id": r["id"],
        "name": r["name"],
        "category": r["category"],
        "description": r["description"],
        "precautions": json.loads(r["precautions"]),
        "medications": json.loads(r["medications"]),
        "diet": json.loads(r["diet"]),
        "workout": json.loads(r["workout"])
    }

@router.get("/symptoms")
async def get_all_symptoms(
    category: Optional[str] = Query(None, description="Filter by body system category"),
    search: Optional[str] = Query(None, description="Search query")
):
    """Returns the complete catalogue of 132 symptoms grouped with body categories and severity weights."""
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT id, name, category, weight FROM symptoms"
    params = []
    conditions = []
    
    if category:
        conditions.append("category = ?")
        params.append(category)
    if search:
        conditions.append("name LIKE ?")
        params.append(f"%{search}%")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY name ASC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    symptoms = []
    categories = set()
    for r in rows:
        categories.add(r["category"])
        symptoms.append({
            "id": r["id"],
            "name": r["name"],
            "display_name": r["name"].replace("_", " ").title(),
            "category": r["category"],
            "weight": r["weight"]
        })
    conn.close()
    
    return {
        "total": len(symptoms),
        "categories": sorted(list(categories)),
        "symptoms": symptoms
    }
