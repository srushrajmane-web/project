"""
Healthcare Provider and Specialist Clinic Locator API.
"""

import math
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query
from database import get_db

router = APIRouter(prefix="/api/hospitals", tags=["Hospital Locator"])

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two geographic coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

@router.get("")
async def find_hospitals(
    department: Optional[str] = Query(None, description="Filter by specialty/department"),
    emergency_only: Optional[bool] = Query(False, description="Filter only emergency-ready centers"),
    user_lat: Optional[float] = Query(37.7749, description="User latitude (default: San Francisco / Metro)"),
    user_lng: Optional[float] = Query(-122.4194, description="User longitude"),
    query: Optional[str] = Query(None, description="Search keyword")
):
    """Finds matching healthcare clinics, hospitals, and specialists with distance calculation."""
    conn = get_db()
    cursor = conn.cursor()
    
    sql = "SELECT id, name, department, address, phone, lat, lng, rating, emergency_ready FROM hospitals WHERE 1=1"
    params = []
    
    if department:
        sql += " AND department LIKE ?"
        params.append(f"%{department}%")
        
    if emergency_only:
        sql += " AND emergency_ready = 1"
        
    if query:
        sql += " AND (name LIKE ? OR department LIKE ? OR address LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
        
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        dist_km = haversine_distance(user_lat or 37.7749, user_lng or -122.4194, r["lat"], r["lng"])
        results.append({
            "id": r["id"],
            "name": r["name"],
            "department": r["department"],
            "departments_list": [d.strip() for d in r["department"].split(",")],
            "address": r["address"],
            "phone": r["phone"],
            "lat": r["lat"],
            "lng": r["lng"],
            "rating": r["rating"],
            "emergency_ready": bool(r["emergency_ready"]),
            "distance_km": dist_km,
            "distance_miles": round(dist_km * 0.621371, 1)
        })
        
    results.sort(key=lambda x: x["distance_km"])
    return {"total": len(results), "hospitals": results}
