"""
Authentication Endpoints for User Registration, Login, and Profile Management.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, HTTPException, Depends, status
from database import get_db
from auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = Field("patient", description="Role: 'patient', 'doctor', or 'admin'")
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[str] = Field(None)
    blood_group: Optional[str] = Field(None)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class DemoLoginRequest(BaseModel):
    role: str = Field("patient", description="'patient', 'doctor', or 'admin'")

@router.post("/register")
async def register(req: RegisterRequest):
    """Registers a new user account."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if email exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (req.email.lower(),))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
        
    hashed_pw = hash_password(req.password)
    user_role = req.role if req.role in ["patient", "doctor", "admin"] else "patient"
    
    cursor.execute("""
    INSERT INTO users (name, email, password_hash, role, age, gender, blood_group)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (req.name, req.email.lower(), hashed_pw, user_role, req.age, req.gender, req.blood_group))
    
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    
    token = create_access_token({
        "id": user_id,
        "name": req.name,
        "email": req.email.lower(),
        "role": user_role
    })
    
    return {
        "status": "success",
        "message": "User account created successfully",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": req.name,
            "email": req.email.lower(),
            "role": user_role
        }
    }

@router.post("/login")
async def login(req: LoginRequest):
    """Authenticates user and returns JWT token."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash, role, age, gender, blood_group FROM users WHERE email = ?", (req.email.lower(),))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    token = create_access_token({
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    })
    
    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "age": user["age"],
            "gender": user["gender"],
            "blood_group": user["blood_group"]
        }
    }

@router.post("/demo")
async def demo_login(req: DemoLoginRequest):
    """One-click instant login for Demo roles (Patient, Doctor, Admin)."""
    conn = get_db()
    cursor = conn.cursor()
    role = req.role.lower()
    
    cursor.execute("SELECT id, name, email, role, age, gender, blood_group FROM users WHERE role = ? LIMIT 1", (role,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=404, detail=f"No demo account found for role '{role}'.")
        
    token = create_access_token({
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    })
    
    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "age": user["age"],
            "gender": user["gender"],
            "blood_group": user["blood_group"]
        }
    }

@router.get("/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Fetches currently logged-in user profile."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, age, gender, blood_group, created_at FROM users WHERE id = ?", (current_user["id"],))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found")
        
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "age": user["age"],
        "gender": user["gender"],
        "blood_group": user["blood_group"],
        "created_at": user["created_at"]
    }
