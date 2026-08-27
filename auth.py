"""
Authentication and JWT security module for Human Disease Diagnosis System.
Supports bcrypt password hashing and JWT token issuance and validation.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
import bcrypt
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "medicare_super_secure_clinical_key_2026_x89a")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

security_scheme = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Optional[Dict[str, Any]]:
    """Extracts user from token if present, returns None for guest."""
    if not credentials:
        return None
    token = credentials.credentials
    payload = decode_token(token)
    return payload

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Dict[str, Any]:
    """Requires authenticated user with valid JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"}
        )
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return payload

def require_role(allowed_roles: list[str]):
    """Role-based access guard."""
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "patient")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {', '.join(allowed_roles)}"
            )
        return current_user
    return role_checker
