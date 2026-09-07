"""
API Routers Package for Human Disease Diagnosis System.
"""

from backend.api.predict import router as predict_router
from backend.api.diseases import router as diseases_router
from backend.api.auth import router as auth_router
from backend.api.history import router as history_router
from backend.api.chat import router as chat_router
from backend.api.locator import router as locator_router
from backend.api.admin import router as admin_router

__all__ = [
    "predict_router",
    "diseases_router",
    "auth_router",
    "history_router",
    "chat_router",
    "locator_router",
    "admin_router"
]
