"""
Database Layer Package for Human Disease Diagnosis System.
"""

from backend.database.database import get_db, init_db

__all__ = ["get_db", "init_db"]
