"""
Main FastAPI Application Entry Point for Human Disease Diagnosis System.
A Full-Stack Python Machine Learning Healthcare Application.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Automatically load environment variables from .env file
load_dotenv()

from backend.database.database import init_db
from machine_learning.predictor import predictor_engine
from backend.api.predict import router as predict_router
from backend.api.diseases import router as diseases_router
from backend.api.auth import router as auth_router
from backend.api.history import router as history_router
from backend.api.chat import router as chat_router
from backend.api.locator import router as locator_router
from backend.api.admin import router as admin_router

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "frontend", "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and seeds
    print("[Startup] Initializing SQLite Database & Knowledge Base...")
    init_db()
    print("[Startup] Ready! Predictive engines loaded.")
    yield
    print("[Shutdown] Cleaning up server resources.")

app = FastAPI(
    title="Human Disease Diagnosis System API",
    description="End-to-end Machine Learning Powered Clinical Decision Support & Symptom Prediction Platform.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(predict_router)
app.include_router(diseases_router)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(chat_router)
app.include_router(locator_router)
app.include_router(admin_router)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main Full-Stack Healthcare Single Page Application."""
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")

@app.get("/health")
async def health_check():
    """Health check endpoint for containerization & deployment readiness."""
    return {
        "status": "healthy",
        "service": "Human Disease Diagnosis System",
        "version": "2.0.0",
        "engine_status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
