"""
Database layer for Human Disease Diagnosis System.
Uses SQLite with automatic table initialization, migrations, and seed data.
"""

import os
import json
import sqlite3
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "diagnosis_system.db")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def get_db():
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'patient',
        age INTEGER,
        gender TEXT,
        blood_group TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Predictions History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_email TEXT,
        input_symptoms TEXT NOT NULL,
        predicted_disease TEXT NOT NULL,
        confidence REAL NOT NULL,
        top_predictions TEXT NOT NULL,
        severity_level TEXT NOT NULL,
        triage_urgency TEXT DEFAULT 'Standard',
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 3. Specialized Reports Table (Diabetes, Heart, Parkinson's)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS specialized_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_email TEXT,
        type TEXT NOT NULL,
        input_values TEXT NOT NULL,
        result TEXT NOT NULL,
        probability REAL NOT NULL,
        risk_score TEXT NOT NULL,
        insights TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 4. Reference Diseases Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diseases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        description TEXT NOT NULL,
        precautions TEXT NOT NULL,
        medications TEXT NOT NULL,
        diet TEXT NOT NULL,
        workout TEXT NOT NULL
    )
    """)
    
    # 5. Reference Symptoms Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS symptoms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        weight INTEGER NOT NULL DEFAULT 3
    )
    """)
    
    # 6. Healthcare Providers & Hospitals Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hospitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        address TEXT NOT NULL,
        phone TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        rating REAL NOT NULL,
        emergency_ready INTEGER DEFAULT 1
    )
    """)
    
    conn.commit()
    seed_reference_data(conn)
    seed_default_users(conn)
    seed_hospitals(conn)
    conn.close()
    print("Database initialized and seeded successfully.")

def seed_reference_data(conn):
    cursor = conn.cursor()
    
    # Check if diseases already populated
    cursor.execute("SELECT COUNT(*) FROM diseases")
    if cursor.fetchone()[0] == 0:
        desc_file = os.path.join(RAW_DATA_DIR, "symptom_Description.csv")
        prec_file = os.path.join(RAW_DATA_DIR, "symptom_precaution.csv")
        med_file = os.path.join(RAW_DATA_DIR, "disease_medications.csv")
        life_file = os.path.join(RAW_DATA_DIR, "disease_lifestyle.csv")
        
        if os.path.exists(desc_file) and os.path.exists(prec_file):
            df_desc = pd.read_csv(desc_file)
            df_prec = pd.read_csv(prec_file)
            df_med = pd.read_csv(med_file) if os.path.exists(med_file) else None
            df_life = pd.read_csv(life_file) if os.path.exists(life_file) else None
            
            for _, r in df_desc.iterrows():
                dis_name = str(r["Disease"]).strip()
                desc = str(r["Description"]).strip()
                
                # precautions
                p_row = df_prec[df_prec["Disease"].str.strip() == dis_name]
                prec_list = []
                if not p_row.empty:
                    p = p_row.iloc[0]
                    prec_list = [p[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(p.get(f"Precaution_{i}")) and str(p.get(f"Precaution_{i}")).strip()]
                
                # medications
                med_list = ["Standard prescribed clinical medication", "Consult physician for targeted dosage"]
                if df_med is not None:
                    m_row = df_med[df_med["Disease"].str.strip() == dis_name]
                    if not m_row.empty and pd.notna(m_row.iloc[0]["Medication"]):
                        med_list = json.loads(m_row.iloc[0]["Medication"])
                        
                # diet & workout
                diet_list = ["Nutritious balanced diet", "Stay hydrated", "Avoid processed foods"]
                workout_list = ["Light daily aerobic exercise", "Adequate rest and recovery"]
                cat = "General Medicine"
                if df_life is not None:
                    l_row = df_life[df_life["Disease"].str.strip() == dis_name]
                    if not l_row.empty:
                        if pd.notna(l_row.iloc[0]["Diet"]):
                            diet_list = json.loads(l_row.iloc[0]["Diet"])
                        if pd.notna(l_row.iloc[0]["Workout"]):
                            workout_list = json.loads(l_row.iloc[0]["Workout"])
                        if pd.notna(l_row.iloc[0]["Category"]):
                            cat = str(l_row.iloc[0]["Category"])
                            
                cursor.execute("""
                INSERT OR REPLACE INTO diseases (name, category, description, precautions, medications, diet, workout)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (dis_name, cat, desc, json.dumps(prec_list), json.dumps(med_list), json.dumps(diet_list), json.dumps(workout_list)))
                
    # Symptoms catalogue
    cursor.execute("SELECT COUNT(*) FROM symptoms")
    if cursor.fetchone()[0] == 0:
        sev_file = os.path.join(RAW_DATA_DIR, "symptom_severity.csv")
        if os.path.exists(sev_file):
            df_sev = pd.read_csv(sev_file)
            for _, r in df_sev.iterrows():
                sym = str(r["Symptom"]).strip()
                w = int(r["weight"]) if pd.notna(r["weight"]) else 3
                
                # Determine body system category
                category = "General / Systemic"
                if any(k in sym for k in ["headache", "dizziness", "balance", "speech", "sensorium", "vision", "depression", "irritability"]):
                    category = "Neurological & Mental"
                elif any(k in sym for k in ["cough", "breath", "throat", "sneezing", "nose", "sinus", "chest", "sputum"]):
                    category = "Respiratory & Cardiac"
                elif any(k in sym for k in ["stomach", "vomit", "acidity", "diarrhoea", "bowel", "constipation", "appetite", "nausea", "abdomen"]):
                    category = "Gastrointestinal & Hepatic"
                elif any(k in sym for k in ["rash", "itch", "pimples", "skin", "spots", "blister", "patches", "nails"]):
                    category = "Dermatological"
                elif any(k in sym for k in ["urine", "micturition", "bladder", "kidney"]):
                    category = "Urological & Renal"
                elif any(k in sym for k in ["joint", "muscle", "knee", "back", "neck", "limbs", "walking"]):
                    category = "Musculoskeletal"
                    
                cursor.execute("""
                INSERT OR REPLACE INTO symptoms (name, category, weight)
                VALUES (?, ?, ?)
                """, (sym, category, w))
                
    conn.commit()

def seed_default_users(conn):
    """Creates initial Demo Patient, Doctor, and Admin accounts with bcrypt hashed passwords."""
    import bcrypt
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        def hash_pw(pw: str) -> str:
            return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
        users_to_add = [
            ("Dr. Sarah Jenkins", "doctor@medicare.io", hash_pw("Doctor@123"), "doctor", 42, "Female", "O+"),
            ("Alex Morgan", "patient@medicare.io", hash_pw("Patient@123"), "patient", 34, "Male", "A+"),
            ("System Administrator", "admin@medicare.io", hash_pw("Admin@123"), "admin", 38, "Other", "B+")
        ]
        
        cursor.executemany("""
        INSERT INTO users (name, email, password_hash, role, age, gender, blood_group)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, users_to_add)
        conn.commit()

def seed_hospitals(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM hospitals")
    if cursor.fetchone()[0] == 0:
        providers = [
            ("Apollo Multi-Specialty Health City", "Cardiology, Emergency, Neurology", "45 Healthcare Blvd, Central City", "+1 (800) 555-0199", 37.7749, -122.4194, 4.9, 1),
            ("Metropolitan University Medical Center", "Gastroenterology, Hepatology, Pulmonology", "120 Academic Row, University District", "+1 (800) 555-0142", 37.7833, -122.4167, 4.8, 1),
            ("St. Jude Institute of Neurological Sciences", "Neurology, Parkinson's Care, Spine Surgery", "88 Neural Way, North Hills", "+1 (800) 555-0188", 37.7651, -122.4350, 4.9, 1),
            ("Hope Endocrinology & Diabetes Clinic", "Endocrinology, Metabolic Disorders, Dietetics", "210 Wellness Plaza, Eastside", "+1 (800) 555-0163", 37.7915, -122.4050, 4.7, 0),
            ("Apex Advanced Dermatology Center", "Dermatology, Immunology, Skin Allergy", "34 Derma Suites, West End", "+1 (800) 555-0111", 37.7562, -122.4200, 4.8, 0),
            ("City Care 24x7 Urgent Care & Trauma", "Emergency Medicine, General Practice, Pediatrics", "10 Rapid Response Ave, Downtown", "+1 (800) 555-0100", 37.7700, -122.4400, 4.6, 1),
            ("Premier Rheumatology & Orthopedic Institute", "Rheumatology, Orthopedics, Physical Therapy", "55 Mobility Court, South Bay", "+1 (800) 555-0177", 37.7500, -122.4100, 4.7, 0)
        ]
        cursor.executemany("""
        INSERT INTO hospitals (name, department, address, phone, lat, lng, rating, emergency_ready)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, providers)
        conn.commit()
