"""
Specialized Parkinson's Disease Prediction Model Training.
Uses UCI Parkinson's Biomedical Voice Measurements:
[MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP,
 MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5,
 MDVP:APQ, Shimmer:DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
Trains and compares Random Forest, Logistic Regression, and SVM with StandardScaler.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_parkinsons_model():
    print("=== Training Specialized Parkinson's Disease Classifier ===")
    df = pd.read_csv(os.path.join(DATA_DIR, "parkinsons.csv"))
    
    feature_cols = [c for c in df.columns if c != "status"]
    X = df[feature_cols]
    y = df["status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42),
        "SVM (RBF)": SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42)
    }
    
    comparison = {}
    best_name = None
    best_f1 = -1
    best_clf = None
    
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cv = cross_val_score(clf, X_train_scaled, y_train, cv=5).mean()
        
        comparison[name] = {
            "test_accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "roc_auc": round(float(auc) * 100, 2),
            "cv_accuracy": round(float(cv) * 100, 2)
        }
        print(f"[{name}] Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_clf = clf
            
    bundle = {
        "model": best_clf,
        "model_name": best_name,
        "scaler": scaler,
        "features": feature_cols,
        "feature_importances": dict(zip(feature_cols, [round(float(v), 4) for v in getattr(best_clf, "feature_importances_", np.ones(len(feature_cols)) / len(feature_cols))])) if hasattr(best_clf, "feature_importances_") else {},
        "target_names": ["Healthy (No Parkinson's Indicators)", "Parkinson's Indicators Detected"],
        "version": "1.0.0"
    }
    
    save_path = os.path.join(MODELS_DIR, "parkinsons_model.joblib")
    joblib.dump(bundle, save_path)
    
    metrics = {
        "model_type": "Specialized Parkinson's Predictor",
        "selected_model": best_name,
        "comparison": comparison,
        "features": feature_cols,
        "samples_count": len(df)
    }
    with open(os.path.join(MODELS_DIR, "parkinsons_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved parkinsons model to {save_path}\n")
    return metrics

if __name__ == "__main__":
    train_parkinsons_model()
