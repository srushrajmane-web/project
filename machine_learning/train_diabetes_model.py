"""
Specialized Diabetes Prediction Model Training.
Uses Pima Indians Clinical Dataset format:
[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
Trains and compares Logistic Regression, Random Forest, and SVM with StandardScaler.
Saves model and metrics to ml_models/.
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
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_diabetes_model():
    print("=== Training Specialized Diabetes Classifier ===")
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    
    feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df[feature_cols]
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "SVM (RBF)": SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
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
        "target_names": ["Non-Diabetic", "Diabetic"],
        "version": "1.0.0"
    }
    
    save_path = os.path.join(MODELS_DIR, "diabetes_model.joblib")
    joblib.dump(bundle, save_path)
    
    metrics = {
        "model_type": "Specialized Diabetes Predictor",
        "selected_model": best_name,
        "comparison": comparison,
        "features": feature_cols,
        "samples_count": len(df)
    }
    with open(os.path.join(MODELS_DIR, "diabetes_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved diabetes model to {save_path}\n")
    return metrics

if __name__ == "__main__":
    train_diabetes_model()
