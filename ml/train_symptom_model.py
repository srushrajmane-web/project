"""
Multi-Class Symptom-Based Disease Prediction Model Training Pipeline.
Trains and compares:
- Random Forest Classifier
- Multinomial / Gaussian Naive Bayes
- Decision Tree Classifier
- Logistic Regression
- Support Vector Machine (SVC)
Saves the top-performing serialized model, feature definitions, and full evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_symptom_models():
    print("=== Training Symptom-Disease Multi-Class Classifier ===")
    train_path = os.path.join(DATA_DIR, "Training.csv")
    test_path = os.path.join(DATA_DIR, "Testing.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Feature columns and label
    feature_cols = [c for c in df_train.columns if c != "prognosis"]
    X_train = df_train[feature_cols]
    y_train = df_train["prognosis"]
    
    X_test = df_test[feature_cols]
    y_test = df_test["prognosis"]
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=25, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "SVM (RBF)": SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
    }
    
    model_comparison = {}
    best_model_name = None
    best_f1 = -1
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train_enc)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test_enc, preds)
        prec = precision_score(y_test_enc, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test_enc, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test_enc, preds, average="weighted", zero_division=0)
        
        cv_scores = cross_val_score(model, X_train, y_train_enc, cv=5)
        
        model_comparison[name] = {
            "test_accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "cv_accuracy_mean": round(float(cv_scores.mean()) * 100, 2),
            "cv_accuracy_std": round(float(cv_scores.std()) * 100, 2)
        }
        
        print(f"[{name}] Test Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | CV: {cv_scores.mean()*100:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    print(f"\n--> Selected Best Model: {best_model_name} (F1: {best_f1*100:.2f}%)")
    
    # Per-disease classification report
    final_preds = best_model.predict(X_test)
    report_dict = classification_report(
        y_test_enc,
        final_preds,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    
    # Save model artifact bundle
    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "encoder": le,
        "feature_names": feature_cols,
        "classes": list(le.classes_),
        "version": "1.0.0"
    }
    
    model_save_path = os.path.join(MODELS_DIR, "symptom_model.joblib")
    joblib.dump(bundle, model_save_path)
    
    # Save performance metrics summary
    metrics_summary = {
        "model_type": "Multi-Class Symptom Classifier",
        "selected_model": best_model_name,
        "num_features": len(feature_cols),
        "num_classes": len(le.classes_),
        "total_training_samples": len(df_train),
        "total_testing_samples": len(df_test),
        "comparison": model_comparison,
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"] * 100, 1),
                "recall": round(report_dict[cls]["recall"] * 100, 1),
                "f1-score": round(report_dict[cls]["f1-score"] * 100, 1),
                "support": report_dict[cls]["support"]
            }
            for cls in le.classes_ if cls in report_dict
        }
    }
    
    with open(os.path.join(MODELS_DIR, "symptom_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Model saved to {model_save_path}")
    print(f"Metrics saved to {os.path.join(MODELS_DIR, 'symptom_metrics.json')}")
    return metrics_summary

if __name__ == "__main__":
    train_symptom_models()
