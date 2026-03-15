import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt

def load_processed_data():
    print("Loading processed features and labels...")
    if not os.path.exists("X_train_processed.csv"):
        raise FileNotFoundError("Processed data not found. Run data_processing.py first.")
        
    X = pd.read_csv("X_train_processed.csv")
    y = pd.read_csv("y_train_processed.csv").squeeze("columns") # Convert to Series
    return X, y

def train_baseline(X_train, X_test, y_train, y_test):
    print("\n--- Training Baseline Model (Logistic Regression on TF-IDF only) ---")
    # Extract only the TF-IDF features (drop our custom engineered ones for the baseline)
    engineered_cols = ['review_length', 'exclamation_count', 'verified_purchase', 
                       'reviewer_review_count', 'rating_deviation']
    
    X_train_base = X_train.drop(columns=engineered_cols, errors='ignore')
    X_test_base = X_test.drop(columns=engineered_cols, errors='ignore')
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_base, y_train)
    
    y_pred = model.predict(X_test_base)
    print(f"Baseline F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Baseline Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Baseline Recall: {recall_score(y_test, y_pred):.4f}")

def train_champion(X_train, X_test, y_train, y_test):
    print("\n--- Training Champion Model (Random Forest on ALL features) ---")
    
    # We use fewer estimators here just to keep training fast for the demo
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    print("\nChampion Model Performance:")
    print(f"Advanced F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Advanced Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Advanced Recall: {recall_score(y_test, y_pred):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the Champion Model
    print("\nSaving Champion model structure to random_forest.pkl...")
    with open("random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
        
    return rf_model

def generate_shap_explainer(model, X_train):
    print("\nGenerating SHAP TreeExplainer for UI Interpretability...")
    # SHAP can be slow on large datasets, so we fit the explainer on a background sample
    background_sample = shap.sample(X_train, 100)
    explainer = shap.TreeExplainer(model)
    
    with open("shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("Saved shap_explainer.pkl!")

def run_training_pipeline():
    X, y = load_processed_data()
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]} rows. Test set: {X_test.shape[0]} rows.")
    
    # Train Models
    train_baseline(X_train, X_test, y_train, y_test)
    champion_model = train_champion(X_train, X_test, y_train, y_test)
    
    # Build SHAP
    generate_shap_explainer(champion_model, X_train)
    print("\nModel pipeline complete! Project is ready for Streamlit UI.")

if __name__ == "__main__":
    run_training_pipeline()
