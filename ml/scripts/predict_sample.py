"""
Sample Prediction Script for Clinical Decision Support
Loads trained XGBoost model and preprocessor to predict 30-day readmission risk probability for a patient encounter.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def predict_patient_risk(sample_dict=None):
    model_path = os.path.join(MODELS_DIR, "xgboost_readmission.joblib")
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        print("[Inference] Model or preprocessor not found. Running training first...")
        from train import train_readmission_model
        train_readmission_model()
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    
    feature_names = preprocessor['feature_names']
    scaler = preprocessor['scaler']
    numeric_cols = preprocessor['numeric_cols']
    
    # Default high-risk diabetic patient profile for demo
    if sample_dict is None:
        sample_dict = {
            'time_in_hospital': 8,
            'num_lab_procedures': 65,
            'num_procedures': 2,
            'num_medications': 22,
            'number_outpatient': 1,
            'number_emergency': 2,
            'number_inpatient': 3,
            'number_diagnoses': 9,
            'age_num': 75,
            'gender': 1, # Male
            'change': 1,
            'diabetesMed': 1
        }
        
    # Build dataframe with all expected features initialized to 0
    row_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    for col, val in sample_dict.items():
        if col in row_df.columns:
            row_df.at[0, col] = val
            
    # Scale numeric columns
    present_num_cols = [c for c in numeric_cols if c in row_df.columns]
    row_df[present_num_cols] = scaler.transform(row_df[present_num_cols])
    
    prob = model.predict_proba(row_df)[0][1]
    prediction = int(prob >= 0.5)
    risk_level = "High" if prob >= 0.60 else ("Moderate" if prob >= 0.35 else "Low")
    
    print("\n" + "=" * 50)
    print("AI-CDSS: 30-Day Readmission Risk Assessment")
    print("=" * 50)
    print(f"Predicted Risk Probability: {prob * 100:.1f}%")
    print(f"Stratified Risk Level:      {risk_level}")
    print(f"Model Recommendation:       Doctor review recommended for discharge planning.")
    print("Notice: Clinical decision support tool only. Does not replace physician judgement.")
    print("=" * 50)
    
    return {
        "risk_probability": round(float(prob), 4),
        "risk_percentage": round(float(prob * 100), 1),
        "risk_level": risk_level,
        "flagged_for_review": bool(prob >= 0.40)
    }

if __name__ == "__main__":
    predict_patient_risk()
