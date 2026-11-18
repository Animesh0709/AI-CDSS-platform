"""
Preprocessing Pipeline for UCI Diabetes 30-Day Hospital Readmission Prediction
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

AGE_MAP = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}

def map_icd9_to_category(code):
    if pd.isna(code) or code == '?':
        return 'Other'
    code_str = str(code).strip()
    try:
        val = float(code_str)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif int(val) == 250:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

def preprocess_data(df: pd.DataFrame):
    print("[Preprocessing] Starting data preprocessing...")
    df = df.copy()
    
    # 1. Target Variable: 30-day readmission (<30 is 1, else 0)
    df['target'] = (df['readmitted'] == '<30').astype(int)
    
    # 2. Drop high-missing or pure identifier columns
    drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'readmitted']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 3. Clean Gender
    df = df[df['gender'].isin(['Male', 'Female'])]
    df['gender'] = (df['gender'] == 'Male').astype(int)
    
    # 4. Age mapping
    df['age_num'] = df['age'].map(AGE_MAP).fillna(55)
    df.drop(columns=['age'], inplace=True, errors='ignore')
    
    # 5. ICD-9 Categorization
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        if diag in df.columns:
            df[f'{diag}_cat'] = df[diag].apply(map_icd9_to_category)
            df.drop(columns=[diag], inplace=True)
            
    # 6. Binary mappings for change and diabetesMed
    if 'change' in df.columns:
        df['change'] = (df['change'] == 'Ch').astype(int)
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
        
    # 7. Categorical One-Hot Encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"[Preprocessing] Encoding categorical columns: {cat_cols}")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop(columns=['target'])
    y = df_encoded['target']
    
    # 8. Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 9. Scaling Numeric Features
    numeric_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'age_num'
    ]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    feature_names = X_train.columns.tolist()
    
    # Save preprocessor artifacts
    preprocessor = {
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'feature_names': feature_names,
        'age_map': AGE_MAP
    }
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    outputs_dir = os.path.join(os.path.dirname(MODELS_DIR), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    with open(os.path.join(outputs_dir, "feature_list.json"), "w") as f:
        json.dump({"feature_count": len(feature_names), "features": feature_names}, f, indent=2)
    print(f"[Preprocessing] Complete. Features: {len(feature_names)}. Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_names
