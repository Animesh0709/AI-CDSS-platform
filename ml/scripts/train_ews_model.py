"""
Train the Early Warning System XGBoost model.

Usage
-----
    cd <repo_root>
    python ml/scripts/train_ews_model.py

Generates
---------
    ml/models/ews_xgboost.pkl   — pickled dict {model, feature_names, label_encoder}
    ml/outputs/ews_eval.json    — hold-out evaluation report
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Paths and Backend Service Import
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]
_MODEL_DIR = _BASE_DIR / "models"
_OUTPUT_DIR = _BASE_DIR / "outputs"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add backend to sys.path so we reuse the production feature extraction & rules
_repo_root = Path(__file__).resolve().parents[2]
_backend_dir = os.path.join(os.path.dirname(_repo_root), "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.services.early_warning import _extract_window_features, compute_news2


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# Simulate patients with known risk levels based on NEWS2 components.
# ─────────────────────────────────────────────────────────────────────────────

random.seed(42)
np.random.seed(42)


def _sample_vitals(risk_class: int) -> list[dict]:
    """
    Generate a sequence of 6 vital-sign readings for a synthetic patient with timestamps.
    risk_class: 0 = Low/Medium (NEWS2 < 5), 1 = High/Critical (NEWS2 >= 5)
    """
    n = random.randint(3, 6)
    readings = []
    base_time = datetime(2026, 1, 1, 8, 0, 0)

    for i in range(n):
        rec_time = base_time + timedelta(hours=i * 2)
        if risk_class == 0:
            hr = np.random.normal(78, 12)
            rr = np.random.normal(16, 2)
            spo2 = np.random.normal(97, 1.5)
            temp = np.random.normal(37.0, 0.4)
            sbp = np.random.normal(125, 15)
            dbp = np.random.normal(80, 10)
        else:
            # Deteriorating — progressively worse over time
            factor = i / (n - 1) if n > 1 else 0.5
            hr = np.random.normal(105 + factor * 25, 10)
            rr = np.random.normal(22 + factor * 6, 2)
            spo2 = np.random.normal(93 - factor * 5, 1.5)
            temp = np.random.normal(38.5 + factor * 0.8, 0.4)
            sbp = np.random.normal(100 - factor * 20, 12)
            dbp = np.random.normal(65 - factor * 10, 8)

        # Compute NEWS2 score for this reading using the shared service
        try:
            news2 = compute_news2({
                "heart_rate": hr, "respiratory_rate": rr, "spo2": spo2,
                "temperature_c": temp, "systolic_bp": sbp,
            })
            news2_score = news2["score"]
        except Exception:
            news2_score = max(0, int((abs(hr - 75) / 15) + (abs(rr - 16) / 4) + max(0, (96 - spo2) / 2)))

        readings.append({
            "recorded_at": rec_time,
            "heart_rate": max(30, hr),
            "respiratory_rate": max(4, rr),
            "spo2": min(100, max(70, spo2)),
            "temperature_c": max(34, min(42, temp)),
            "systolic_bp": max(60, sbp),
            "diastolic_bp": max(40, dbp),
            "news2_score": news2_score,
        })
    return readings


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return

    print("Generating synthetic training data …")
    N_PATIENTS = 2000
    X_rows, y_rows = [], []

    for _ in range(N_PATIENTS):
        risk_class = random.choice([0, 1])
        readings = _sample_vitals(risk_class)
        feats = _extract_window_features(readings, window_hours=24.0)
        X_rows.append(feats)
        y_rows.append(risk_class)

    feature_names = list(X_rows[0].keys())
    X = np.array([[row[k] for k in feature_names] for row in X_rows])
    y = np.array(y_rows)

    print(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"Class distribution: Low/Medium={sum(y==0)}, High/Critical={sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    print("Training XGBoost classifier …")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nTest AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 important features:")
    for name, imp in top_features:
        print(f"  {name:<35} {imp:.4f}")

    # Save model
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "auc": auc,
        "classification_report": report,
    }
    model_path = _MODEL_DIR / "ews_xgboost.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nModel saved to {model_path}")

    # Save eval report
    eval_report = {
        "auc_roc": round(auc, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": len(feature_names),
        "top_features": [{"name": n, "importance": round(i, 4)} for n, i in top_features],
        "classification_report": report,
        "model_type": "XGBoostClassifier",
        "task": "binary (Low/Medium vs High/Critical deterioration risk)",
        "note": "Trained on synthetic NEWS2-labeled vital-sign time-series. For research/demo only.",
    }
    eval_path = _OUTPUT_DIR / "ews_eval.json"
    with open(eval_path, "w") as fh:
        json.dump(eval_report, fh, indent=2)
    print(f"Evaluation report saved to {eval_path}")


if __name__ == "__main__":
    main()
