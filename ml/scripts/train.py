"""
XGBoost 30-Day Hospital Readmission Model Training Pipeline (Week 1 Baseline)

Fixes applied:
  - reg_lambda=2, min_child_weight=4  : L2 regularisation to tighten leaf weights
  - early_stopping_rounds=20          : stops at optimal number of trees using a
                                        held-out 10 % validation set (no data leak)
  - Optimal decision threshold        : tuned from precision-recall curve on the
                                        validation set, saved alongside the model
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
import joblib

from download_or_generate_dataset import fetch_or_create_dataset
from preprocess import preprocess_data

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def best_f1_threshold(y_true, y_prob):
    """Return the probability threshold that maximises F1 on a validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # avoid div-by-zero for the final sentinel point where threshold is undefined
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_readmission_model():
    print("=" * 60)
    print("AI-CDSS: Training XGBoost 30-Day Readmission Model")
    print("=" * 60)

    # 1. Fetch / generate dataset
    df = fetch_or_create_dataset()

    # 2. Preprocess  (returns stratified 80/20 train/test split)
    X_train_full, X_test, y_train_full, y_test, feature_names = preprocess_data(df)

    # 3. Carve out a 10 % validation set from the training pool for early stopping
    #    and threshold tuning — this keeps X_test completely held-out.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.10, random_state=42, stratify=y_train_full,
    )

    # 4. Class-balance weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"[Training] Non-Readmitted: {neg_count}, Readmitted: {pos_count}")
    print(f"[Training] scale_pos_weight: {scale_pos_weight:.2f}")

    # 5. Model — now with L2 regularisation and min_child_weight
    model = XGBClassifier(
        n_estimators=500,           # high ceiling; early stopping will cut this down
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,               # FIX 1: L2 leaf-weight regularisation
        min_child_weight=4,         # FIX 1: minimum sum of instance weight per leaf
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,   # FIX 1: stop when val loss stops improving
    )

    print("[Training] Fitting XGBoost (early stopping on validation logloss)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # FIX 1: dedicated validation set
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"[Training] Best iteration: {best_iter} (out of 500 ceiling)")

    # 6. Tune decision threshold on the validation set (FIX 3)
    val_prob = model.predict_proba(X_val)[:, 1]
    opt_threshold, val_f1 = best_f1_threshold(y_val, val_prob)
    print(f"[Threshold] Optimal threshold: {opt_threshold:.4f}  (val F1 = {val_f1:.4f})")

    # 7. Evaluate on the held-out test set with BOTH thresholds so results are comparable
    y_prob = model.predict_proba(X_test)[:, 1]

    def metric_block(y_true, y_pred_binary, y_prob_scores):
        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred_binary)), 4),
            "precision": round(float(precision_score(y_true, y_pred_binary, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred_binary, zero_division=0)), 4),
            "f1_score":  round(float(f1_score(y_true, y_pred_binary, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_true, y_prob_scores)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred_binary).tolist(),
        }

    metrics_default   = metric_block(y_test, (y_prob >= 0.5).astype(int),          y_prob)
    metrics_tuned     = metric_block(y_test, (y_prob >= opt_threshold).astype(int), y_prob)

    # 8. Feature importances
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in top_indices
    ]

    metrics = {
        "model_name": "XGBoost 30-Day Readmission Classifier (Fixed)",
        "dataset": "UCI Diabetes 130-US Hospitals",
        "total_samples": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "num_features": len(feature_names),
        "best_iteration": int(best_iter),
        "fixes_applied": [
            "reg_lambda=2 (L2 regularisation)",
            "min_child_weight=4",
            "early_stopping_rounds=20 on held-out validation set",
            "decision threshold tuned from PR curve on validation set",
        ],
        "decision_threshold": {
            "default_0_5": metrics_default,
            "tuned": metrics_tuned,
            "optimal_threshold": round(opt_threshold, 4),
        },
        # Keep backward-compat key pointing at tuned metrics
        "evaluation_metrics": metrics_tuned,
        "confusion_matrix": metrics_tuned["confusion_matrix"],
        "top_10_features": top_features,
    }

    # 9. Save model + threshold
    model_path     = os.path.join(MODELS_DIR, "xgboost_readmission.joblib")
    threshold_path = os.path.join(MODELS_DIR, "threshold.json")
    metrics_path   = os.path.join(OUTPUTS_DIR, "metrics.json")

    joblib.dump(model, model_path)
    with open(threshold_path, "w") as f:
        json.dump({"optimal_threshold": round(opt_threshold, 4), "tuned_on": "validation_set_pr_curve"}, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION SUMMARY:")
    print(f"  * Model saved  : {model_path}")
    print(f"  * Threshold    : {threshold_path}  ({opt_threshold:.4f})")
    print(f"  * Metrics      : {metrics_path}")
    print()
    print("  -- Threshold = 0.50 (default) --------------------")
    print(f"     ROC-AUC  : {metrics_default['roc_auc']}")
    print(f"     F1-Score : {metrics_default['f1_score']}")
    print(f"     Recall   : {metrics_default['recall']}")
    print()
    print(f"  -- Threshold = {opt_threshold:.4f} (tuned) -------------------")
    print(f"     ROC-AUC  : {metrics_tuned['roc_auc']}")
    print(f"     F1-Score : {metrics_tuned['f1_score']}")
    print(f"     Recall   : {metrics_tuned['recall']}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train_readmission_model()
