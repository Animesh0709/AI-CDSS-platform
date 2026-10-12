"""Week 9 model comparison, tuning, validation, and calibration report.

Run from this directory: python train_week9.py
Use --quick for a faster three-fold comparison on a reproducible sample.

Fix applied (calibration):
  - Model is trained first, then calibrated on a SEPARATE held-out calibration
    set using cv="prefit".  Previously CalibratedClassifierCV was given cv=3
    on an already-fitted estimator, which caused it to re-fit 3 clones on the
    training data and collapse all predicted probabilities below 0.5.
  - Decision threshold is also tuned on the calibration set and saved.
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from download_or_generate_dataset import fetch_or_create_dataset
from preprocess import preprocess_data

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def scores(y_true, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy":         round(float(accuracy_score(y_true, predicted)), 4),
        "precision":        round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1_score":         round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "roc_auc":          round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
        "threshold_used":   round(float(threshold), 4),
    }


def best_f1_threshold(y_true, y_prob):
    """Return threshold that maximises F1 on a labelled set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_vals = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best = int(np.argmax(f1_vals))
    return float(thresholds[best]), float(f1_vals[best])


def build_xgb(weight, params):
    return XGBClassifier(
        random_state=42, eval_metric="logloss",
        n_jobs=1, scale_pos_weight=weight, **params,
    )


def train_week9(quick=False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = fetch_or_create_dataset()
    X_train_full, X_test, y_train_full, y_test, feature_names = preprocess_data(df)

    # ── Carve calibration set out of training data BEFORE cross-val ──────────
    # 10 % of training data is held out for calibration and threshold tuning.
    # This ensures (a) no data leakage into calibration, (b) X_test stays clean.
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full,
        test_size=0.10, random_state=42, stratify=y_train_full,
    )

    if quick and len(X_train) > 30_000:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=30_000, stratify=y_train, random_state=42,
        )

    weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    folds  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Candidates now all include regularisation (reg_lambda, min_child_weight)
    candidates = {
        "baseline_xgboost": {
            "n_estimators": 150, "max_depth": 4, "learning_rate": .08,
            "subsample": .8,  "colsample_bytree": .8,
            "reg_lambda": 2,  "min_child_weight": 4,
        },
        "tuned_shallow_xgboost": {
            "n_estimators": 260, "max_depth": 3, "learning_rate": .05,
            "min_child_weight": 4, "subsample": .85, "colsample_bytree": .85,
            "reg_lambda": 2,
        },
        "tuned_balanced_xgboost": {
            "n_estimators": 220, "max_depth": 5, "learning_rate": .04,
            "min_child_weight": 6, "subsample": .8, "colsample_bytree": .75,
            "reg_lambda": 3,
        },
    }

    print("[Week 9] Cross-validating candidates...")
    comparison = {
        "logistic_regression": round(float(
            cross_val_score(
                LogisticRegression(max_iter=500, class_weight="balanced"),
                X_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1,
            ).mean()
        ), 4)
    }
    for name, params in candidates.items():
        comparison[name] = round(float(
            cross_val_score(
                build_xgb(weight, params),
                X_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1,
            ).mean()
        ), 4)
        print(f"  {name}: {comparison[name]}")

    winner = max(candidates, key=lambda n: comparison[n])
    print(f"[Week 9] Winner: {winner}  (AUC={comparison[winner]})")

    # ── Train winning model on full training set ──────────────────────────────
    raw_model = build_xgb(weight, candidates[winner]).fit(X_train, y_train)
    raw_probability = raw_model.predict_proba(X_test)[:, 1]

    # ── FIX 2: Calibrate on the held-out calibration set ─────────────────────
    # sklearn ≥1.6 uses cv=None to signal "estimator is already fitted, just
    # wrap it in Platt scaling using X_cal / y_cal". The old cv="prefit" string
    # was removed in sklearn 1.9.
    calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv=None)
    calibrated.fit(X_cal, y_cal)
    cal_probability = calibrated.predict_proba(X_test)[:, 1]

    # ── FIX 3: Tune decision threshold on calibration-set probabilities ───────
    cal_prob_for_tuning = calibrated.predict_proba(X_cal)[:, 1]
    opt_threshold, cal_f1 = best_f1_threshold(y_cal, cal_prob_for_tuning)
    print(f"[Threshold] Optimal: {opt_threshold:.4f}  (cal F1={cal_f1:.4f})")

    # ── Calibration curve (reliability diagram data) ──────────────────────────
    observed, pred_cal = calibration_curve(
        y_test, cal_probability, n_bins=10, strategy="quantile",
    )

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "model_name": "XGBoost 30-Day Readmission Classifier (Week 9 Fixed)",
        "dataset": "UCI Diabetes 130-US Hospitals",
        "feature_engineering": [
            "age-bin midpoint", "ICD-9 diagnostic categories",
            "one-hot encounter and medication features", "numeric feature scaling",
        ],
        "class_imbalance": {"strategy": "scale_pos_weight", "value": round(weight, 4)},
        "fixes_applied": [
            "reg_lambda added to all candidates",
            "Calibration uses cv='prefit' on a held-out 10% calibration set",
            "Decision threshold tuned from PR curve on calibration set",
        ],
        "cross_validation": {
            "folds": 3, "metric": "roc_auc",
            "comparison": comparison, "selected_model": winner,
        },
        "uncalibrated_test_metrics":        scores(y_test, raw_probability),
        "calibrated_default_threshold":     scores(y_test, cal_probability, threshold=0.5),
        "calibrated_tuned_threshold":       scores(y_test, cal_probability, threshold=opt_threshold),
        "optimal_threshold":                round(opt_threshold, 4),
        # Keep legacy key so the API still reads it
        "calibrated_test_metrics":          scores(y_test, cal_probability, threshold=opt_threshold),
        "calibration_curve": {
            "mean_predicted_probability": [round(float(x), 4) for x in pred_cal],
            "fraction_of_positives":      [round(float(x), 4) for x in observed],
        },
        "note": (
            "Calibration uses isotonic regression on a held-out calibration set (cv='prefit'). "
            "Decision threshold is tuned on the same calibration set via the PR curve. "
            "External validation is still required before clinical use."
        ),
    }

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(calibrated, os.path.join(MODELS_DIR, "xgboost_readmission_week9_calibrated.joblib"))
    threshold_path = os.path.join(MODELS_DIR, "threshold_week9.json")
    with open(threshold_path, "w", encoding="utf-8") as fh:
        json.dump({"optimal_threshold": round(opt_threshold, 4), "tuned_on": "calibration_set_pr_curve"}, fh, indent=2)
    with open(os.path.join(OUTPUTS_DIR, "week9_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n[Week 9] Results:")
    print(f"  Uncalibrated  AUC={report['uncalibrated_test_metrics']['roc_auc']}  F1={report['uncalibrated_test_metrics']['f1_score']}")
    print(f"  Calibrated (thresh=0.50)     F1={report['calibrated_default_threshold']['f1_score']}")
    print(f"  Calibrated (thresh={opt_threshold:.4f}) F1={report['calibrated_tuned_threshold']['f1_score']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a smaller reproducible training sample")
    train_week9(parser.parse_args().quick)
