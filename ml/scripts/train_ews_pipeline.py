"""
Early Warning System — Real-Data Training Pipeline (Phase 2).
=============================================================
Supports:
1. Loading PhysioNet 2019 .psv dataset (or preprocessed patient vitals).
2. Leakage-safe sliding observation windows using production `_extract_window_features`.
3. Patient-level train / val / test splitting.
4. Training a Logistic Regression baseline and an XGBoost deterioration risk model.
5. Calibrating probabilities via CalibratedClassifierCV.
6. Recording complete metadata: feature schema, metrics (AUROC, AUPRC, Sensitivity,
   Specificity, Brier score), versions, and preprocessing parameters.
7. Safe saving: preserves current model unless explicitly directed to replace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Add backend directory to sys.path
_repo_root = Path(__file__).resolve().parents[2]
_backend_dir = os.path.join(str(_repo_root), "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Add ml/scripts directory to sys.path
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from app.services.early_warning import _extract_window_features
from dataset_loader import generate_patient_windows, parse_psv_patient, split_patients_by_id
from evaluation import evaluate_ews_predictions, find_optimal_threshold

logger = logging.getLogger(__name__)


def build_feature_matrix(
    samples: List[Dict[str, float]],
    ordered_feature_names: List[str],
) -> np.ndarray:
    """Convert list of feature dicts into a 2D numpy array strictly matching ordered_feature_names."""
    matrix = np.zeros((len(samples), len(ordered_feature_names)), dtype=np.float32)
    for i, s in enumerate(samples):
        for j, feat_name in enumerate(ordered_feature_names):
            matrix[i, j] = float(s.get(feat_name, 0.0))
    return matrix


def train_models_pipeline(
    train_samples: List[Dict[str, float]],
    train_labels: List[int],
    val_samples: List[Dict[str, float]],
    val_labels: List[int],
    test_samples: List[Dict[str, float]],
    test_labels: List[int],
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = False,
    artifact_prefix: str = "phase2_candidate",
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression and XGBoost classifiers.
    
    Parameters
    ----------
    train_samples, train_labels : Training dataset
    val_samples, val_labels     : Validation dataset
    test_samples, test_labels   : Held-out test dataset
    feature_names               : Canonical feature schema. If None, derived from first sample.
    output_dir                  : Directory to save candidate artifacts
    save_artifacts              : If True, saves candidate model pickles and eval reports
    artifact_prefix             : Filename prefix (never overwrites ews_xgboost.pkl directly)
    """
    if feature_names is None:
        if len(train_samples) > 0:
            feature_names = list(train_samples[0].keys())
        else:
            raise ValueError("No training samples provided to determine feature schema.")

    X_train = build_feature_matrix(train_samples, feature_names)
    y_train = np.array(train_labels, dtype=int)

    X_val = build_feature_matrix(val_samples, feature_names)
    y_val = np.array(val_labels, dtype=int)

    X_test = build_feature_matrix(test_samples, feature_names)
    y_test = np.array(test_labels, dtype=int)

    # Impute any residual non-finite values (safety guarantee)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Baseline: Logistic Regression (Regularized with C=0.01 to prevent collinear slope divergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_train_scaled
    X_test_scaled = scaler.transform(X_test)

    # Standard regularized baseline
    lr_model = LogisticRegression(
        C=0.01,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_probs_val = lr_model.predict_proba(X_val_scaled)[:, 1] if len(X_val) > 0 else None
    lr_probs_test = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Select operating threshold strictly on validation data
    lr_opt_thresh = find_optimal_threshold(y_val, lr_probs_val, metric="f1") if lr_probs_val is not None else 0.5
    lr_eval_default = evaluate_ews_predictions(y_test, lr_probs_test, operating_threshold=0.5)
    lr_eval_optimal = evaluate_ews_predictions(y_test, lr_probs_test, operating_threshold=lr_opt_thresh)

    # 2. Main Candidate: XGBoost Classifier
    from xgboost import XGBClassifier

    # Calculate class balance weight
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos = (neg_count / pos_count) if pos_count > 0 else 1.0

    xgb_raw = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_raw.fit(X_train, y_train)

    # Calibrate probability predictions using cross-validation or validation set
    final_model = xgb_raw
    is_calibrated = False

    min_class_count = min(pos_count, neg_count)
    if min_class_count >= 5:
        try:
            cv_folds = min(3, min_class_count)
            calibrated_xgb = CalibratedClassifierCV(
                estimator=xgb_raw,
                method="sigmoid",
                cv=cv_folds,
            )
            calibrated_xgb.fit(X_train, y_train)
            final_model = calibrated_xgb
            is_calibrated = True
        except Exception as exc:
            logger.warning("Calibration fallback to raw XGBoost: %s", exc)
            final_model = xgb_raw

    xgb_probs_val = final_model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else None
    xgb_probs_test = final_model.predict_proba(X_test)[:, 1]

    # Select operating threshold strictly on validation data
    xgb_opt_thresh = find_optimal_threshold(y_val, xgb_probs_val, metric="f1") if xgb_probs_val is not None else 0.5
    xgb_eval_default = evaluate_ews_predictions(y_test, xgb_probs_test, operating_threshold=0.5)
    xgb_eval_optimal = evaluate_ews_predictions(y_test, xgb_probs_test, operating_threshold=xgb_opt_thresh)

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "baseline_logistic_regression": {
            "eval_metrics": lr_eval_optimal,
            "eval_metrics_default_threshold_0_5": lr_eval_default,
            "eval_metrics_val_selected_threshold": lr_eval_optimal,
            "selected_threshold": round(lr_opt_thresh, 4),
        },
        "xgboost_model": {
            "eval_metrics": xgb_eval_optimal,
            "eval_metrics_default_threshold_0_5": xgb_eval_default,
            "eval_metrics_val_selected_threshold": xgb_eval_optimal,
            "selected_threshold": round(xgb_opt_thresh, 4),
            "calibrated": is_calibrated,
        },
    }

    if save_artifacts and output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": final_model,
            "feature_names": feature_names,
            "scaler": scaler if isinstance(final_model, LogisticRegression) else None,
            "eval_metrics": xgb_eval_optimal,
            "eval_metrics_default_threshold": xgb_eval_default,
            "model_version": "phase2_v1_physionet_compatible",
            "training_timestamp": results["timestamp"],
            "preprocessing_config": {
                "window_hours": 6.0,
                "min_readings": 2,
                "prediction_horizon_hours": 6,
            },
        }

        # Save to candidate path — NEVER overwrite ews_xgboost.pkl directly
        artifact_path = out_dir / f"{artifact_prefix}_ews_xgboost.pkl"
        with open(artifact_path, "wb") as fh:
            pickle.dump(bundle, fh)
        results["saved_model_artifact"] = str(artifact_path)

        eval_path = out_dir / f"{artifact_prefix}_eval_report.json"
        with open(eval_path, "w") as fh:
            json.dump(results, fh, indent=2)
        results["saved_eval_report"] = str(eval_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2 EWS Model Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="ml/data/physionet_2019", help="Directory of .psv files")
    parser.add_argument("--output_dir", type=str, default="ml/models", help="Output directory for candidate artifacts")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit number of patients for quick testing")
    parser.add_argument("--save", action="store_true", help="Save candidate artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory '{data_dir}' not found.")
        print("Note: In Phase 2, dataset download is manual or via dedicated download script.")
        return

    psv_files = sorted(list(data_dir.glob("*.psv")))
    if not psv_files:
        print(f"No .psv files found in '{data_dir}'.")
        return

    if args.max_patients:
        psv_files = psv_files[: args.max_patients]

    patient_ids = [f.stem for f in psv_files]
    train_ids, val_ids, test_ids = split_patients_by_id(patient_ids)

    print(f"Loaded {len(psv_files)} patient files.")
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test.")

    # Process files into windowed datasets
    train_samples, train_labels = [], []
    val_samples, val_labels = [], []
    test_samples, test_labels = [], []

    for f in psv_files:
        pid, vitals, sepsis_labels = parse_psv_patient(f)
        samples, labels, _ = generate_patient_windows(vitals, sepsis_labels)
        if pid in train_ids:
            train_samples.extend(samples)
            train_labels.extend(labels)
        elif pid in val_ids:
            val_samples.extend(samples)
            val_labels.extend(labels)
        else:
            test_samples.extend(samples)
            test_labels.extend(labels)

    print(f"Windows extracted: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test.")

    results = train_models_pipeline(
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        test_samples=test_samples,
        test_labels=test_labels,
        output_dir=Path(args.output_dir),
        save_artifacts=args.save,
    )
    print("\nTraining complete.")
    lr_res = results.get("baseline_logistic_regression", {})
    xgb_res = results.get("xgboost_model", {})

    lr_metrics = lr_res.get("eval_metrics") or lr_res.get("eval_metrics_val_selected_threshold") or lr_res.get("eval_metrics_default_threshold_0_5") or {}
    xgb_metrics = xgb_res.get("eval_metrics") or xgb_res.get("eval_metrics_val_selected_threshold") or xgb_res.get("eval_metrics_default_threshold_0_5") or {}

    lr_auroc = lr_metrics.get("auroc")
    xgb_auroc = xgb_metrics.get("auroc")
    xgb_auprc = xgb_metrics.get("auprc")
    xgb_brier = xgb_metrics.get("brier_score")
    xgb_thresh = xgb_res.get("selected_threshold", 0.5)

    print(f"Baseline Logistic Regression AUROC: {f'{lr_auroc:.4f}' if isinstance(lr_auroc, (int, float)) else lr_auroc}")
    print(f"XGBoost AUROC: {f'{xgb_auroc:.4f}' if isinstance(xgb_auroc, (int, float)) else xgb_auroc}")
    print(f"XGBoost AUPRC: {f'{xgb_auprc:.4f}' if isinstance(xgb_auprc, (int, float)) else xgb_auprc}")
    print(f"XGBoost Brier Score: {f'{xgb_brier:.4f}' if isinstance(xgb_brier, (int, float)) else xgb_brier}")
    print(f"Selected Operating Threshold: {xgb_thresh}")
    if "saved_model_artifact" in results:
        print(f"Saved Candidate Model: {results['saved_model_artifact']}")
    if "saved_eval_report" in results:
        print(f"Saved Evaluation Report: {results['saved_eval_report']}")


if __name__ == "__main__":
    main()
