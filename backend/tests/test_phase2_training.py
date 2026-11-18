"""
Tests for Phase 2 Real-Data EWS Pipeline.
=========================================
Verifies:
1. PhysioNet 2019 PSV parsing and column mapping.
2. Leakage-free observation windowing (anti-leakage guarantees).
3. 6-hour prediction horizon target labeling.
4. Patient-level split isolation (no cross-split patient overlap).
5. Comprehensive evaluation metrics handling (AUROC, AUPRC, Brier score, edge cases).
6. Model training pipeline on synthetic test cohorts (labeled explicitly as synthetic).
7. Feature schema consistency with production `_extract_window_features`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# Ensure backend and ml/scripts paths are on sys.path
_repo_root = Path(__file__).resolve().parents[2]
_backend_dir = os.path.join(str(_repo_root), "backend")
_scripts_dir = os.path.join(str(_repo_root), "ml", "scripts")
for p in (_backend_dir, _scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.early_warning import _extract_window_features
from dataset_loader import (
    PHYSIONET_VITAL_MAP,
    generate_patient_windows,
    parse_psv_patient,
    split_patients_by_id,
)
from evaluation import evaluate_ews_predictions
from train_ews_pipeline import build_feature_matrix, train_models_pipeline


# ─── 1. PSV Parser & Vital Mapping Tests ─────────────────────────────────────

def test_parse_psv_patient_synthetic_file():
    """Verify PSV parser reads headers, extracts vitals and timestamps correctly."""
    # Synthetic test fixture labeled as SYNTHETIC with 40 pipe-separated columns matching PhysioNet 2019
    columns = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_total",
        "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_direct",
        "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"
    ]
    header_line = "|".join(columns)
    
    def make_row(hr, spo2, temp, sbp, dbp, resp, iculos, label):
        row_dict = {c: "" for c in columns}
        row_dict.update({
            "HR": str(hr),
            "O2Sat": str(spo2),
            "Temp": str(temp),
            "SBP": str(sbp),
            "DBP": str(dbp),
            "Resp": str(resp),
            "Age": "65",
            "Gender": "1",
            "ICULOS": str(iculos),
            "SepsisLabel": str(label),
        })
        return "|".join(row_dict[c] for c in columns)

    psv_lines = [
        header_line,
        make_row(82, 98, 37.1, 122, 68, 16, 1, 0),
        make_row(88, 97, 37.3, 118, 65, 18, 2, 0),
        make_row(105, 93, 38.6, 95, 50, 24, 3, 1),
    ]
    psv_content = "\n".join(psv_lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".psv", delete=False) as tmp:
        tmp.write(psv_content)
        tmp_path = tmp.name

    try:
        pid, vitals, labels = parse_psv_patient(tmp_path)
        assert len(vitals) == 3
        assert len(labels) == 3
        assert labels == [0, 0, 1]

        # Verify mapped fields
        v0 = vitals[0]
        assert v0["heart_rate"] == 82.0
        assert v0["spo2"] == 98.0
        assert v0["temperature_c"] == 37.1
        assert v0["systolic_bp"] == 122.0
        assert v0["diastolic_bp"] == 68.0
        assert v0["respiratory_rate"] == 16.0
        assert "news2_score" in v0
        assert isinstance(v0["recorded_at"], datetime)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── 2. Window Generation & Anti-Leakage Tests ───────────────────────────────

def test_generate_patient_windows_prevents_future_leakage():
    """Verify that features at step i never see observations from step > i."""
    base_t = datetime(2026, 1, 1, 0, 0, 0)
    # Synthetic patient with 5 readings; vital spikes dramatically at step 4
    vitals = [
        {"recorded_at": base_t + timedelta(hours=i), "heart_rate": 70.0 + i}
        for i in range(4)
    ]
    # At step 4, a huge spike happens
    vitals.append({"recorded_at": base_t + timedelta(hours=4), "heart_rate": 200.0})
    labels = [0, 0, 0, 0, 0]

    samples, y_out, _ = generate_patient_windows(vitals, labels, window_hours=6.0, min_readings=2)

    # Sample at step 2 (i=2, count=3 readings: hr 70, 71, 72)
    # Must NOT have been influenced by step 4's spike (heart_rate = 200.0)
    assert len(samples) > 0
    sample_step_2 = samples[1]  # min_readings=2 skips i=0, so index 0 is i=1, index 1 is i=2
    assert sample_step_2["heart_rate_last"] == 72.0
    assert sample_step_2["heart_rate_mean"] < 75.0


def test_target_labeling_6h_horizon_comprehensive_boundaries():
    """
    Verify Target B definition: Predict clinical sepsis onset within the next 6 hours.
    
    PhysioNet 2019 shifts SepsisLabel by 6 hours:
        raw_sepsis_label[t] = 1  if t >= t_sepsis - 6
        raw_sepsis_label[t] = 0  if t < t_sepsis - 6
        
    Example:
    Suppose clinical sepsis occurs at t = 10 (t_sepsis = 10).
    Then raw SepsisLabel transitions from 0 to 1 at t = 4 (10 - 6).
    
    Hour-by-hour expected Target B (sepsis within next 6 hours?):
    - t = 0..3: Clinical onset is at t=10. Time to onset is 10 - t > 6 hours.
      Target B must be 0 (raw_sepsis_label is 0).
    - t = 4: Exactly 6 hours before clinical onset (10 - 4 = 6).
      Target B must be 1 (raw_sepsis_label is 1).
    - t = 5..10: Time to clinical onset is <= 5 hours.
      Target B must be 1 (raw_sepsis_label is 1).
    - t = 11: Post-onset (clinical sepsis already occurred).
      Must be excluded when include_post_onset=False.
    """
    base_t = datetime(2026, 1, 1, 0, 0, 0)
    total_hours = 14
    vitals = [
        {"recorded_at": base_t + timedelta(hours=i), "heart_rate": 75.0, "systolic_bp": 120.0}
        for i in range(total_hours)
    ]
    # SepsisLabel turns 1 at index 4 (representing t_sepsis = 10)
    raw_labels = [0]*4 + [1]*10

    # 1. Default: exclude post-onset (stops after t_clinical_onset = index 10)
    samples, targets, timestamps = generate_patient_windows(
        vitals, raw_labels, window_hours=6.0, min_readings=1, include_post_onset=False
    )

    # Evaluates hours 0 through 10 (total 11 steps)
    assert len(targets) == 11
    # Hours 0..3: > 6 hours before clinical onset -> 0
    assert targets[0:4] == [0, 0, 0, 0]
    # Hour 4: exactly 6 hours before clinical onset -> 1
    assert targets[4] == 1
    # Hours 5..10: within 6 hours before clinical onset -> 1
    assert targets[5:11] == [1, 1, 1, 1, 1, 1]
    # No post-onset samples (hour 11+) included
    assert len(targets) == 11

    # 2. Test patient who never develops sepsis
    non_septic_labels = [0] * total_hours
    samples_clean, targets_clean, _ = generate_patient_windows(
        vitals, non_septic_labels, window_hours=6.0, min_readings=1
    )
    assert len(targets_clean) == total_hours
    assert all(t == 0 for t in targets_clean)


# ─── 3. Patient-Level Splitting Tests ─────────────────────────────────────────

def test_split_patients_by_id_no_overlap():
    """Verify strictly zero patient overlap between train, validation, and test splits."""
    patient_ids = [f"patient_{i:04d}" for i in range(100)]
    train_ids, val_ids, test_ids = split_patients_by_id(
        patient_ids, test_size=0.2, val_size=0.1, random_seed=42
    )

    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    # Disjoint sets
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0

    # Exhaustive coverage
    assert len(set_train) + len(set_val) + len(set_test) == 100
    assert set_train.union(set_val).union(set_test) == set(patient_ids)


# ─── 4. Evaluation Metrics Tests ─────────────────────────────────────────────

def test_evaluation_metrics_normal():
    """Verify AUROC, AUPRC, Sensitivity, Specificity, Brier Score, Baselines, and Threshold Grids."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    metrics = evaluate_ews_predictions(y_true, y_prob, operating_threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert metrics["confusion_matrix"]["true_positive"] == 4
    assert metrics["confusion_matrix"]["true_negative"] == 4

    # Baselines
    assert "baselines" in metrics
    assert metrics["baselines"]["all_negative_brier_score"] == 0.5  # mean((0 - [0,0,0,0,1,1,1,1])^2) = 4/8 = 0.5
    assert metrics["baselines"]["random_auprc"] == 0.5

    # Multi-threshold evaluations
    assert "threshold_evaluations" in metrics
    assert len(metrics["threshold_evaluations"]) > 0
    t05 = next(item for item in metrics["threshold_evaluations"] if item["threshold"] == 0.5)
    assert t05["sensitivity"] == 1.0
    assert t05["specificity"] == 1.0


def test_recent_dev_clipping_and_no_zero_division():
    """Verify recent_dev handles constant baselines without producing extreme infinity/spikes."""
    from datetime import datetime, timedelta
    base_t = datetime(2026, 1, 1, 0, 0, 0)
    # Three identical measurements followed by a 1-unit step
    constant_history = [
        {"recorded_at": base_t, "heart_rate": 80.0},
        {"recorded_at": base_t + timedelta(hours=1), "heart_rate": 80.0},
        {"recorded_at": base_t + timedelta(hours=2), "heart_rate": 81.0},
    ]
    feats = _extract_window_features(constant_history)
    # With b_std floor of 0.5, recent_dev should be (81 - 80) / 0.5 = 2.0 (not 1,000,000)
    assert feats["heart_rate_recent_dev"] == pytest.approx(2.0, abs=0.1)
    assert -10.0 <= feats["heart_rate_recent_dev"] <= 10.0


def test_evaluation_metrics_single_class_edge_case():
    """Verify graceful handling when y_true contains only negative samples (no crashes)."""
    y_true = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.1, 0.3]

    metrics = evaluate_ews_predictions(y_true, y_prob)
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert "note" in metrics


# ─── 5. Training Pipeline on Synthetic Cohort ─────────────────────────────────

def test_train_models_pipeline_synthetic_cohort():
    """Verify train_models_pipeline trains both LR baseline and XGBoost on a synthetic test cohort."""
    # Synthetic cohort fixture explicitly labeled as SYNTHETIC
    rng = np.random.RandomState(42)
    feature_template = list(_extract_window_features([{"heart_rate": 80.0}]).keys())

    def make_synthetic_samples(n, label):
        samples = []
        for _ in range(n):
            s = {}
            for k in feature_template:
                # Give subtle signal difference for synthetic testing
                shift = 1.0 if (label == 1 and "mean" in k) else 0.0
                s[k] = float(rng.normal(loc=0.0 + shift, scale=1.0))
            samples.append(s)
        return samples

    train_samples = make_synthetic_samples(30, 0) + make_synthetic_samples(30, 1)
    train_labels = [0] * 30 + [1] * 30

    val_samples = make_synthetic_samples(10, 0) + make_synthetic_samples(10, 1)
    val_labels = [0] * 10 + [1] * 10

    test_samples = make_synthetic_samples(10, 0) + make_synthetic_samples(10, 1)
    test_labels = [0] * 10 + [1] * 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = train_models_pipeline(
            train_samples=train_samples,
            train_labels=train_labels,
            val_samples=val_samples,
            val_labels=val_labels,
            test_samples=test_samples,
            test_labels=test_labels,
            feature_names=feature_template,
            output_dir=Path(tmp_dir),
            save_artifacts=True,
            artifact_prefix="test_run",
        )

        assert "baseline_logistic_regression" in results
        assert "xgboost_model" in results
        assert results["n_features"] == len(feature_template)
        # Verify backwards-compatible eval_metrics key is present and contains valid metrics
        assert "eval_metrics" in results["baseline_logistic_regression"]
        assert "eval_metrics" in results["xgboost_model"]
        assert results["baseline_logistic_regression"]["eval_metrics"]["auroc"] is not None
        assert results["xgboost_model"]["eval_metrics"]["auroc"] is not None
        assert results["xgboost_model"]["eval_metrics"]["auprc"] is not None

        # Verify threshold-specific keys
        assert results["baseline_logistic_regression"]["eval_metrics_default_threshold_0_5"]["auroc"] is not None
        assert results["xgboost_model"]["eval_metrics_default_threshold_0_5"]["auroc"] is not None
        assert "all_negative_brier_score" in results["xgboost_model"]["eval_metrics_default_threshold_0_5"]["baselines"]
        assert "threshold_evaluations" in results["xgboost_model"]["eval_metrics_default_threshold_0_5"]
        assert results["baseline_logistic_regression"]["selected_threshold"] is not None
        assert results["xgboost_model"]["selected_threshold"] is not None

        # Verify candidate artifact was written without affecting production model
        saved_pkl = results["saved_model_artifact"]
        assert os.path.exists(saved_pkl)
        assert "test_run_ews_xgboost.pkl" in saved_pkl

        # Verify candidate evaluation JSON report structure
        saved_json = results["saved_eval_report"]
        assert os.path.exists(saved_json)
        import json
        with open(saved_json) as f:
            report_data = json.load(f)
        assert "baseline_logistic_regression" in report_data
        assert "eval_metrics" in report_data["baseline_logistic_regression"]
        assert "xgboost_model" in report_data
        assert "eval_metrics" in report_data["xgboost_model"]
