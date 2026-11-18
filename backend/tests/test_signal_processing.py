"""
Unit tests for the improved EWS signal-processing and feature-engineering
pipeline (services/early_warning.py — Feature Engineering V4).

Coverage
--------
1.  Empty history → no crash, all zeros / missingness = 1.0
2.  Timestamp-based window excludes old readings
3.  Duplicate timestamps → deduplicated (last entry wins)
4.  Out-of-bounds vital excluded from features, counted in missingness
5.  Vital always absent → missing_frac = 1.0
6.  velocity_last arithmetic correctness
7.  slope_per_hr arithmetic correctness
8.  recent_dev positive when last value exceeds baseline
9.  shock_index computed when HR + SBP both valid
10. shock_index = 0.0 when SBP is missing
11. spo2_rr_ratio computed correctly
12. Future-leakage prevention (t_ref = max ts; older readings dropped)
13. has_sufficient_history = 1.0 (≥ 3 readings over ≥ 2 h)
14. has_sufficient_history = 0.0 (only 2 readings)
15. All original 30 feature names present (backward compat)
16. obs_density_per_hr arithmetic
17. No-timestamp history uses all entries
18. _parse_ts handles None / empty / NaN / ISO-8601 / datetime object
19. _validate_vital accepts valid values, rejects OOB / None / NaN / string
20. news2_velocity equals news2_score_slope_per_hr
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta

import pytest

# Make sure backend package is importable whether pytest is run from
# the repo root or the backend/ directory.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)

from app.services.early_warning import (
    VITAL_FEATURES,
    _empty_features,
    _extract_window_features,
    _linear_slope,
    _parse_ts,
    _time_aware_slope,
    _validate_vital,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_reading(ts: datetime, **vitals) -> dict:
    """Create a minimal vitals dict with a timestamp."""
    return {"recorded_at": ts, **vitals}


BASE = datetime(2024, 1, 10, 8, 0, 0)   # arbitrary fixed reference point


# ─── 1. Empty history ────────────────────────────────────────────────────────

def test_empty_history_returns_zero_dict():
    feats = _extract_window_features([])
    assert feats["n_readings"] == 0.0
    assert feats["has_sufficient_history"] == 0.0
    assert feats["overall_missingness"] == 1.0
    for vk in VITAL_FEATURES:
        assert feats[f"{vk}_missing_frac"] == 1.0, f"{vk}_missing_frac should be 1.0"
        assert feats[f"{vk}_mean"] == 0.0


# ─── 2. Timestamp-based window excludes old readings ─────────────────────────

def test_window_excludes_old_readings():
    t_ref  = BASE
    t_old  = BASE - timedelta(hours=48)  # well outside 24-h window
    t_near = BASE - timedelta(hours=4)

    history = [
        _make_reading(t_old,  heart_rate=999.0),   # out of window (also OOB value)
        _make_reading(t_near, heart_rate=75.0),
        _make_reading(t_ref,  heart_rate=80.0),
    ]
    feats = _extract_window_features(history, window_hours=24.0)

    assert feats["n_readings"] == 2.0, "Old reading must be excluded"
    assert abs(feats["heart_rate_mean"] - 77.5) < 0.01


# ─── 3. Duplicate timestamps → deduplicated (last wins) ──────────────────────

def test_duplicate_timestamps_keep_last():
    t = BASE
    history = [
        _make_reading(t, heart_rate=70.0),
        _make_reading(t, heart_rate=80.0),   # same ts — this should win
    ]
    feats = _extract_window_features(history)

    assert feats["n_readings"] == 1.0, "Duplicate ts must collapse to 1 reading"
    assert feats["heart_rate_last"] == 80.0


# ─── 4. Out-of-bounds vital excluded from features, counted in missingness ────

def test_oob_vital_excluded_counted_as_missing():
    history = [
        _make_reading(BASE - timedelta(hours=2), heart_rate=500.0),  # OOB (> 300)
        _make_reading(BASE,                       heart_rate=75.0),   # valid
    ]
    feats = _extract_window_features(history)

    assert feats["n_readings"] == 2.0           # both entries in window
    assert feats["heart_rate_last"] == 75.0     # only valid value
    assert feats["heart_rate_mean"] == 75.0     # single valid value
    assert feats["heart_rate_missing_frac"] == 0.5  # 1 of 2 slots invalid


# ─── 5. Vital always absent → missing_frac = 1.0 ─────────────────────────────

def test_absent_vital_gives_missing_frac_one():
    history = [
        _make_reading(BASE - timedelta(hours=1), heart_rate=75.0),
        _make_reading(BASE,                       heart_rate=78.0),
    ]
    feats = _extract_window_features(history)

    assert feats["spo2_missing_frac"] == 1.0
    assert feats["spo2_mean"] == 0.0
    assert feats["spo2_last"] == 0.0


# ─── 6. velocity_last arithmetic ─────────────────────────────────────────────

def test_velocity_last_correct():
    # HR rises from 70 → 80 over exactly 2 hours → velocity = 5 bpm/hr
    t1 = BASE
    t2 = BASE + timedelta(hours=2)
    history = [
        _make_reading(t1, heart_rate=70.0),
        _make_reading(t2, heart_rate=80.0),
    ]
    feats = _extract_window_features(history)

    assert abs(feats["heart_rate_velocity_last"] - 5.0) < 0.01, (
        f"Expected 5.0 bpm/hr, got {feats['heart_rate_velocity_last']}"
    )


def test_velocity_last_negative_for_decreasing():
    t1 = BASE
    t2 = BASE + timedelta(hours=1)
    history = [
        _make_reading(t1, heart_rate=100.0),
        _make_reading(t2, heart_rate=80.0),   # −20 over 1 hr = −20 bpm/hr
    ]
    feats = _extract_window_features(history)
    assert feats["heart_rate_velocity_last"] == pytest.approx(-20.0, abs=0.1)


# ─── 7. slope_per_hr arithmetic ──────────────────────────────────────────────

def test_slope_per_hr_correct_linear_series():
    # 3 equally spaced readings, HR 70 → 75 → 80 over 0, 1, 2 hours → slope = 5 /hr
    history = [
        _make_reading(BASE,                        heart_rate=70.0),
        _make_reading(BASE + timedelta(hours=1),   heart_rate=75.0),
        _make_reading(BASE + timedelta(hours=2),   heart_rate=80.0),
    ]
    feats = _extract_window_features(history)
    assert abs(feats["heart_rate_slope_per_hr"] - 5.0) < 0.1


def test_slope_per_hr_zero_for_flat_series():
    history = [
        _make_reading(BASE,                      heart_rate=75.0),
        _make_reading(BASE + timedelta(hours=3), heart_rate=75.0),
    ]
    feats = _extract_window_features(history)
    assert feats["heart_rate_slope_per_hr"] == pytest.approx(0.0, abs=1e-9)


# ─── 8. recent_dev positive when last > baseline ─────────────────────────────

def test_recent_dev_positive_for_spike():
    history = [
        _make_reading(BASE,                      heart_rate=70.0),
        _make_reading(BASE + timedelta(hours=1), heart_rate=72.0),
        _make_reading(BASE + timedelta(hours=2), heart_rate=120.0),  # spike
    ]
    feats = _extract_window_features(history)
    assert feats["heart_rate_recent_dev"] > 0


def test_recent_dev_zero_for_single_reading():
    history = [_make_reading(BASE, heart_rate=75.0)]
    feats = _extract_window_features(history)
    assert feats["heart_rate_recent_dev"] == 0.0


# ─── 9. shock_index when both HR and SBP are valid ───────────────────────────

def test_shock_index_computed_correctly():
    # HR=100, SBP=100 → SI = 1.0
    history = [_make_reading(BASE, heart_rate=100.0, systolic_bp=100.0)]
    feats = _extract_window_features(history)
    assert abs(feats["shock_index"] - 1.0) < 1e-6


def test_shock_index_normal_range():
    # HR=72, SBP=120 → SI ≈ 0.6 (normal)
    history = [_make_reading(BASE, heart_rate=72.0, systolic_bp=120.0)]
    feats = _extract_window_features(history)
    assert abs(feats["shock_index"] - 72 / 120) < 1e-6


# ─── 10. shock_index = 0.0 when SBP is missing ───────────────────────────────

def test_shock_index_zero_when_sbp_absent():
    history = [_make_reading(BASE, heart_rate=100.0)]   # no SBP
    feats = _extract_window_features(history)
    assert feats["shock_index"] == 0.0


def test_shock_index_zero_when_hr_absent():
    history = [_make_reading(BASE, systolic_bp=120.0)]   # no HR
    feats = _extract_window_features(history)
    assert feats["shock_index"] == 0.0


# ─── 11. spo2_rr_ratio computed correctly ────────────────────────────────────

def test_spo2_rr_ratio_correct():
    # SpO₂=90, RR=30 → ratio = 3.0
    history = [_make_reading(BASE, spo2=90.0, respiratory_rate=30.0)]
    feats = _extract_window_features(history)
    assert abs(feats["spo2_rr_ratio"] - 3.0) < 1e-6


def test_spo2_rr_ratio_zero_when_rr_absent():
    history = [_make_reading(BASE, spo2=98.0)]
    feats = _extract_window_features(history)
    assert feats["spo2_rr_ratio"] == 0.0


# ─── 12. Future-leakage prevention ───────────────────────────────────────────

def test_no_future_leakage_t_ref_is_max_timestamp():
    """t_ref must be max(recorded_at); readings > window_hours before t_ref dropped."""
    t_early = BASE
    t_late  = BASE + timedelta(hours=20)  # this becomes t_ref

    # With window=12h, t_early is 20h before t_ref → must be excluded
    history = [
        _make_reading(t_early, heart_rate=70.0),
        _make_reading(t_late,  heart_rate=80.0),
    ]
    feats = _extract_window_features(history, window_hours=12.0)

    assert feats["n_readings"] == 1.0, "Reading outside window must be excluded"
    assert feats["heart_rate_last"] == 80.0


# ─── 13. has_sufficient_history = 1.0 ────────────────────────────────────────

def test_sufficient_history_flag_true():
    # 3 readings spanning 3 hours → both conditions met
    history = [
        _make_reading(BASE,                      heart_rate=70.0),
        _make_reading(BASE + timedelta(hours=1), heart_rate=72.0),
        _make_reading(BASE + timedelta(hours=3), heart_rate=74.0),
    ]
    feats = _extract_window_features(history)
    assert feats["has_sufficient_history"] == 1.0


# ─── 14. has_sufficient_history = 0.0 ────────────────────────────────────────

def test_sufficient_history_flag_false_too_few_readings():
    history = [
        _make_reading(BASE,                      heart_rate=70.0),
        _make_reading(BASE + timedelta(hours=3), heart_rate=72.0),
    ]
    feats = _extract_window_features(history)
    assert feats["has_sufficient_history"] == 0.0  # only 2 readings (< 3)


def test_sufficient_history_flag_false_too_short_span():
    # 3 readings but only 1-hour span (< 2-hour threshold)
    history = [
        _make_reading(BASE,                        heart_rate=70.0),
        _make_reading(BASE + timedelta(minutes=30), heart_rate=71.0),
        _make_reading(BASE + timedelta(hours=1),    heart_rate=72.0),
    ]
    feats = _extract_window_features(history)
    assert feats["has_sufficient_history"] == 0.0


# ─── 15. Backward compat: all original 30 features still present ──────────────

def test_backward_compat_original_30_features_present():
    history = [
        _make_reading(BASE,                      heart_rate=75.0, spo2=98.0, news2_score=2),
        _make_reading(BASE + timedelta(hours=1), heart_rate=78.0, spo2=97.0, news2_score=3),
    ]
    feats = _extract_window_features(history)

    # 28 vital features: 4 suffixes × 7 vitals
    for vk in VITAL_FEATURES:
        for sfx in ("mean", "std", "slope", "last"):
            key = f"{vk}_{sfx}"
            assert key in feats, f"Missing backward-compat feature: {key}"

    # 2 meta features
    assert "n_readings" in feats
    assert "window_hours" in feats


# ─── 16. obs_density_per_hr ──────────────────────────────────────────────────

def test_obs_density_correct():
    # 3 readings over 4 hours → density = 0.75 /hr
    history = [
        _make_reading(BASE,                      heart_rate=70.0),
        _make_reading(BASE + timedelta(hours=2), heart_rate=72.0),
        _make_reading(BASE + timedelta(hours=4), heart_rate=74.0),
    ]
    feats = _extract_window_features(history)
    assert abs(feats["obs_density_per_hr"] - 0.75) < 0.01


# ─── 17. No-timestamp history uses all entries ───────────────────────────────

def test_no_timestamps_uses_entire_history():
    history = [
        {"heart_rate": 70.0},
        {"heart_rate": 80.0},
        {"heart_rate": 90.0},
    ]
    feats = _extract_window_features(history)
    assert feats["n_readings"] == 3.0
    assert abs(feats["heart_rate_mean"] - 80.0) < 0.01
    assert feats["heart_rate_last"] == 90.0


# ─── 18. _parse_ts handles all expected input forms ──────────────────────────

def test_parse_ts_none_returns_none():
    assert _parse_ts(None) is None


def test_parse_ts_empty_string_returns_none():
    assert _parse_ts("") is None


def test_parse_ts_nan_float_returns_none():
    assert _parse_ts(float("nan")) is None


def test_parse_ts_iso8601_string():
    expected = datetime(2024, 1, 15, 10, 30, 0)
    assert _parse_ts("2024-01-15T10:30:00") == expected


def test_parse_ts_iso8601_with_z():
    expected = datetime(2024, 1, 15, 10, 30, 0)
    assert _parse_ts("2024-01-15T10:30:00Z") == expected


def test_parse_ts_space_separated_string():
    expected = datetime(2024, 1, 15, 10, 30, 0)
    assert _parse_ts("2024-01-15 10:30:00") == expected


def test_parse_ts_datetime_object_passthrough():
    dt = datetime(2024, 3, 20, 14, 0, 0)
    assert _parse_ts(dt) == dt


def test_parse_ts_unparseable_returns_none():
    assert _parse_ts("not-a-date") is None


# ─── 19. _validate_vital bounds ──────────────────────────────────────────────

def test_validate_vital_accepts_normal_value():
    assert _validate_vital(75.0, "heart_rate") == 75.0


def test_validate_vital_rejects_above_max():
    assert _validate_vital(500.0, "heart_rate") is None


def test_validate_vital_rejects_below_min():
    assert _validate_vital(10.0, "heart_rate") is None  # < 20 bpm


def test_validate_vital_rejects_none():
    assert _validate_vital(None, "heart_rate") is None


def test_validate_vital_rejects_nan():
    assert _validate_vital(float("nan"), "spo2") is None


def test_validate_vital_rejects_non_numeric_string():
    assert _validate_vital("abc", "heart_rate") is None


def test_validate_vital_accepts_numeric_string():
    # "75.0" is a valid numeric string and within bounds
    assert _validate_vital("75.0", "heart_rate") == 75.0


def test_validate_vital_boundary_values_inclusive():
    # Exact boundary values should be accepted
    assert _validate_vital(20.0, "heart_rate") == 20.0   # min
    assert _validate_vital(300.0, "heart_rate") == 300.0  # max
    assert _validate_vital(50.0, "spo2") == 50.0
    assert _validate_vital(100.0, "spo2") == 100.0


# ─── 20. news2_velocity equals news2_score_slope_per_hr ──────────────────────

def test_news2_velocity_equals_slope_per_hr():
    history = [
        _make_reading(BASE,                      news2_score=2),
        _make_reading(BASE + timedelta(hours=2), news2_score=6),
    ]
    feats = _extract_window_features(history)

    assert feats["news2_velocity"] == pytest.approx(
        feats["news2_score_slope_per_hr"], abs=1e-9
    )
    assert feats["news2_velocity"] > 0   # score rising = positive velocity


# ─── extra: _time_aware_slope and _linear_slope unit tests ───────────────────

def test_time_aware_slope_linear():
    # t=[0,1,2], v=[10,15,20] → slope = 5 /hr
    assert abs(_time_aware_slope([0.0, 1.0, 2.0], [10.0, 15.0, 20.0]) - 5.0) < 1e-9


def test_time_aware_slope_single_point_returns_zero():
    assert _time_aware_slope([0.0], [42.0]) == 0.0


def test_linear_slope_increasing():
    assert _linear_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_linear_slope_flat():
    assert _linear_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)


# ─── Training / Inference Parity & Schema Mismatch Tests ─────────────────────

def test_feature_parity_with_training_extractor():
    """Verify that train_ews_model imports and reuses the exact same feature extraction function."""
    sys_path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if sys_path_root not in sys.path:
        sys.path.insert(0, sys_path_root)

    import ml.scripts.train_ews_model as train_script
    assert hasattr(train_script, "_extract_window_features")
    assert train_script._extract_window_features is _extract_window_features


def test_schema_mismatch_raises_on_missing_model_feature(monkeypatch):
    """Verify that inference strictly raises ValueError on missing expected features instead of zero-filling."""
    from app.services.early_warning import predict_deterioration_risk

    # Mock a bundle that expects a feature name that does NOT exist in extracted features
    fake_bundle = {
        "model": None,
        "feature_names": ["heart_rate_mean", "non_existent_biomarker_xyz"],
        "auc": 0.9,
    }
    monkeypatch.setattr("app.services.early_warning._load_model", lambda: fake_bundle)

    # Calling predict_deterioration_risk should catch the schema mismatch and fallback safely with error logging
    res = predict_deterioration_risk([
        {"heart_rate": 80.0, "recorded_at": "2026-10-01T08:00:00"}
    ])
    assert res["model_used"] == "news2_fallback_error"


def test_schema_validation_passes_when_all_features_present(monkeypatch):
    """Verify inference proceeds when all model features are present in extracted features."""
    from app.services.early_warning import predict_deterioration_risk

    class MockModel:
        def predict_proba(self, X):
            assert X.shape[1] == 2
            return [[0.8, 0.2]]

    fake_bundle = {
        "model": MockModel(),
        "feature_names": ["heart_rate_mean", "systolic_bp_last"],
        "auc": 0.9,
    }
    monkeypatch.setattr("app.services.early_warning._load_model", lambda: fake_bundle)

    res = predict_deterioration_risk([
        {"heart_rate": 80.0, "systolic_bp": 120.0, "recorded_at": "2026-10-01T08:00:00"}
    ])
    assert res["model_used"] == "xgboost"
    assert res["ml_risk_score"] == 0.2
    assert res["feature_names"] == ["heart_rate_mean", "systolic_bp_last"]

