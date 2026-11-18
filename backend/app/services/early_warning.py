"""
Early Warning System — Business Logic Service (V1–V4).

Implements
----------
V2  compute_news2()                  – Rule-based National Early Warning Score 2
V3  predict_deterioration_risk()     – XGBoost sliding-window risk prediction
V4  explain_ews_prediction()         – SHAP explainability
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[3]  # repo root
_MODEL_DIR = _BASE_DIR / "ml" / "models"
_EWS_MODEL_PATH = _MODEL_DIR / "ews_xgboost.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# V2  NEWS2 RULE-BASED SCORER
# ─────────────────────────────────────────────────────────────────────────────

# NEWS2 scoring tables (Royal College of Physicians, 2017)
# Each vital maps to: [(upper_threshold_exclusive, score), ...] sorted descending
# If value <= threshold the score is assigned.

_NEWS2_HR = [
    (40,   3),  # ≤ 40
    (50,   1),  # 41-50
    (90,   0),  # 51-90  (normal)
    (110,  1),  # 91-110
    (130,  2),  # 111-130
    (None, 3),  # > 130
]

_NEWS2_RR = [
    (8,    3),  # ≤ 8
    (11,   1),  # 9-11
    (20,   0),  # 12-20
    (24,   2),  # 21-24
    (None, 3),  # > 24
]

_NEWS2_SPO2_NO_O2 = [
    (91,   3),  # ≤ 91
    (93,   2),  # 92-93
    (95,   1),  # 94-95
    (None, 0),  # ≥ 96
]

_NEWS2_SPO2_ON_O2 = [
    (83,   3),
    (85,   2),
    (87,   1),
    (92,   0),
    (94,   1),
    (96,   2),
    (None, 3),
]

_NEWS2_TEMP = [
    (35.0, 3),
    (36.0, 1),
    (38.0, 0),
    (39.0, 1),
    (None, 2),
]

_NEWS2_SBP = [
    (90,   3),
    (100,  2),
    (110,  1),
    (219,  0),
    (None, 3),
]

_NEWS2_CONSCIOUSNESS = {
    "A": 0,  # Alert
    "V": 3,  # Voice
    "P": 3,  # Pain
    "U": 3,  # Unresponsive
}


def _score_table(value: float, table: list) -> int:
    for threshold, score in table:
        if threshold is None or value <= threshold:
            return score
    return 0


def compute_news2(vitals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the NEWS2 score and risk level from a vitals dictionary.

    Parameters
    ----------
    vitals : dict with optional keys:
        heart_rate, respiratory_rate, spo2, supplemental_o2,
        temperature_c, systolic_bp, consciousness_level

    Returns
    -------
    dict with: score, risk_level, subscores, recommendations
    """
    subscores: Dict[str, int] = {}
    total = 0

    hr = vitals.get("heart_rate")
    if hr is not None:
        s = _score_table(hr, _NEWS2_HR)
        subscores["heart_rate"] = s
        total += s

    rr = vitals.get("respiratory_rate")
    if rr is not None:
        s = _score_table(rr, _NEWS2_RR)
        subscores["respiratory_rate"] = s
        total += s

    spo2 = vitals.get("spo2")
    on_o2 = bool(vitals.get("supplemental_o2", False))
    if spo2 is not None:
        table = _NEWS2_SPO2_ON_O2 if on_o2 else _NEWS2_SPO2_NO_O2
        s = _score_table(spo2, table)
        subscores["spo2"] = s
        total += s
    if on_o2:
        subscores["supplemental_o2"] = 2
        total += 2

    temp = vitals.get("temperature_c")
    if temp is not None:
        s = _score_table(temp, _NEWS2_TEMP)
        subscores["temperature_c"] = s
        total += s

    sbp = vitals.get("systolic_bp")
    if sbp is not None:
        s = _score_table(sbp, _NEWS2_SBP)
        subscores["systolic_bp"] = s
        total += s

    cons = (vitals.get("consciousness_level") or "A").upper()
    cons_score = _NEWS2_CONSCIOUSNESS.get(cons, 0)
    subscores["consciousness_level"] = cons_score
    total += cons_score

    # Determine risk level
    # Single subscore ≥ 3 → at least Medium regardless of total
    max_subscore = max(subscores.values()) if subscores else 0

    if total == 0:
        risk_level = "Low"
    elif total <= 4 and max_subscore < 3:
        risk_level = "Low"
    elif total <= 4 and max_subscore == 3:
        risk_level = "Medium"
    elif 5 <= total <= 6:
        risk_level = "Medium"
    elif total == 7:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return {
        "score": total,
        "risk_level": risk_level,
        "subscores": subscores,
        "recommendations": _get_news2_recommendations(total, risk_level, max_subscore),
    }


def _get_news2_recommendations(score: int, risk_level: str, max_subscore: int) -> str:
    if risk_level == "Low":
        return (
            "Routine monitoring. Reassess every 12 hours or per ward protocol. "
            "Escalate if NEWS2 increases."
        )
    elif risk_level == "Medium":
        if max_subscore == 3:
            return (
                "Urgent assessment by a clinician with core competencies in acute illness. "
                "Consider 4-hourly monitoring. Inform the doctor in charge."
            )
        return (
            "Increased monitoring frequency (minimum 4-6 hourly). "
            "Doctor review required. Consider causes and treatment."
        )
    elif risk_level == "High":
        return (
            "Urgent clinical review by a doctor or advanced nurse practitioner. "
            "Continuous monitoring. Consider HDU/critical care referral."
        )
    else:  # Critical
        return (
            "EMERGENCY — Immediate assessment by a critical care team. "
            "Continuous monitoring. ICU referral strongly recommended."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering (V4 — improved signal pipeline)
# ─────────────────────────────────────────────────────────────────────────────

VITAL_FEATURES = [
    "heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp",
    "spo2", "temperature_c", "news2_score",
]

# Physiological validity bounds — readings outside these ranges are treated as
# measurement errors and excluded from ML features (not from NEWS2 scoring).
_VITAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "heart_rate":       (20.0, 300.0),
    "respiratory_rate": (4.0,  70.0),
    "systolic_bp":      (50.0, 300.0),
    "diastolic_bp":     (20.0, 200.0),
    "spo2":             (50.0, 100.0),
    "temperature_c":    (25.0, 45.0),
    "news2_score":      (0.0,  36.0),
}

# All per-vital suffixes produced by _extract_window_features
_FEATURE_SUFFIXES: Tuple[str, ...] = (
    "mean", "std", "slope", "last",
    "slope_per_hr", "velocity_last", "missing_frac", "recent_dev",
)

# Build display names programmatically so they stay in sync with VITAL_FEATURES
_VITAL_LABELS: Dict[str, str] = {
    "heart_rate":       "Heart Rate",
    "respiratory_rate": "Resp. Rate",
    "systolic_bp":      "Systolic BP",
    "diastolic_bp":     "Diastolic BP",
    "spo2":             "SpO\u2082",
    "temperature_c":    "Temperature",
    "news2_score":      "NEWS2",
}
_SUFFIX_LABELS: Dict[str, str] = {
    "mean":          "(mean)",
    "std":           "(variability)",
    "slope":         "(ordinal trend)",
    "last":          "(latest)",
    "slope_per_hr":  "(trend/hr)",
    "velocity_last": "(velocity)",
    "missing_frac":  "(missingness)",
    "recent_dev":    "(recent \u0394)",
}

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    f"{vk}_{sk}": f"{vl} {sl}"
    for vk, vl in _VITAL_LABELS.items()
    for sk, sl in _SUFFIX_LABELS.items()
}
FEATURE_DISPLAY_NAMES.update({
    "n_readings":             "# Readings",
    "window_hours":           "Observation Window (h)",
    "obs_density_per_hr":     "Observation Density (/hr)",
    "has_sufficient_history": "Sufficient History Flag",
    "overall_missingness":    "Overall Missingness",
    "shock_index":            "Shock Index (HR/SBP)",
    "spo2_rr_ratio":          "SpO\u2082/RR Ratio",
    "news2_velocity":         "NEWS2 Velocity (/hr)",
})


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _parse_ts(val: Any) -> Optional[datetime]:
    """Safely parse a recorded_at value to a tz-naive datetime.

    Accepts datetime objects, ISO-8601 strings (with or without Z/offset),
    and ``%Y-%m-%d %H:%M:%S`` / ``%Y-%m-%d %H:%M`` strings.
    Returns None for None, empty strings, NaN floats, and unparseable values.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo else val
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
    except (TypeError, ValueError):
        pass
    text = str(val).strip()
    if not text:
        return None
    # ISO-8601 with optional Z or +HH:MM offset
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except ValueError:
        pass
    # Common non-ISO formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _validate_vital(value: Any, key: str) -> Optional[float]:
    """Return the value as float if it is within physiological bounds.

    Returns None for None, non-numeric, NaN, Inf, or out-of-bounds values.
    Bounds are checked only for vitals listed in _VITAL_BOUNDS; unknown keys
    are accepted as-is after numeric conversion.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    lo, hi = _VITAL_BOUNDS.get(key, (-1e9, 1e9))
    return v if lo <= v <= hi else None


def _linear_slope(values: List[float]) -> float:
    """Ordinal least-squares slope over equally-spaced index positions.

    Preserved unchanged for backward compatibility with the saved XGBoost model
    which was trained on this unitless slope.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _time_aware_slope(times_hours: List[float], values: List[float]) -> float:
    """Least-squares slope in vital-units **per hour** over actual timestamps.

    ``times_hours`` should be offsets relative to a fixed reference (e.g. all
    negative, relative to t_ref).  Sign of offsets cancels out in the formula.
    """
    n = len(times_hours)
    if n < 2:
        return 0.0
    xm = sum(times_hours) / n
    ym = sum(values) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(times_hours, values))
    den = sum((x - xm) ** 2 for x in times_hours)
    return num / den if abs(den) > 1e-12 else 0.0


def _empty_features() -> Dict[str, float]:
    """Zero-valued feature dict returned for empty or all-invalid histories."""
    feats: Dict[str, float] = {}
    for vk in VITAL_FEATURES:
        for sfx in _FEATURE_SUFFIXES:
            feats[f"{vk}_{sfx}"] = 0.0
        feats[f"{vk}_missing_frac"] = 1.0   # explicit: all missing
    feats.update({
        "n_readings": 0.0,
        "window_hours": 0.0,
        "obs_density_per_hr": 0.0,
        "has_sufficient_history": 0.0,
        "overall_missingness": 1.0,
        "shock_index": 0.0,
        "spo2_rr_ratio": 0.0,
        "news2_velocity": 0.0,
    })
    return feats


def _extract_window_features(
    vitals_history: List[Dict],
    window_hours: float = 24.0,
) -> Dict[str, float]:
    """Extract sliding-window features from a list of vital-sign dicts.

    Improvements over V3
    --------------------
    Timestamp-based window
        Only readings within ``window_hours`` of the latest recorded timestamp
        are included (default 24 h, configurable).

    Anti-leakage
        ``t_reference = max(recorded_at)`` — wall-clock time is never used.

    Deduplication
        If two entries share the same timestamp the **last** one in the
        original list is kept (consistent with "most recent write wins").

    Bounds validation
        Physiologically impossible values (see ``_VITAL_BOUNDS``) are
        excluded from ML features; they are still present in the raw dict so
        NEWS2 scoring (called separately) is unaffected.

    Backward compatibility
        The original 30 features (``{vital}_mean/std/slope/last``,
        ``n_readings``, ``window_hours``) are preserved with the same
        computation so the saved XGBoost model continues to work unchanged.
        ``predict_deterioration_risk`` uses ``feat_dict.get(name, 0.0)``
        to select features by name — new extras are silently ignored.

    New per-vital features (× 7 vitals)
        ``slope_per_hr``   — least-squares slope in vital-units/hour
        ``velocity_last``  — (v[-1] − v[-2]) / Δt in units/hour
        ``missing_frac``   — fraction of window slots without a valid value
        ``recent_dev``     — z-score of last value vs baseline (all but last)

    New derived features
        ``shock_index``    — HR / SBP (only when both valid; normal < 0.9)
        ``spo2_rr_ratio``  — SpO₂ / RR (only when both valid and RR > 0)
        ``news2_velocity`` — alias of ``news2_score_slope_per_hr``

    New meta features
        ``obs_density_per_hr``     — readings per hour in window
        ``has_sufficient_history`` — 1.0 when ≥ 3 readings over ≥ 2 h
        ``overall_missingness``    — mean of per-vital missing_frac values

    Parameters
    ----------
    vitals_history:
        List of dicts, each optionally containing vital keys and
        ``recorded_at`` (datetime | ISO-8601 str | None).
    window_hours:
        Look-back window in hours (default 24).  Pass a smaller value
        (e.g. 6.0) to tighten the window for high-frequency monitoring.

    Returns
    -------
    dict mapping feature name → float.
    """
    if not vitals_history:
        return _empty_features()

    # ── 1. Parse timestamps ───────────────────────────────────────────────────
    enriched: List[Tuple[Optional[datetime], Dict]] = [
        (_parse_ts(v.get("recorded_at")), v) for v in vitals_history
    ]

    # ── 2. Anti-leakage: t_ref = latest recorded timestamp (never wall-clock) ─
    valid_ts: List[datetime] = [ts for ts, _ in enriched if ts is not None]
    if valid_ts:
        t_ref: Optional[datetime] = max(valid_ts)
        t_cutoff: Optional[datetime] = t_ref - timedelta(hours=window_hours)
        windowed: List[Tuple[Optional[datetime], Dict]] = [
            (ts, v) for ts, v in enriched
            if ts is None or ts >= t_cutoff
        ]
    else:
        # No timestamps at all — use entire history (graceful fallback)
        t_ref = None
        t_cutoff = None
        windowed = list(enriched)

    if not windowed:
        return _empty_features()

    # ── 3. Sort ascending by timestamp (no-ts entries go last) ───────────────
    windowed.sort(key=lambda x: (x[0] is None, x[0] or datetime.min))

    # ── 4. Deduplicate: same timestamp → keep last in original list order ─────
    # Iterating sorted windowed and overwriting means the last occurrence of
    # each ts key wins (stable sort preserves original list order for ties).
    seen: Dict[str, Tuple[Optional[datetime], Dict]] = {}
    for ts, v in windowed:
        key = ts.isoformat() if ts is not None else str(id(v))
        seen[key] = (ts, v)
    deduped: List[Tuple[Optional[datetime], Dict]] = sorted(
        seen.values(),
        key=lambda x: (x[0] is None, x[0] or datetime.min),
    )

    n_in_window = len(deduped)

    # ── 5. Actual window duration from timestamps ─────────────────────────────
    ts_list: List[datetime] = [ts for ts, _ in deduped if ts is not None]
    if len(ts_list) >= 2:
        actual_window_hours = (ts_list[-1] - ts_list[0]).total_seconds() / 3600.0
    elif t_ref is not None:
        actual_window_hours = window_hours   # single ts reading: assume full span
    else:
        actual_window_hours = float(n_in_window)  # no timestamps: use count proxy

    # ── 6. Per-vital feature extraction ──────────────────────────────────────
    features: Dict[str, float] = {}
    missing_fracs: List[float] = []

    for vkey in VITAL_FEATURES:
        # Collect (time_offset_hours_from_t_ref, validated_value) pairs
        timed: List[Tuple[Optional[float], float]] = []
        raw_vals: List[float] = []

        for ts, v in deduped:
            validated = _validate_vital(v.get(vkey), vkey)
            if validated is None:
                continue
            t_offset: Optional[float] = (
                (ts - t_ref).total_seconds() / 3600.0
                if (ts is not None and t_ref is not None)
                else None
            )
            timed.append((t_offset, validated))
            raw_vals.append(validated)

        n_present = len(raw_vals)
        miss_frac = 1.0 - n_present / n_in_window if n_in_window > 0 else 1.0
        missing_fracs.append(miss_frac)

        if n_present == 0:
            for sfx in _FEATURE_SUFFIXES:
                features[f"{vkey}_{sfx}"] = 0.0
            features[f"{vkey}_missing_frac"] = 1.0
            continue

        mean_val  = float(np.mean(raw_vals))
        std_val   = float(np.std(raw_vals)) if n_present > 1 else 0.0
        last_val  = raw_vals[-1]
        ord_slope = _linear_slope(raw_vals)   # backward-compat unitless slope

        # Time-aware slope (vital-units / hour) — needs ≥ 2 timed readings
        timed_pairs: List[Tuple[float, float]] = [
            (t, val) for t, val in timed if t is not None
        ]
        slope_per_hr = (
            _time_aware_slope(
                [p[0] for p in timed_pairs],
                [p[1] for p in timed_pairs],
            )
            if len(timed_pairs) >= 2 else 0.0
        )

        # Velocity of most-recent change: (v_last − v_prev) / |Δt| (units/hr)
        if len(timed_pairs) >= 2:
            t2, v2 = timed_pairs[-1]
            t1, v1 = timed_pairs[-2]
            dt_hr = abs(t2 - t1)
            velocity_last = (v2 - v1) / dt_hr if dt_hr > 1e-9 else 0.0
        else:
            velocity_last = 0.0

        # Recent deviation: (last − baseline_mean) / baseline_std
        # Guard against zero-variance baseline creating massive artificial spikes
        if n_present >= 2:
            baseline = raw_vals[:-1]
            b_mean = float(np.mean(baseline))
            b_std  = max(
                float(np.std(baseline)) if len(baseline) > 1 else 1.0,
                0.5,
            )
            raw_dev = (last_val - b_mean) / b_std
            # Clip z-score to clinically plausible range [-10.0, 10.0] to prevent downstream numerical instability
            recent_dev = float(np.clip(raw_dev, -10.0, 10.0))
        else:
            recent_dev = 0.0

        features[f"{vkey}_mean"]          = mean_val
        features[f"{vkey}_std"]           = std_val
        features[f"{vkey}_slope"]         = ord_slope      # ← backward compat
        features[f"{vkey}_last"]          = last_val
        features[f"{vkey}_slope_per_hr"]  = slope_per_hr
        features[f"{vkey}_velocity_last"] = velocity_last
        features[f"{vkey}_missing_frac"]  = miss_frac
        features[f"{vkey}_recent_dev"]    = recent_dev

    # ── 7. Meta features ──────────────────────────────────────────────────────
    features["n_readings"]   = float(n_in_window)
    features["window_hours"] = actual_window_hours
    features["obs_density_per_hr"] = (
        n_in_window / actual_window_hours if actual_window_hours > 0 else 0.0
    )
    features["has_sufficient_history"] = (
        1.0 if (actual_window_hours >= 2.0 and n_in_window >= 3) else 0.0
    )
    features["overall_missingness"] = (
        float(np.mean(missing_fracs)) if missing_fracs else 1.0
    )

    # ── 8. Derived features (only when required inputs are valid) ─────────────
    hr_missing   = features.get("heart_rate_missing_frac",       1.0) >= 1.0
    sbp_missing  = features.get("systolic_bp_missing_frac",      1.0) >= 1.0
    rr_missing   = features.get("respiratory_rate_missing_frac", 1.0) >= 1.0
    spo2_missing = features.get("spo2_missing_frac",             1.0) >= 1.0

    hr_last   = features.get("heart_rate_last",       0.0)
    sbp_last  = features.get("systolic_bp_last",      0.0)
    spo2_last = features.get("spo2_last",             0.0)
    rr_last   = features.get("respiratory_rate_last", 0.0)

    # Shock index: HR / SBP  (normal < 0.9; ≥ 1.0 suggests haemodynamic shock)
    features["shock_index"] = (
        hr_last / sbp_last
        if (not hr_missing and not sbp_missing and sbp_last >= 1.0)
        else 0.0
    )

    # SpO₂/RR ratio: low value = high RR with low SpO₂ (respiratory distress)
    features["spo2_rr_ratio"] = (
        spo2_last / rr_last
        if (not rr_missing and not spo2_missing and rr_last > 0.0)
        else 0.0
    )

    # NEWS2 change-rate (alias for the time-aware slope already computed)
    features["news2_velocity"] = features.get("news2_score_slope_per_hr", 0.0)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# V3  XGBoost ML Inference
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: Optional[Dict] = None


def _load_model() -> Optional[Dict]:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _EWS_MODEL_PATH.exists():
        return None
    try:
        with open(_EWS_MODEL_PATH, "rb") as fh:
            _model_cache = pickle.load(fh)
        logger.info("EWS XGBoost model loaded from %s", _EWS_MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load EWS model: %s", exc)
        return None
    return _model_cache


def predict_deterioration_risk(vitals_history: List[Dict]) -> Dict[str, Any]:
    """
    V3: Run XGBoost risk prediction on a list of vitals dicts.
    Falls back to NEWS2-derived score if model is unavailable.

    Returns
    -------
    dict with: ml_risk_score, ml_risk_label, feature_vector, feature_names,
               model_used, shap_values (empty list if SHAP unavailable)
    """
    if len(vitals_history) < 1:
        return {
            "ml_risk_score": 0.0,
            "ml_risk_label": "Insufficient Data",
            "feature_vector": [],
            "feature_names": [],
            "model_used": "none",
            "shap_values": [],
        }

    features = _extract_window_features(vitals_history)
    feature_names = list(features.keys())
    feature_vector = [features[k] for k in feature_names]

    bundle = _load_model()

    if bundle is None:
        # Fallback: derive risk from latest NEWS2 score
        latest_news2 = vitals_history[-1].get("news2_score") or 0
        score = min(latest_news2 / 20.0, 1.0)
        label = _score_to_label(score)
        return {
            "ml_risk_score": round(score, 3),
            "ml_risk_label": label,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "model_used": "news2_fallback",
            "shap_values": [],
        }

    try:
        model = bundle["model"]
        ordered_names = bundle.get("feature_names", feature_names)

        # Strict validation: verify that all expected model features are present
        missing_features = [n for n in ordered_names if n not in features]
        if missing_features:
            raise ValueError(
                f"Feature schema mismatch: missing required model features: {missing_features[:5]}"
                + (f" (...and {len(missing_features) - 5} more)" if len(missing_features) > 5 else "")
            )

        # Align features to model's expected order
        feat_dict = dict(zip(feature_names, feature_vector))
        X = np.array([[feat_dict[n] for n in ordered_names]])

        proba = model.predict_proba(X)[0]
        # Model trained as binary: 0=low-risk, 1=high-risk
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        label = _score_to_label(score)

        return {
            "ml_risk_score": round(score, 4),
            "ml_risk_label": label,
            "feature_vector": X[0].tolist(),
            "feature_names": ordered_names,
            "model_used": "xgboost",
            "shap_values": [],
        }
    except Exception as exc:
        logger.error("XGBoost inference error: %s", exc)
        # Fallback
        latest_news2 = vitals_history[-1].get("news2_score") or 0
        score = min(latest_news2 / 20.0, 1.0)
        return {
            "ml_risk_score": round(score, 3),
            "ml_risk_label": _score_to_label(score),
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "model_used": "news2_fallback_error",
            "shap_values": [],
        }


def _score_to_label(score: float) -> str:
    if score < 0.25:
        return "Low"
    elif score < 0.50:
        return "Medium"
    elif score < 0.75:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────────────────────────────────────
# V4  SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain_ews_prediction(
    feature_vector: List[float],
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    V4: Compute SHAP values for a single prediction and return the
    top-5 feature contributions, sorted by |shap|.
    """
    bundle = _load_model()
    if bundle is None or not feature_vector:
        return _heuristic_shap(feature_vector, feature_names)

    try:
        import shap as shap_lib

        model = bundle["model"]
        X = np.array([feature_vector])
        explainer = shap_lib.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # For binary XGBoost, shap_values is shape (n_samples, n_features)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # class-1 (high risk)
        else:
            sv = shap_vals[0]

        results = []
        for i, (name, val, sv_i) in enumerate(zip(feature_names, feature_vector, sv)):
            display = FEATURE_DISPLAY_NAMES.get(name, name)
            direction = "↑ Risk" if sv_i > 0 else "↓ Risk"
            results.append({
                "feature": name,
                "display_name": display,
                "value": round(val, 3),
                "shap_value": round(float(sv_i), 4),
                "direction": direction,
            })

        # Sort by |shap| descending, take top 5
        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:5]

    except ImportError:
        logger.warning("shap library not available — using heuristic explanations")
        return _heuristic_shap(feature_vector, feature_names)
    except Exception as exc:
        logger.error("SHAP error: %s", exc)
        return _heuristic_shap(feature_vector, feature_names)


# ─────────────────────────────────────────────────────────────────────────────
# CSV ingestion (optional bulk upload)
# ─────────────────────────────────────────────────────────────────────────────

_CSV_ALIASES = {
    "recorded_at": ["recorded_at", "timestamp", "time", "datetime", "date"],
    "heart_rate": ["heart_rate", "hr", "heartrate", "pulse"],
    "respiratory_rate": ["respiratory_rate", "rr", "resp_rate", "respiration"],
    "systolic_bp": ["systolic_bp", "sbp", "sys_bp", "systolic"],
    "diastolic_bp": ["diastolic_bp", "dbp", "dia_bp", "diastolic"],
    "spo2": ["spo2", "sp02", "oxygen_saturation", "o2sat", "o2_sat"],
    "temperature_c": ["temperature_c", "temp", "temperature", "temp_c"],
    "consciousness_level": ["consciousness_level", "avpu", "consciousness"],
    "supplemental_o2": ["supplemental_o2", "on_o2", "oxygen", "o2"],
    "urine_output_ml_hr": ["urine_output_ml_hr", "urine", "urine_output"],
    "notes": ["notes", "comment", "comments"],
}

_CSV_MAX_ROWS = 500

_NUMERIC_BOUNDS = {
    "heart_rate": (0, 300),
    "respiratory_rate": (0, 100),
    "systolic_bp": (0, 300),
    "diastolic_bp": (0, 200),
    "spo2": (0, 100),
    "temperature_c": (25.0, 45.0),
    "urine_output_ml_hr": (0, 5000),
}


def _normalize_header(name: str) -> str:
    return "".join(ch for ch in name.strip().lower().replace("-", "_") if ch.isalnum() or ch == "_")


def _map_csv_columns(headers: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    normalized = {h: _normalize_header(h) for h in headers}
    for canonical, aliases in _CSV_ALIASES.items():
        alias_set = {_normalize_header(a) for a in aliases}
        for original, norm in normalized.items():
            if norm in alias_set and canonical not in mapping:
                mapping[canonical] = original
    return mapping


def _parse_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(text.replace("Z", ""), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def parse_vitals_csv(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parse a vitals CSV without inventing missing values.

    Returns dict with keys: rows, errors, warnings.
    Each valid row is a dict suitable for VitalSignCreate (minus patient_id).
    """
    import io

    import pandas as pd

    errors: List[str] = []
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"rows": [], "errors": ["CSV must be UTF-8 encoded."], "warnings": []}

    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return {"rows": [], "errors": [f"Could not parse CSV: {exc}"], "warnings": []}

    if frame.empty:
        return {"rows": [], "errors": ["CSV contains no data rows."], "warnings": []}

    if len(frame) > _CSV_MAX_ROWS:
        errors.append(f"CSV has {len(frame)} rows; only the first {_CSV_MAX_ROWS} will be considered.")
        frame = frame.head(_CSV_MAX_ROWS)

    mapping = _map_csv_columns([str(c) for c in frame.columns])
    if "heart_rate" not in mapping and "respiratory_rate" not in mapping and "spo2" not in mapping:
        errors.append(
            "CSV is missing recognised vital-sign columns. "
            "Expected headers such as recorded_at, heart_rate, respiratory_rate, spo2, temperature_c, systolic_bp."
        )
        return {"rows": [], "errors": errors, "warnings": warnings}

    seen_timestamps: set = set()
    for idx, record in frame.iterrows():
        line = int(idx) + 2  # header is line 1
        parsed: Dict[str, Any] = {}

        raw_ts = record.get(mapping["recorded_at"]) if "recorded_at" in mapping else None
        ts = _parse_timestamp(raw_ts)
        if "recorded_at" in mapping and raw_ts is not None and not (isinstance(raw_ts, float) and math.isnan(raw_ts)) and str(raw_ts).strip() and ts is None:
            errors.append(f"Row {line}: malformed timestamp '{raw_ts}'.")
            continue
        parsed["recorded_at"] = ts

        if ts is not None:
            if ts in seen_timestamps:
                errors.append(f"Row {line}: duplicate timestamp {ts.isoformat()} in this file.")
                continue
            seen_timestamps.add(ts)

        has_measurement = False
        for field, bounds in _NUMERIC_BOUNDS.items():
            if field not in mapping:
                continue
            raw = record.get(mapping[field])
            if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
                warnings.append(f"Row {line}: missing {field} (left blank, not imputed).")
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"Row {line}: {field} is not a number ('{raw}').")
                continue
            low, high = bounds
            if value < low or value > high:
                errors.append(f"Row {line}: {field}={value} is outside accepted range {low}–{high}.")
                continue
            parsed[field] = value
            has_measurement = True

        if "consciousness_level" in mapping:
            raw = record.get(mapping["consciousness_level"])
            if raw is not None and not (isinstance(raw, float) and math.isnan(raw)) and str(raw).strip():
                avpu = str(raw).strip().upper()[:1]
                if avpu not in {"A", "V", "P", "U"}:
                    errors.append(f"Row {line}: consciousness_level must be A, V, P, or U.")
                else:
                    parsed["consciousness_level"] = avpu

        if "supplemental_o2" in mapping:
            parsed["supplemental_o2"] = _parse_bool(record.get(mapping["supplemental_o2"]))
        else:
            parsed["supplemental_o2"] = False

        if "notes" in mapping:
            raw = record.get(mapping["notes"])
            if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                parsed["notes"] = str(raw).strip() or None

        if not has_measurement:
            errors.append(f"Row {line}: no usable vital-sign measurements.")
            continue

        rows.append(parsed)

    return {"rows": rows, "errors": errors, "warnings": warnings}


def _heuristic_shap(
    feature_vector: List[float], feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Fallback: approximate importance using z-score-like deviation from
    a nominal healthy range.
    """
    healthy_means = {
        "heart_rate_last": 75.0,
        "respiratory_rate_last": 16.0,
        "spo2_last": 98.0,
        "temperature_c_last": 37.0,
        "systolic_bp_last": 120.0,
        "news2_score_last": 1.0,
    }
    healthy_stds = {
        "heart_rate_last": 15.0,
        "respiratory_rate_last": 4.0,
        "spo2_last": 2.0,
        "temperature_c_last": 0.5,
        "systolic_bp_last": 20.0,
        "news2_score_last": 1.5,
    }

    results = []
    for name, val in zip(feature_names, feature_vector):
        if name in healthy_means:
            z = (val - healthy_means[name]) / healthy_stds.get(name, 1.0)
            # For SpO2 low is bad; invert sign
            if "spo2" in name:
                z = -z
            display = FEATURE_DISPLAY_NAMES.get(name, name)
            direction = "↑ Risk" if z > 0 else "↓ Risk"
            results.append({
                "feature": name,
                "display_name": display,
                "value": round(val, 3),
                "shap_value": round(z * 0.1, 4),
                "direction": direction,
            })

    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return results[:5]
