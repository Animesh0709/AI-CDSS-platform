"""
PhysioNet 2019 Sepsis Challenge Dataset Loader & Sliding Window Ingestion.
========================================================================
Loads individual patient .psv files, maps vital columns to the AI-CDSS format,
computes observation windows without future-data leakage, and assigns binary
labels for onset of sepsis within the prediction horizon.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Add backend directory to sys.path if not present so we reuse production functions
import sys
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.services.early_warning import _extract_window_features, compute_news2

logger = logging.getLogger(__name__)

# Column mapping from PhysioNet 2019 PSV to AI-CDSS EWS naming
PHYSIONET_VITAL_MAP = {
    "HR": "heart_rate",
    "O2Sat": "spo2",
    "Temp": "temperature_c",
    "SBP": "systolic_bp",
    "DBP": "diastolic_bp",
    "Resp": "respiratory_rate",
}


def parse_psv_patient(
    file_path: str | Path,
    base_timestamp: Optional[datetime] = None,
) -> Tuple[str, List[Dict[str, Any]], List[int]]:
    """
    Parse a single PhysioNet 2019 .psv patient file.
    
    Parameters
    ----------
    file_path : str | Path
        Path to the patient's .psv file.
    base_timestamp : datetime, optional
        Reference start datetime for ICULOS=1. Defaults to 2026-01-01 00:00:00.
        
    Returns
    -------
    patient_id : str
        Extracted patient identifier (e.g., 'p00101').
    vitals_records : list of dicts
        List of hourly vitals dictionaries suitable for `_extract_window_features`.
    raw_sepsis_labels : list of int
        Raw hourly SepsisLabel from the file (0 or 1).
    """
    path = Path(file_path)
    patient_id = path.stem
    if base_timestamp is None:
        base_timestamp = datetime(2026, 1, 1, 0, 0, 0)

    # Read pipe-delimited file
    df = pd.read_csv(path, sep="|")

    vitals_records: List[Dict[str, Any]] = []
    raw_sepsis_labels: List[int] = []

    for idx, row in df.iterrows():
        # Hour offset based on ICULOS or row index
        iculos = row.get("ICULOS")
        if pd.notna(iculos) and iculos > 0:
            rec_time = base_timestamp + timedelta(hours=float(iculos) - 1.0)
        else:
            rec_time = base_timestamp + timedelta(hours=float(idx))

        record: Dict[str, Any] = {
            "recorded_at": rec_time,
            "patient_id": patient_id,
        }

        # Map vitals
        for src_col, target_col in PHYSIONET_VITAL_MAP.items():
            if src_col in row and pd.notna(row[src_col]):
                try:
                    record[target_col] = float(row[src_col])
                except (ValueError, TypeError):
                    pass

        # Calculate NEWS2 on the available raw measurements
        try:
            news_res = compute_news2(record)
            record["news2_score"] = news_res.get("score", 0)
        except Exception:
            record["news2_score"] = 0

        vitals_records.append(record)

        label_val = row.get("SepsisLabel", 0)
        raw_sepsis_labels.append(int(label_val) if pd.notna(label_val) else 0)

    return patient_id, vitals_records, raw_sepsis_labels


def generate_patient_windows(
    vitals_records: List[Dict[str, Any]],
    raw_sepsis_labels: List[int],
    window_hours: float = 6.0,
    min_readings: int = 2,
    include_post_onset: bool = False,
) -> Tuple[List[Dict[str, float]], List[int], List[datetime]]:
    """
    Generate sliding observation windows and target labels for a patient.
    
    Target Definition:
    ------------------
    We predict: "Will clinical sepsis onset occur within the next 6 hours?" (Target B).
    
    In the PhysioNet 2019 dataset, `SepsisLabel` is already shifted by 6 hours:
        SepsisLabel(t) = 1  if t >= t_sepsis - 6 hours
        SepsisLabel(t) = 0  if t < t_sepsis - 6 hours
        
    Therefore:
    - At any hour t before t_sepsis - 6: clinical sepsis is > 6 hours away -> target = 0.
    - At any hour t in [t_sepsis - 6, t_sepsis]: clinical sepsis is within <= 6 hours -> target = 1.
      This corresponds directly to `raw_sepsis_labels[t] == 1`!
    - At any hour t > t_sepsis: clinical sepsis has already occurred (post-onset).
      By default, post-onset time steps are excluded (`include_post_onset=False`)
      to prevent learning post-onset artifacts or treating established sepsis as a pre-warning case.
      
    If a patient never develops sepsis, raw_sepsis_labels is 0 throughout, so target is always 0.
    
    Anti-leakage:
    -------------
    - Features for prediction at step `i` are computed strictly from vitals observed at or
      before step `i` (`vitals_records[: i + 1]`).
    - No future observations or future labels are leaked into the feature extractor.
    """
    X_samples: List[Dict[str, float]] = []
    y_labels: List[int] = []
    timestamps: List[datetime] = []

    total_steps = len(vitals_records)
    if total_steps == 0:
        return X_samples, y_labels, timestamps

    # Find the first index where SepsisLabel is 1 (which is t_first_1 = t_sepsis - 6)
    first_1_idx = None
    for idx, lbl in enumerate(raw_sepsis_labels):
        if lbl == 1:
            first_1_idx = idx
            break

    # If sepsis occurs, clinical onset t_sepsis is estimated as first_1_idx + 6 hours
    # because PhysioNet shifted the label activation 6 hours before t_sepsis.
    t_clinical_onset = (first_1_idx + 6) if first_1_idx is not None else None

    for i in range(total_steps):
        # Once clinical sepsis has already occurred (post-onset), do not include as pre-onset warnings
        if not include_post_onset and t_clinical_onset is not None and i > t_clinical_onset:
            break

        current_time = vitals_records[i]["recorded_at"]
        # History strictly up to current step (anti-leakage)
        history_up_to_i = vitals_records[: i + 1]

        # Extract features using production window extractor
        feats = _extract_window_features(history_up_to_i, window_hours=window_hours)

        # Enforce minimum data sufficiency
        if feats.get("n_readings", 0) < min_readings:
            continue

        # Target B: Clinical sepsis onset within the next 6 hours
        # In PhysioNet 2019, raw_sepsis_labels[i] == 1 exactly during [t_sepsis - 6, t_sepsis].
        # For hours i <= t_clinical_onset:
        # If raw_sepsis_labels[i] == 1, onset is within the next 6 hours (target = 1).
        # If raw_sepsis_labels[i] == 0, onset is > 6 hours away or will not occur (target = 0).
        target = int(raw_sepsis_labels[i] == 1)

        X_samples.append(feats)
        y_labels.append(target)
        timestamps.append(current_time)

    return X_samples, y_labels, timestamps


def split_patients_by_id(
    patient_ids: Sequence[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient IDs into Train, Validation, and Test sets without cross-contamination.
    Each patient belongs strictly to one split.
    """
    unique_ids = sorted(list(set(patient_ids)))
    rng = np.random.RandomState(random_seed)
    shuffled_ids = list(unique_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))
    n_train = n_total - n_test - n_val

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    return train_ids, val_ids, test_ids
