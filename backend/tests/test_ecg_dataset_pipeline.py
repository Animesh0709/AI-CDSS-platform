import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.scripts.ecg.ptbxl_loader import PTBXLDatasetLoader, STANDARD_12_LEADS, DIAGNOSTIC_SUPERCLASSES
from ml.scripts.ecg.ludb_loader import LUDBDatasetLoader
from ml.scripts.ecg.mitdb_loader import MITDBDatasetLoader
from ml.scripts.ecg.render_ecg_image import (
    render_12lead_ecg_image,
    CONVENTIONAL_3X4_GRID,
    STANDARD_LEAD_NAMES,
)


@pytest.fixture
def ecg_data_root():
    return REPO_ROOT / "ml" / "data" / "ecg"


# ── 1. PTB-XL Dataset Integrity & Record Count Verification ─────────────────

def test_ptbxl_exact_record_count_and_provenance(ecg_data_root):
    """
    Verify exact 21,799 records in ptbxl_database.csv.
    Identifies the exact 38 excluded raw IDs (range 1 to 21837) due to PhysioNet's
    upstream exclusion of duplicate/triplicate records across releases.
    """
    ptb_loader = PTBXLDatasetLoader(ecg_data_root / "ptbxl")
    df = ptb_loader.df

    assert len(df) == 21799
    assert df["patient_id"].nunique() == 18869

    # Max ECG ID is 21837; verify exactly 38 gaps exist upstream due to duplicate/triplicate exclusions
    all_possible_ids = set(range(1, 21838))
    actual_ids = set(df.index)
    excluded_ids = sorted(list(all_possible_ids - actual_ids))
    assert len(excluded_ids) == 38
    assert 137 in excluded_ids
    assert 456 in excluded_ids


def test_ptbxl_manifest_integrity_and_patient_isolation(ecg_data_root):
    ptb_loader = PTBXLDatasetLoader(ecg_data_root / "ptbxl")
    df = ptb_loader.map_diagnostic_superclasses()

    # Verify official 5 superclass counts match PhysioNet benchmark:
    # NORM: 9514, MI: 5469, STTC: 5235, CD: 4898, HYP: 2649
    expected_counts = {
        "NORM": 9514,
        "MI": 5469,
        "STTC": 5235,
        "CD": 4898,
        "HYP": 2649,
    }
    for sc, exp_count in expected_counts.items():
        assert f"label_{sc}" in df.columns
        assert df[f"label_{sc}"].isin([0, 1]).all()
        assert int(df[f"label_{sc}"].sum()) == exp_count

    train_df, val_df, test_df = ptb_loader.get_split_manifests()

    # Zero patient overlap across splits
    train_pts = set(train_df["patient_id"])
    val_pts = set(val_df["patient_id"])
    test_pts = set(test_df["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0
    assert len(train_pts.intersection(test_pts)) == 0
    assert len(val_pts.intersection(test_pts)) == 0

    assert len(train_df) == 17418
    assert len(val_df) == 2183
    assert len(test_df) == 2198


def test_ptbxl_sampling_frequency_and_signal_loading(ecg_data_root):
    ptb_loader = PTBXLDatasetLoader(ecg_data_root / "ptbxl")
    # Low-res is 100 Hz (1000 samples for 10s)
    sig_lr, fs_lr, leads_lr = ptb_loader.load_record_signal(1, resolution="100")
    assert fs_lr == 100
    assert sig_lr.shape == (1000, 12)
    assert leads_lr == STANDARD_12_LEADS
    assert not np.isnan(sig_lr).any()


# ── 2. Conventional 3x4 Layout & Signal-to-Lead Time Alignment ──────────────

def test_conventional_3x4_lead_ordering_and_mapping(tmp_path):
    """
    Verify conventional 12-lead layout:
      Row 1: I,   aVR, V1, V4
      Row 2: II,  aVL, V2, V5
      Row 3: III, aVF, V3, V6
    """
    assert CONVENTIONAL_3X4_GRID[0] == ["I", "aVR", "V1", "V4"]
    assert CONVENTIONAL_3X4_GRID[1] == ["II", "aVL", "V2", "V5"]
    assert CONVENTIONAL_3X4_GRID[2] == ["III", "aVF", "V3", "V6"]

    # Construct identifiable synthetic signal: channel i has constant value i + 1.0
    fs = 100
    num_samples = 1000  # 10s
    test_sig = np.zeros((num_samples, 12))
    for i in range(12):
        test_sig[:, i] = (i + 1) * 0.1

    out_png = tmp_path / "conventional_test.png"
    meta = render_12lead_ecg_image(
        signal=test_sig,
        sampling_rate=fs,
        output_image_path=out_png,
        lead_names=STANDARD_LEAD_NAMES,
    )

    assert out_png.exists()
    assert meta["is_synthetic"] is True
    assert meta["layout_format"] == "Conventional 3x4 + Lead II Rhythm Strip"
    assert meta["row_layout"]["row_1"] == ["I", "aVR", "V1", "V4"]
    assert meta["row_layout"]["row_2"] == ["II", "aVL", "V2", "V5"]
    assert meta["row_layout"]["row_3"] == ["III", "aVF", "V3", "V6"]


def test_time_segment_alignment_and_duration(tmp_path):
    """
    Verify 2.5-second time window per column and 10-second total duration:
      Col 0: 0.0s to 2.5s (samples 0..250)
      Col 1: 2.5s to 5.0s (samples 250..500)
      Col 2: 5.0s to 7.5s (samples 500..750)
      Col 3: 7.5s to 10.0s (samples 750..1000)
    """
    fs = 100
    test_sig = np.ones((1000, 12)) * 0.2
    out_png = tmp_path / "time_align_test.png"
    meta = render_12lead_ecg_image(
        signal=test_sig,
        sampling_rate=fs,
        output_image_path=out_png,
    )

    segs = meta["rendered_segments"]
    # Row 1 check
    assert segs["I"]["t_start_sec"] == 0.0 and segs["I"]["t_end_sec"] == 2.5
    assert segs["aVR"]["t_start_sec"] == 2.5 and segs["aVR"]["t_end_sec"] == 5.0
    assert segs["V1"]["t_start_sec"] == 5.0 and segs["V1"]["t_end_sec"] == 7.5
    assert segs["V4"]["t_start_sec"] == 7.5 and segs["V4"]["t_end_sec"] == 10.0

    # Row 2 check
    assert segs["II"]["t_start_sec"] == 0.0 and segs["II"]["t_end_sec"] == 2.5
    assert segs["aVL"]["t_start_sec"] == 2.5 and segs["aVL"]["t_end_sec"] == 5.0
    assert segs["V2"]["t_start_sec"] == 5.0 and segs["V2"]["t_end_sec"] == 7.5
    assert segs["V5"]["t_start_sec"] == 7.5 and segs["V5"]["t_end_sec"] == 10.0

    # Row 3 check
    assert segs["III"]["t_start_sec"] == 0.0 and segs["III"]["t_end_sec"] == 2.5
    assert segs["aVF"]["t_start_sec"] == 2.5 and segs["aVF"]["t_end_sec"] == 5.0
    assert segs["V3"]["t_start_sec"] == 5.0 and segs["V3"]["t_end_sec"] == 7.5
    assert segs["V6"]["t_start_sec"] == 7.5 and segs["V6"]["t_end_sec"] == 10.0


def test_calibration_and_signal_metadata_fidelity(tmp_path):
    """
    Verify paper speed (25 mm/s), voltage gain (10 mm/mV), and source signal stats.
    """
    fs = 100
    test_sig = np.random.randn(1000, 12) * 0.5
    out_png = tmp_path / "fidelity_test.png"
    meta = render_12lead_ecg_image(
        signal=test_sig,
        sampling_rate=fs,
        output_image_path=out_png,
        paper_speed_mm_s=25.0,
        voltage_gain_mm_mv=10.0,
    )

    assert meta["paper_speed_mm_s"] == 25.0
    assert meta["voltage_gain_mm_mv"] == 10.0
    assert meta["duration_seconds"] == 10.0
    assert "source_signal_stats" in meta

    # Check signal stats match numpy source exactly
    for idx, lead in enumerate(STANDARD_LEAD_NAMES):
        stats = meta["source_signal_stats"][lead]
        assert pytest.approx(stats["mean_mv"], abs=1e-4) == float(np.mean(test_sig[:, idx]))
        assert pytest.approx(stats["min_mv"], abs=1e-4) == float(np.min(test_sig[:, idx]))
        assert pytest.approx(stats["max_mv"], abs=1e-4) == float(np.max(test_sig[:, idx]))


# ── 3. LUDB and MIT-BIH Regression Checks ───────────────────────────────────

def test_ludb_regression_check(ecg_data_root):
    ludb_loader = LUDBDatasetLoader(ecg_data_root / "ludb")
    assert len(ludb_loader.list_records()) == 200
    sig, fs, leads = ludb_loader.load_record(1)
    assert fs == 500
    assert leads == STANDARD_12_LEADS


def test_mitdb_regression_check(ecg_data_root):
    mit_loader = MITDBDatasetLoader(ecg_data_root / "mitdb")
    assert len(mit_loader.list_records()) == 48
    sig, fs, leads = mit_loader.load_record("100", sampto=360)
    assert fs == 360
    assert sig.shape == (360, 2)
