import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.ecg.reconstruction import ECGWaveformReconstructor
from app.services.ecg.image_processing import ECGImageProcessor, CONVENTIONAL_3X4_LEADS, ALL_12_LEADS
from ml.scripts.ecg.ptbxl_loader import PTBXLDatasetLoader


@pytest.fixture
def reconstructor():
    return ECGWaveformReconstructor(
        target_sampling_rate=100,
        lead_segment_duration_sec=2.5,
        rhythm_strip_duration_sec=10.0,
    )


@pytest.fixture
def sample1_ecg_path():
    path = REPO_ROOT / "ml" / "data" / "ecg" / "sample_synthetic_12lead.png"
    if not path.exists():
        pytest.skip(f"Synthetic ECG sample image not found at {path}")
    return path


@pytest.fixture
def sample2_ecg_path():
    path = REPO_ROOT / "ml" / "data" / "ecg" / "sample_synthetic_12lead_eval.png"
    if not path.exists():
        pytest.skip(f"Evaluation ECG sample image not found at {path}")
    return path


@pytest.fixture
def ptbxl_loader():
    path = REPO_ROOT / "ml" / "data" / "ecg" / "ptbxl"
    if not path.exists():
        pytest.skip("PTB-XL data directory not found")
    return PTBXLDatasetLoader(path)


# 1. Test all 12 lead extractions
def test_all_12_lead_extractions(reconstructor, sample1_ecg_path):
    result = reconstructor.extract_traces(sample1_ecg_path)

    assert result["reconstruction_status"] == "SUCCESS"
    assert result["is_calibrated"] is True
    assert result["voltage_units"] == "millivolts"
    assert result["time_units"] == "seconds"

    # All 12 leads present
    for lead in ALL_12_LEADS:
        assert lead in result["signals"], f"Missing reconstructed lead: {lead}"
        sig = result["signals"][lead]
        assert isinstance(sig, np.ndarray)
        assert len(sig) == 250  # 2.5s at 100Hz
        assert not np.all(np.isnan(sig)), f"Signal for {lead} is all NaNs"

    # Rhythm strip present
    assert result["rhythm_strip"] is not None
    assert len(result["rhythm_strip"]) == 1000  # 10s at 100Hz


# 2. Test correct lead-to-region mapping
def test_lead_to_region_mapping(reconstructor, sample1_ecg_path):
    result = reconstructor.extract_traces(sample1_ecg_path)
    lead_meta = result["lead_metadata"]

    for r in range(3):
        for c in range(4):
            expected_lead = CONVENTIONAL_3X4_LEADS[r][c]
            meta = lead_meta[expected_lead]
            assert meta["row"] == r, f"Expected {expected_lead} in row {r}, got {meta['row']}"
            assert meta["col"] == c, f"Expected {expected_lead} in col {c}, got {meta['col']}"


# 3. Test positive and negative waveform polarity
def test_waveform_polarity_preservation(reconstructor, sample1_ecg_path):
    result = reconstructor.extract_traces(sample1_ecg_path)
    signals = result["signals"]

    # Lead I in sample 1 has positive R-peak (should be > 0.3 mV)
    assert np.max(signals["I"]) > 0.3

    # Lead V1 in sample 1 has predominantly negative deflection (min < -0.3 mV)
    assert np.min(signals["V1"]) < -0.3

    # Lead aVR in standard ECG is typically inverted (min < -0.2 mV)
    assert np.min(signals["aVR"]) < -0.2


# 4. Test grid removal without trace destruction
def test_grid_removal_without_trace_destruction(reconstructor, sample1_ecg_path):
    result = reconstructor.extract_traces(sample1_ecg_path)
    for lead, meta in result["lead_metadata"].items():
        # Missing pixel fraction should be tiny (< 5%) on calibrated clean scans
        assert meta["missing_pixel_fraction"] < 0.05
        # Dynamic confidence score reflects lead quality (clean leads >= 0.88, mean >= 0.90)
        assert meta["confidence_score"] >= 0.85
    assert result["mean_lead_confidence"] >= 0.90


# 5. Test calibration-based voltage and time conversion
def test_calibration_based_voltage_and_time_conversion(reconstructor, sample1_ecg_path):
    result = reconstructor.extract_traces(sample1_ecg_path)
    assert result["is_calibrated"] is True
    assert result["voltage_units"] == "millivolts"
    assert result["time_units"] == "seconds"

    # Physiological range check for normal human ECG: voltages should lie within [-5.0, 5.0] mV
    for lead, sig in result["signals"].items():
        assert np.min(sig) >= -5.0, f"{lead} voltage too negative: {np.min(sig)}"
        assert np.max(sig) <= 5.0, f"{lead} voltage too high: {np.max(sig)}"


# 6. Test unknown calibration fallback handling
def test_unknown_calibration_handling(reconstructor, sample1_ecg_path):
    # Simulated audit with calibration_valid=False and no pulse
    fake_audit = {
        "overall_status": "WARNING",
        "image_quality_valid": True,
        "layout_geometry_valid": True,
        "lead_labels_valid": True,
        "calibration_valid": False,
        "calibration_detection": {
            "calibration_pulse_detected": False,
            "estimated_paper_speed_mm_s": None,
            "estimated_gain_mm_mv": None,
        },
    }

    result = reconstructor.extract_traces(sample1_ecg_path, validation_audit=fake_audit)
    assert result["is_calibrated"] is False
    assert result["voltage_units"] == "pixels"
    assert "UNKNOWN" in result["warnings"][0]
    assert any("pixel-domain" in w for w in result["warnings"])


# 7. Test missing waveform segments / interpolation
def test_missing_waveform_segments_handling(reconstructor, sample1_ecg_path):
    orig = Image.open(sample1_ecg_path).convert("RGB")
    arr = np.array(orig)

    # Intentionally erase a 30-pixel vertical chunk from Lead I (x in [100, 130])
    arr[180:330, 100:130] = [255, 255, 255]
    corrupted_img = Image.fromarray(arr)

    result = reconstructor.extract_traces(corrupted_img)
    sig_i = result["signals"]["I"]

    # Even with missing chunk, interpolation should ensure no NaNs in final time-series
    assert not np.any(np.isnan(sig_i))
    assert len(sig_i) == 250
    assert result["lead_metadata"]["I"]["missing_pixel_fraction"] > 0.05


# 8. Test synthetic source-versus-reconstructed signal comparison (Record 1)
def test_synthetic_ground_truth_comparison_sample1(reconstructor, sample1_ecg_path, ptbxl_loader):
    result = reconstructor.extract_traces(sample1_ecg_path)
    gt_sig, fs, leads = ptbxl_loader.load_record_signal(1, resolution="100")

    eval_result = reconstructor.evaluate_against_ground_truth(result, gt_sig, leads, fs_gt=fs)

    # High correlation across 12 leads
    assert eval_result["mean_correlation"] >= 0.75, f"Mean correlation low: {eval_result['mean_correlation']}"
    assert eval_result["mean_mae_mv"] <= 0.05, f"Mean MAE too high: {eval_result['mean_mae_mv']}"

    # Check key leads individually
    metrics = eval_result["per_lead_metrics"]
    assert metrics["I"]["correlation"] > 0.80
    assert metrics["aVR"]["correlation"] > 0.85
    assert metrics["V1"]["correlation"] > 0.85
    assert metrics["II"]["correlation"] > 0.85
    assert metrics["Lead_II_Rhythm"]["correlation"] > 0.85


# 9. Test separate evaluation on independent sample 2 (Reported separately)
def test_independent_evaluation_sample2(reconstructor, sample2_ecg_path, ptbxl_loader):
    result = reconstructor.extract_traces(sample2_ecg_path)
    gt_sig, fs, leads = ptbxl_loader.load_record_signal(2, resolution="100")

    eval_result = reconstructor.evaluate_against_ground_truth(result, gt_sig, leads, fs_gt=fs)
    # Mean correlation should exceed 0.65 with Lead I resolution
    assert eval_result["mean_correlation"] >= 0.65
    assert len(eval_result["per_lead_metrics"]) == 13  # 12 leads + continuous rhythm

    # Verify Lead I accuracy on Sample 2
    metrics = eval_result["per_lead_metrics"]
    assert metrics["I"]["correlation"] > 0.85, f"Lead I correlation too low: {metrics['I']['correlation']}"
    assert metrics["I"]["mae_mv"] < 0.025, f"Lead I MAE too high: {metrics['I']['mae_mv']}"


# 10. Test dynamic confidence scoring and clipping penalty
def test_dynamic_confidence_scoring(reconstructor, sample1_ecg_path, sample2_ecg_path):
    res1 = reconstructor.extract_traces(sample1_ecg_path)
    res2 = reconstructor.extract_traces(sample2_ecg_path)

    # Clean scan should have high overall confidence
    assert res1["mean_lead_confidence"] >= 0.90
    assert res1["reconstruction_status"] == "SUCCESS"

    # In Sample 2, high variance/clipping leads like V2/V3 should have lower confidence than clean leads
    meta2 = res2["lead_metadata"]
    assert meta2["I"]["confidence_score"] > meta2["V2"]["confidence_score"]
    assert meta2["aVR"]["confidence_score"] > meta2["V3"]["confidence_score"]


# 11. Test bundle persistence (NumPy NPZ and JSON metadata)
def test_save_reconstruction_bundle(reconstructor, sample1_ecg_path, tmp_path):
    result = reconstructor.extract_traces(sample1_ecg_path)
    bundle = reconstructor.save_reconstruction_bundle(
        result, output_dir=tmp_path / "recon_bundle", base_name="test_ecg"
    )

    assert os.path.exists(bundle["signals_npz"])
    assert os.path.exists(bundle["metadata_json"])

    # Load npz and verify arrays
    loaded_npz = np.load(bundle["signals_npz"])
    for lead in ALL_12_LEADS:
        assert f"lead_{lead}" in loaded_npz
        assert len(loaded_npz[f"lead_{lead}"]) == 250
    assert "lead_II_rhythm" in loaded_npz
    assert len(loaded_npz["lead_II_rhythm"]) == 1000

    # Load json and verify metadata
    with open(bundle["metadata_json"]) as f:
        meta = json.load(f)
    assert meta["reconstruction_status"] == "SUCCESS"
    assert meta["is_calibrated"] is True
    assert len(meta["lead_metadata"]) == 12
