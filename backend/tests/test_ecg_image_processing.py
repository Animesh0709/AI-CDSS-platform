import io
import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.ecg.image_processing import ECGImageProcessor


@pytest.fixture
def processor():
    return ECGImageProcessor(min_width=400, min_height=300, max_skew_degrees=15.0)


@pytest.fixture
def synthetic_ecg_path():
    path = REPO_ROOT / "ml" / "data" / "ecg" / "sample_synthetic_12lead.png"
    if not path.exists():
        pytest.skip(f"Synthetic ECG sample image not found at {path}")
    return path


# 1. Clean synthetic 12-lead image validation
def test_clean_synthetic_12lead_image(processor, synthetic_ecg_path, tmp_path):
    report = processor.process_and_audit(
        synthetic_ecg_path,
        output_dir=tmp_path / "ecg_out",
        base_filename="clean_test",
    )

    assert report["overall_status"] == "PASSED"
    assert report["image_quality_valid"] is True
    assert report["layout_geometry_valid"] is True
    assert report["lead_labels_valid"] is True
    assert report["calibration_valid"] is True
    assert report["quality_score"] >= 70
    assert report["format_validation"]["is_valid"] is True
    assert report["format_validation"]["width"] > 1000
    assert report["format_validation"]["height"] > 600
    assert report["grid_detection"]["grid_detected"] is True
    assert report["layout_validation"]["is_layout_valid"] is True
    assert report["layout_validation"]["all_12_leads_identified"] is True
    assert len(report["layout_validation"]["identified_lead_labels"]) == 12
    assert os.path.exists(report["processed_image_path"])
    assert os.path.exists(report["validation_report_path"])


# 1b. Positive calibration pulse detection test
def test_calibration_pulse_detected_on_synthetic_image(processor, synthetic_ecg_path):
    orig = Image.open(synthetic_ecg_path)
    grid_props = processor.detect_grid_properties(orig)
    cal_meta = processor.detect_calibration_pulse(orig, grid_props)

    assert cal_meta["calibration_pulse_detected"] is True
    assert cal_meta["pulse_width_px"] is not None and cal_meta["pulse_width_px"] > 0
    assert cal_meta["pulse_height_px"] is not None and cal_meta["pulse_height_px"] > 0
    assert cal_meta["estimated_paper_speed_mm_s"] is not None
    assert cal_meta["estimated_gain_mm_mv"] is not None

    # Paper speed within reasonable tolerance of nominal 25.0 mm/s (+/- 5.0)
    assert abs(cal_meta["estimated_paper_speed_mm_s"] - 25.0) <= 5.0
    # Voltage gain within reasonable tolerance of nominal 10.0 mm/mV (+/- 2.0)
    assert abs(cal_meta["estimated_gain_mm_mv"] - 10.0) <= 2.0


# 1c. Positive 12-lead recognition and layout validation test
def test_all_12_leads_identified_on_synthetic_image(processor, synthetic_ecg_path):
    orig = Image.open(synthetic_ecg_path)
    layout_meta = processor.validate_12lead_layout(orig)

    assert layout_meta["all_12_leads_identified"] is True
    assert layout_meta["is_layout_valid"] is True
    assert layout_meta["rhythm_strip_detected"] is True
    assert layout_meta["rhythm_strip_lead"] == "II"
    assert len(layout_meta["identified_lead_labels"]) == 12
    assert len(layout_meta["missing_or_unconfirmed_labels"]) == 0

    # Verify conventional 3x4 layout mapping
    mapping = layout_meta["lead_quadrant_mapping"]
    assert mapping["I"]["row"] == 0 and mapping["I"]["col"] == 0
    assert mapping["aVR"]["row"] == 0 and mapping["aVR"]["col"] == 1
    assert mapping["V1"]["row"] == 0 and mapping["V1"]["col"] == 2
    assert mapping["V4"]["row"] == 0 and mapping["V4"]["col"] == 3
    assert mapping["II"]["row"] == 1 and mapping["II"]["col"] == 0
    assert mapping["aVL"]["row"] == 1 and mapping["aVL"]["col"] == 1
    assert mapping["V2"]["row"] == 1 and mapping["V2"]["col"] == 2
    assert mapping["V5"]["row"] == 1 and mapping["V5"]["col"] == 3
    assert mapping["III"]["row"] == 2 and mapping["III"]["col"] == 0
    assert mapping["aVF"]["row"] == 2 and mapping["aVF"]["col"] == 1
    assert mapping["V3"]["row"] == 2 and mapping["V3"]["col"] == 2
    assert mapping["V6"]["row"] == 2 and mapping["V6"]["col"] == 3


# 1d. Semantic status separation test
def test_semantic_status_flags_separation(processor, synthetic_ecg_path):
    # On clean synthetic image: all four flags are True
    report = processor.process_and_audit(synthetic_ecg_path)
    assert report["overall_status"] == "PASSED"
    assert report["image_quality_valid"] is True
    assert report["layout_geometry_valid"] is True
    assert report["lead_labels_valid"] is True
    assert report["calibration_valid"] is True

    # Image without calibration pulse: image_quality_valid and layout_geometry_valid are True,
    # but calibration_valid is False -> overall_status becomes "WARNING" (not silently "PASSED")
    orig = Image.open(synthetic_ecg_path).copy()
    # Blank out top-left region to remove pulse
    arr = np.array(orig)
    arr[:300, :200] = 255
    no_cal_img = Image.fromarray(arr)

    report_no_cal = processor.process_and_audit(no_cal_img)
    assert report_no_cal["calibration_valid"] is False
    assert report_no_cal["calibration_detection"]["calibration_pulse_detected"] is False
    assert report_no_cal["overall_status"] == "WARNING"


# 2. Rotated and skewed image detection & conservative correction
def test_rotated_and_skewed_image(processor, synthetic_ecg_path, tmp_path):
    orig = Image.open(synthetic_ecg_path)
    # Apply a known 4.0 degree clockwise rotation
    rotated = orig.rotate(-4.0, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    rot_path = tmp_path / "skewed.png"
    rotated.save(rot_path)

    report = processor.process_and_audit(rot_path, output_dir=tmp_path)
    assert report["geometric_correction"]["detected_skew_degrees"] != 0.0
    # Detected angle should be close to -4.0 (within +/- 1.5 degrees)
    assert abs(report["geometric_correction"]["detected_skew_degrees"] - (-4.0)) <= 1.5
    assert report["geometric_correction"]["applied_correction_degrees"] != 0.0


# 3. Different image resolutions (Too low resolution rejected)
def test_low_resolution_image_rejection(processor, tmp_path):
    low_res = Image.new("RGB", (200, 150), color="white")
    low_res_path = tmp_path / "tiny_ecg.png"
    low_res.save(low_res_path)

    report = processor.process_and_audit(low_res_path)
    assert report["overall_status"] == "FAILED"
    assert report["format_validation"]["is_valid"] is False
    assert any("too low" in r for r in report["reasons_for_failure"])


# 4. Low-contrast and noisy image handling
def test_low_contrast_and_noisy_image(processor, synthetic_ecg_path, tmp_path):
    img = Image.open(synthetic_ecg_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    # Compress contrast dramatically and add Gaussian noise
    low_contrast = (arr * 0.2 + 100).clip(0, 255).astype(np.uint8)
    noise = np.random.normal(0, 15, low_contrast.shape)
    noisy_arr = np.clip(low_contrast + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    report = processor.process_and_audit(noisy_img, output_dir=tmp_path)
    # Noise and contrast should still be processed gracefully without crash
    assert report["enhancement"]["contrast_enhanced"] is True
    assert report["enhancement"]["median_denoised"] is True


# 5. Missing or ambiguous lead labels
def test_missing_or_ambiguous_lead_labels(processor, tmp_path):
    # Blank white page with simple lines, no lead labels text
    blank_grid = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    # Add horizontal lines simulating traces
    for y in [200, 400, 600]:
        blank_grid[y : y + 2, :] = 0
    blank_img = Image.fromarray(blank_grid)

    layout_meta = processor.validate_12lead_layout(blank_img)
    # Should flag missing lead labels
    assert len(layout_meta["identified_lead_labels"]) == 0
    assert len(layout_meta["missing_or_unconfirmed_labels"]) == 12


# 6. Missing calibration pulse detection
def test_missing_calibration_pulse_handling(processor, tmp_path):
    # An image with no calibration pulse on the left margin
    grid_img = Image.new("RGB", (1000, 600), color="white")
    grid_props = {"grid_detected": True, "estimated_major_grid_px": 50.0}

    cal_meta = processor.detect_calibration_pulse(grid_img, grid_props)
    assert cal_meta["calibration_pulse_detected"] is False
    assert cal_meta["estimated_paper_speed_mm_s"] is None
    assert cal_meta["estimated_gain_mm_mv"] is None
    assert any("UNKNOWN" in w for w in cal_meta["warnings"])


# 7. Corrupted and unsupported images
def test_corrupted_image_handling(processor):
    corrupt_bytes = b"\x89PNG\r\n\x1a\n" + b"incomplete_truncated_bytes"
    report = processor.process_and_audit(corrupt_bytes)
    assert report["overall_status"] == "FAILED"
    assert len(report["reasons_for_failure"]) > 0


# 8. Unreliable grid detection
def test_unreliable_grid_detection(processor):
    # Pure flat gray image with zero grid lines
    flat_img = Image.new("RGB", (800, 600), color=(200, 200, 200))
    grid_meta = processor.detect_grid_properties(flat_img)
    assert grid_meta["grid_detected"] is False
    assert grid_meta["grid_detection_confidence"] <= 0.3
    assert len(grid_meta["warnings"]) > 0
