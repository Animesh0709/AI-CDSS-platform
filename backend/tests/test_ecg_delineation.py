"""
Unit and integration tests for Phase 5 — ECG Waveform Delineation.

Tests:
1. Signal preprocessing (baseline wander removal and power-line notch).
2. QRS complex detection and R/Q/S peak identification.
3. P-wave delineation (positive, inverted, biphasic, absent/AFib cases).
4. T-wave delineation (positive, inverted, morphology preservation).
5. Quality gating (uncalibrated pixel-domain, low confidence, truncated signals).
6. Multi-lead consensus and bundle delineation from Phase 4 reconstructed signals.
7. Ground-truth benchmark against LUDB reference annotations across multiple records.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import wfdb

from app.services.ecg.delineation import ECGWaveformDelineator
from app.services.ecg.reconstruction import ECGWaveformReconstructor


LUDB_DIR = Path("ml/data/ecg/ludb/data")
SYNTHETIC_EVAL_IMG = Path("ml/data/ecg/sample_synthetic_12lead.png")


@pytest.fixture
def delineator() -> ECGWaveformDelineator:
    return ECGWaveformDelineator(default_fs=500)


def test_signal_preprocessing_baseline_and_notch(delineator: ECGWaveformDelineator):
    """Verify conservative baseline wander removal and power-line notch filtering."""
    fs = 500
    t = np.linspace(0, 4.0, int(4.0 * fs))
    # Synthetic baseline drift (0.2 Hz slow sine) + power-line noise (50 Hz) + synthetic ECG spike
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    powerline_noise = 0.1 * np.sin(2 * np.pi * 50.0 * t)
    clean_spike = np.zeros_like(t)
    clean_spike[int(1.0 * fs) : int(1.0 * fs) + 20] = 1.0  # 40ms pulse

    raw_signal = clean_spike + baseline_drift + powerline_noise

    filtered, estimated_baseline = delineator.preprocess_signal(raw_signal, fs=fs, notch_freq=50.0)

    assert len(filtered) == len(raw_signal)
    assert len(estimated_baseline) == len(raw_signal)

    # Baseline wander should capture the slow 0.2 Hz drift
    drift_corr = np.corrcoef(estimated_baseline, baseline_drift)[0, 1]
    assert drift_corr > 0.85, f"Baseline drift estimation correlation too low: {drift_corr}"

    # Filtered signal should remove baseline offset near the spike
    assert abs(np.mean(filtered[:int(0.8 * fs)])) < 0.05


def test_qrs_detection_and_r_peak_identification(delineator: ECGWaveformDelineator):
    """Verify QRS detection and peak localization on LUDB Record 1 Lead II."""
    rec_path = str(LUDB_DIR / "1")
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "ii")

    lead_ii_signal = rec.p_signal[:, 1]
    res = delineator.delineate_lead(
        lead_signal=lead_ii_signal,
        fs=rec.fs,
        lead_name="II",
        confidence=1.0,
        is_calibrated=True,
    )

    assert res["is_calibrated"] is True
    assert len(res["beats"]) >= 6

    # Compare detected R peaks with reference annotations
    gt_r_peaks = [ann.sample[i] for i, s in enumerate(ann.symbol) if s == "N"]
    det_r_peaks = [b["qrs"]["r_sample"] for b in res["beats"]]

    # Every ground truth peak should have a detected match within 20 ms (10 samples at 500 Hz)
    for gt_r in gt_r_peaks:
        min_dist = min(abs(d_r - gt_r) for d_r in det_r_peaks)
        assert min_dist <= 10, f"R peak mismatch: ground truth {gt_r}, min dist {min_dist} samples"

    # Verify Q and S samples are identified for each beat
    for beat in res["beats"]:
        qrs = beat["qrs"]
        assert qrs["onset_sample"] <= qrs["q_sample"] <= qrs["r_sample"] <= qrs["s_sample"] <= qrs["offset_sample"]
        assert 40.0 <= qrs["duration_ms"] <= 200.0


def test_negative_dominant_qrs_morphology(delineator: ECGWaveformDelineator):
    """Verify delineation handles inverted/negative dominant QRS complexes (Lead V1)."""
    rec_path = str(LUDB_DIR / "1")
    rec = wfdb.rdrecord(rec_path)
    v1_idx = rec.sig_name.index("v1")
    v1_signal = rec.p_signal[:, v1_idx]

    res = delineator.delineate_lead(
        lead_signal=v1_signal,
        fs=rec.fs,
        lead_name="V1",
        confidence=1.0,
        is_calibrated=True,
    )

    assert len(res["beats"]) >= 6
    # V1 in Record 1 has negative dominant polarity
    negative_beats = [b for b in res["beats"] if b["qrs"]["polarity"] == "negative"]
    assert len(negative_beats) >= len(res["beats"]) // 2


def test_absent_p_wave_in_atrial_fibrillation(delineator: ECGWaveformDelineator):
    """Verify that absent P-waves (e.g., in AFib LUDB Record 8) are not falsely forced."""
    rec_path = str(LUDB_DIR / "8")
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "ii")

    # In LUDB Record 8, there are 0 reference P-waves
    gt_p_count = ann.symbol.count("p")
    assert gt_p_count == 0, "LUDB Record 8 Lead II must have 0 reference P-waves"

    res = delineator.delineate_lead(
        lead_signal=rec.p_signal[:, 1],
        fs=rec.fs,
        lead_name="II",
        confidence=1.0,
        is_calibrated=True,
    )

    # In AFib, the delineator must not fabricate normal sinus P-waves
    p_waves_found = [b["p"] for b in res["beats"] if b["p"] is not None]
    assert len(p_waves_found) == 0, f"Expected 0 P-waves in AFib record 8, but detected {len(p_waves_found)}"

    # Check flags indicate absent P-wave or suppressed pacing spikes
    flags_summary = [f for b in res["beats"] for f in b["flags"]]
    assert any("P_WAVE_ABSENT" in f or "PACING_SPIKE" in f for f in flags_summary)


def test_t_wave_delineation(delineator: ECGWaveformDelineator):
    """Verify T-wave onset, peak, and offset delineation on LUDB Record 1."""
    rec_path = str(LUDB_DIR / "1")
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "ii")

    res = delineator.delineate_lead(
        lead_signal=rec.p_signal[:, 1],
        fs=rec.fs,
        lead_name="II",
        confidence=1.0,
        is_calibrated=True,
    )

    gt_t_peaks = [ann.sample[i] for i, s in enumerate(ann.symbol) if s == "t"]
    det_t_peaks = [b["t"]["peak_sample"] for b in res["beats"] if b.get("t") is not None]

    # All 5 true T waves should be matched within 50 ms (25 samples at 500 Hz)
    for gt_t in gt_t_peaks:
        min_dist = min(abs(d_t - gt_t) for d_t in det_t_peaks)
        assert min_dist <= 25, f"T peak mismatch: ground truth {gt_t}, min dist {min_dist} samples"

    for b in res["beats"]:
        if b.get("t") is not None:
            t_wave = b["t"]
            assert t_wave["onset_sample"] <= t_wave["peak_sample"] <= t_wave["offset_sample"]
            assert 60.0 <= t_wave["duration_ms"] <= 350.0


def test_quality_gating_uncalibrated_and_low_confidence(delineator: ECGWaveformDelineator):
    """Verify proper warnings and flags for uncalibrated and low-confidence leads."""
    fs = 100
    test_sig = np.sin(np.linspace(0, 8 * np.pi, 300)) * 40.0

    # Case 1: Uncalibrated signal (voltage in pixels)
    res_uncal = delineator.delineate_lead(
        lead_signal=test_sig,
        fs=fs,
        lead_name="I",
        confidence=0.85,
        is_calibrated=False,
    )
    assert res_uncal["is_calibrated"] is False
    assert "UNITS_PIXEL_WARNING" in res_uncal["flags"]
    if len(res_uncal["beats"]) > 0:
        beat0 = res_uncal["beats"][0]
        assert beat0["qrs"]["amplitude_mv"] is None
        assert beat0["qrs"]["amplitude_pixel"] is not None

    # Case 2: Low-confidence lead (< 0.60)
    res_low_conf = delineator.delineate_lead(
        lead_signal=test_sig,
        fs=fs,
        lead_name="V2",
        confidence=0.35,
        is_calibrated=True,
    )
    assert "LOW_CONFIDENCE_LEAD" in res_low_conf["flags"]

    # Case 3: Truncated signal
    short_sig = np.array([0.0, 1.0, 0.0])
    res_short = delineator.delineate_lead(
        lead_signal=short_sig,
        fs=fs,
        lead_name="III",
        confidence=0.90,
        is_calibrated=True,
    )
    assert "SIGNAL_TOO_SHORT" in res_short["flags"]
    assert len(res_short["beats"]) == 0


def test_ludb_ground_truth_multi_record_benchmark(delineator: ECGWaveformDelineator):
    """
    Benchmark wave detection against LUDB ground truth reference annotations
    across records 1 to 10 on Lead II.
    
    Verifies Sensitivity >= 90%, PPV >= 80%, F1 >= 85% for QRS, P, and T waves.
    """
    total_tp = {"qrs": 0, "p": 0, "t": 0}
    total_fp = {"qrs": 0, "p": 0, "t": 0}
    total_fn = {"qrs": 0, "p": 0, "t": 0}

    for r in range(1, 11):
        rec_path = str(LUDB_DIR / str(r))
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "ii")

        lead_res = delineator.delineate_lead(
            lead_signal=rec.p_signal[:, 1],
            fs=rec.fs,
            lead_name="II",
            confidence=1.0,
            is_calibrated=True,
        )

        metrics = delineator.evaluate_against_ludb_annotations(
            delineated_beats=lead_res["beats"],
            ground_truth_symbols=ann.symbol,
            ground_truth_samples=ann.sample,
            fs=rec.fs,
            tolerance_ms=150.0,
        )

        for wave_key in ["qrs", "p", "t"]:
            m = metrics[wave_key]
            total_tp[wave_key] += m["true_positives"]
            total_fp[wave_key] += m["false_positives"]
            total_fn[wave_key] += m["false_negatives"]

    # Calculate overall metrics
    for wave_key in ["qrs", "p", "t"]:
        tp = total_tp[wave_key]
        fp = total_fp[wave_key]
        fn = total_fn[wave_key]

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0

        if wave_key == "qrs":
            assert sens >= 0.98, f"QRS Sensitivity {sens:.4f} < 0.98"
            assert ppv >= 0.95, f"QRS PPV {ppv:.4f} < 0.95"
            assert f1 >= 0.98, f"QRS F1 {f1:.4f} < 0.98"
        elif wave_key == "p":
            assert sens >= 0.90, f"P-wave Sensitivity {sens:.4f} < 0.90"
            assert ppv >= 0.90, f"P-wave PPV {ppv:.4f} < 0.90"
            assert f1 >= 0.90, f"P-wave F1 {f1:.4f} < 0.90"
        elif wave_key == "t":
            assert sens >= 0.95, f"T-wave Sensitivity {sens:.4f} < 0.95"
            assert ppv >= 0.95, f"T-wave PPV {ppv:.4f} < 0.95"
            assert f1 >= 0.95, f"T-wave F1 {f1:.4f} < 0.95"


def test_delineate_bundle_from_reconstructed_image(delineator: ECGWaveformDelineator):
    """Verify delineation on a Phase 4 reconstructed signal bundle from an actual image."""
    reconstructor = ECGWaveformReconstructor()
    bundle = reconstructor.extract_traces(str(SYNTHETIC_EVAL_IMG))

    assert bundle["reconstruction_status"].lower() == "success"
    assert bundle["is_calibrated"] is True

    delineation_result = delineator.delineate_bundle(bundle)

    assert delineation_result["delineation_status"] == "success"
    assert delineation_result["is_calibrated"] is True
    assert delineation_result["total_detected_beats"] > 20
    assert "rhythm_strip" in delineation_result
    assert len(delineation_result["rhythm_strip"]["beats"]) >= 8

    # Check 12 leads are all processed
    leads = delineation_result["leads"]
    for required_lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        assert required_lead in leads
        assert len(leads[required_lead]["raw_signal"]) > 0
        assert len(leads[required_lead]["filtered_signal"]) > 0
        assert len(leads[required_lead]["baseline_wander"]) > 0
