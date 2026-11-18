"""
Unit and integration tests for Phase 6 — ECG Clinical Measurements Engine.

Tests:
1. Synthetic validation of RR intervals, instantaneous HR, mean/median HR, and irregular rhythms.
2. Synthetic validation of PR intervals, absent P-wave handling, and prolonged/short PR warnings.
3. Synthetic validation of QRS duration, normal and wide QRS complexes.
4. Synthetic validation of QT and corrected QTc (Bazett and Fridericia formulas) and irregular rhythm gating.
5. Synthetic validation of ST-segment deviation (positive elevation, negative depression, isoelectric baseline).
6. Quality and provenance tracking (units, formulas, warnings, pixel-domain vs. millivolt signals).
7. Benchmark on LUDB ground-truth digital recordings (Interval timing validation).
8. End-to-end clinical measurement integration on image-reconstructed 12-lead bundles.
"""

from pathlib import Path
import numpy as np
import pytest
import wfdb

from app.services.ecg.delineation import ECGWaveformDelineator
from app.services.ecg.measurements import ECGMeasurementEngine
from app.services.ecg.reconstruction import ECGWaveformReconstructor

LUDB_DIR = Path("ml/data/ecg/ludb/data")
SYNTHETIC_EVAL_IMG = Path("ml/data/ecg/sample_synthetic_12lead.png")


@pytest.fixture
def engine() -> ECGMeasurementEngine:
    return ECGMeasurementEngine(default_fs=500, st_j_offset_ms=60.0)


def test_synthetic_rr_and_heart_rate(engine: ECGMeasurementEngine):
    """Verify exact calculation of RR intervals, instantaneous HR, mean HR, and rhythm regularity."""
    fs = 500
    # Simulate regular R peaks at 75 bpm (RR = 800 ms = 400 samples)
    r_peaks_regular = [400, 800, 1200, 1600, 2000]
    res_reg = engine.calculate_rr_and_heart_rate(r_peaks_regular, fs, "Lead_II")

    assert res_reg["status"] == "VALID"
    assert res_reg["total_beats"] == 5
    assert len(res_reg["rr_intervals_ms"]) == 4
    for rr in res_reg["rr_intervals_ms"]:
        assert rr == 800.0
    for hr in res_reg["instantaneous_hr_bpm"]:
        assert hr == 75.0
    assert res_reg["mean_hr_bpm"] == 75.0
    assert res_reg["median_hr_bpm"] == 75.0
    assert res_reg["is_regular_rhythm"] is True
    assert len(res_reg["warnings"]) == 0

    # Simulate irregular R peaks (e.g. AFib or severe arrhythmia)
    r_peaks_irregular = [400, 750, 1300, 1650, 2200]  # RRs: 700ms, 1100ms, 700ms, 1100ms
    res_irreg = engine.calculate_rr_and_heart_rate(r_peaks_irregular, fs, "Lead_II")
    assert res_irreg["is_regular_rhythm"] is False
    assert "IRREGULAR_RR_INTERVALS_DETECTED" in res_irreg["warnings"]


def test_synthetic_pr_interval(engine: ECGMeasurementEngine):
    """Verify PR interval calculation, absence warnings, and physiological checks."""
    fs = 500

    # 1. Normal beat: P onset at 100ms (50 samples), QRS onset at 260ms (130 samples) -> PR = 160 ms
    beat_normal = {
        "lead": "II",
        "beat_index": 0,
        "p": {"onset_sample": 50, "peak_sample": 70, "offset_sample": 90},
        "qrs": {"onset_sample": 130, "r_sample": 160, "offset_sample": 180},
    }
    pr_normal = engine.calculate_pr_interval(beat_normal, fs)
    assert pr_normal["status"] == "VALID"
    assert pr_normal["value_ms"] == 160.0
    assert pr_normal["unit"] == "ms"
    assert len(pr_normal["warnings"]) == 0

    # 2. Absent P-wave (e.g. Atrial Fibrillation or junctional escape)
    beat_no_p = {
        "lead": "II",
        "beat_index": 1,
        "p": None,
        "qrs": {"onset_sample": 130, "r_sample": 160, "offset_sample": 180},
    }
    pr_no_p = engine.calculate_pr_interval(beat_no_p, fs)
    assert pr_no_p["status"] == "UNAVAILABLE"
    assert pr_no_p["value_ms"] is None
    assert "P_WAVE_ABSENT_OR_UNAVAILABLE" in pr_no_p["warnings"]

    # 3. Prolonged PR interval (> 220 ms)
    beat_first_degree_block = {
        "lead": "II",
        "beat_index": 2,
        "p": {"onset_sample": 0, "peak_sample": 20, "offset_sample": 40},
        "qrs": {"onset_sample": 130, "r_sample": 160, "offset_sample": 180},  # PR = 260 ms
    }
    pr_block = engine.calculate_pr_interval(beat_first_degree_block, fs)
    assert pr_block["value_ms"] == 260.0
    assert "PROLONGED_PR_INTERVAL" in pr_block["warnings"]


def test_synthetic_qrs_duration(engine: ECGMeasurementEngine):
    """Verify QRS duration measurement and wide QRS flagging."""
    fs = 500

    # Normal QRS duration: onset at 100 samples, offset at 145 samples -> 45 samples = 90 ms
    beat_normal = {
        "lead": "V1",
        "beat_index": 0,
        "qrs": {"onset_sample": 100, "r_sample": 125, "offset_sample": 145},
    }
    qrs_norm = engine.calculate_qrs_duration(beat_normal, fs)
    assert qrs_norm["status"] == "VALID"
    assert qrs_norm["value_ms"] == 90.0
    assert len(qrs_norm["warnings"]) == 0

    # Wide QRS duration (bundle branch block): 70 samples = 140 ms
    beat_wide = {
        "lead": "V1",
        "beat_index": 1,
        "qrs": {"onset_sample": 100, "r_sample": 135, "offset_sample": 170},
    }
    qrs_wide = engine.calculate_qrs_duration(beat_wide, fs)
    assert qrs_wide["value_ms"] == 140.0
    assert "WIDE_QRS_DURATION" in qrs_wide["warnings"]


def test_synthetic_qt_and_qtc_formulas(engine: ECGMeasurementEngine):
    """Verify exact Bazett and Fridericia correction formulas and irregular gating."""
    fs = 500
    # Beat: QRS onset at 100 samples, T offset at 300 samples -> QT = 200 samples = 400 ms (0.40 s)
    # Reference RR = 800 ms (0.80 s)
    beat = {
        "lead": "II",
        "beat_index": 0,
        "qrs": {"onset_sample": 100, "r_sample": 130, "offset_sample": 150},
        "t": {"onset_sample": 220, "peak_sample": 260, "offset_sample": 300},
    }
    qt_res = engine.calculate_qt_and_qtc(beat, fs, prior_rr_ms=800.0, is_irregular_rhythm=False)

    assert qt_res["status"] == "VALID"
    assert qt_res["qt_ms"] == 400.0

    # Bazett: 400 / sqrt(0.8) = 447.21 ms
    expected_bazett = round((0.400 / np.sqrt(0.800)) * 1000.0, 2)
    assert qt_res["qtc_bazett_ms"] == expected_bazett
    assert qt_res["qtc_bazett_ms"] == 447.21

    # Fridericia: 400 / cbrt(0.8) = 430.89 ms
    expected_fridericia = round((0.400 / np.cbrt(0.800)) * 1000.0, 2)
    assert qt_res["qtc_fridericia_ms"] == expected_fridericia
    assert qt_res["qtc_fridericia_ms"] == 430.89

    # Irregular rhythm gating: QTc must be flagged UNRELIABLE while raw QT remains VALID
    qt_irreg = engine.calculate_qt_and_qtc(beat, fs, prior_rr_ms=800.0, is_irregular_rhythm=True)
    assert qt_irreg["qt_status"] == "VALID"
    assert qt_irreg["qtc_status"] == "UNRELIABLE"
    assert "QTC_UNRELIABLE_DUE_TO_IRREGULAR_RHYTHM" in qt_irreg["warnings"]

    # Missing prior RR: raw QT must remain VALID while QTc is UNAVAILABLE
    qt_no_rr = engine.calculate_qt_and_qtc(beat, fs, prior_rr_ms=None, is_irregular_rhythm=False)
    assert qt_no_rr["qt_status"] == "VALID"
    assert qt_no_rr["qt_ms"] == 400.0
    assert qt_no_rr["qtc_status"] == "UNAVAILABLE"
    assert qt_no_rr["qtc_bazett_ms"] is None
    assert "PRIOR_RR_UNAVAILABLE_FOR_QTC" in qt_no_rr["warnings"]


def test_record4_beat0_qt_and_qtc_separation(engine: ECGMeasurementEngine):
    """Verify that Record 4 Beat 0 has valid raw QT despite missing preceding RR for QTc."""
    rec_path = str(LUDB_DIR / "4")
    rec = wfdb.rdrecord(rec_path)
    delineator = ECGWaveformDelineator()
    delin = delineator.delineate_lead(rec.p_signal[:, 1], rec.fs, "II", is_calibrated=True)
    meas = engine.measure_lead(delin, is_calibrated=True)

    b0_meas = meas["beat_measurements"][0]["qt_qtc"]
    # Raw QT must be valid and physiological (~408 ms)
    assert b0_meas["qt_status"] == "VALID"
    assert b0_meas["status"] == "VALID"
    assert b0_meas["qt_ms"] == 408.0
    # QTc must be unavailable due to absence of preceding RR interval
    assert b0_meas["qtc_status"] == "UNAVAILABLE"
    assert b0_meas["qtc_bazett_ms"] is None
    assert b0_meas["qtc_fridericia_ms"] is None
    assert "PRIOR_RR_UNAVAILABLE_FOR_QTC" in b0_meas["warnings"]


def test_synthetic_st_segment_elevation_and_depression(engine: ECGMeasurementEngine):
    """Verify ST deviation relative to the PR segment isoelectric baseline at J + 60 ms."""
    fs = 500
    signal_len = 1000
    # J offset = 60 ms = 30 samples

    # Case 1: ST Elevation (+0.25 mV)
    sig_elev = np.zeros(signal_len, dtype=np.float64)
    # PR segment baseline at 0.0 mV
    sig_elev[150:180] = 0.0
    # J point at sample 200, J+60ms is sample 230
    sig_elev[230] = 0.25

    beat_elev = {
        "lead": "V2",
        "beat_index": 0,
        "p": {"onset_sample": 100, "offset_sample": 150},
        "qrs": {"onset_sample": 180, "offset_sample": 200},
        "flags": [],
    }
    st_elev = engine.measure_st_deviation(beat_elev, sig_elev, fs, is_calibrated=True)
    assert st_elev["status"] == "VALID"
    assert st_elev["st_deviation_mv"] == 0.25
    assert "ST_ELEVATION_EXCEEDS_100UV" in st_elev["warnings"]

    # Case 2: ST Depression (-0.18 mV)
    sig_dep = np.zeros(signal_len, dtype=np.float64)
    sig_dep[150:180] = 0.05  # PR baseline at +0.05 mV
    sig_dep[230] = -0.13     # Voltage at J+60ms -> deviation is -0.13 - (+0.05) = -0.18 mV

    beat_dep = {
        "lead": "V5",
        "beat_index": 0,
        "p": {"onset_sample": 100, "offset_sample": 150},
        "qrs": {"onset_sample": 180, "offset_sample": 200},
        "flags": [],
    }
    st_dep = engine.measure_st_deviation(beat_dep, sig_dep, fs, is_calibrated=True)
    assert st_dep["status"] == "VALID"
    assert st_dep["st_deviation_mv"] == -0.18
    assert "ST_DEPRESSION_EXCEEDS_100UV" in st_dep["warnings"]


def test_st_insufficient_pr_sampling_resolution(engine: ECGMeasurementEngine):
    """Verify that fewer than 3 samples in PR segment triggers INSUFFICIENT_SAMPLING_RESOLUTION_FOR_PR_BASELINE and UNRELIABLE status."""
    fs = 100  # 10ms per sample
    signal_len = 500
    sig = np.zeros(signal_len, dtype=np.float64)
    # PR segment with only 2 samples (offset 58 to onset 60)
    beat = {
        "lead": "II",
        "beat_index": 0,
        "p": {"onset_sample": 40, "offset_sample": 58},
        "qrs": {"onset_sample": 60, "offset_sample": 70},
        "flags": [],
    }
    st_res = engine.measure_st_deviation(beat, sig, fs, is_calibrated=True)
    assert st_res["status"] == "UNRELIABLE"
    assert "INSUFFICIENT_SAMPLING_RESOLUTION_FOR_PR_BASELINE" in st_res["warnings"]


def test_quality_and_pixel_domain_safety(engine: ECGMeasurementEngine):
    """Verify that uncalibrated pixel-domain signals do not report millivolts."""
    fs = 100
    signal_len = 500
    pixel_sig = np.zeros(signal_len, dtype=np.float64)
    pixel_sig[100] = 50.0  # 50 pixels

    beat = {
        "lead": "Lead_I",
        "beat_index": 0,
        "p": {"onset_sample": 40, "offset_sample": 60},
        "qrs": {"onset_sample": 70, "offset_sample": 90},
        "flags": [],
    }
    st_pixel = engine.measure_st_deviation(beat, pixel_sig, fs, is_calibrated=False)
    assert st_pixel["st_deviation_mv"] is None
    assert st_pixel["st_deviation_pixel"] is not None
    assert st_pixel["unit"] == "pixel"
    assert "UNITS_PIXEL_WARNING" in st_pixel["warnings"]


def test_ludb_clinical_intervals_benchmark(engine: ECGMeasurementEngine):
    """Validate interval measurements on digital LUDB recordings against true physiological ranges."""
    delineator = ECGWaveformDelineator()

    for r in [1, 2, 3]:
        rec_path = str(LUDB_DIR / str(r))
        rec = wfdb.rdrecord(rec_path)
        delineation = delineator.delineate_lead(rec.p_signal[:, 1], rec.fs, "II", is_calibrated=True)
        meas = engine.measure_lead(delineation, is_calibrated=True)

        summary = meas["summary_metrics"]
        assert summary["mean_pr_ms"] is not None
        assert 100.0 <= summary["mean_pr_ms"] <= 240.0
        assert summary["mean_qrs_duration_ms"] is not None
        assert 60.0 <= summary["mean_qrs_duration_ms"] <= 140.0
        assert summary["mean_qt_ms"] is not None
        assert 300.0 <= summary["mean_qt_ms"] <= 550.0
        assert summary["mean_qtc_bazett_ms"] is not None
        assert 350.0 <= summary["mean_qtc_bazett_ms"] <= 500.0


def test_measure_bundle_on_reconstructed_image(engine: ECGMeasurementEngine):
    """Verify full 12-lead + rhythm clinical measurement bundle on image-reconstructed ECG."""
    reconstructor = ECGWaveformReconstructor()
    bundle = reconstructor.extract_traces(str(SYNTHETIC_EVAL_IMG))
    delineator = ECGWaveformDelineator()
    delineation_res = delineator.delineate_bundle(bundle)

    meas_bundle = engine.measure_bundle(delineation_res)

    assert meas_bundle["measurement_status"] == "SUCCESS"
    assert meas_bundle["is_calibrated"] is True
    assert meas_bundle["sampling_rate_hz"] == 100

    # Global rhythm
    global_rhythm = meas_bundle["global_rhythm"]
    assert global_rhythm["status"] == "VALID"
    assert 50.0 <= global_rhythm["mean_hr_bpm"] <= 100.0
    assert global_rhythm["is_regular_rhythm"] is True

    # 12 quadrant leads
    leads = meas_bundle["leads"]
    for lead_name in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        assert lead_name in leads
        assert "summary_metrics" in leads[lead_name]
        assert "beat_measurements" in leads[lead_name]
