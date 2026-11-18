"""
ECG Clinical Measurements Engine.

Computes standard clinical electrocardiographic intervals and metrics using
Phase 5 waveform delineation outputs:
1. Heart Rate & RR Intervals (beat-to-beat RR, instantaneous HR, mean/median HR, irregularity).
2. PR Interval (P onset to QRS onset, with support for absent P/AFib/pacing warnings).
3. QRS Duration (QRS onset to QRS offset, per-beat and summary).
4. QT and QTc (Bazett and Fridericia correction formulas, RR provenance, irregularity gating).
5. ST-Segment Analysis (J-point + 60ms isoelectric reference, per-lead elevation/depression).
6. Quality & Provenance (units, formulas, confidence, warnings, no diagnostic overreach).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("ai_cdss.ecg.measurements")


class ECGMeasurementEngine:
    """
    Engine for extracting clinical ECG measurements from waveform delineation results.
    """

    def __init__(
        self,
        default_fs: int = 500,
        st_j_offset_ms: float = 60.0,
        rr_irregularity_threshold: float = 0.15,
    ):
        self.default_fs = default_fs
        self.st_j_offset_ms = st_j_offset_ms
        self.rr_irregularity_threshold = rr_irregularity_threshold

    def calculate_rr_and_heart_rate(
        self,
        r_peak_samples: List[int],
        fs: int,
        lead_name: str = "Lead",
    ) -> Dict[str, Any]:
        """
        Calculate beat-to-beat RR intervals, instantaneous heart rates, and summary HR metrics.

        Args:
            r_peak_samples: Ordered sample indices of detected R-peaks.
            fs: Sampling rate in Hz.
            lead_name: Name of lead analyzed.

        Returns:
            Structured dictionary with RR intervals, HR values, and rhythm regularity flags.
        """
        warnings: List[str] = []
        n_peaks = len(r_peak_samples)

        if n_peaks < 2:
            return {
                "lead": lead_name,
                "status": "UNAVAILABLE",
                "total_beats": n_peaks,
                "rr_intervals_ms": [],
                "instantaneous_hr_bpm": [],
                "mean_hr_bpm": None,
                "median_hr_bpm": None,
                "min_hr_bpm": None,
                "max_hr_bpm": None,
                "is_regular_rhythm": None,
                "rr_cov": None,
                "warnings": ["INSUFFICIENT_BEATS_FOR_RR_INTERVALS"],
            }

        # Calculate RR intervals in milliseconds
        rr_diffs = np.diff(r_peak_samples)
        rr_intervals_ms = [round(float(diff) / fs * 1000.0, 2) for diff in rr_diffs]

        # Instantaneous Heart Rate in BPM: 60,000 / RR_ms
        instantaneous_hr = [
            round(60000.0 / rr_ms, 1) if rr_ms > 0 else 0.0
            for rr_ms in rr_intervals_ms
        ]

        mean_rr = float(np.mean(rr_intervals_ms))
        std_rr = float(np.std(rr_intervals_ms))
        median_rr = float(np.median(rr_intervals_ms))

        # Coefficient of variation of RR intervals
        rr_cov = round(std_rr / mean_rr, 4) if mean_rr > 0 else 0.0

        # Irregular rhythm detection (e.g. AFib, sinus arrhythmia, ectopy)
        is_regular = rr_cov <= self.rr_irregularity_threshold
        if not is_regular:
            warnings.append("IRREGULAR_RR_INTERVALS_DETECTED")

        # Summary Heart Rate metrics
        mean_hr = round(60000.0 / mean_rr, 1) if mean_rr > 0 else None
        median_hr = round(60000.0 / median_rr, 1) if median_rr > 0 else None
        min_hr = round(float(np.min(instantaneous_hr)), 1)
        max_hr = round(float(np.max(instantaneous_hr)), 1)

        return {
            "lead": lead_name,
            "status": "VALID",
            "total_beats": n_peaks,
            "rr_intervals_ms": rr_intervals_ms,
            "instantaneous_hr_bpm": instantaneous_hr,
            "mean_hr_bpm": mean_hr,
            "median_hr_bpm": median_hr,
            "min_hr_bpm": min_hr,
            "max_hr_bpm": max_hr,
            "mean_rr_ms": round(mean_rr, 2),
            "std_rr_ms": round(std_rr, 2),
            "median_rr_ms": round(median_rr, 2),
            "rr_cov": rr_cov,
            "is_regular_rhythm": is_regular,
            "warnings": warnings,
        }

    def calculate_pr_interval(
        self,
        beat: Dict[str, Any],
        fs: int,
    ) -> Dict[str, Any]:
        """
        Calculate PR interval (P-wave onset to QRS onset) for an individual beat.

        Returns UNAVAILABLE if P-wave is absent, uncertain, or suppressed (e.g., AFib).
        """
        lead = beat.get("lead", "Unknown")
        beat_idx = beat.get("beat_index", 0)
        p_wave = beat.get("p")
        qrs = beat.get("qrs", {})

        if p_wave is None:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNAVAILABLE",
                "value_ms": None,
                "unit": "ms",
                "formula": "QRS_onset - P_onset",
                "source_boundaries": {
                    "p_onset_sample": None,
                    "qrs_onset_sample": qrs.get("onset_sample"),
                },
                "warnings": ["P_WAVE_ABSENT_OR_UNAVAILABLE"],
            }

        p_on = p_wave.get("onset_sample")
        qrs_on = qrs.get("onset_sample")

        if p_on is None or qrs_on is None or qrs_on <= p_on:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNRELIABLE",
                "value_ms": None,
                "unit": "ms",
                "formula": "QRS_onset - P_onset",
                "source_boundaries": {
                    "p_onset_sample": p_on,
                    "qrs_onset_sample": qrs_on,
                },
                "warnings": ["INVALID_PR_SEGMENT_ORDERING"],
            }

        pr_ms = round((qrs_on - p_on) / fs * 1000.0, 2)
        warnings: List[str] = []

        # Physiologic boundaries check: normal PR is typically 120-200 ms
        if pr_ms < 100.0:
            warnings.append("SHORT_PR_INTERVAL")
        elif pr_ms > 220.0:
            warnings.append("PROLONGED_PR_INTERVAL")

        return {
            "beat_index": beat_idx,
            "lead": lead,
            "status": "VALID",
            "value_ms": pr_ms,
            "unit": "ms",
            "formula": "QRS_onset - P_onset",
            "source_boundaries": {
                "p_onset_sample": int(p_on),
                "qrs_onset_sample": int(qrs_on),
            },
            "warnings": warnings,
        }

    def calculate_qrs_duration(
        self,
        beat: Dict[str, Any],
        fs: int,
    ) -> Dict[str, Any]:
        """
        Calculate QRS duration (QRS onset to QRS offset / J point) for an individual beat.
        """
        lead = beat.get("lead", "Unknown")
        beat_idx = beat.get("beat_index", 0)
        qrs = beat.get("qrs", {})

        qrs_on = qrs.get("onset_sample")
        qrs_off = qrs.get("offset_sample")

        if qrs_on is None or qrs_off is None or qrs_off <= qrs_on:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNAVAILABLE",
                "value_ms": None,
                "unit": "ms",
                "formula": "QRS_offset - QRS_onset",
                "source_boundaries": {
                    "qrs_onset_sample": qrs_on,
                    "qrs_offset_sample": qrs_off,
                },
                "warnings": ["INVALID_QRS_BOUNDARIES"],
            }

        duration_ms = round((qrs_off - qrs_on) / fs * 1000.0, 2)
        warnings: List[str] = []

        if duration_ms < 50.0:
            warnings.append("ATYPICAL_SHORT_QRS_DURATION")
        elif duration_ms > 120.0:
            warnings.append("WIDE_QRS_DURATION")

        return {
            "beat_index": beat_idx,
            "lead": lead,
            "status": "VALID",
            "value_ms": duration_ms,
            "unit": "ms",
            "formula": "QRS_offset - QRS_onset",
            "source_boundaries": {
                "qrs_onset_sample": int(qrs_on),
                "qrs_offset_sample": int(qrs_off),
            },
            "warnings": warnings,
        }

    def calculate_qt_and_qtc(
        self,
        beat: Dict[str, Any],
        fs: int,
        prior_rr_ms: Optional[float] = None,
        is_irregular_rhythm: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate QT interval (QRS onset to T-wave offset) and corrected QTc.

        Implements Bazett: QTc = QT / sqrt(RR_sec)
        Implements Fridericia: QTc = QT / cbrt(RR_sec)

        Flags unreliable QTc calculations during irregular rhythms or when prior RR is missing.
        """
        lead = beat.get("lead", "Unknown")
        beat_idx = beat.get("beat_index", 0)
        qrs = beat.get("qrs", {})
        t_wave = beat.get("t")

        if t_wave is None:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNAVAILABLE",
                "qt_status": "UNAVAILABLE",
                "qtc_status": "UNAVAILABLE",
                "qt_ms": None,
                "qtc_bazett_ms": None,
                "qtc_fridericia_ms": None,
                "unit": "ms",
                "formulas": {
                    "qt": "T_offset - QRS_onset",
                    "qtc_bazett": "QT / sqrt(RR_sec)",
                    "qtc_fridericia": "QT / cbrt(RR_sec)",
                },
                "reference_rr_ms": None,
                "source_boundaries": {
                    "qrs_onset_sample": qrs.get("onset_sample"),
                    "t_offset_sample": None,
                },
                "warnings": ["T_WAVE_ABSENT_OR_UNAVAILABLE"],
            }

        qrs_on = qrs.get("onset_sample")
        t_off = t_wave.get("offset_sample")

        if qrs_on is None or t_off is None or t_off <= qrs_on:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNRELIABLE",
                "qt_status": "UNRELIABLE",
                "qtc_status": "UNRELIABLE",
                "qt_ms": None,
                "qtc_bazett_ms": None,
                "qtc_fridericia_ms": None,
                "unit": "ms",
                "formulas": {
                    "qt": "T_offset - QRS_onset",
                    "qtc_bazett": "QT / sqrt(RR_sec)",
                    "qtc_fridericia": "QT / cbrt(RR_sec)",
                },
                "reference_rr_ms": None,
                "source_boundaries": {
                    "qrs_onset_sample": qrs_on,
                    "t_offset_sample": t_off,
                },
                "warnings": ["INVALID_QT_SEGMENT_BOUNDARIES"],
            }

        qt_ms = round((t_off - qrs_on) / fs * 1000.0, 2)
        warnings: List[str] = []
        qt_status = "VALID"

        # QTc calculation
        qtc_bazett: Optional[float] = None
        qtc_fridericia: Optional[float] = None
        qtc_status = "VALID"

        if prior_rr_ms is None or prior_rr_ms <= 0:
            qtc_status = "UNAVAILABLE"
            warnings.append("PRIOR_RR_UNAVAILABLE_FOR_QTC")
        else:
            rr_sec = prior_rr_ms / 1000.0
            qt_sec = qt_ms / 1000.0

            # Bazett: QT / sqrt(RR)
            qtc_bazett = round((qt_sec / np.sqrt(rr_sec)) * 1000.0, 2)

            # Fridericia: QT / cbrt(RR)
            qtc_fridericia = round((qt_sec / np.cbrt(rr_sec)) * 1000.0, 2)

            if is_irregular_rhythm:
                qtc_status = "UNRELIABLE"
                warnings.append("QTC_UNRELIABLE_DUE_TO_IRREGULAR_RHYTHM")

            # Extreme heart rate warnings for Bazett formula (known to overcorrect at HR > 100 and undercorrect at HR < 60)
            hr_prior = 60000.0 / prior_rr_ms
            if hr_prior > 100.0:
                warnings.append("EXTREME_TACHYCARDIA_BAZETT_OVERCORRECTION_RISK")
            elif hr_prior < 50.0:
                warnings.append("EXTREME_BRADYCARDIA_BAZETT_UNDERCORRECTION_RISK")

            if qtc_bazett > 470.0:
                warnings.append("PROLONGED_QTC_BAZETT")
            elif qtc_bazett < 350.0:
                warnings.append("SHORT_QTC_BAZETT")

        return {
            "beat_index": beat_idx,
            "lead": lead,
            "status": qt_status,
            "qt_status": qt_status,
            "qtc_status": qtc_status,
            "qt_ms": qt_ms,
            "qtc_bazett_ms": qtc_bazett,
            "qtc_fridericia_ms": qtc_fridericia,
            "unit": "ms",
            "formulas": {
                "qt": "T_offset - QRS_onset",
                "qtc_bazett": "QT / sqrt(RR_sec)",
                "qtc_fridericia": "QT / cbrt(RR_sec)",
            },
            "reference_rr_ms": prior_rr_ms,
            "source_boundaries": {
                "qrs_onset_sample": int(qrs_on),
                "t_offset_sample": int(t_off),
            },
            "warnings": warnings,
        }

    def measure_st_deviation(
        self,
        beat: Dict[str, Any],
        filtered_signal: np.ndarray,
        fs: int,
        is_calibrated: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure ST-segment deviation relative to the PR segment isoelectric baseline.

        Measurement point: J point (QRS offset) + 60 ms.
        Baseline: Median voltage of the PR segment preceding QRS onset.
        """
        lead = beat.get("lead", "Unknown")
        beat_idx = beat.get("beat_index", 0)
        qrs = beat.get("qrs", {})
        p_wave = beat.get("p")
        flags = beat.get("flags", [])

        warnings: List[str] = []
        if not is_calibrated:
            warnings.append("UNITS_PIXEL_WARNING")

        qrs_on = qrs.get("onset_sample")
        j_point = qrs.get("offset_sample")

        if j_point is None or qrs_on is None:
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNAVAILABLE",
                "st_deviation_mv": None,
                "st_deviation_pixel": None,
                "unit": "mV" if is_calibrated else "pixel",
                "j_point_sample": None,
                "measurement_sample": None,
                "isoelectric_baseline_mv": None,
                "warnings": ["QRS_BOUNDARIES_UNAVAILABLE"],
            }

        # 1. Estimate Isoelectric Baseline from PR segment
        # If P-wave offset is available, use [P_off, QRS_on].
        # Otherwise, use [QRS_on - 40ms, QRS_on].
        t_wave = beat.get("t")
        if p_wave is not None and p_wave.get("offset_sample") is not None:
            pr_start = p_wave["offset_sample"]
            pr_end = qrs_on
            # Check PR segment duration
            pr_seg_duration_ms = (pr_end - pr_start) / fs * 1000.0
            if pr_seg_duration_ms < 15.0:
                warnings.append("SHORT_PR_SEGMENT_FOR_ISOELECTRIC_BASELINE")
            # Sampling resolution check: at low fs (e.g. 100 Hz), PR segment spans very few samples
            if (pr_end - pr_start) < 3:
                warnings.append("INSUFFICIENT_SAMPLING_RESOLUTION_FOR_PR_BASELINE")
        else:
            pr_start = max(0, qrs_on - int(0.04 * fs))
            pr_end = qrs_on
            warnings.append("ISOELECTRIC_BASELINE_ESTIMATED_WITHOUT_P_OFFSET")

        if pr_end > pr_start:
            pr_sub = filtered_signal[pr_start:pr_end]
            baseline_val = float(np.median(pr_sub))
            # Baseline stability check (high variance or drift in isoelectric segment)
            if len(pr_sub) >= 3 and float(np.std(pr_sub)) > 0.05:
                warnings.append("UNSTABLE_ISOELECTRIC_BASELINE")
        else:
            baseline_val = float(filtered_signal[max(0, qrs_on - 1)])
            warnings.append("UNAVAILABLE_ISOELECTRIC_SEGMENT_FALLBACK_TO_ONSET")

        # 2. ST measurement point at J + 60 ms
        j_offset_samples = int(self.st_j_offset_ms / 1000.0 * fs)
        meas_sample = j_point + j_offset_samples

        # Guard: check if J+60ms extends beyond signal boundary or beyond T-wave peak
        if meas_sample >= len(filtered_signal):
            return {
                "beat_index": beat_idx,
                "lead": lead,
                "status": "UNAVAILABLE",
                "st_deviation_mv": None,
                "st_deviation_pixel": None,
                "unit": "mV" if is_calibrated else "pixel",
                "j_point_sample": j_point,
                "measurement_sample": meas_sample,
                "isoelectric_baseline_mv": round(baseline_val, 4) if is_calibrated else None,
                "warnings": warnings + ["ST_MEASUREMENT_POINT_TRUNCATED"],
            }

        # Guard: check if measurement sample encroaches past T wave peak
        if t_wave is not None and t_wave.get("peak_sample") is not None:
            t_pk = t_wave["peak_sample"]
            if meas_sample >= t_pk:
                warnings.append("ST_MEASUREMENT_POINT_EXTENDS_INTO_T_WAVE")

        st_voltage = float(filtered_signal[meas_sample])
        st_dev = st_voltage - baseline_val

        status = "VALID"
        if "LOW_CONFIDENCE_LEAD" in flags or "LOW_CONFIDENCE_LEAD_MEASUREMENT" in flags:
            status = "UNRELIABLE"
            warnings.append("LOW_CONFIDENCE_LEAD_MEASUREMENT")
        if "UNSTABLE_ISOELECTRIC_BASELINE" in warnings:
            status = "UNRELIABLE"
        if "INSUFFICIENT_SAMPLING_RESOLUTION_FOR_PR_BASELINE" in warnings:
            status = "UNRELIABLE"

        if is_calibrated:
            st_dev_mv = round(st_dev, 4)
            st_dev_px = None
            if st_dev_mv >= 0.10:
                warnings.append("ST_ELEVATION_EXCEEDS_100UV")
            elif st_dev_mv <= -0.10:
                warnings.append("ST_DEPRESSION_EXCEEDS_100UV")
        else:
            st_dev_mv = None
            st_dev_px = round(st_dev, 2)

        return {
            "beat_index": beat_idx,
            "lead": lead,
            "status": status,
            "st_deviation_mv": st_dev_mv,
            "st_deviation_pixel": st_dev_px,
            "unit": "mV" if is_calibrated else "pixel",
            "j_point_sample": int(j_point),
            "measurement_sample": int(meas_sample),
            "measurement_delay_ms": self.st_j_offset_ms,
            "isoelectric_baseline_mv": round(baseline_val, 4) if is_calibrated else None,
            "warnings": warnings,
        }

    def measure_lead(
        self,
        lead_delineation: Dict[str, Any],
        is_calibrated: bool = True,
    ) -> Dict[str, Any]:
        """
        Process clinical measurements for an individual lead's delineation output.
        """
        lead_name = lead_delineation.get("lead", "Unknown")
        fs = int(lead_delineation.get("sampling_rate_hz", self.default_fs))
        confidence = float(lead_delineation.get("confidence", 1.0))
        beats = lead_delineation.get("beats", [])
        filt_sig = np.array(lead_delineation.get("filtered_signal", []), dtype=np.float64)

        # 1. R-peaks and RR intervals
        r_peaks = [b["qrs"]["r_sample"] for b in beats if "qrs" in b and "r_sample" in b["qrs"]]
        rr_results = self.calculate_rr_and_heart_rate(r_peaks, fs, lead_name)
        rr_intervals = rr_results.get("rr_intervals_ms", [])
        is_irregular = rr_results.get("is_regular_rhythm") is False

        # 2. Per-beat measurements
        beat_measurements: List[Dict[str, Any]] = []
        valid_prs: List[float] = []
        valid_qrss: List[float] = []
        valid_qts: List[float] = []
        valid_qtcs_bazett: List[float] = []
        valid_qtcs_fridericia: List[float] = []
        valid_st_devs: List[float] = []

        for b_idx, beat in enumerate(beats):
            # Prior RR interval for QTc
            prior_rr = rr_intervals[b_idx - 1] if b_idx > 0 and b_idx - 1 < len(rr_intervals) else None

            pr_meas = self.calculate_pr_interval(beat, fs)
            qrs_meas = self.calculate_qrs_duration(beat, fs)
            qt_meas = self.calculate_qt_and_qtc(beat, fs, prior_rr, is_irregular)
            st_meas = self.measure_st_deviation(beat, filt_sig, fs, is_calibrated)

            if pr_meas["status"] == "VALID" and pr_meas["value_ms"] is not None:
                valid_prs.append(pr_meas["value_ms"])
            if qrs_meas["status"] == "VALID" and qrs_meas["value_ms"] is not None:
                valid_qrss.append(qrs_meas["value_ms"])
            if qt_meas["qt_ms"] is not None:
                valid_qts.append(qt_meas["qt_ms"])
            if qt_meas["qtc_bazett_ms"] is not None:
                valid_qtcs_bazett.append(qt_meas["qtc_bazett_ms"])
            if qt_meas["qtc_fridericia_ms"] is not None:
                valid_qtcs_fridericia.append(qt_meas["qtc_fridericia_ms"])
            if st_meas["status"] == "VALID" and st_meas["st_deviation_mv"] is not None:
                valid_st_devs.append(st_meas["st_deviation_mv"])

            beat_measurements.append({
                "beat_index": b_idx,
                "r_peak_sample": beat["qrs"]["r_sample"],
                "r_peak_ms": beat["qrs"]["r_ms"],
                "pr_interval": pr_meas,
                "qrs_duration": qrs_meas,
                "qt_qtc": qt_meas,
                "st_segment": st_meas,
            })

        # Summary statistics
        summary = {
            "mean_pr_ms": round(float(np.mean(valid_prs)), 2) if valid_prs else None,
            "median_pr_ms": round(float(np.median(valid_prs)), 2) if valid_prs else None,
            "mean_qrs_duration_ms": round(float(np.mean(valid_qrss)), 2) if valid_qrss else None,
            "median_qrs_duration_ms": round(float(np.median(valid_qrss)), 2) if valid_qrss else None,
            "mean_qt_ms": round(float(np.mean(valid_qts)), 2) if valid_qts else None,
            "mean_qtc_bazett_ms": round(float(np.mean(valid_qtcs_bazett)), 2) if valid_qtcs_bazett else None,
            "mean_qtc_fridericia_ms": round(float(np.mean(valid_qtcs_fridericia)), 2) if valid_qtcs_fridericia else None,
            "mean_st_deviation_mv": round(float(np.mean(valid_st_devs)), 4) if valid_st_devs else None,
        }

        return {
            "lead": lead_name,
            "confidence": confidence,
            "is_calibrated": is_calibrated,
            "sampling_rate_hz": fs,
            "rhythm_summary": rr_results,
            "summary_metrics": summary,
            "beat_measurements": beat_measurements,
        }

    def measure_bundle(
        self,
        delineation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a full clinical measurement bundle across all 12 leads and rhythm strip.
        """
        is_calibrated = delineation_result.get("is_calibrated", False)
        fs = int(delineation_result.get("sampling_rate_hz", self.default_fs))
        mean_conf = float(delineation_result.get("mean_lead_confidence", 1.0))

        # Rhythm strip measurements (Lead II 10s)
        rhythm_delineation = delineation_result.get("rhythm_strip")
        rhythm_meas: Optional[Dict[str, Any]] = None
        global_rr_summary: Optional[Dict[str, Any]] = None

        if rhythm_delineation is not None:
            rhythm_meas = self.measure_lead(rhythm_delineation, is_calibrated)
            global_rr_summary = rhythm_meas.get("rhythm_summary")

        # 12 quadrant leads measurements
        leads_delineation = delineation_result.get("leads", {})
        leads_meas: Dict[str, Any] = {}

        for lead_name, lead_delin in leads_delineation.items():
            leads_meas[lead_name] = self.measure_lead(lead_delin, is_calibrated)

        return {
            "measurement_status": "SUCCESS" if (rhythm_meas or leads_meas) else "FAILED",
            "is_calibrated": is_calibrated,
            "mean_extraction_confidence": mean_conf,
            "sampling_rate_hz": fs,
            "global_rhythm": global_rr_summary,
            "rhythm_strip": rhythm_meas,
            "leads": leads_meas,
        }
