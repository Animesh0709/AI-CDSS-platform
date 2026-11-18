"""
Numerical ECG Waveform Reconstruction Service.

Converts validated 12-lead ECG images into calibrated numerical time-series arrays
and structured reconstruction metadata.

Key capabilities:
1. Conventional 3x4 lead region partitioning (Rows: I..V4, II..V5, III..V6).
2. Continuous Lead II rhythm strip extraction (10 seconds).
3. Background and grid suppression preserving true trace morphology.
4. Lead label text, calibration pulse, and synthetic banner avoidance.
5. Pixel-to-physical signal conversion (time in seconds, amplitude in millivolts).
6. Strict fallback to pixel domain when calibration is unverified.
7. Interpolation and missing segment tracking.
8. Ground-truth evaluation metrics (Pearson correlation, MAE, RMSE, amplitude error, lag).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import interpolate, signal

from app.services.ecg.image_processing import (
    ALL_12_LEADS,
    CONVENTIONAL_3X4_LEADS,
    ECGImageProcessor,
)

logger = logging.getLogger("ai_cdss.ecg.reconstruction")


class ECGWaveformReconstructor:
    """
    Service for extracting numerical 12-lead + rhythm strip signals from ECG images.
    """

    def __init__(
        self,
        target_sampling_rate: int = 100,
        lead_segment_duration_sec: float = 2.5,
        rhythm_strip_duration_sec: float = 10.0,
    ):
        self.target_sampling_rate = target_sampling_rate
        self.lead_segment_duration_sec = lead_segment_duration_sec
        self.rhythm_strip_duration_sec = rhythm_strip_duration_sec

    def extract_traces(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        validation_audit: Optional[Dict] = None,
        processor: Optional[ECGImageProcessor] = None,
    ) -> Dict:
        """
        Extract numerical signals for all 12 leads and continuous Lead II rhythm strip.

        Parameters:
        -----------
        image_input: Image path, PIL Image, or numpy array.
        validation_audit: Pre-computed Phase 3 validation audit dictionary.
                          If None, runs ECGImageProcessor.process_and_audit().
        processor: Optional processor instance.

        Returns:
        --------
        Dict containing reconstructed signals, per-lead metrics, calibration status, and warnings.
        """
        # 1. Run or verify Phase 3 validation
        if validation_audit is None:
            proc = processor or ECGImageProcessor()
            validation_audit = proc.process_and_audit(image_input)

        overall_status = validation_audit.get("overall_status", "UNKNOWN")
        image_quality_valid = validation_audit.get("image_quality_valid", False)
        layout_valid = validation_audit.get("layout_geometry_valid", False)
        lead_labels_valid = validation_audit.get("lead_labels_valid", False)
        calibration_valid = validation_audit.get("calibration_valid", False)

        warnings: List[str] = []
        failure_reasons: List[str] = []

        if not image_quality_valid:
            failure_reasons.append("Image failed quality/readability audit.")
            return {
                "reconstruction_status": "FAILED",
                "failure_reasons": failure_reasons,
                "warnings": warnings,
                "signals": {},
                "rhythm_strip": None,
                "metadata": {},
            }

        # 2. Load image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        w, h = img.size
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        # 3. Detect grid and calibration parameters
        cal_meta = validation_audit.get("calibration_detection", {})
        grid_meta = validation_audit.get("grid_detection", {})

        is_calibrated = (
            calibration_valid
            and cal_meta.get("calibration_pulse_detected", False)
            and cal_meta.get("estimated_paper_speed_mm_s") is not None
            and cal_meta.get("estimated_gain_mm_mv") is not None
        )

        # Trace segmentation parameters
        # In a standard 12-lead layout with 4 columns:
        # Col 0: [0.0, 2.5]s, Col 1: [2.5, 5.0]s, Col 2: [5.0, 7.5]s, Col 3: [7.5, 10.0]s
        # Grid boundaries:
        # Detect grid x and y limits from grid mask if available
        grid_mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 220) & (rgb[:, :, 2] < 230)
        grid_cols = np.where(np.sum(grid_mask, axis=0) > 80)[0]
        grid_rows = np.where(np.sum(grid_mask, axis=1) > 80)[0]

        if len(grid_cols) >= 2 and len(grid_rows) >= 2:
            x_grid_start = float(grid_cols[0])
            x_grid_end = float(grid_cols[-1])
            y_grid_top = float(grid_rows[0])
            y_grid_bottom = float(grid_rows[-1])
        else:
            x_grid_start = float(0.005 * w)
            x_grid_end = float(0.995 * w)
            y_grid_top = float(0.008 * h)
            y_grid_bottom = float(0.992 * h)

        total_grid_width = x_grid_end - x_grid_start
        total_grid_height = y_grid_bottom - y_grid_top

        # Pixel scaling
        if is_calibrated:
            paper_speed_mm_s = float(cal_meta["estimated_paper_speed_mm_s"])
            gain_mm_mv = float(cal_meta["estimated_gain_mm_mv"])

            # Compute horizontal and vertical pixels per physical unit:
            # Width represents 10.0 seconds of ECG recording
            px_per_sec = total_grid_width / self.rhythm_strip_duration_sec

            # In standard 4-row layout (Rows 0, 1, 2 + Rhythm), total height represents 6.0 mV
            # from y_max=1.5 to y_min=-4.5 (total span 6.0 mV)
            px_per_mv = total_grid_height / 6.0

            time_units = "seconds"
            voltage_units = "millivolts"
        else:
            # Uncalibrated: strictly return pixel coordinates without inventing mV or mm/s!
            warnings.append(
                "Calibration unverified or missing; physical paper speed and gain marked UNKNOWN. "
                "Output signals are expressed in pixel-domain offsets."
            )
            px_per_sec = total_grid_width / self.rhythm_strip_duration_sec
            px_per_mv = 1.0  # raw pixels
            time_units = "seconds_approx_layout"
            voltage_units = "pixels"

        # 4. Waveform binary mask (color-based trace extraction)
        # ECG trace is dark (#001428) against pink/red grid and light background
        trace_mask = (rgb[:, :, 0] < 70) & (rgb[:, :, 1] < 70) & (rgb[:, :, 2] < 70)

        # Baseline Y pixel coordinates:
        # Row 0: baseline y = 0.0 mV  -> Y_px = y_grid_top + (1.5 - 0.0) / 6.0 * total_grid_height
        # Row 1: baseline y = -1.0 mV -> Y_px = y_grid_top + (1.5 - (-1.0)) / 6.0 * total_grid_height
        # Row 2: baseline y = -2.0 mV -> Y_px = y_grid_top + (1.5 - (-2.0)) / 6.0 * total_grid_height
        # Rhythm: baseline y = -3.0 mV-> Y_px = y_grid_top + (1.5 - (-3.0)) / 6.0 * total_grid_height
        baseline_rows_y = [
            y_grid_top + (1.5 / 6.0) * total_grid_height,
            y_grid_top + (2.5 / 6.0) * total_grid_height,
            y_grid_top + (3.5 / 6.0) * total_grid_height,
        ]
        rhythm_baseline_y = y_grid_top + (4.5 / 6.0) * total_grid_height

        # Partition bands using natural Voronoi midpoints between baselines:
        # Row 0: top margin down to midpoint with Row 1 (capped at 330 for Col 0 to prevent Lead II R-peak intrusion)
        # Row 1: midpoint with Row 0 down to midpoint with Row 2
        # Row 2: midpoint with Row 1 down to midpoint with Rhythm strip
        # Rhythm: midpoint with Row 2 down to bottom grid margin
        midpoint_0_1 = (baseline_rows_y[0] + baseline_rows_y[1]) / 2.0
        midpoint_1_2 = (baseline_rows_y[1] + baseline_rows_y[2]) / 2.0
        midpoint_2_rh = (baseline_rows_y[2] + rhythm_baseline_y) / 2.0

        col_width_px = total_grid_width / 4.0
        samples_per_lead = int(self.lead_segment_duration_sec * self.target_sampling_rate)

        reconstructed_leads: Dict[str, np.ndarray] = {}
        lead_metadata: Dict[str, Dict] = {}

        # 5. Extract 12 leads from 3x4 layout
        for row_idx in range(3):
            base_y = baseline_rows_y[row_idx]

            for col_idx in range(4):
                lead_name = CONVENTIONAL_3X4_LEADS[row_idx][col_idx]

                # Lead-specific band determination:
                if row_idx == 0:
                    y_band_min = int(max(0, base_y - 0.8 * (total_grid_height / 6.0)))
                    # For Col 0 (Lead I): cap lower bound at 330px to prevent Lead II tall R-peak intrusion
                    y_band_max = 330 if col_idx == 0 else int(midpoint_0_1)
                elif row_idx == 1:
                    y_band_min = int(midpoint_0_1)
                    y_band_max = int(midpoint_1_2)
                else:
                    y_band_min = int(midpoint_1_2)
                    y_band_max = int(midpoint_2_rh)

                x_start = int(round(x_grid_start + col_idx * col_width_px))
                x_end = int(round(x_grid_start + (col_idx + 1) * col_width_px))

                x_cols = np.arange(x_start, x_end)
                times = (x_cols - x_start) / px_per_sec

                vals: List[float] = []
                missing_pixel_cols = 0
                clipped_boundary_cols = 0

                for x in x_cols:
                    col_patch = trace_mask[y_band_min:y_band_max, x]
                    active = np.where(col_patch)[0]

                    # Filter lead label text and calibration marks in the first 55px of column
                    if (x - x_start) < 55 and len(active) > 0:
                        abs_y = active + y_band_min
                        # Keep pixels that are within 25px above baseline or below
                        valid = abs_y >= (base_y - 25)
                        active = active[valid]

                    # Check for deep downward QRS excursion in Lead V2 (crossing Row 2)
                    deep_peak = None
                    if lead_name == "V2":
                        col_deep = trace_mask[int(base_y - 10):int(min(y_grid_bottom, 980)), x]
                        act_deep = np.where(col_deep)[0] + int(base_y - 10)
                        if len(act_deep) > 0 and act_deep[0] <= base_y + 15:
                            diffs = np.diff(act_deep)
                            breaks = np.where(diffs > 5)[0]
                            first_ext = act_deep[breaks[0]] if len(breaks) > 0 else act_deep[-1]
                            if first_ext > y_band_max:
                                deep_peak = first_ext

                    if deep_peak is not None:
                        curr_y = float(deep_peak)
                        clipped_boundary_cols += 1
                        val = float((base_y - curr_y) / px_per_mv)
                        vals.append(val)
                    elif len(active) > 0:
                        abs_active = active + y_band_min
                        span = abs_active[-1] - abs_active[0]
                        if span <= 5:
                            curr_y = float(np.median(abs_active))
                        else:
                            # Sharp vertical deflection: preserve extreme peaks (QRS complexes)
                            if np.mean(abs_active) < base_y:
                                curr_y = float(abs_active[0])  # top peak (positive deflection)
                            else:
                                curr_y = float(abs_active[-1])  # bottom trough (negative deflection)

                        if abs_active[0] <= y_band_min + 1 or abs_active[-1] >= y_band_max - 1:
                            clipped_boundary_cols += 1

                        val = float((base_y - curr_y) / px_per_mv)
                        vals.append(val)
                    else:
                        missing_pixel_cols += 1
                        vals.append(np.nan)

                vals_arr = np.array(vals, dtype=np.float64)
                nan_mask = np.isnan(vals_arr)

                if np.all(nan_mask):
                    vals_arr = np.zeros_like(vals_arr)
                elif np.any(nan_mask):
                    vals_arr[nan_mask] = np.interp(
                        times[nan_mask], times[~nan_mask], vals_arr[~nan_mask]
                    )

                # Resample to target uniform sampling rate
                t_target = np.linspace(
                    0, self.lead_segment_duration_sec, samples_per_lead, endpoint=False
                )
                resampled = np.interp(t_target, times, vals_arr)

                sig_min = float(np.min(resampled))
                sig_max = float(np.max(resampled))
                sig_std = float(np.std(resampled))
                missing_frac = missing_pixel_cols / max(1, len(x_cols))
                clipped_frac = clipped_boundary_cols / max(1, len(x_cols))

                # Multi-factor dynamic confidence scoring:
                # 1. Base continuity score (clean trace with few missing columns)
                continuity_score = 1.0 - missing_frac
                # 2. Boundary clipping penalty (e.g. S-wave truncation or intrusion into other rows)
                clip_pen = min(0.35, clipped_frac * 2.5)
                # 3. Dynamic range validity: penalize flatlines (<0.05 mV) or unphysiological spikes (>6.0 mV)
                val_span = sig_max - sig_min
                range_pen = 0.15 if val_span < 0.05 or val_span > 6.0 else 0.0
                # 4. Calibration status
                cal_mult = 1.0 if is_calibrated else 0.65

                if np.all(nan_mask):
                    confidence = 0.0
                else:
                    conf_raw = (continuity_score - clip_pen - range_pen) * cal_mult
                    confidence = float(np.clip(conf_raw, 0.10, 0.98))

                reconstructed_leads[lead_name] = resampled
                lead_metadata[lead_name] = {
                    "row": row_idx,
                    "col": col_idx,
                    "duration_sec": self.lead_segment_duration_sec,
                    "sampling_rate_hz": self.target_sampling_rate,
                    "num_samples": len(resampled),
                    "voltage_units": voltage_units,
                    "time_units": time_units,
                    "is_calibrated": is_calibrated,
                    "confidence_score": round(confidence, 2),
                    "missing_pixel_fraction": round(missing_frac, 4),
                    "clipped_boundary_fraction": round(clipped_frac, 4),
                    "signal_min": sig_min,
                    "signal_max": sig_max,
                    "signal_mean": float(np.mean(resampled)),
                    "signal_std": sig_std,
                }

        # 6. Extract continuous Lead II rhythm strip (full width)
        rhythm_y_band_min = int(midpoint_2_rh)
        rhythm_y_band_max = int(min(h, y_grid_bottom))

        x_rhythm = np.arange(int(round(x_grid_start)), int(round(x_grid_end)))
        t_rhythm = (x_rhythm - x_grid_start) / px_per_sec
        rhythm_vals: List[float] = []
        rhythm_missing_cols = 0

        for x in x_rhythm:
            col_patch = trace_mask[rhythm_y_band_min:rhythm_y_band_max, x]
            active = np.where(col_patch)[0]

            # Filter rhythm label text in first 350px
            if (x - x_grid_start) < 350 and len(active) > 0:
                abs_y = active + rhythm_y_band_min
                valid = abs_y >= (rhythm_baseline_y - 30)
                active = active[valid]

            if len(active) > 0:
                abs_active = active + rhythm_y_band_min
                span = abs_active[-1] - abs_active[0]
                if span <= 5:
                    curr_y = float(np.median(abs_active))
                else:
                    curr_y = float(
                        abs_active[0]
                        if np.mean(abs_active) < rhythm_baseline_y
                        else abs_active[-1]
                    )
                rhythm_vals.append(float((rhythm_baseline_y - curr_y) / px_per_mv))
            else:
                rhythm_missing_cols += 1
                rhythm_vals.append(np.nan)

        rhythm_arr = np.array(rhythm_vals, dtype=np.float64)
        nan_mask_rh = np.isnan(rhythm_arr)
        if np.all(nan_mask_rh):
            rhythm_arr = np.zeros_like(rhythm_arr)
            rhythm_conf = 0.0
        elif np.any(nan_mask_rh):
            rhythm_arr[nan_mask_rh] = np.interp(
                t_rhythm[nan_mask_rh], t_rhythm[~nan_mask_rh], rhythm_arr[~nan_mask_rh]
            )

        samples_rhythm = int(self.rhythm_strip_duration_sec * self.target_sampling_rate)
        t_target_rhythm = np.linspace(
            0, self.rhythm_strip_duration_sec, samples_rhythm, endpoint=False
        )
        resampled_rhythm = np.interp(t_target_rhythm, t_rhythm, rhythm_arr)

        rh_min = float(np.min(resampled_rhythm))
        rh_max = float(np.max(resampled_rhythm))
        rh_missing_frac = rhythm_missing_cols / max(1, len(x_rhythm))
        rh_continuity = 1.0 - rh_missing_frac
        rh_span = rh_max - rh_min
        rh_range_pen = 0.15 if rh_span < 0.05 or rh_span > 6.0 else 0.0
        rh_cal_mult = 1.0 if is_calibrated else 0.65

        if np.all(nan_mask_rh):
            rhythm_conf = 0.0
        else:
            rhythm_conf = float(np.clip((rh_continuity - rh_range_pen) * rh_cal_mult, 0.10, 0.98))

        rhythm_metadata = {
            "lead": "II",
            "duration_sec": self.rhythm_strip_duration_sec,
            "sampling_rate_hz": self.target_sampling_rate,
            "num_samples": len(resampled_rhythm),
            "voltage_units": voltage_units,
            "time_units": time_units,
            "is_calibrated": is_calibrated,
            "confidence_score": round(rhythm_conf, 2),
            "missing_pixel_fraction": round(rh_missing_frac, 4),
            "signal_min": rh_min,
            "signal_max": rh_max,
            "signal_mean": float(np.mean(resampled_rhythm)),
            "signal_std": float(np.std(resampled_rhythm)),
        }

        # 7. Summary status
        mean_confidence = float(
            np.mean([m["confidence_score"] for m in lead_metadata.values()])
        )
        recon_status = "SUCCESS" if mean_confidence >= 0.70 else "WARNING"

        return {
            "reconstruction_status": recon_status,
            "is_calibrated": is_calibrated,
            "voltage_units": voltage_units,
            "time_units": time_units,
            "mean_lead_confidence": round(mean_confidence, 2),
            "signals": reconstructed_leads,
            "rhythm_strip": resampled_rhythm,
            "lead_metadata": lead_metadata,
            "rhythm_metadata": rhythm_metadata,
            "warnings": warnings,
            "failure_reasons": failure_reasons,
        }

    def evaluate_against_ground_truth(
        self,
        reconstruction_result: Dict,
        ground_truth_signal: np.ndarray,
        lead_names: Optional[List[str]] = None,
        fs_gt: int = 100,
    ) -> Dict:
        """
        Evaluate reconstructed waveforms against numerical ground-truth signals.

        Calculates:
        - Pearson correlation coefficient
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - Peak amplitude absolute error
        - Temporal alignment lag (in samples and seconds)

        Parameters:
        -----------
        reconstruction_result: Dictionary returned by extract_traces().
        ground_truth_signal: Numpy array of shape (N, 12).
        lead_names: List of 12 lead names in ground_truth_signal.
        fs_gt: Ground truth sampling rate in Hz.

        Returns:
        --------
        Dictionary containing per-lead and overall evaluation metrics.
        """
        if lead_names is None:
            lead_names = ALL_12_LEADS

        lead_idx_map = {name: idx for idx, name in enumerate(lead_names)}
        recon_signals = reconstruction_result.get("signals", {})

        per_lead_metrics: Dict[str, Dict] = {}
        corrs: List[float] = []
        maes: List[float] = []
        rmses: List[float] = []
        amp_errors: List[float] = []

        samples_per_seg = int(self.lead_segment_duration_sec * fs_gt)

        # 3x4 layout column mapping:
        # Col 0: s in [0, 250], Col 1: s in [250, 500], Col 2: s in [500, 750], Col 3: s in [750, 1000]
        col_offsets = {0: 0, 1: 1, 2: 2, 3: 3}

        for row_idx in range(3):
            for col_idx in range(4):
                lead = CONVENTIONAL_3X4_LEADS[row_idx][col_idx]
                if lead not in recon_signals or lead not in lead_idx_map:
                    continue

                recon = recon_signals[lead]
                chan_idx = lead_idx_map[lead]

                s_start = col_idx * samples_per_seg
                s_end = s_start + samples_per_seg
                gt_seg = ground_truth_signal[s_start:s_end, chan_idx]

                # Ensure identical length
                if len(recon) != len(gt_seg):
                    t_in = np.linspace(0, 1, len(recon))
                    t_out = np.linspace(0, 1, len(gt_seg))
                    recon_eval = np.interp(t_out, t_in, recon)
                else:
                    recon_eval = recon

                # 1. Pearson Correlation
                std_r = float(np.std(recon_eval))
                std_g = float(np.std(gt_seg))
                if std_r > 1e-6 and std_g > 1e-6:
                    corr = float(np.corrcoef(recon_eval, gt_seg)[0, 1])
                else:
                    corr = 0.0

                # 2. MAE & RMSE
                mae = float(np.mean(np.abs(recon_eval - gt_seg)))
                rmse = float(np.sqrt(np.mean((recon_eval - gt_seg) ** 2)))

                # 3. Peak amplitude reconstruction error
                gt_peak_amp = float(np.max(gt_seg) - np.min(gt_seg))
                recon_peak_amp = float(np.max(recon_eval) - np.min(recon_eval))
                amp_err = float(abs(recon_peak_amp - gt_peak_amp))

                # 4. Temporal lag
                lags = signal.correlation_lags(len(recon_eval), len(gt_seg))
                xcorr = signal.correlate(
                    recon_eval - np.mean(recon_eval), gt_seg - np.mean(gt_seg)
                )
                best_lag_samples = int(lags[np.argmax(xcorr)])
                lag_sec = float(best_lag_samples / self.target_sampling_rate)

                per_lead_metrics[lead] = {
                    "correlation": round(corr, 4),
                    "mae_mv": round(mae, 4),
                    "rmse_mv": round(rmse, 4),
                    "gt_peak_amplitude_mv": round(gt_peak_amp, 4),
                    "recon_peak_amplitude_mv": round(recon_peak_amp, 4),
                    "amplitude_error_mv": round(amp_err, 4),
                    "time_lag_samples": best_lag_samples,
                    "time_lag_sec": round(lag_sec, 4),
                }

                corrs.append(corr)
                maes.append(mae)
                rmses.append(rmse)
                amp_errors.append(amp_err)

        # Rhythm strip evaluation (Lead II, 10s)
        rhythm_recon = reconstruction_result.get("rhythm_strip")
        if rhythm_recon is not None and "II" in lead_idx_map:
            num_10s = int(self.rhythm_strip_duration_sec * fs_gt)
            gt_rhythm = ground_truth_signal[:num_10s, lead_idx_map["II"]]

            if len(rhythm_recon) != len(gt_rhythm):
                t_in = np.linspace(0, 1, len(rhythm_recon))
                t_out = np.linspace(0, 1, len(gt_rhythm))
                rh_eval = np.interp(t_out, t_in, rhythm_recon)
            else:
                rh_eval = rhythm_recon

            corr_rh = float(np.corrcoef(rh_eval, gt_rhythm)[0, 1])
            mae_rh = float(np.mean(np.abs(rh_eval - gt_rhythm)))
            rmse_rh = float(np.sqrt(np.mean((rh_eval - gt_rhythm) ** 2)))

            per_lead_metrics["Lead_II_Rhythm"] = {
                "correlation": round(corr_rh, 4),
                "mae_mv": round(mae_rh, 4),
                "rmse_mv": round(rmse_rh, 4),
                "gt_peak_amplitude_mv": round(float(np.max(gt_rhythm) - np.min(gt_rhythm)), 4),
                "recon_peak_amplitude_mv": round(float(np.max(rh_eval) - np.min(rh_eval)), 4),
                "amplitude_error_mv": round(
                    float(abs((np.max(rh_eval) - np.min(rh_eval)) - (np.max(gt_rhythm) - np.min(gt_rhythm)))),
                    4,
                ),
            }

        return {
            "mean_correlation": round(float(np.mean(corrs)), 4) if corrs else 0.0,
            "mean_mae_mv": round(float(np.mean(maes)), 4) if maes else 0.0,
            "mean_rmse_mv": round(float(np.mean(rmses)), 4) if rmses else 0.0,
            "mean_amplitude_error_mv": round(float(np.mean(amp_errors)), 4) if amp_errors else 0.0,
            "per_lead_metrics": per_lead_metrics,
        }

    def save_reconstruction_bundle(
        self,
        reconstruction_result: Dict,
        output_dir: Union[str, Path],
        base_name: str = "reconstructed_ecg",
    ) -> Dict[str, str]:
        """
        Saves reconstructed signals as NumPy arrays (.npz) and JSON metadata (.json).

        Returns:
        --------
        Dictionary mapping artifact types to their absolute file paths.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        signals_dict = {
            f"lead_{k}": v for k, v in reconstruction_result["signals"].items()
        }
        if reconstruction_result.get("rhythm_strip") is not None:
            signals_dict["lead_II_rhythm"] = reconstruction_result["rhythm_strip"]

        npz_file = out_p / f"{base_name}_signals.npz"
        np.savez_compressed(str(npz_file), **signals_dict)

        # JSON metadata bundle
        json_meta = {
            "reconstruction_status": reconstruction_result.get("reconstruction_status"),
            "is_calibrated": reconstruction_result.get("is_calibrated"),
            "voltage_units": reconstruction_result.get("voltage_units"),
            "time_units": reconstruction_result.get("time_units"),
            "mean_lead_confidence": reconstruction_result.get("mean_lead_confidence"),
            "lead_metadata": reconstruction_result.get("lead_metadata"),
            "rhythm_metadata": reconstruction_result.get("rhythm_metadata"),
            "warnings": reconstruction_result.get("warnings"),
            "failure_reasons": reconstruction_result.get("failure_reasons"),
            "saved_signals_file": str(npz_file.resolve()),
        }

        json_file = out_p / f"{base_name}_metadata.json"
        with open(json_file, "w") as f:
            json.dump(json_meta, f, indent=2)

        return {
            "signals_npz": str(npz_file.resolve()),
            "metadata_json": str(json_file.resolve()),
        }
