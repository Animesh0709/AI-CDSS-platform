"""
ECG Waveform Delineation Service.

Provides robust, multi-lead P-wave, QRS-complex, and T-wave delineation
using reconstructed numerical ECG signals or standard digital recordings.

Conforms to Phase 5 requirements:
1. Input and quality gating (sampling rate, units, calibration flags, confidence).
2. Conservative preprocessing (two-stage median baseline removal, power-line notch, bandpass).
3. Robust QRS detection with R, Q, S peak identification and QRS onset/offset estimation.
4. P-wave delineation (positive, inverted, biphasic, and absent/nodal rhythms).
5. T-wave delineation (positive, inverted, biphasic, and distinction from drift).
6. Multi-lead processing with Lead II rhythm strip cross-lead consensus.
7. Ground-truth evaluation against LUDB reference annotations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal

logger = logging.getLogger("ai_cdss.ecg.delineation")


class ECGWaveformDelineator:
    """
    ECG Waveform Delineator for P-wave, QRS-complex, and T-wave identification.
    """

    def __init__(
        self,
        default_fs: int = 500,
        min_qrs_distance_sec: float = 0.25,
        p_search_window_sec: Tuple[float, float] = (0.26, 0.03),
        t_search_window_sec: Tuple[float, float] = (0.10, 0.48),
        min_p_amplitude_mv: float = 0.035,
        min_t_amplitude_mv: float = 0.040,
    ):
        self.default_fs = default_fs
        self.min_qrs_distance_sec = min_qrs_distance_sec
        self.p_search_window_sec = p_search_window_sec
        self.t_search_window_sec = t_search_window_sec
        self.min_p_amplitude_mv = min_p_amplitude_mv
        self.min_t_amplitude_mv = min_t_amplitude_mv

    def preprocess_signal(
        self,
        raw_signal: np.ndarray,
        fs: int,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conservative ECG signal preprocessing.
        
        Removes baseline wander using a two-stage median filter (200 ms and 600 ms)
        to isolate baseline drift without eroding ST segments or P/T wave morphology.
        Applies a zero-phase 50 Hz notch filter to suppress power-line interference.

        Returns:
            (filtered_signal, baseline_wander)
        """
        sig_data = np.asarray(raw_signal, dtype=np.float64)
        n = len(sig_data)
        if n == 0:
            return sig_data, sig_data

        # 1. Baseline wander estimation via two-stage median filter
        # Window 1: ~200 ms (removes QRS/P/T spikes)
        m1 = int(0.20 * fs)
        if m1 % 2 == 0:
            m1 += 1
        m1 = max(3, min(m1, n if n % 2 == 1 else n - 1))

        # Window 2: ~600 ms (smoothes baseline)
        m2 = int(0.60 * fs)
        if m2 % 2 == 0:
            m2 += 1
        m2 = max(3, min(m2, n if n % 2 == 1 else n - 1))

        baseline_stage1 = signal.medfilt(sig_data, m1)
        baseline = signal.medfilt(baseline_stage1, m2)
        sig_bc = sig_data - baseline

        # 2. Power-line notch filter (zero-phase IIR)
        if fs > 2 * notch_freq:
            b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs)
            filtered = signal.filtfilt(b_notch, a_notch, sig_bc)
        else:
            filtered = sig_bc

        return filtered, baseline

    def _detect_qrs_candidates(
        self,
        filtered_sig: np.ndarray,
        fs: int,
        min_rr_sec: float = 0.25,
    ) -> List[int]:
        """
        Adaptive Pan-Tompkins QRS energy detector.
        """
        n = len(filtered_sig)
        if n < int(0.3 * fs):
            return []

        nyq = 0.5 * fs
        lowcut = max(0.01, 5.0 / nyq)
        highcut = min(0.95, 18.0 / nyq)

        # Zero-phase Butterworth bandpass (5-18 Hz)
        b_bp, a_bp = signal.butter(2, [lowcut, highcut], btype="bandpass")
        sig_bp = signal.filtfilt(b_bp, a_bp, filtered_sig)

        # 5-point central derivative approximation
        diff_sig = np.diff(sig_bp, prepend=sig_bp[0])
        sq_sig = diff_sig ** 2

        # Moving window integration (120 ms)
        win_len = max(3, int(0.12 * fs))
        kernel = np.ones(win_len) / win_len
        mwa = np.convolve(sq_sig, kernel, mode="same")

        # Peak detection with refractory period
        min_dist = max(3, int(min_rr_sec * fs))
        energy_mean = float(np.mean(mwa))
        threshold = energy_mean * 0.70

        peaks, _ = signal.find_peaks(mwa, distance=min_dist, height=threshold)

        # Refine candidate peaks to true local extrema in the filtered ECG
        refined_peaks: List[int] = []
        margin = int(0.08 * fs)

        for p in peaks:
            if p < margin or p >= n - margin:
                continue
            sub = filtered_sig[p - margin : p + margin]
            max_idx = int(np.argmax(sub))
            min_idx = int(np.argmin(sub))

            # Choose dominant absolute deflection (supports positive, negative, biphasic QRS)
            if abs(sub[max_idx]) >= abs(sub[min_idx]):
                r_pk = (p - margin) + max_idx
            else:
                r_pk = (p - margin) + min_idx

            if refined_peaks and (r_pk - refined_peaks[-1]) < min_dist:
                # Merge or keep the higher-energy peak
                if abs(filtered_sig[r_pk]) > abs(filtered_sig[refined_peaks[-1]]):
                    refined_peaks[-1] = r_pk
            else:
                refined_peaks.append(r_pk)

        return refined_peaks

    def delineate_lead(
        self,
        lead_signal: np.ndarray,
        fs: int,
        lead_name: str = "Lead",
        confidence: float = 1.0,
        is_calibrated: bool = True,
        external_r_peaks: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Delineate P-wave, QRS-complex, and T-wave for an individual ECG lead.

        Args:
            lead_signal: 1D array of signal samples.
            fs: Sampling rate in Hz.
            lead_name: Name of lead (e.g., 'II', 'V1').
            confidence: Reconstruction / extraction confidence score [0.0, 1.0].
            is_calibrated: True if signal is calibrated millivolts; False if pixel-domain.
            external_r_peaks: Optional pre-aligned R-peak timestamps from rhythm strip consensus.

        Returns:
            Dictionary with filtered signal, baseline wander, and detected beats list.
        """
        n = len(lead_signal)
        flags: List[str] = []

        if not is_calibrated:
            flags.append("UNITS_PIXEL_WARNING")

        if confidence < 0.60:
            flags.append("LOW_CONFIDENCE_LEAD")

        if n < int(0.5 * fs):
            return {
                "lead": lead_name,
                "sampling_rate_hz": fs,
                "is_calibrated": is_calibrated,
                "confidence": confidence,
                "flags": flags + ["SIGNAL_TOO_SHORT"],
                "raw_signal": lead_signal.tolist(),
                "filtered_signal": lead_signal.tolist(),
                "baseline_wander": [0.0] * n,
                "beats": [],
            }

        filtered, baseline = self.preprocess_signal(lead_signal, fs)

        # 1. Detect or adapt R-peaks
        if external_r_peaks is not None and len(external_r_peaks) > 0:
            # Snap external peaks to local extrema in this specific lead
            margin = int(0.06 * fs)
            r_peaks: List[int] = []
            for ext_r in external_r_peaks:
                if ext_r < margin or ext_r >= n - margin:
                    continue
                sub = filtered[ext_r - margin : ext_r + margin]
                max_i = int(np.argmax(sub))
                min_i = int(np.argmin(sub))
                pk = (ext_r - margin) + (max_i if abs(sub[max_i]) >= abs(sub[min_i]) else min_i)
                r_peaks.append(pk)
        else:
            r_peaks = self._detect_qrs_candidates(filtered, fs, self.min_qrs_distance_sec)

        diff_filt = np.diff(filtered, prepend=filtered[0])
        diff_filt_abs = np.abs(diff_filt)
        global_slope_floor = float(np.percentile(diff_filt_abs, 25))

        beats: List[Dict[str, Any]] = []

        for b_idx, r in enumerate(r_peaks):
            beat_flags = list(flags)
            prev_r = r_peaks[b_idx - 1] if b_idx > 0 else None
            next_r = r_peaks[b_idx + 1] if b_idx < len(r_peaks) - 1 else None

            rr_prev = (r - prev_r) if prev_r is not None else int(0.80 * fs)
            rr_next = (next_r - r) if next_r is not None else int(0.80 * fs)

            # --- QRS Complex Delineation ---
            q_search_start = max(0, r - int(0.08 * fs))
            s_search_end = min(n - 1, r + int(0.08 * fs))

            r_val = float(filtered[r])
            polarity = "positive" if r_val >= 0 else "negative"

            # Q-peak is local extrema before R
            if r > q_search_start:
                if polarity == "positive":
                    q_pk = q_search_start + int(np.argmin(filtered[q_search_start : r + 1]))
                else:
                    q_pk = q_search_start + int(np.argmax(filtered[q_search_start : r + 1]))
            else:
                q_pk = r

            # S-peak is local extrema after R
            if s_search_end > r:
                if polarity == "positive":
                    s_pk = r + int(np.argmin(filtered[r : s_search_end + 1]))
                else:
                    s_pk = r + int(np.argmax(filtered[r : s_search_end + 1]))
            else:
                s_pk = r

            # QRS onset: scan backwards from Q-peak
            qrs_on_lim = max(0, r - int(0.12 * fs))
            qrs_slope_max = float(np.max(diff_filt_abs[q_pk : s_pk + 1])) if s_pk > q_pk else 0.10
            qrs_th = max(global_slope_floor, qrs_slope_max * 0.08)

            qrs_on = q_pk
            for k in range(q_pk, qrs_on_lim, -1):
                if diff_filt_abs[k] <= qrs_th:
                    qrs_on = k
                    break
            else:
                qrs_on = qrs_on_lim

            # QRS offset: scan forwards from S-peak
            qrs_off_lim = min(n - 1, r + int(0.14 * fs))
            qrs_off = s_pk
            for k in range(s_pk, qrs_off_lim):
                if diff_filt_abs[k] <= qrs_th:
                    qrs_off = k
                    break
            else:
                qrs_off = qrs_off_lim

            qrs_duration_ms = (qrs_off - qrs_on) / fs * 1000.0
            if qrs_duration_ms < 40.0 or qrs_duration_ms > 220.0:
                beat_flags.append("ATYPICAL_QRS_DURATION")

            qrs_dict = {
                "onset_sample": int(qrs_on),
                "onset_ms": round(qrs_on / fs * 1000.0, 2),
                "q_sample": int(q_pk),
                "q_ms": round(q_pk / fs * 1000.0, 2),
                "r_sample": int(r),
                "r_ms": round(r / fs * 1000.0, 2),
                "s_sample": int(s_pk),
                "s_ms": round(s_pk / fs * 1000.0, 2),
                "offset_sample": int(qrs_off),
                "offset_ms": round(qrs_off / fs * 1000.0, 2),
                "duration_ms": round(qrs_duration_ms, 2),
                "amplitude_mv": round(r_val, 4) if is_calibrated else None,
                "amplitude_pixel": round(r_val, 2) if not is_calibrated else None,
                "polarity": polarity,
            }

            # --- P-wave Delineation ---
            p_win_start = max(0, qrs_on - min(int(self.p_search_window_sec[0] * fs), int(0.45 * rr_prev)))
            p_win_end = max(0, qrs_on - int(self.p_search_window_sec[1] * fs))

            p_dict: Optional[Dict[str, Any]] = None
            if p_win_end > p_win_start + int(0.04 * fs):
                p_sub = filtered[p_win_start:p_win_end]
                p_max = float(np.max(p_sub))
                p_min = float(np.min(p_sub))
                p_amp = max(abs(p_max), abs(p_min))

                noise_floor = float(np.std(p_sub[: int(0.02 * fs)])) if len(p_sub) >= int(0.02 * fs) else 0.01

                # Physiologic gating: P-wave threshold
                min_p_thresh = self.min_p_amplitude_mv if is_calibrated else 3.0
                if p_amp >= min_p_thresh and p_amp >= 2.5 * noise_floor:
                    p_local_idx = int(np.argmax(p_sub)) if abs(p_max) >= abs(p_min) else int(np.argmin(p_sub))
                    p_pk = p_win_start + p_local_idx
                    p_pk_val = float(filtered[p_pk])

                    # Onset & offset via threshold
                    th_p = p_pk_val * 0.12
                    p_on = max(p_win_start, p_pk - int(0.05 * fs))
                    for k in range(p_pk, max(0, p_pk - int(0.08 * fs)), -1):
                        if abs(filtered[k]) <= abs(th_p) or (filtered[k] * p_pk_val <= 0):
                            p_on = k
                            break

                    p_off = min(p_win_end, p_pk + int(0.05 * fs))
                    for k in range(p_pk, min(n - 1, p_pk + int(0.08 * fs))):
                        if abs(filtered[k]) <= abs(th_p) or (filtered[k] * p_pk_val <= 0):
                            p_off = k
                            break

                    p_duration_ms = (p_off - p_on) / fs * 1000.0

                    # Physiologic validation: True P-waves have duration >= 40 ms.
                    # Sharp spikes (< 35 ms) are unipolar ventricular/atrial pacing spikes or artifacts.
                    if p_duration_ms >= 35.0:
                        p_polarity = "positive" if p_pk_val > 0 else "negative"
                        p_dict = {
                            "onset_sample": int(p_on),
                            "onset_ms": round(p_on / fs * 1000.0, 2),
                            "peak_sample": int(p_pk),
                            "peak_ms": round(p_pk / fs * 1000.0, 2),
                            "offset_sample": int(p_off),
                            "offset_ms": round(p_off / fs * 1000.0, 2),
                            "duration_ms": round(p_duration_ms, 2),
                            "amplitude_mv": round(p_pk_val, 4) if is_calibrated else None,
                            "amplitude_pixel": round(p_pk_val, 2) if not is_calibrated else None,
                            "polarity": p_polarity,
                        }
                    else:
                        beat_flags.append("PACING_SPIKE_OR_HIGH_FREQUENCY_ARTIFACT_SUPPRESSED")
                else:
                    beat_flags.append("P_WAVE_ABSENT_OR_BELOW_NOISE_FLOOR")
            else:
                beat_flags.append("P_WAVE_WINDOW_TRUNCATED")

            # --- T-wave Delineation ---
            # Physiological T-wave window: searches after ST segment (~120 ms after R or 0.22*RR)
            t_win_start = min(n - 1, r + max(int(0.12 * fs), int(0.22 * rr_next)))
            t_win_end = min(n - 1, r + min(int(0.50 * fs), int(0.60 * rr_next)))

            t_dict: Optional[Dict[str, Any]] = None
            if t_win_end > t_win_start + int(0.06 * fs):
                t_sub = filtered[t_win_start:t_win_end]
                t_max = float(np.max(t_sub))
                t_min = float(np.min(t_sub))
                t_amp = max(abs(t_max), abs(t_min))

                min_t_thresh = self.min_t_amplitude_mv if is_calibrated else 3.5
                if t_amp >= min_t_thresh:
                    t_local_idx = int(np.argmax(t_sub)) if abs(t_max) >= abs(t_min) else int(np.argmin(t_sub))
                    t_pk = t_win_start + t_local_idx
                    t_pk_val = float(filtered[t_pk])

                    # T-wave onset and offset via baseline intersection
                    th_t_on = t_pk_val * 0.08
                    th_t_off = t_pk_val * 0.10

                    t_on = max(t_win_start, t_pk - int(0.12 * fs))
                    for k in range(t_pk, max(0, t_pk - int(0.22 * fs)), -1):
                        if abs(filtered[k]) <= abs(th_t_on) or (filtered[k] * t_pk_val <= 0):
                            t_on = k
                            break

                    t_off = min(t_win_end, t_pk + int(0.09 * fs))
                    for k in range(t_pk, min(n - 1, t_pk + int(0.16 * fs))):
                        if abs(filtered[k]) <= abs(th_t_off) or (filtered[k] * t_pk_val <= 0):
                            t_off = k
                            break

                    t_polarity = "positive" if t_pk_val > 0 else "negative"
                    t_dict = {
                        "onset_sample": int(t_on),
                        "onset_ms": round(t_on / fs * 1000.0, 2),
                        "peak_sample": int(t_pk),
                        "peak_ms": round(t_pk / fs * 1000.0, 2),
                        "offset_sample": int(t_off),
                        "offset_ms": round(t_off / fs * 1000.0, 2),
                        "duration_ms": round((t_off - t_on) / fs * 1000.0, 2),
                        "amplitude_mv": round(t_pk_val, 4) if is_calibrated else None,
                        "amplitude_pixel": round(t_pk_val, 2) if not is_calibrated else None,
                        "polarity": t_polarity,
                    }
                else:
                    beat_flags.append("T_WAVE_ABSENT_OR_FLAT")
            else:
                beat_flags.append("T_WAVE_WINDOW_TRUNCATED")

            beats.append({
                "beat_index": b_idx,
                "lead": lead_name,
                "qrs": qrs_dict,
                "p": p_dict,
                "t": t_dict,
                "confidence": round(float(confidence), 3),
                "flags": beat_flags,
            })

        return {
            "lead": lead_name,
            "sampling_rate_hz": fs,
            "is_calibrated": is_calibrated,
            "confidence": confidence,
            "flags": flags,
            "raw_signal": lead_signal.tolist(),
            "filtered_signal": filtered.tolist(),
            "baseline_wander": baseline.tolist(),
            "beats": beats,
        }

    def delineate_bundle(
        self,
        reconstruction_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Delineate all leads from a Phase 4/4.2 reconstruction bundle.

        Extracts continuous Lead II rhythm strip first to establish 10-second
        global beat timestamps and RR consensus, then delineates the 12 quadrant leads
        with consensus timing mapping.
        """
        is_calibrated = reconstruction_bundle.get("is_calibrated", False)
        mean_conf = float(reconstruction_bundle.get("mean_lead_confidence", 0.8))
        bundle_warnings = list(reconstruction_bundle.get("warnings", []))

        # Check rhythm strip
        rhythm_sig = reconstruction_bundle.get("rhythm_strip")
        rhythm_meta = reconstruction_bundle.get("rhythm_metadata", {})
        fs = int(rhythm_meta.get("sampling_rate_hz", self.default_fs))

        lead_results: Dict[str, Any] = {}
        rhythm_result: Optional[Dict[str, Any]] = None
        global_r_peaks: List[int] = []

        if rhythm_sig is not None and len(rhythm_sig) > 0:
            rh_arr = np.array(rhythm_sig, dtype=np.float64)
            rh_conf = float(rhythm_meta.get("confidence_score", mean_conf))
            rhythm_result = self.delineate_lead(
                lead_signal=rh_arr,
                fs=fs,
                lead_name="Lead_II_Rhythm",
                confidence=rh_conf,
                is_calibrated=is_calibrated,
            )
            global_r_peaks = [b["qrs"]["r_sample"] for b in rhythm_result["beats"]]

        # Process 12 quadrant leads
        signals = reconstruction_bundle.get("signals", {})
        lead_metadata = reconstruction_bundle.get("lead_metadata", {})

        # Conventional quadrant lead temporal offsets:
        # Row 1 (I, aVR, V1, V4) -> [0.0-2.5s, 2.5-5.0s, 5.0-7.5s, 7.5-10.0s]
        # Row 2 (II, aVL, V2, V5) -> [0.0-2.5s, 2.5-5.0s, 5.0-7.5s, 7.5-10.0s]
        # Row 3 (III, aVF, V3, V6) -> [0.0-2.5s, 2.5-5.0s, 5.0-7.5s, 7.5-10.0s]
        col_time_offsets = {
            "I": 0.0, "II": 0.0, "III": 0.0,
            "aVR": 2.5, "aVL": 2.5, "aVF": 2.5,
            "V1": 5.0, "V2": 5.0, "V3": 5.0,
            "V4": 7.5, "V5": 7.5, "V6": 7.5,
        }

        for lead_name, sig_list in signals.items():
            if sig_list is None or len(sig_list) == 0:
                continue
            sig_arr = np.array(sig_list, dtype=np.float64)
            lead_meta = lead_metadata.get(lead_name, {})
            lead_conf = float(lead_meta.get("confidence_score", mean_conf))

            # Project global R peaks onto this quadrant's local coordinate frame
            offset_sec = col_time_offsets.get(lead_name, 0.0)
            offset_sample = int(offset_sec * fs)
            quadrant_duration = int(2.5 * fs)

            local_r_peaks = [
                (r_pk - offset_sample)
                for r_pk in global_r_peaks
                if offset_sample <= r_pk < offset_sample + quadrant_duration
            ]

            lead_res = self.delineate_lead(
                lead_signal=sig_arr,
                fs=fs,
                lead_name=lead_name,
                confidence=lead_conf,
                is_calibrated=is_calibrated,
                external_r_peaks=local_r_peaks if local_r_peaks else None,
            )
            lead_results[lead_name] = lead_res

        total_beats = sum(len(res["beats"]) for res in lead_results.values())
        if rhythm_result:
            total_beats += len(rhythm_result["beats"])

        return {
            "delineation_status": "success" if total_beats > 0 else "failed",
            "is_calibrated": is_calibrated,
            "mean_lead_confidence": mean_conf,
            "sampling_rate_hz": fs,
            "warnings": bundle_warnings,
            "total_detected_beats": total_beats,
            "rhythm_strip": rhythm_result,
            "leads": lead_results,
        }

    def evaluate_against_ludb_annotations(
        self,
        delineated_beats: List[Dict[str, Any]],
        ground_truth_symbols: List[str],
        ground_truth_samples: List[int],
        fs: int,
        tolerance_ms: float = 150.0,
    ) -> Dict[str, Any]:
        """
        Evaluate delineation accuracy against LUDB ground truth annotations.

        Calculates Sensitivity, PPV, F1-score, and onset/offset timing errors in ms
        for QRS-complexes, P-waves, and T-waves.
        """
        tol_samples = int(tolerance_ms / 1000.0 * fs)

        # Parse ground-truth waves: '(', sym, ')' triples
        gt_waves: Dict[str, List[Tuple[int, int, int]]] = {"N": [], "p": [], "t": []}
        for i in range(len(ground_truth_symbols) - 2):
            if ground_truth_symbols[i] == "(" and ground_truth_symbols[i + 2] == ")":
                sym = ground_truth_symbols[i + 1]
                if sym in gt_waves:
                    gt_waves[sym].append((
                        int(ground_truth_samples[i]),
                        int(ground_truth_samples[i + 1]),
                        int(ground_truth_samples[i + 2]),
                    ))

        # Annotated evaluation boundary
        sample_list = [int(s) for s in ground_truth_samples]
        min_gt_sample = min(sample_list) if len(sample_list) > 0 else 0
        max_gt_sample = max(sample_list) if len(sample_list) > 0 else 0

        # Filter detected waves to the annotated segment
        det_qrs = [
            (b["qrs"]["onset_sample"], b["qrs"]["r_sample"], b["qrs"]["offset_sample"])
            for b in delineated_beats
            if min_gt_sample <= b["qrs"]["r_sample"] <= max_gt_sample
        ]
        det_p = [
            (b["p"]["onset_sample"], b["p"]["peak_sample"], b["p"]["offset_sample"])
            for b in delineated_beats
            if b.get("p") is not None and min_gt_sample <= b["p"]["peak_sample"] <= max_gt_sample
        ]
        det_t = [
            (b["t"]["onset_sample"], b["t"]["peak_sample"], b["t"]["offset_sample"])
            for b in delineated_beats
            if b.get("t") is not None and min_gt_sample <= b["t"]["peak_sample"] <= max_gt_sample
        ]

        metrics: Dict[str, Any] = {}

        for wave_key, det_list, gt_list in [
            ("qrs", det_qrs, gt_waves["N"]),
            ("p", det_p, gt_waves["p"]),
            ("t", det_t, gt_waves["t"]),
        ]:
            matched_det = set()
            tp = 0
            err_on: List[float] = []
            err_pk: List[float] = []
            err_off: List[float] = []

            for g_on, g_pk, g_off in gt_list:
                best_di = None
                min_dist = 999999
                for di, (d_on, d_pk, d_off) in enumerate(det_list):
                    if di in matched_det:
                        continue
                    dist = abs(d_pk - g_pk)
                    if dist < min_dist and dist <= tol_samples:
                        min_dist = dist
                        best_di = di

                if best_di is not None:
                    matched_det.add(best_di)
                    tp += 1
                    d_on, d_pk, d_off = det_list[best_di]
                    err_on.append((d_on - g_on) / fs * 1000.0)
                    err_pk.append((d_pk - g_pk) / fs * 1000.0)
                    err_off.append((d_off - g_off) / fs * 1000.0)

            fn = len(gt_list) - tp
            fp = len(det_list) - len(matched_det)

            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f1 = float(2 * sens * ppv / (sens + ppv)) if (sens + ppv) > 0 else 0.0

            metrics[wave_key] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "sensitivity": round(sens, 4),
                "positive_predictive_value": round(ppv, 4),
                "f1_score": round(f1, 4),
                "mean_onset_error_ms": round(float(np.mean(err_on)), 2) if err_on else None,
                "std_onset_error_ms": round(float(np.std(err_on)), 2) if err_on else None,
                "mean_peak_error_ms": round(float(np.mean(err_pk)), 2) if err_pk else None,
                "std_peak_error_ms": round(float(np.std(err_pk)), 2) if err_pk else None,
                "mean_offset_error_ms": round(float(np.mean(err_off)), 2) if err_off else None,
                "std_offset_error_ms": round(float(np.std(err_off)), 2) if err_off else None,
            }

        return metrics
