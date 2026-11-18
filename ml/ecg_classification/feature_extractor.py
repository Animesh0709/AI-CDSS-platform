"""
Feature Extraction and Training-Only Preprocessing for Multi-Lead ECG Signals.

Extracts:
1. 10 statistical & morphological features per lead:
   - mean, std, min, max, peak-to-peak amplitude, median, skewness, kurtosis, RMS, IQR
   - 10 features * 12 leads = 120 features
2. Standardized waveform profile:
   - 100 uniformly spaced temporal points per lead (decimated by 10 from 100 Hz signal)
   - 100 points * 12 leads = 1200 features
Total feature dimension per 12-lead ECG: 1320 features.

Preprocessing:
- Lead-wise z-score normalization parameters (mean, std) are strictly fitted on training records only.
- Validation and test signals are transformed using the frozen training statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class ECGFeatureExtractor:
    """
    Standardized feature extractor for 12-lead ECG signals.
    """

    def __init__(self, target_len: int = 1000, decimation_factor: int = 10):
        self.target_len = target_len
        self.decimation_factor = decimation_factor
        self.training_mean: Optional[np.ndarray] = None
        self.training_std: Optional[np.ndarray] = None

    def fit_training_statistics(self, training_signals: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit channel-wise mean and std strictly on training set signals.
        Never include validation or test signals.
        """
        if not training_signals:
            raise ValueError("training_signals cannot be empty")

        channel_means = [np.mean(sig, axis=0) for sig in training_signals]
        channel_stds = [np.std(sig, axis=0) for sig in training_signals]

        self.training_mean = np.mean(channel_means, axis=0)
        self.training_std = np.mean(channel_stds, axis=0)
        # Avoid division by zero
        self.training_std[self.training_std < 1e-6] = 1.0

        return {
            "channel_mean": self.training_mean,
            "channel_std": self.training_std,
            "num_training_signals": len(training_signals),
        }

    def set_training_statistics(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Explicitly set frozen training statistics."""
        self.training_mean = np.array(mean, dtype=np.float32)
        self.training_std = np.array(std, dtype=np.float32)

    def extract_features_from_signal(
        self,
        signal: np.ndarray,
        apply_normalization: bool = True,
    ) -> np.ndarray:
        """
        Extract fixed-length feature vector from a (samples, 12) signal array.
        """
        if signal.ndim != 2 or signal.shape[1] != 12:
            raise ValueError(f"Expected signal shape (samples, 12), got {signal.shape}")

        sig = signal.copy().astype(np.float32)

        # Apply training-fitted normalization if available and requested
        if apply_normalization and self.training_mean is not None and self.training_std is not None:
            sig = (sig - self.training_mean) / self.training_std

        features: List[float] = []

        # 1. 10 statistical features per lead (12 leads * 10 = 120 features)
        for ch in range(12):
            lead = sig[:, ch]
            mean_val = float(np.mean(lead))
            std_val = float(np.std(lead))
            min_val = float(np.min(lead))
            max_val = float(np.max(lead))
            p2p_val = max_val - min_val
            median_val = float(np.median(lead))
            skew_val = float(stats.skew(lead))
            kurt_val = float(stats.kurtosis(lead))
            rms_val = float(np.sqrt(np.mean(lead**2)))
            q75, q25 = np.percentile(lead, [75, 25])
            iqr_val = float(q75 - q25)

            features.extend([
                mean_val, std_val, min_val, max_val, p2p_val,
                median_val, skew_val, kurt_val, rms_val, iqr_val,
            ])

        # 2. Downsampled waveform summary (decimated by 10: 100 samples * 12 leads = 1200 features)
        downsampled = sig[::self.decimation_factor, :].flatten()
        features.extend(downsampled.tolist())

        arr = np.array(features, dtype=np.float32)
        # Nan / Inf check & clean replacement if numeric instability occurs
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr
