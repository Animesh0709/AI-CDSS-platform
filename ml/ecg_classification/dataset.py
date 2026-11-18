"""
PyTorch Dataset and Preprocessing for 12-Lead ECG Signals.

Loads 100 Hz 10-second waveforms, applies training-only channel normalization,
and returns tensors of shape (12, 1000) with binary multi-hot targets (5,).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES, PTBXLDatasetLoader


class PTBXLWaveformDataset(Dataset):
    """PyTorch Dataset yielding (12, 1000) float32 signal tensors and multi-hot target vectors."""

    def __init__(
        self,
        loader: PTBXLDatasetLoader,
        record_ids: List[int],
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
        target_classes: Optional[List[str]] = None,
    ):
        self.loader = loader
        self.record_ids = record_ids
        self.channel_mean = np.array(channel_mean, dtype=np.float32) if channel_mean is not None else None
        self.channel_std = np.array(channel_std, dtype=np.float32) if channel_std is not None else None

        self.target_classes = target_classes or DIAGNOSTIC_SUPERCLASSES
        self.label_cols = [f"label_{sc}" for sc in self.target_classes]

        # Extract target matrix
        subset_df = self.loader.df.loc[self.record_ids]
        self.targets = subset_df[self.label_cols].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eid = self.record_ids[idx]
        sig, fs, leads = self.loader.load_record_signal(eid, resolution="100")

        # sig shape: (1000 samples, 12 channels)
        if sig.shape != (1000, 12):
            raise ValueError(f"Record {eid} has unexpected shape {sig.shape}, expected (1000, 12)")

        # Normalization using frozen parameters
        if self.channel_mean is not None and self.channel_std is not None:
            sig = (sig - self.channel_mean) / self.channel_std

        # PyTorch Conv1d expects (channels, samples) -> (12, 1000)
        tensor_x = torch.from_numpy(sig.T).float()
        tensor_y = torch.from_numpy(self.targets[idx]).float()

        return tensor_x, tensor_y
