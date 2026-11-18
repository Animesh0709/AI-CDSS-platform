"""
MIT-BIH Arrhythmia Database Loader (Supplementary Beat Analysis).

Dataset: MIT-BIH Arrhythmia Database (PhysioNet)
License: Open Data Commons Public Domain Dedication and License (PDDL v1.0)
Structure: 48 half-hour 2-lead ambulatory ECG recordings sampled at 360 Hz.
Leads: Modified limb lead II (MLII) and one modified precordial lead (typically V1, V2, V4, or V5).

Beat Annotations (atr):
- 'N': Normal beat
- 'L': Left bundle branch block
- 'R': Right bundle branch block
- 'A': Atrial premature beat (PAC)
- 'V': Premature ventricular contraction (PVC)
- 'F': Fusion of ventricular and normal beat
- '/': Paced beat
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb


class MITDBDatasetLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.records_file = self.data_dir / "RECORDS"
        if not self.records_file.exists():
            raise FileNotFoundError(f"MIT-BIH RECORDS file not found at {self.records_file}")

        with open(self.records_file, "r") as f:
            self.record_names = [line.strip() for line in f if line.strip()]

    def list_records(self) -> List[str]:
        return list(self.record_names)

    def load_record(
        self,
        record_id: str,
        sampfrom: int = 0,
        sampto: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, List[str]]:
        """Load 2-lead ambulatory signal, sampling rate (360 Hz), and lead names."""
        rec_path = self.data_dir / str(record_id)
        record = wfdb.rdrecord(str(rec_path), sampfrom=sampfrom, sampto=sampto)
        return record.p_signal, int(record.fs), list(record.sig_name)

    def load_annotations(
        self,
        record_id: str,
        sampfrom: int = 0,
        sampto: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load beat annotation sample indices and beat symbols.
        Returns: (sample_indices, beat_symbols)
        """
        rec_path = self.data_dir / str(record_id)
        ann = wfdb.rdann(str(rec_path), "atr", sampfrom=sampfrom, sampto=sampto)
        return ann.sample, list(ann.symbol)
