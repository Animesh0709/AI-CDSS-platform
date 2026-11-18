"""
LUDB (Lobachevsky University Electrocardiography Database) Loader.

Dataset: LUDB (PhysioNet)
License: Open Data Commons Attribution License (ODC-By 1.0)
Structure: 200 10-second 12-lead ECG records at 500 Hz with manual cardiologists'
annotations for P, QRS, and T wave boundaries across all 12 leads independently.

Annotation Symbols:
- '(': Wave onset
- 'p', 't', 'N': Wave peak (P-peak, T-peak, QRS peak/R-peak)
- ')': Wave offset
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_MAP = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
}


class LUDBDatasetLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.records_file = self.data_dir / "RECORDS"
        if not self.records_file.exists():
            raise FileNotFoundError(f"LUDB RECORDS file not found at {self.records_file}")

        with open(self.records_file, "r") as f:
            self.record_names = [line.strip() for line in f if line.strip()]

    def list_records(self) -> List[str]:
        return list(self.record_names)

    def load_record(
        self,
        record_id: str | int,
    ) -> Tuple[np.ndarray, int, List[str]]:
        """Load 12-lead signal, sampling rate, and standardized lead names."""
        rec_path = self._resolve_record_path(record_id)
        record = wfdb.rdrecord(str(rec_path))

        raw_leads = record.sig_name
        std_leads = [LEAD_NAME_MAP.get(name.lower(), name) for name in raw_leads]

        signal = record.p_signal
        if set(std_leads) == set(STANDARD_12_LEADS):
            indices = [std_leads.index(lead) for lead in STANDARD_12_LEADS]
            signal = signal[:, indices]
            ordered_leads = STANDARD_12_LEADS
        else:
            ordered_leads = std_leads

        return signal, int(record.fs), ordered_leads

    def load_lead_annotations(
        self,
        record_id: str | int,
        lead_name: str,
    ) -> Optional[Dict[str, List]]:
        """
        Load waveform boundary annotations for a specific lead (e.g. 'i', 'ii', 'v1').
        Parses onsets, peaks, offsets for P, QRS, and T waves.
        """
        rec_path = self._resolve_record_path(record_id)
        annotator = lead_name.lower()

        try:
            ann = wfdb.rdann(str(rec_path), annotator)
        except (FileNotFoundError, OSError):
            return None

        samples = ann.sample
        symbols = ann.symbol

        waves = {
            "P": [],    # list of dicts: {"onset": idx, "peak": idx, "offset": idx}
            "QRS": [],
            "T": [],
        }

        # LUDB format: '(' then wave_type then ')'
        i = 0
        while i < len(symbols) - 2:
            if symbols[i] == "(" and symbols[i + 2] == ")":
                wave_type = symbols[i + 1]
                onset = samples[i]
                peak = samples[i + 1]
                offset = samples[i + 2]

                if wave_type == "p":
                    waves["P"].append({"onset": int(onset), "peak": int(peak), "offset": int(offset)})
                elif wave_type == "N":  # QRS
                    waves["QRS"].append({"onset": int(onset), "peak": int(peak), "offset": int(offset)})
                elif wave_type == "t":
                    waves["T"].append({"onset": int(onset), "peak": int(peak), "offset": int(offset)})
                i += 3
            else:
                i += 1

        return waves

    def _resolve_record_path(self, record_id: str | int) -> Path:
        s_id = str(record_id)
        if s_id.startswith("data/"):
            target = self.data_dir / s_id
        else:
            target = self.data_dir / "data" / s_id
        if not target.with_suffix(".hea").exists():
            alt = self.data_dir / s_id
            if alt.with_suffix(".hea").exists():
                return alt
        return target
