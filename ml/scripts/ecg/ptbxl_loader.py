"""
PTB-XL Dataset Loader and Multi-Label Manifest Generator.

Dataset: PTB-XL (PhysioNet)
License: Creative Commons Attribution 4.0 International (CC-BY 4.0)
Structure: 21,799 12-lead clinical ECG recordings from 18,869 patients (10 seconds each).

Splitting Protocol:
- Uses official patient-stratified folds (strat_fold 1-10).
- Folds 1-8: Training set (~80%)
- Fold 9: Validation set (~10%)
- Fold 10: Test set (~10%)
- Guarantees 0 patient overlap across splits (patient-aware).

Diagnostic Superclasses:
- NORM: Normal ECG
- MI: Myocardial Infarction
- STTC: ST/T Change
- CD: Conduction Disturbance
- HYP: Hypertrophy
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_MAP = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
}
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLDatasetLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.db_csv_path = self.data_dir / "ptbxl_database.csv"
        self.scp_csv_path = self.data_dir / "scp_statements.csv"

        if not self.db_csv_path.exists():
            raise FileNotFoundError(f"PTB-XL database metadata not found at {self.db_csv_path}")
        if not self.scp_csv_path.exists():
            raise FileNotFoundError(f"SCP statements mapping not found at {self.scp_csv_path}")

        self.df = pd.read_csv(self.db_csv_path, index_col="ecg_id")
        self.df["scp_codes_dict"] = self.df["scp_codes"].apply(ast.literal_eval)

        self.scp_df = pd.read_csv(self.scp_csv_path, index_col=0)
        self.diag_statements = self.scp_df[self.scp_df["diagnostic"] == 1.0]

    def map_diagnostic_superclasses(self, min_weight_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Map raw SCP codes to 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).
        Follows standard PhysioNet benchmark aggregation matching scp_statements.csv:
          - If min_weight_threshold is None: standard PhysioNet raw aggregation:
            NORM: 9514, MI: 5469, STTC: 5235, CD: 4898, HYP: 2649
          - If min_weight_threshold is specified (e.g. 0.0): filters out statements with weight <= threshold:
            NORM: 9514, MI: 5469, STTC: 5108, CD: 4898, HYP: 2649
        """
        diag_dict = self.diag_statements["diagnostic_class"].to_dict()

        def get_superclasses(scp_dict: Dict[str, float]) -> List[str]:
            classes = set()
            for code, weight in scp_dict.items():
                if min_weight_threshold is not None and weight <= min_weight_threshold:
                    continue
                if code in diag_dict and pd.notna(diag_dict[code]):
                    classes.add(diag_dict[code])
            return sorted(list(classes))

        self.df["superclasses"] = self.df["scp_codes_dict"].apply(get_superclasses)

        # Multi-hot binary columns
        for sc in DIAGNOSTIC_SUPERCLASSES:
            self.df[f"label_{sc}"] = self.df["superclasses"].apply(lambda classes: 1 if sc in classes else 0)

        return self.df

    def get_split_manifests(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Produce patient-level train, val, test manifests using official folds:
        - Train: Folds 1..8
        - Val: Fold 9
        - Test: Fold 10
        """
        if "label_NORM" not in self.df.columns:
            self.map_diagnostic_superclasses()

        train_df = self.df[self.df["strat_fold"].between(1, 8)].copy()
        val_df = self.df[self.df["strat_fold"] == 9].copy()
        test_df = self.df[self.df["strat_fold"] == 10].copy()

        # Verify zero patient overlap
        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        test_patients = set(test_df["patient_id"])

        assert len(train_patients.intersection(val_patients)) == 0, "Patient overlap between train and val!"
        assert len(train_patients.intersection(test_patients)) == 0, "Patient overlap between train and test!"
        assert len(val_patients.intersection(test_patients)) == 0, "Patient overlap between val and test!"

        return train_df, val_df, test_df

    def audit_physical_dataset_completeness(self, resolution: str = "100") -> Dict[str, Any]:
        """
        Audit physical file presence on disk against the 21,799 database records across splits.
        Returns detailed report of expected vs physical files, missing IDs, and completeness status.
        """
        splits = {
            "train": self.df[self.df["strat_fold"].between(1, 8)],
            "val": self.df[self.df["strat_fold"] == 9],
            "test": self.df[self.df["strat_fold"] == 10],
        }

        col_name = "filename_lr" if resolution == "100" else "filename_hr"
        split_reports = {}
        all_missing_records = {}

        for s_name, s_df in splits.items():
            expected_count = len(s_df)
            expected_patients = int(s_df["patient_id"].nunique())
            physical_records = []
            missing_records = []

            for ecg_id, row in s_df.iterrows():
                rel_path = row[col_name]
                rec_path = self.data_dir / rel_path
                dat_exists = rec_path.with_suffix(".dat").is_file()
                hea_exists = rec_path.with_suffix(".hea").is_file()

                if dat_exists and hea_exists:
                    physical_records.append(ecg_id)
                else:
                    missing_records.append({
                        "ecg_id": int(ecg_id),
                        "expected_dat": str(rec_path.with_suffix(".dat")),
                        "expected_hea": str(rec_path.with_suffix(".hea")),
                        "dat_present": dat_exists,
                        "hea_present": hea_exists,
                    })

            is_complete = len(physical_records) == expected_count
            split_reports[s_name] = {
                "expected_records": expected_count,
                "physical_records_found": len(physical_records),
                "missing_records_count": len(missing_records),
                "expected_unique_patients": expected_patients,
                "is_complete": is_complete,
                "physical_record_ids": physical_records,
            }
            all_missing_records[s_name] = missing_records

        dataset_complete = all(r["is_complete"] for r in split_reports.values())

        return {
            "dataset_complete": dataset_complete,
            "resolution": resolution,
            "splits": split_reports,
            "missing_records_sample": {
                s: m[:10] for s, m in all_missing_records.items()
            },
            "total_missing_count": sum(r["missing_records_count"] for r in split_reports.values()),
        }

    def load_record_signal(
        self,
        ecg_id: int,
        resolution: str = "100",  # "100" or "500"
    ) -> Tuple[np.ndarray, int, List[str]]:
        """
        Load signal array (shape: samples, 12), sampling frequency, and standardized lead names.
        """
        row = self.df.loc[ecg_id]
        rel_path = row["filename_lr"] if resolution == "100" else row["filename_hr"]
        record_path = self.data_dir / rel_path

        record = wfdb.rdrecord(str(record_path))
        raw_leads = record.sig_name
        std_leads = [LEAD_NAME_MAP.get(name, name) for name in raw_leads]

        # Re-order to canonical standard 12-lead layout if possible
        signal = record.p_signal
        if set(std_leads) == set(STANDARD_12_LEADS):
            indices = [std_leads.index(lead) for lead in STANDARD_12_LEADS]
            signal = signal[:, indices]
            ordered_leads = STANDARD_12_LEADS
        else:
            ordered_leads = std_leads

        return signal, int(record.fs), ordered_leads
