"""
Auditor for PTB-XL Dataset, Multi-Label Splits, and Preprocessing Isolation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ml.ecg_classification.label_mapping import (
    DIAGNOSTIC_SUBCLASSES,
    DIAGNOSTIC_SUPERCLASSES,
    PTBXLLabelMapper,
)


class PTBXLDatasetAuditor:
    """
    Performs comprehensive data auditing on PTB-XL:
    - Integrity of records, patient isolation across 10 folds.
    - Class distributions for superclasses and subclasses.
    - Multi-label statistics.
    - Leakage-free split standardization statistics.
    """

    def __init__(self, ptbxl_dir: str | Path):
        self.ptbxl_dir = Path(ptbxl_dir)
        self.db_csv_path = self.ptbxl_dir / "ptbxl_database.csv"
        self.scp_csv_path = self.ptbxl_dir / "scp_statements.csv"

        if not self.db_csv_path.exists():
            raise FileNotFoundError(f"Missing ptbxl_database.csv at {self.db_csv_path}")
        if not self.scp_csv_path.exists():
            raise FileNotFoundError(f"Missing scp_statements.csv at {self.scp_csv_path}")

        self.df = pd.read_csv(self.db_csv_path, index_col="ecg_id")
        self.mapper = PTBXLLabelMapper(self.scp_csv_path)

    def audit_splits_and_patient_isolation(self) -> Dict[str, Any]:
        """Verify strict 0-patient overlap across train (1-8), val (9), and test (10)."""
        folds = {f: self.df[self.df["strat_fold"] == f] for f in range(1, 11)}
        fold_patients: Dict[int, Set[int]] = {f: set(df_f["patient_id"]) for f, df_f in folds.items()}

        pairwise_overlaps: Dict[str, int] = {}
        for i in range(1, 11):
            for j in range(i + 1, 11):
                inter = len(fold_patients[i].intersection(fold_patients[j]))
                if inter > 0:
                    pairwise_overlaps[f"fold_{i}_vs_fold_{j}"] = inter

        train_df = self.df[self.df["strat_fold"].between(1, 8)]
        val_df = self.df[self.df["strat_fold"] == 9]
        test_df = self.df[self.df["strat_fold"] == 10]

        train_pts = set(train_df["patient_id"])
        val_pts = set(val_df["patient_id"])
        test_pts = set(test_df["patient_id"])

        split_isolation = {
            "train_records": len(train_df),
            "train_patients": len(train_pts),
            "val_records": len(val_df),
            "val_patients": len(val_pts),
            "test_records": len(test_df),
            "test_patients": len(test_pts),
            "total_records": len(self.df),
            "total_unique_patients": self.df["patient_id"].nunique(),
            "train_val_overlap": len(train_pts.intersection(val_pts)),
            "train_test_overlap": len(train_pts.intersection(test_pts)),
            "val_test_overlap": len(val_pts.intersection(test_pts)),
            "pairwise_fold_overlaps": pairwise_overlaps,
            "is_leakage_free": (
                len(train_pts.intersection(val_pts)) == 0
                and len(train_pts.intersection(test_pts)) == 0
                and len(val_pts.intersection(test_pts)) == 0
                and len(pairwise_overlaps) == 0
            ),
        }
        return split_isolation

    def audit_label_distributions(self, weight_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Audit multi-label distributions for 5 diagnostic superclasses
        and 23 diagnostic subclasses.
        """
        parsed_records = [
            self.mapper.map_record_statements(scp, min_weight_threshold=weight_threshold)
            for scp in self.df["scp_codes"]
        ]

        sc_counts = {sc: 0 for sc in DIAGNOSTIC_SUPERCLASSES}
        sub_counts = {sub: 0 for sub in DIAGNOSTIC_SUBCLASSES}
        label_cardinality = []

        for pr in parsed_records:
            for sc in pr["superclasses"]:
                sc_counts[sc] += 1
            for sub in pr["subclasses"]:
                sub_counts[sub] += 1
            label_cardinality.append(pr["num_superclasses"])

        cardinality_series = pd.Series(label_cardinality)

        return {
            "weight_threshold": weight_threshold,
            "diagnostic_superclasses": sc_counts,
            "diagnostic_subclasses": sub_counts,
            "label_cardinality_distribution": cardinality_series.value_counts().sort_index().to_dict(),
            "mean_label_cardinality": round(float(cardinality_series.mean()), 4),
            "records_with_zero_diagnostic_superclasses": int((cardinality_series == 0).sum()),
        }

    def compute_training_signal_statistics(
        self,
        loader_func,
        train_ecg_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        Compute standardization mean and standard deviation strictly on the training set.
        Guarantees that validation and test signals are NEVER used to fit normalization parameters.
        """
        channel_means = []
        channel_stds = []

        for ecg_id in train_ecg_ids:
            sig, fs, leads = loader_func(ecg_id)
            channel_means.append(np.mean(sig, axis=0))
            channel_stds.append(np.std(sig, axis=0))

        grand_mean = np.mean(channel_means, axis=0)
        grand_std = np.mean(channel_stds, axis=0)
        grand_std[grand_std < 1e-6] = 1.0

        return {
            "channel_mean": grand_mean,
            "channel_std": grand_std,
            "num_training_samples": len(train_ecg_ids),
        }
