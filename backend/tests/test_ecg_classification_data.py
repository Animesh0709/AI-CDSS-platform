"""
Unit tests for PTB-XL multi-label target mapping and patient split isolation.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from ml.ecg_classification.label_mapping import (
    PTBXLLabelMapper,
    DIAGNOSTIC_SUPERCLASSES,
    DIAGNOSTIC_SUBCLASSES,
)
from ml.ecg_classification.dataset_auditor import PTBXLDatasetAuditor
from ml.scripts.ecg.ptbxl_loader import PTBXLDatasetLoader

DATA_DIR = Path("ml/data/ecg/ptbxl")
SCP_CSV = DATA_DIR / "scp_statements.csv"
DB_CSV = DATA_DIR / "ptbxl_database.csv"


@pytest.fixture
def mapper() -> PTBXLLabelMapper:
    return PTBXLLabelMapper(SCP_CSV)


@pytest.fixture
def auditor() -> PTBXLDatasetAuditor:
    return PTBXLDatasetAuditor(DATA_DIR)


def test_scp_statement_category_partition(mapper: PTBXLLabelMapper):
    """Verify that statements are partitioned cleanly into diagnostic, form, and rhythm categories."""
    assert len(mapper.diagnostic_df) == 44
    assert len(mapper.form_df) == 19
    assert len(mapper.rhythm_df) == 12

    # Verify statement categories
    cat_norm = mapper.get_statement_category("NORM")
    assert cat_norm["diagnostic"] is True
    assert cat_norm["form"] is False
    assert cat_norm["rhythm"] is False

    cat_afib = mapper.get_statement_category("AFIB")
    assert cat_afib["diagnostic"] is False
    assert cat_afib["rhythm"] is True

    cat_qwave = mapper.get_statement_category("QWAVE")
    assert cat_qwave["diagnostic"] is False
    assert cat_qwave["form"] is True


def test_zero_weight_label_exclusion(mapper: PTBXLLabelMapper):
    """Verify that zero-weight (ruled out / uncertain) statements are excluded from positive targets."""
    # NST_ is a diagnostic code under STTC. If weight is 0.0, it must be excluded.
    scp_input = "{'NORM': 100.0, 'NST_': 0.0, 'SR': 0.0}"
    res = mapper.map_record_statements(scp_input, min_weight_threshold=0.0)

    assert "NORM" in res["superclasses"]
    assert "STTC" not in res["superclasses"]
    assert "NST_" in res["zero_weight_statements"]
    assert "SR" in res["zero_weight_statements"]
    assert res["superclass_vector"]["label_NORM"] == 1
    assert res["superclass_vector"]["label_STTC"] == 0


def test_multi_label_representation(mapper: PTBXLLabelMapper):
    """Verify records with multiple diagnostic superclasses and subclasses are multi-hot encoded."""
    # Composite case: Antero-septal MI + Complete RBBB + Left Ventricular Hypertrophy
    scp_input = "{'ASMI': 100.0, 'CRBBB': 100.0, 'LVH': 80.0}"
    res = mapper.map_record_statements(scp_input)

    assert set(res["superclasses"]) == {"MI", "CD", "HYP"}
    assert "AMI" in res["subclasses"]
    assert "CRBBB" in res["subclasses"]
    assert "LVH" in res["subclasses"]

    # Verify multi-hot vector integrity
    vec = res["superclass_vector"]
    assert vec["label_MI"] == 1
    assert vec["label_CD"] == 1
    assert vec["label_HYP"] == 1
    assert vec["label_NORM"] == 0
    assert vec["label_STTC"] == 0


def test_pure_rhythm_zero_diagnostic_record(mapper: PTBXLLabelMapper):
    """Verify that pure rhythm records (e.g., AFIB or Pacemaker) have all-zero diagnostic superclasses."""
    scp_input = "{'AFIB': 100.0, 'PACE': 100.0}"
    res = mapper.map_record_statements(scp_input)

    assert len(res["superclasses"]) == 0
    assert len(res["subclasses"]) == 0
    assert "AFIB" in res["rhythm_statements"]
    assert "PACE" in res["rhythm_statements"]
    assert all(v == 0 for v in res["superclass_vector"].values())


def test_ptbxl_official_splits_strict_patient_isolation(auditor: PTBXLDatasetAuditor):
    """Verify zero patient overlap across official train (1-8), val (9), and test (10) splits."""
    iso = auditor.audit_splits_and_patient_isolation()

    assert iso["is_leakage_free"] is True
    assert iso["total_records"] == 21799
    assert iso["total_unique_patients"] == 18869
    assert iso["train_records"] == 17418
    assert iso["val_records"] == 2183
    assert iso["test_records"] == 2198

    # Overlaps MUST be strictly 0
    assert iso["train_val_overlap"] == 0
    assert iso["train_test_overlap"] == 0
    assert iso["val_test_overlap"] == 0
    assert len(iso["pairwise_fold_overlaps"]) == 0


def test_training_signal_standardization_isolation(auditor: PTBXLDatasetAuditor):
    """Verify that normalization statistics are strictly derived from training records only."""
    loader = PTBXLDatasetLoader(DATA_DIR)
    # Use available training records from records100 (e.g. records 1, 2, 3)
    train_ids = [1, 2, 3]

    stats = auditor.compute_training_signal_statistics(
        lambda eid: loader.load_record_signal(eid, resolution="100"),
        train_ids,
    )

    assert stats["num_training_samples"] == 3
    assert stats["channel_mean"].shape == (12,)
    assert stats["channel_std"].shape == (12,)
    assert np.all(stats["channel_std"] > 0)
