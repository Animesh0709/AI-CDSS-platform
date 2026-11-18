"""
Validation and Quality Check Script for ECG Datasets and Loaders.

Executes:
1. PTB-XL manifest integrity check (records count, patient splits, label distribution).
2. Signal loading and unit validation for sample PTB-XL records.
3. LUDB waveform delineation parsing check.
4. MIT-BIH beat annotation parsing check.
5. Synthetic 12-lead paper ECG rendering test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ECG_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ecg"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.scripts.ecg.ptbxl_loader import PTBXLDatasetLoader
from ml.scripts.ecg.ludb_loader import LUDBDatasetLoader
from ml.scripts.ecg.mitdb_loader import MITDBDatasetLoader
from ml.scripts.ecg.render_ecg_image import render_12lead_ecg_image


def validate_all_ecg_datasets():
    results = {}
    print("=" * 60)
    print("ECG DATASET PIPELINE VALIDATION AUDIT")
    print("=" * 60)

    # 1. PTB-XL
    print("\n[1/4] Auditing PTB-XL Dataset...")
    ptb_loader = PTBXLDatasetLoader(ECG_DATA_DIR / "ptbxl")
    df = ptb_loader.map_diagnostic_superclasses()
    train_df, val_df, test_df = ptb_loader.get_split_manifests()

    ptb_summary = {
        "total_records": len(df),
        "total_patients": int(df["patient_id"].nunique()),
        "train_records": len(train_df),
        "train_patients": int(train_df["patient_id"].nunique()),
        "val_records": len(val_df),
        "val_patients": int(val_df["patient_id"].nunique()),
        "test_records": len(test_df),
        "test_patients": int(test_df["patient_id"].nunique()),
        "superclass_distribution": {
            "NORM": int(df["label_NORM"].sum()),
            "MI": int(df["label_MI"].sum()),
            "STTC": int(df["label_STTC"].sum()),
            "CD": int(df["label_CD"].sum()),
            "HYP": int(df["label_HYP"].sum()),
        },
    }
    print(f"  Records: {ptb_summary['total_records']}, Patients: {ptb_summary['total_patients']}")
    print(f"  Train: {ptb_summary['train_records']}, Val: {ptb_summary['val_records']}, Test: {ptb_summary['test_records']}")
    print(f"  Superclasses: {ptb_summary['superclass_distribution']}")

    # Test signal loading on available sample
    sample_signal, fs, leads = ptb_loader.load_record_signal(1, resolution="100")
    print(f"  Sample Record 1: shape={sample_signal.shape}, fs={fs}Hz, leads={leads}")
    assert sample_signal.shape == (1000, 12), f"Unexpected shape {sample_signal.shape}"
    results["ptbxl"] = ptb_summary

    # Save manifests
    manifest_dir = ECG_DATA_DIR / "ptbxl" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(manifest_dir / "ptbxl_train_manifest.csv")
    val_df.to_csv(manifest_dir / "ptbxl_val_manifest.csv")
    test_df.to_csv(manifest_dir / "ptbxl_test_manifest.csv")
    print(f"  Manifests saved to {manifest_dir}")

    # 2. LUDB
    print("\n[2/4] Auditing LUDB Dataset...")
    ludb_loader = LUDBDatasetLoader(ECG_DATA_DIR / "ludb")
    ludb_records = ludb_loader.list_records()
    print(f"  Total catalogued records in RECORDS: {len(ludb_records)}")

    ludb_sig, ludb_fs, ludb_leads = ludb_loader.load_record(1)
    ludb_waves_i = ludb_loader.load_lead_annotations(1, "i")
    print(f"  Sample Record 1: shape={ludb_sig.shape}, fs={ludb_fs}Hz, leads={ludb_leads}")
    print(f"  Wave boundaries Lead I: P={len(ludb_waves_i['P'])}, QRS={len(ludb_waves_i['QRS'])}, T={len(ludb_waves_i['T'])}")
    assert len(ludb_waves_i["QRS"]) > 0, "No QRS detected in LUDB lead i"

    results["ludb"] = {
        "catalogued_records": len(ludb_records),
        "sampling_frequency_hz": ludb_fs,
        "sample_waves_count": {
            "P": len(ludb_waves_i["P"]),
            "QRS": len(ludb_waves_i["QRS"]),
            "T": len(ludb_waves_i["T"]),
        },
    }

    # 3. MIT-BIH
    print("\n[3/4] Auditing MIT-BIH Arrhythmia Database...")
    mit_loader = MITDBDatasetLoader(ECG_DATA_DIR / "mitdb")
    mit_records = mit_loader.list_records()
    print(f"  Catalogued records: {len(mit_records)}")

    mit_sig, mit_fs, mit_sig_names = mit_loader.load_record("100", sampto=3600)
    mit_samples, mit_symbols = mit_loader.load_annotations("100", sampto=3600)
    print(f"  Record 100 (first 10s): shape={mit_sig.shape}, fs={mit_fs}Hz, leads={mit_sig_names}")
    print(f"  Beat annotations: {len(mit_symbols)} beats, unique types={set(mit_symbols)}")

    results["mitdb"] = {
        "catalogued_records": len(mit_records),
        "sampling_frequency_hz": mit_fs,
        "sample_100_beats_in_10s": len(mit_symbols),
        "unique_beat_types_sample": sorted(list(set(mit_symbols))),
    }

    # 4. Synthetic ECG Image Renderer
    print("\n[4/4] Validating 12-Lead Paper ECG Rendering...")
    synth_output = ECG_DATA_DIR / "sample_synthetic_12lead.png"
    meta = render_12lead_ecg_image(
        signal=sample_signal,
        sampling_rate=fs,
        output_image_path=synth_output,
        lead_names=leads,
    )
    print(f"  Generated synthetic ECG scan: {synth_output}")
    print(f"  Image metadata: {meta}")
    assert synth_output.exists() and synth_output.stat().st_size > 50000, "Image generation failed"

    # Save summary report
    report_path = ECG_DATA_DIR / "ecg_datasets_summary.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit complete. Full summary report written to {report_path}")

    return results


if __name__ == "__main__":
    validate_all_ecg_datasets()
