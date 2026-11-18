"""
LUMIERE Longitudinal Glioblastoma MRI — Progression Preprocessing Pipeline
===========================================================================

Transforms longitudinal MRI scans and expert RANO assessments into reproducible
transition examples (T_baseline -> T_followup) for Tumor Progression Prediction.

Strict Guidelines Enforced:
1. Ground Truth Target:
   - Progressive Disease (PD, Post-Op/PD) -> 1
   - Stable Disease (SD), Partial Response (PR), Complete Response (CR) -> 0
   - Pre-Op / Post-Op baseline scans without prior comparator -> excluded as targets
2. Anti-Leakage Patient Grouping:
   - Split strictly by patient_id BEFORE generating examples so no patient's scans
     ever cross train/val/test splits.
3. Feature Boundaries:
   - Only features available up to prediction time T_followup are included.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "ml" / "data" / "lumiere"


def parse_week(date_str: str) -> float:
    """Convert 'week-000-1' or 'week-044' to numeric weeks."""
    clean = str(date_str).replace("week-", "").strip()
    parts = clean.split("-")
    try:
        week = float(parts[0])
        sub = float(parts[1]) if len(parts) > 1 else 0.0
        return week + sub * 0.01
    except Exception:
        return 0.0


def load_and_clean_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rating_file = data_dir / "LUMIERE-ExpertRating-v202211.csv"
    demo_file = data_dir / "LUMIERE-Demographics_Pathology.csv"
    rad_file = data_dir / "LUMIERE-pyradiomics-hdglioauto-features.csv"

    if not rating_file.exists():
        raise FileNotFoundError(f"Missing {rating_file}. Please run download script first.")

    ratings = pd.read_csv(rating_file)
    ratings.columns = [c.strip() for c in ratings.columns]
    rating_col = [c for c in ratings.columns if "Rating (" in c][0]
    ratings["RANO"] = ratings[rating_col].astype(str).str.strip()
    ratings["week_num"] = ratings["Date"].apply(parse_week)

    demo = pd.read_csv(demo_file)
    demo.columns = [c.strip() for c in demo.columns]

    # Load select radiomics columns to optimize memory & reproducibility
    feature_cols = [
        "Patient", "Time point", "Sequence", "Label name",
        "original_shape_MeshVolume", "original_shape_VoxelVolume",
        "original_shape_SurfaceArea", "original_shape_Sphericity",
        "original_firstorder_Entropy", "original_firstorder_Mean",
        "original_glcm_Contrast", "original_glcm_JointEntropy"
    ]
    rad = pd.read_csv(rad_file, usecols=lambda c: c in feature_cols)
    rad.columns = [c.strip() for c in rad.columns]

    return ratings, demo, rad


def pivot_radiomics(rad: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot radiomics so each (Patient, Time point) is a single row with sequence and label prefixes.
    """
    rad["label_prefix"] = rad["Label name"].replace({
        "Contrast-enhancing": "enhancing",
        "Non-enhancing": "edema"
    })
    rad["feat_prefix"] = rad["Sequence"].str.lower() + "_" + rad["label_prefix"]

    numeric_cols = [c for c in rad.columns if c.startswith("original_")]
    pivoted = rad.pivot_table(
        index=["Patient", "Time point"],
        columns="feat_prefix",
        values=numeric_cols,
        aggfunc="first"
    )
    # Flatten multiindex columns: e.g. "ct1_enhancing_MeshVolume"
    pivoted.columns = [f"{col[1]}_{col[0].replace('original_', '')}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    return pivoted


def build_transition_dataset(
    ratings: pd.DataFrame,
    demo: pd.DataFrame,
    rad_pivoted: pd.DataFrame
) -> pd.DataFrame:
    """
    Build longitudinal transition pairs (T_0 -> T_1) for each patient.
    """
    # Demo lookup
    demo_dict = demo.set_index("Patient").to_dict(orient="index")
    rad_idx = set(zip(rad_pivoted["Patient"], rad_pivoted["Time point"]))
    rad_lookup = rad_pivoted.set_index(["Patient", "Time point"])

    # Sort visits by patient and week count
    ratings = ratings.sort_values(["Patient", "week_num"]).reset_index(drop=True)

    rows = []
    for patient, p_df in ratings.groupby("Patient"):
        p_dates = p_df["Date"].tolist()
        p_ranos = p_df["RANO"].tolist()
        p_weeks = p_df["week_num"].tolist()
        p_less3 = p_df["LessThan3Months"].fillna("").tolist()

        p_info = demo_dict.get(patient, {})
        age = p_info.get("Age at surgery (years)", np.nan)
        sex = 1.0 if str(p_info.get("Sex", "")).lower() == "m" else 0.0
        mgmt = 1.0 if "methylated" in str(p_info.get("MGMT qualitative", "")).lower() and "not" not in str(p_info.get("MGMT qualitative", "")).lower() else 0.0
        idh = 1.0 if "mut" in str(p_info.get("IDH (WT: wild type)", "")).lower() else 0.0

        for i in range(1, len(p_dates)):
            curr_rano = p_ranos[i]
            # Exclude non-evaluable progression states
            if curr_rano not in ["PD", "SD", "PR", "CR", "Post-Op/PD"]:
                continue

            prev_date = p_dates[i - 1]
            curr_date = p_dates[i]
            has_prev = (patient, prev_date) in rad_idx
            has_curr = (patient, curr_date) in rad_idx

            if not (has_prev and has_curr):
                continue

            delta_weeks = p_weeks[i] - p_weeks[i - 1]
            is_pseudo_window = 1.0 if p_less3[i] == "x" else 0.0
            target = 1 if ("PD" in curr_rano) else 0

            # Retrieve radiomics for both
            r_prev = rad_lookup.loc[(patient, prev_date)]
            r_curr = rad_lookup.loc[(patient, curr_date)]

            row = {
                "patient_id": patient,
                "baseline_date": prev_date,
                "followup_date": curr_date,
                "delta_weeks": delta_weeks,
                "is_pseudo_progression_window": is_pseudo_window,
                "age": float(age) if not pd.isna(age) else 60.0,
                "sex_male": sex,
                "mgmt_methylated": mgmt,
                "idh_mutant": idh,
                "rano_label": curr_rano,
                "progression_target": target
            }

            # Baseline absolute volumes (cm^3 = mm^3 / 1000)
            v0_enh = r_prev.get("ct1_enhancing_shape_VoxelVolume", np.nan) / 1000.0 if "ct1_enhancing_shape_VoxelVolume" in r_prev else np.nan
            v1_enh = r_curr.get("ct1_enhancing_shape_VoxelVolume", np.nan) / 1000.0 if "ct1_enhancing_shape_VoxelVolume" in r_curr else np.nan
            v0_edema = r_prev.get("flair_edema_shape_VoxelVolume", np.nan) / 1000.0 if "flair_edema_shape_VoxelVolume" in r_prev else np.nan
            v1_edema = r_curr.get("flair_edema_shape_VoxelVolume", np.nan) / 1000.0 if "flair_edema_shape_VoxelVolume" in r_curr else np.nan

            row["baseline_enhancing_vol_cm3"] = v0_enh
            row["followup_enhancing_vol_cm3"] = v1_enh
            row["delta_enhancing_vol_cm3"] = v1_enh - v0_enh
            row["rel_change_enhancing_vol"] = (v1_enh - v0_enh) / (v0_enh + 0.1)

            row["baseline_edema_vol_cm3"] = v0_edema
            row["followup_edema_vol_cm3"] = v1_edema
            row["delta_edema_vol_cm3"] = v1_edema - v0_edema
            row["rel_change_edema_vol"] = (v1_edema - v0_edema) / (v0_edema + 0.1)

            # Compute deltas for all numeric features
            for col in r_curr.index:
                val0 = float(r_prev[col]) if not pd.isna(r_prev[col]) else 0.0
                val1 = float(r_curr[col]) if not pd.isna(r_curr[col]) else 0.0
                row[f"delta_{col}"] = val1 - val0
                row[f"curr_{col}"] = val1

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def split_by_patient(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified train / validation / test split.
    Guarantees no patient's scans appear across multiple splits.
    """
    unique_patients = df["patient_id"].unique()
    # Patient-level outcome: does the patient ever progress in this dataset?
    patient_outcomes = df.groupby("patient_id")["progression_target"].max().loc[unique_patients].values

    # Step 1: 5-fold StratifiedGroupKFold on patients
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df, df["progression_target"], groups=df["patient_id"]))

    train_val_idx, test_idx = splits[0]
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Step 2: Split train_val into train and val (4-fold gives 75% / 25% of train_val -> ~60% / ~20% of total)
    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    tv_splits = list(sgkf_val.split(train_val_df, train_val_df["progression_target"], groups=train_val_df["patient_id"]))
    train_idx, val_idx = tv_splits[0]

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Assert strict zero leakage
    train_pts = set(train_df["patient_id"])
    val_pts = set(val_df["patient_id"])
    test_pts = set(test_df["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test!"

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess LUMIERE dataset for Progression Prediction")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw LUMIERE tables...")
    ratings, demo, rad = load_and_clean_data(DATA_DIR)
    print("Pivoting multi-sequence radiomics...")
    rad_pivoted = pivot_radiomics(rad)
    print("Constructing longitudinal transition pairs...")
    dataset = build_transition_dataset(ratings, demo, rad_pivoted)

    print(f"Total transition examples: {len(dataset)}")
    print(f"Total unique patients: {dataset['patient_id'].nunique()}")
    print("Class distribution:")
    print(dataset["progression_target"].value_counts())

    print("\nSplitting by patient_id...")
    train_df, val_df, test_df = split_by_patient(dataset)
    print(f"Train set: {len(train_df)} transitions across {train_df['patient_id'].nunique()} patients (Prog: {train_df['progression_target'].sum()})")
    print(f"Val set:   {len(val_df)} transitions across {val_df['patient_id'].nunique()} patients (Prog: {val_df['progression_target'].sum()})")
    print(f"Test set:  {len(test_df)} transitions across {test_df['patient_id'].nunique()} patients (Prog: {test_df['progression_target'].sum()})")

    dataset.to_csv(out_dir / "progression_full.csv", index=False)
    train_df.to_csv(out_dir / "progression_train.csv", index=False)
    val_df.to_csv(out_dir / "progression_val.csv", index=False)
    test_df.to_csv(out_dir / "progression_test.csv", index=False)
    print(f"Saved preprocessed splits to {out_dir}")


if __name__ == "__main__":
    main()
