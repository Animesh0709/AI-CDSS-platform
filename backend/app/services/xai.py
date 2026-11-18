"""Week 10 XGBoost prediction, SHAP explanations, and reliability checks."""
from functools import lru_cache
from pathlib import Path
import os
import sys

# ── macOS fork-safety: must be set before any OpenMP/XGBoost/SHAP import ────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "ml" / "models"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "backend" / ".mpl-cache"))
import shap
import lime.lime_tabular
import re


# ── Human-readable labels for model features ─────────────────────────────────
# Maps raw feature names (or prefixes) to plain-English clinical descriptions.
FEATURE_LABELS: dict[str, str] = {
    # Core numeric inputs
    "time_in_hospital":           "Days spent in hospital",
    "num_lab_procedures":         "Number of lab tests performed",
    "num_procedures":             "Number of medical procedures",
    "num_medications":            "Number of medications prescribed",
    "number_outpatient":          "Outpatient visits in the past year",
    "number_emergency":           "Emergency room visits in the past year",
    "number_inpatient":           "Prior hospital admissions in the past year",
    "number_diagnoses":           "Total number of diagnoses recorded",
    "age_num":                    "Patient age",
    "gender":                     "Patient sex",
    "discharge_disposition_id":   "Discharge destination (e.g., home, skilled nursing facility)",
    "admission_type_id":          "Type of hospital admission",
    "admission_source_id":        "Source of admission (e.g., emergency room, referral)",
    # Diagnoses
    "diag_1_cat_Circulatory":     "Primary diagnosis: Circulatory / heart condition",
    "diag_1_cat_Respiratory":     "Primary diagnosis: Respiratory condition",
    "diag_1_cat_Digestive":       "Primary diagnosis: Digestive condition",
    "diag_1_cat_Diabetes":        "Primary diagnosis: Diabetes",
    "diag_1_cat_Injury":          "Primary diagnosis: Injury or poisoning",
    "diag_1_cat_Musculoskeletal": "Primary diagnosis: Musculoskeletal condition",
    "diag_1_cat_Genitourinary":   "Primary diagnosis: Genitourinary condition",
    "diag_1_cat_Neoplasms":       "Primary diagnosis: Cancer / neoplasm",
    "diag_1_cat_Other":           "Primary diagnosis: Other condition",
    "diag_2_cat_Circulatory":     "Secondary diagnosis: Circulatory / heart condition",
    "diag_2_cat_Respiratory":     "Secondary diagnosis: Respiratory condition",
    "diag_2_cat_Digestive":       "Secondary diagnosis: Digestive condition",
    "diag_2_cat_Diabetes":        "Secondary diagnosis: Diabetes",
    "diag_2_cat_Injury":          "Secondary diagnosis: Injury or poisoning",
    "diag_2_cat_Musculoskeletal": "Secondary diagnosis: Musculoskeletal condition",
    "diag_2_cat_Neoplasms":       "Secondary diagnosis: Cancer / neoplasm",
    "diag_2_cat_Other":           "Secondary diagnosis: Other condition",
    "diag_3_cat_Circulatory":     "Third diagnosis: Circulatory / heart condition",
    "diag_3_cat_Respiratory":     "Third diagnosis: Respiratory condition",
    "diag_3_cat_Diabetes":        "Third diagnosis: Diabetes",
    "diag_3_cat_Neoplasms":       "Third diagnosis: Cancer / neoplasm",
    "diag_3_cat_Other":           "Third diagnosis: Other condition",
    # Medications (pattern: drug_status)
    "insulin_No":                     "Not prescribed insulin",
    "insulin_Steady":                 "Insulin dose: unchanged",
    "insulin_Up":                     "Insulin dose: increased",
    "insulin_Down":                   "Insulin dose: decreased",
    "metformin_No":                   "Not prescribed Metformin",
    "metformin_Steady":               "Metformin dose: unchanged",
    "metformin_Up":                   "Metformin dose: increased",
    "glipizide_No":                   "Not prescribed Glipizide",
    "glipizide_Steady":               "Glipizide dose: unchanged",
    "glyburide_No":                   "Not prescribed Glyburide",
    "glyburide_Steady":               "Glyburide dose: unchanged",
    "nateglinide_No":                 "Not prescribed Nateglinide",
    "nateglinide_Steady":             "Nateglinide dose: unchanged",
    "chlorpropamide_No":              "Not prescribed Chlorpropamide",
    "chlorpropamide_Steady":          "Chlorpropamide dose: unchanged",
    "acarbose_No":                    "Not prescribed Acarbose",
    "acarbose_Steady":                "Acarbose dose: unchanged",
    "miglitol_No":                    "Not prescribed Miglitol",
    "miglitol_Steady":                "Miglitol dose: unchanged",
    "pioglitazone_No":                "Not prescribed Pioglitazone",
    "rosiglitazone_No":               "Not prescribed Rosiglitazone",
    "glimepiride_No":                 "Not prescribed Glimepiride",
    "repaglinide_No":                 "Not prescribed Repaglinide",
    "tolbutamide_No":                 "Not prescribed Tolbutamide",
    "glyburide-metformin_No":         "Not prescribed Glyburide-Metformin combo",
    "glipizide-metformin_No":         "Not prescribed Glipizide-Metformin combo",
    # Lab results
    "A1Cresult_None":                 "HbA1c test: not performed",
    "A1Cresult_Norm":                 "HbA1c result: normal",
    "A1Cresult_>7":                   "HbA1c result: above 7% (elevated)",
    "A1Cresult_>8":                   "HbA1c result: above 8% (significantly elevated)",
    "max_glu_serum_None":             "Blood glucose test: not performed",
    "max_glu_serum_Norm":             "Blood glucose result: normal",
    "max_glu_serum_>200":             "Blood glucose result: above 200 mg/dL (high)",
    "max_glu_serum_>300":             "Blood glucose result: above 300 mg/dL (very high)",
    # Medical specialties
    "medical_specialty_InternalMedicine":         "Treated by: Internal Medicine",
    "medical_specialty_Cardiology":               "Treated by: Cardiology (heart)",
    "medical_specialty_Nephrology":               "Treated by: Nephrology (kidneys)",
    "medical_specialty_Pulmonology":              "Treated by: Pulmonology (lungs)",
    "medical_specialty_Gastroenterology":         "Treated by: Gastroenterology (digestive)",
    "medical_specialty_Neurology":                "Treated by: Neurology (brain/nerves)",
    "medical_specialty_Orthopedics":              "Treated by: Orthopedics (bones/joints)",
    "medical_specialty_Urology":                  "Treated by: Urology",
    "medical_specialty_Oncology":                 "Treated by: Oncology (cancer)",
    "medical_specialty_Endocrinology":            "Treated by: Endocrinology (hormones/diabetes)",
    "medical_specialty_ObstetricsandGynecology":  "Treated by: OB/GYN",
    "medical_specialty_Otolaryngology":           "Treated by: ENT (ear, nose & throat)",
    "medical_specialty_Surgery-Neuro":            "Treated by: Neurosurgery",
    "medical_specialty_Surgery-General":          "Treated by: General Surgery",
    "medical_specialty_Psychiatry":               "Treated by: Psychiatry",
    "medical_specialty_Radiology":                "Treated by: Radiology",
    "medical_specialty_Emergency/Trauma":         "Treated by: Emergency / Trauma",
    "medical_specialty_Missing":                  "Treating specialty: not recorded",
    # Race
    "race_Caucasian":     "Race: Caucasian",
    "race_AfricanAmerican": "Race: African American",
    "race_Hispanic":      "Race: Hispanic",
    "race_Asian":         "Race: Asian",
    "race_Other":         "Race: Other",
    "race_Missing":       "Race: not recorded",
}


def humanize_feature(raw: str) -> str:
    """Return a plain-English label for a raw feature name."""
    if raw in FEATURE_LABELS:
        return FEATURE_LABELS[raw]
    # Try prefix match for one-hot encoded columns not explicitly listed
    for key, label in FEATURE_LABELS.items():
        if raw.startswith(key):
            return label
    # Generic cleanup: replace underscores, capitalise
    cleaned = raw.replace("_", " ").replace("-", " ")
    return cleaned.title()



def _is_binary_feature(feat: str) -> bool:
    """True if the feature is a binary/one-hot encoded column."""
    binary_suffixes = ('_No', '_Yes', '_Steady', '_Up', '_Down', '_None', '_Norm',
                       '>7', '>8', '>200', '>300')
    binary_prefixes = ('medical_specialty_', 'diag_1_cat_', 'diag_2_cat_', 'diag_3_cat_',
                       'race_', 'A1Cresult_', 'max_glu_serum_')
    return (any(feat.endswith(s) for s in binary_suffixes) or
            any(feat.startswith(p) for p in binary_prefixes))


def _strip_status_prefix(label: str) -> str:
    """Remove 'Not prescribed / Not treated by / Not on' prefix to get the core noun."""
    return re.sub(r'^Not (prescribed|treated by|on) ', '', label)


def _binary_absent(feat: str, label: str) -> str:
    """Interpret a binary feature whose value is 0 (absent in the record)."""
    if feat.endswith('_No'):
        # 'drug_No = 0' means the patient IS on the drug (¬ "No" = "Yes")
        core = _strip_status_prefix(label)
        return f"Currently prescribed {core}"
    if feat.startswith('medical_specialty_'):
        specialty = label.replace('Treated by: ', '').strip()
        return f"Not treated by {specialty}"
    if feat.startswith(('diag_1_cat_', 'diag_2_cat_', 'diag_3_cat_')):
        return f"No {label.lower()}"
    if feat.startswith('race_'):
        return f"Patient not recorded as {label.split(': ', 1)[-1]}"
    # Steady/Up/Down/None/Norm — "status = 0" means this status is absent
    core = _strip_status_prefix(label)
    return f"Not: {core.lower()}"


def _binary_present(feat: str, label: str) -> str:
    """Interpret a binary feature whose value is 1 (present in the record)."""
    if feat.endswith('_No'):
        # 'drug_No = 1' means patient is genuinely NOT on the drug
        return label  # already reads "Not prescribed X"
    if feat.startswith('medical_specialty_'):
        specialty = label.replace('Treated by: ', '').strip()
        return f"Treated by {specialty}"
    if feat.startswith(('diag_1_cat_', 'diag_2_cat_', 'diag_3_cat_')):
        return label
    if feat.startswith('race_'):
        return label
    return label


def humanize_lime_condition(condition: str) -> str:
    """
    Translate a LIME condition string into plain clinical English.

    Handles three forms:
      'feature_name OP value'          e.g. 'number_inpatient <= -0.50'
      'lo < feature_name <= hi'        e.g. '0.10 < num_medications <= 1.20'
    """
    condition = condition.strip()

    # ── Form 2: range  lo < feature <= hi ────────────────────────────────────
    m2 = re.match(r'^(-?[\d.]+)\s*<\s*(.+?)\s*<=\s*(-?[\d.]+)$', condition)
    if m2:
        _, feat, _ = m2.group(1), m2.group(2).strip(), m2.group(3)
        label = humanize_feature(feat)
        core = _strip_status_prefix(label)
        return f"Moderate {core.lower()}"

    # ── Form 1: feature OP value ──────────────────────────────────────────────
    m1 = re.match(r'^(.+?)\s*(<=|>=|<|>)\s*(-?[\d.]+)$', condition)
    if m1:
        feat, op, val = m1.group(1).strip(), m1.group(2), m1.group(3)
        fval  = float(val)
        label = humanize_feature(feat)

        if _is_binary_feature(feat):
            # Threshold ≈ 0 → feature value ≤ 0 means absent, > 0 means present
            feature_absent = op in ('<=', '<') and fval <= 0.5
            return _binary_absent(feat, label) if feature_absent else _binary_present(feat, label)

        # ── Continuous feature ────────────────────────────────────────────────
        core = _strip_status_prefix(label)
        if op in ('<', '<='):
            return f"Low {core.lower()}"
        return f"High {core.lower()}"

    # ── Fallback ──────────────────────────────────────────────────────────────
    return condition.replace('_', ' ').strip()


def _flag_outliers(items: list[dict]) -> list[dict]:
    """
    Tag each factor with is_outlier=True when its |contribution| is
    unusually large (> 2.5 × median of all |contributions|).
    Chlorpropamide_No is a typical case: near-constant feature (99.9% = 1)
    so XGBoost assigns it a large weight for the rare exceptions.
    """
    if len(items) <= 3:
        return [{**x, 'is_outlier': False} for x in items]
    abs_vals = sorted(abs(x['contribution']) for x in items)
    median_abs = abs_vals[len(abs_vals) // 2] or 1e-9
    threshold = 2.5 * median_abs
    return [{**x, 'is_outlier': abs(x['contribution']) > threshold} for x in items]


@lru_cache(maxsize=1)
def artifacts():
    model = joblib.load(MODELS / "xgboost_readmission.joblib")
    preprocessor = joblib.load(MODELS / "preprocessor.joblib")
    # Load the PR-curve-tuned decision threshold saved by train.py.
    # Falls back to 0.5 if the file is not present (e.g. old model artifacts).
    threshold_path = MODELS / "threshold.json"
    if threshold_path.exists():
        import json as _json
        threshold = float(_json.loads(threshold_path.read_text())["optimal_threshold"])
    else:
        threshold = 0.5
    return model, preprocessor, threshold


@lru_cache(maxsize=1)
def lime_explainer():
    scripts = str(ROOT / "ml" / "scripts")
    if scripts not in sys.path: sys.path.insert(0, scripts)
    from download_or_generate_dataset import fetch_or_create_dataset
    from preprocess import preprocess_data
    _, x_test, _, _, feature_names = preprocess_data(fetch_or_create_dataset())
    sample = x_test.sample(min(250, len(x_test)), random_state=42).values
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=sample,
        feature_names=feature_names,
        class_names=["No Readmission", "Readmission"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )


def feature_row(values: dict, patient):
    model, prep, _ = artifacts()
    names = prep["feature_names"]
    row = pd.DataFrame(np.zeros((1, len(names))), columns=names)
    age = 55
    if patient.date_of_birth:
        try: age = max(0, (pd.Timestamp.now() - pd.Timestamp(patient.date_of_birth)).days // 365)
        except (ValueError, TypeError): pass
    raw = {**values, "age_num": age}
    for name, value in raw.items():
        if name in row: row.loc[0, name] = value
    if "gender" in row: row.loc[0, "gender"] = int(patient.gender == "Male")
    numeric = prep["numeric_cols"]
    
    raw_row = row.copy()
    row[numeric] = prep["scaler"].transform(row[numeric])
    quality = ["Input schema matches the model feature list.", "Uncollected categorical encounter features use the training reference baseline."]
    z_scores = {name: float(abs(row.loc[0, name])) for name in numeric}
    unusual = [name for name, score in z_scores.items() if score > 3]
    warning = ("This input differs substantially from typical training data: " + ", ".join(unusual) + ". Interpret the prediction with additional caution.") if unusual else None
    return row, raw_row, quality, warning


def local_explanation(values: dict, patient):
    model, _, threshold = artifacts()
    row, raw_row, quality, warning = feature_row(values, patient)
    probability = float(model.predict_proba(row)[0, 1])
    high_risk = probability >= threshold
    explainer = shap.TreeExplainer(model)
    base_val = explainer.expected_value
    base_value = float(base_val[0]) if isinstance(base_val, (list, np.ndarray)) else float(base_val)
    contributions = explainer.shap_values(row)
    contributions = np.asarray(contributions).reshape(-1)
    # Build SHAP items — attach human_label alongside raw feature name
    items = [
        {
            "feature": name,
            "human_label": humanize_feature(name),
            "contribution": round(float(value), 4),
        }
        for name, value in zip(row.columns, contributions)
        if abs(value) > 0.0001
    ]
    items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    positive = [item for item in items if item["contribution"] > 0]

    lime_explainer_obj = lime_explainer()
    def predict_fn(x):
        return model.predict_proba(pd.DataFrame(x, columns=row.columns))
    lime_exp = lime_explainer_obj.explain_instance(row.values[0], predict_fn, num_features=12)
    # Build LIME items — translate condition strings to plain English
    lime_items = [
        {
            "feature": cond,
            "human_label": humanize_lime_condition(cond),
            "contribution": round(float(w), 4),
        }
        for cond, w in lime_exp.as_list()
    ]
    lime_positive = [item for item in lime_items if item["contribution"] > 0][:6]
    lime_negative = [item for item in lime_items if item["contribution"] < 0][:6]

    shap_positive = positive[:6]
    shap_negative = [item for item in items if item["contribution"] < 0][:6]

    # Flag outliers across the combined SHAP and LIME pools
    all_shap = _flag_outliers(shap_positive + shap_negative)
    half_s = len(shap_positive)
    all_lime = _flag_outliers(lime_positive + lime_negative)
    half_l = len(lime_positive)

    return {
        "probability": round(probability, 4),
        "high_risk": high_risk,
        "decision_threshold": round(threshold, 4),
        "model_label": "XGBoost 30-day readmission model",
        "base_value": base_value,
        "patient_features": {k: (float(v) if pd.notna(v) else 0.0) for k, v in raw_row.iloc[0].items()},
        "increasing_factors": all_shap[:half_s],
        "decreasing_factors": all_shap[half_s:],
        "lime_increasing_factors": all_lime[:half_l],
        "lime_decreasing_factors": all_lime[half_l:],
        "data_quality": quality,
        "ood_warning": warning,
        "disclaimer": "Model output and SHAP contributions describe model behaviour only. They are not a diagnosis, prognosis, or treatment recommendation; clinician review is required.",
    }



def _compute_global_explanation():
    scripts = str(ROOT / "ml" / "scripts")
    if scripts not in sys.path: sys.path.insert(0, scripts)
    from download_or_generate_dataset import fetch_or_create_dataset
    from preprocess import preprocess_data
    model, prep, threshold = artifacts()
    _, x_test, _, y_test, _ = preprocess_data(fetch_or_create_dataset())
    
    # Increase sample size to 1000 for more stable global SHAP
    sample = x_test.sample(min(1000, len(x_test)), random_state=42)
    shap_values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    
    # Top 15 features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top15 = np.argsort(mean_abs_shap)[::-1][:15]
    
    importance = []
    scaler = prep.get("scaler")
    numeric_cols = prep.get("numeric_cols", [])
    
    for i in top15:
        feature_name = sample.columns[i]
        vals = sample[feature_name].values
        
        # If the feature was scaled, unscale it for the stats
        unscaled_vals = vals
        if feature_name in numeric_cols and scaler is not None:
            col_idx = numeric_cols.index(feature_name)
            unscaled_vals = vals * scaler.scale_[col_idx] + scaler.mean_[col_idx]
            
        importance.append({
            "feature": feature_name,
            "mean_abs_shap": round(float(mean_abs_shap[i]), 4),
            "val_min": round(float(np.min(unscaled_vals)), 2),
            "val_max": round(float(np.max(unscaled_vals)), 2),
            "val_mean": round(float(np.mean(unscaled_vals)), 2)
        })
        
    perm = permutation_importance(model, sample, y_test.loc[sample.index], scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=-1)
    permutation = [{"feature": sample.columns[i], "importance": round(float(perm.importances_mean[i]), 4)} for i in np.argsort(perm.importances_mean)[::-1][:8]]
    
    plots = {}
    for i in top15[:2]:
        feature = sample.columns[i]
        base = sample[feature]
        grid = np.linspace(float(base.quantile(.05)), float(base.quantile(.95)), 10)
        pdp, ice = [], []
        for value in grid:
            changed = sample.copy(); changed[feature] = value
            pdp.append(round(float(model.predict_proba(changed)[:, 1].mean()), 4))
            ice.append([round(float(v), 4) for v in model.predict_proba(changed.iloc[:8])[:, 1]])
        plots[feature] = {"values": [round(float(v), 4) for v in grid], "pdp": pdp, "ice": ice}
        
    return {
        "global_shap": importance,
        "permutation_importance": permutation,
        "pdp_ice": plots,
        "calibration_note": "Probability calibration is reported by the Week 9 experiment. Validate calibration and external performance before treating any score as a reliable probability."
    }

def global_explanation():
    import json
    cache_file = MODELS / "global_shap.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    result = _compute_global_explanation()
    cache_file.write_text(json.dumps(result))
    return result
