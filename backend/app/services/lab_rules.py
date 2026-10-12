"""Deterministic, review-only reference-range checks for the Week 4 prototype."""
import re


# Generic adult screening ranges. They are explicit rules, not diagnoses, and
# must be checked by a clinician against the reporting laboratory's range.
RULES = {
    "hba1c": ("HbA1c", "%", None, 5.6, "Configured screening reference: below 5.7%"),
    "glucose": ("Glucose", "mg/dL", 70, 99, "Configured fasting reference: 70–99 mg/dL"),
    "hemoglobin": ("Hemoglobin", "g/dL", 12.0, 17.5, "Configured generic adult reference: 12.0–17.5 g/dL"),
    "wbc": ("WBC", "/uL", 4000, 11000, "Configured adult reference: 4,000–11,000 /uL"),
    "platelets": ("Platelets", "/uL", 150000, 450000, "Configured adult reference: 150,000–450,000 /uL"),
    "creatinine": ("Creatinine", "mg/dL", 0.6, 1.3, "Configured generic adult reference: 0.6–1.3 mg/dL"),
    "urea": ("Urea", "mg/dL", 15, 40, "Configured adult reference: 15–40 mg/dL"),
    "sodium": ("Sodium", "mEq/L", 135, 145, "Configured adult reference: 135–145 mEq/L"),
    "potassium": ("Potassium", "mEq/L", 3.5, 5.0, "Configured adult reference: 3.5–5.0 mEq/L"),
    "cholesterol": ("Total Cholesterol", "mg/dL", None, 199, "Configured desirable threshold: below 200 mg/dL"),
    "ldl": ("LDL", "mg/dL", None, 99, "Configured desirable threshold: below 100 mg/dL"),
    "hdl": ("HDL", "mg/dL", 40, None, "Configured generic adult threshold: at least 40 mg/dL"),
    "triglycerides": ("Triglycerides", "mg/dL", None, 149, "Configured desirable threshold: below 150 mg/dL"),
    "tsh": ("TSH", "mIU/L", 0.4, 4.0, "Configured adult reference: 0.4–4.0 mIU/L"),
}
ALIASES = {"a1c": "hba1c", "ha1c": "hba1c", "white blood cells": "wbc", "total cholesterol": "cholesterol"}


def _key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return ALIASES.get(normalized, normalized.replace(" ", ""))


def _unit(unit: str) -> str:
    return unit.lower().replace("μ", "u").replace(" ", "")


def analyse_lab_value(test_name: str, value: float, unit: str | None = None) -> dict:
    """Apply one configured rule and return display-ready, non-diagnostic data."""
    rule = RULES.get(_key(test_name))
    if not rule:
        raise ValueError(f"No configured rule exists for '{test_name}'.")
    display_name, expected_unit, low, high, basis = rule
    unit = (unit or expected_unit).strip()
    if _unit(unit) != _unit(expected_unit):
        raise ValueError(f"{display_name} requires {expected_unit} for this prototype; unit conversion is not performed.")
    if low is not None and value < low:
        status = "Potentially Low"
    elif high is not None and value > high:
        status = "Potentially High"
    else:
        status = "Normal"
    bounds = []
    if low is not None:
        bounds.append(f">= {low:g}")
    if high is not None:
        bounds.append(f"<= {high:g}")
    return {"test_name": display_name, "unit": expected_unit, "status": status,
            "reference_range": f"{' and '.join(bounds)} {expected_unit}", "clinical_basis": basis}
