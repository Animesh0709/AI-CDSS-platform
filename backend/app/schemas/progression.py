"""
Pydantic Schemas for Tumor Progression Prediction.
==================================================
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProgressionPredictionInput(BaseModel):
    patient_id: int
    baseline_date: str = Field(..., description="Baseline scan date or identifier e.g. 'week-000-2'")
    followup_date: str = Field(..., description="Follow-up scan date or identifier e.g. 'week-012'")
    interval_weeks: float = Field(..., ge=0.0, description="Elapsed interval in weeks")

    # Volumetric measurements (cm^3)
    baseline_enhancing_vol_cm3: float = Field(..., ge=0.0, description="Enhancing lesion volume at T0 (cm^3)")
    followup_enhancing_vol_cm3: float = Field(..., ge=0.0, description="Enhancing lesion volume at T1 (cm^3)")
    baseline_edema_vol_cm3: Optional[float] = Field(None, ge=0.0, description="FLAIR edema volume at T0 (cm^3)")
    followup_edema_vol_cm3: Optional[float] = Field(None, ge=0.0, description="FLAIR edema volume at T1 (cm^3)")

    # Clinical Covariates
    steroid_dose_increased: bool = Field(False, description="Did steroid (dexamethasone) dose increase?")
    is_pseudo_progression_window: bool = Field(False, description="Is follow-up within 12 weeks of chemoradiation completion?")
    clinical_notes: Optional[str] = None


class ProgressionImageEstimateOut(BaseModel):
    baseline_enhancing_vol_cm3: float
    followup_enhancing_vol_cm3: float
    delta_enhancing_vol_cm3: float
    rel_change_enhancing_vol_pct: float
    baseline_edema_vol_cm3: float
    followup_edema_vol_cm3: float
    delta_edema_vol_cm3: float
    rel_change_edema_vol_pct: float
    baseline_tumour_detected: bool
    followup_tumour_detected: bool
    baseline_predicted_class: str
    followup_predicted_class: str
    imaging_summary: str


class SingleProgressionPrediction(BaseModel):
    model_name: str
    status: str
    predicted_label: str  # "Progressive Disease" | "Stable / Response"
    progression_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_drivers: Optional[List[Dict[str, Any]]] = None


class ProgressionPredictionOut(BaseModel):
    study_id: Optional[int] = None
    patient_id: int
    baseline_date: str
    followup_date: str
    interval_weeks: float
    rel_change_enhancing_vol_pct: float
    consensus_label: str  # "Progressive Disease" | "Stable / Response"
    consensus_agreement: str  # "Full Consensus" | "Majority Agreement"
    predictions: List[SingleProgressionPrediction]
    clinical_disclaimer: str = (
        "NOTICE: Tumor progression prediction is provided for clinical decision support under doctor-in-the-loop oversight. "
        "Must be correlated with neuroradiological RANO evaluation, corticosteroid stability, and neurological exam."
    )


class ProgressionEvalReportOut(BaseModel):
    status: str
    dataset: str
    held_out_patients_count: int
    held_out_transitions_count: int
    test_class_distribution: Dict[str, int]
    models: Dict[str, Any]
    top_predictive_features: List[Dict[str, Any]]


class ProgressionHistoryItem(BaseModel):
    id: int
    patient_id: int
    baseline_date: str
    followup_date: str
    interval_weeks: float
    baseline_enhancing_vol_cm3: Optional[float]
    followup_enhancing_vol_cm3: Optional[float]
    rel_change_enhancing_vol: Optional[float]
    consensus_label: Optional[str]
    xgb_progression_probability: Optional[float]
    svm_progression_probability: Optional[float]
    dl_progression_probability: Optional[float]
    confirmed_rano_category: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ConfirmRanoInput(BaseModel):
    confirmed_rano_category: str = Field(..., pattern="^(PD|SD|PR|CR)$")
    clinical_notes: Optional[str] = None
