"""
Pydantic schemas for Early-Warning System endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Vital Signs
# ─────────────────────────────────────────────

class VitalSignCreate(BaseModel):
    patient_id: int
    recorded_at: Optional[datetime] = None  # defaults to now on the server
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="bpm")
    respiratory_rate: Optional[float] = Field(None, ge=0, le=100, description="breaths/min")
    systolic_bp: Optional[float] = Field(None, ge=0, le=300, description="mmHg")
    diastolic_bp: Optional[float] = Field(None, ge=0, le=200, description="mmHg")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="% SpO2")
    temperature_c: Optional[float] = Field(None, ge=25.0, le=45.0, description="°C")
    consciousness_level: Optional[str] = Field(None, description="A/V/P/U (AVPU scale)")
    supplemental_o2: bool = False
    urine_output_ml_hr: Optional[float] = Field(None, ge=0, description="mL/hr")
    notes: Optional[str] = None


class VitalSignOut(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    recorded_at: datetime
    heart_rate: Optional[float]
    respiratory_rate: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    spo2: Optional[float]
    temperature_c: Optional[float]
    consciousness_level: Optional[str]
    supplemental_o2: bool
    urine_output_ml_hr: Optional[float]
    news2_score: Optional[int]
    news2_risk_level: Optional[str]
    news2_recommendations: Optional[str] = None
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────
# EWS Prediction
# ─────────────────────────────────────────────

class SHAPEntry(BaseModel):
    feature: str
    display_name: str
    value: float
    shap_value: float
    direction: str  # "↑ Risk" | "↓ Risk"


class EWSPredictionOut(BaseModel):
    assessment_id: int
    patient_id: int
    assessment_dt: datetime
    ml_risk_score: float
    ml_risk_label: str
    news2_score: Optional[int]
    news2_risk_level: Optional[str]
    news2_recommendations: Optional[str] = None
    shap_features: List[SHAPEntry]
    trigger_features: List[str]
    vitals_count: int
    model_used: str = "unknown"
    prediction_target: str = "Deterioration risk (NEWS2-labelled research prototype)"
    prediction_horizon: str = "Next clinical reassessment window (not a timed sepsis endpoint)"
    disclaimer: str = (
        "Research prototype — for clinical decision support only. "
        "Not a substitute for clinical judgement."
    )

    model_config = {"from_attributes": True}


class CSVImportReport(BaseModel):
    imported_count: int
    skipped_count: int
    errors: List[str]
    warnings: List[str]
    clinical_flags: List[str]
    imported: List[VitalSignOut]


# ─────────────────────────────────────────────
# EWS History
# ─────────────────────────────────────────────

class EWSHistoryItem(BaseModel):
    id: int
    patient_id: int
    assessment_dt: datetime
    ml_risk_score: Optional[float]
    ml_risk_label: Optional[str]
    news2_score: Optional[int]
    doctor_acknowledged: bool
    acknowledged_at: Optional[datetime]
    acknowledge_notes: Optional[str]

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────
# Alert Queue
# ─────────────────────────────────────────────

class AlertItem(BaseModel):
    assessment_id: int
    patient_id: int
    patient_name: str
    assessment_dt: datetime
    ml_risk_score: float
    ml_risk_label: str
    news2_score: Optional[int]
    news2_risk_level: Optional[str]
    doctor_acknowledged: bool

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────
# Acknowledge
# ─────────────────────────────────────────────

class AcknowledgeInput(BaseModel):
    acknowledge_notes: Optional[str] = None
