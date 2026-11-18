"""
SQLAlchemy Model for Longitudinal Glioblastoma Tumor Progression Studies.
=========================================================================

Maintains independent longitudinal comparison records (T_baseline -> T_followup)
without modifying existing Patient, Report, or LabResult tables.
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.database.session import Base


class ProgressionStudy(Base):
    __tablename__ = "progression_studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Longitudinal examination dates & interval
    baseline_date = Column(String(64), nullable=False)
    followup_date = Column(String(64), nullable=False)
    interval_weeks = Column(Float, nullable=False, default=0.0)

    # Volumetric measurements (cm^3)
    baseline_enhancing_vol_cm3 = Column(Float, nullable=True)
    followup_enhancing_vol_cm3 = Column(Float, nullable=True)
    delta_enhancing_vol_cm3 = Column(Float, nullable=True)
    rel_change_enhancing_vol = Column(Float, nullable=True)

    baseline_edema_vol_cm3 = Column(Float, nullable=True)
    followup_edema_vol_cm3 = Column(Float, nullable=True)
    delta_edema_vol_cm3 = Column(Float, nullable=True)

    # Clinical covariates
    steroid_dose_increased = Column(Boolean, default=False)
    is_pseudo_progression_window = Column(Boolean, default=False)
    clinical_notes = Column(Text, nullable=True)

    # 3-Model Predictions
    xgb_progression_probability = Column(Float, nullable=True)
    xgb_predicted_label = Column(String(64), nullable=True)

    svm_progression_probability = Column(Float, nullable=True)
    svm_predicted_label = Column(String(64), nullable=True)

    dl_progression_probability = Column(Float, nullable=True)
    dl_predicted_label = Column(String(64), nullable=True)

    # Consensus & Confidence
    consensus_label = Column(String(64), nullable=True)  # "Progressive Disease" vs "Stable / Response"
    consensus_agreement = Column(String(32), nullable=True)  # "Full Consensus", "Majority"

    # Confirmed expert evaluation (ground truth feedback loop)
    confirmed_rano_category = Column(String(32), nullable=True)  # "PD", "SD", "PR", "CR"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to Patient
    patient = relationship("Patient", backref="progression_studies")
    doctor = relationship("User")
