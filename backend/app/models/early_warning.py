"""
SQLAlchemy models for the Early-Warning System (EWS).

Tables
------
vital_signs      – Time-series vital-sign recordings per patient.
ews_assessments  – ML risk assessments derived from a batch of vital signs.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database.session import Base


class VitalSign(Base):
    """One snapshot of a patient's physiological measurements."""

    __tablename__ = "vital_signs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Core NEWS2 parameters
    heart_rate = Column(Float, nullable=True)            # beats per minute
    respiratory_rate = Column(Float, nullable=True)      # breaths per minute
    systolic_bp = Column(Float, nullable=True)           # mmHg
    diastolic_bp = Column(Float, nullable=True)          # mmHg
    spo2 = Column(Float, nullable=True)                  # % oxygen saturation
    temperature_c = Column(Float, nullable=True)         # degrees Celsius
    consciousness_level = Column(String(10), nullable=True)  # A / V / P / U  (AVPU)
    supplemental_o2 = Column(Boolean, default=False)     # on supplemental O2?
    urine_output_ml_hr = Column(Float, nullable=True)    # mL / hour

    # Auto-computed on insert
    news2_score = Column(Integer, nullable=True)
    news2_risk_level = Column(String(20), nullable=True)  # Low / Medium / High / Critical

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="vital_signs")
    doctor = relationship("User")


class EWSAssessment(Base):
    """ML-based early-warning assessment for a patient at a point in time."""

    __tablename__ = "ews_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    assessment_dt = Column(DateTime, default=datetime.utcnow, nullable=False)

    # ML outputs
    ml_risk_score = Column(Float, nullable=True)         # 0.0 – 1.0
    ml_risk_label = Column(String(20), nullable=True)    # Low / Medium / High / Critical
    news2_score = Column(Integer, nullable=True)         # latest NEWS2 at time of assessment

    # JSON blobs
    shap_values_json = Column(Text, nullable=True)       # [{feature, value, shap}]
    trigger_features_json = Column(Text, nullable=True)  # top-3 feature names

    # Doctor acknowledgement
    doctor_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledge_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="ews_assessments")
    doctor = relationship("User")
