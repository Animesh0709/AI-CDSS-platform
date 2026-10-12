from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database.session import Base


class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    test_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(40), nullable=False)
    reference_range = Column(String(120), nullable=False)
    status = Column(String(30), nullable=False)
    clinical_basis = Column(String(255), nullable=False)
    test_date = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="lab_results")
