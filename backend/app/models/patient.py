from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Date
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.session import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(String(50), nullable=True) # Stored as YYYY-MM-DD string or Date
    gender = Column(String(20), nullable=False) # Male, Female, Other
    phone = Column(String(30), nullable=True)
    email = Column(String(100), nullable=True)
    address = Column(String(255), nullable=True)
    blood_group = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    doctor = relationship("User", back_populates="patients")
    medical_history = relationship("MedicalHistory", back_populates="patient", cascade="all, delete-orphan", order_by="desc(MedicalHistory.created_at)")
    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan", order_by="desc(Report.uploaded_at)")
    lab_results = relationship("LabResult", back_populates="patient", cascade="all, delete-orphan", order_by="desc(LabResult.created_at)")

class MedicalHistory(Base):
    __tablename__ = "medical_history"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    condition = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    diagnosis_date = Column(String(50), nullable=True)
    status = Column(String(50), default="Active") # Active, Chronic, In Remission, Resolved
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="medical_history")
