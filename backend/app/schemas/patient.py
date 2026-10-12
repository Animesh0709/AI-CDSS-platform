from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class MedicalHistoryCreate(BaseModel):
    condition: str
    description: Optional[str] = None
    diagnosis_date: Optional[str] = None
    status: Optional[str] = "Active"

class MedicalHistoryOut(BaseModel):
    id: int
    patient_id: int
    condition: str
    description: Optional[str] = None
    diagnosis_date: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class PatientBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    blood_group: Optional[str] = None

class PatientCreate(PatientBase):
    initial_condition: Optional[str] = None
    initial_condition_desc: Optional[str] = None

class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    blood_group: Optional[str] = None

class PatientOut(PatientBase):
    id: int
    doctor_id: int
    created_at: datetime
    updated_at: datetime
    medical_history: List[MedicalHistoryOut] = []

    class Config:
        from_attributes = True

class PatientListOut(BaseModel):
    id: int
    doctor_id: int
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    blood_group: Optional[str] = None
    created_at: datetime
    history_count: int = 0

    class Config:
        from_attributes = True
