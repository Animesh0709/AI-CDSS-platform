from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ReportOut(BaseModel):
    id: int
    patient_id: int
    filename: str
    file_path: str
    report_type: Optional[str]
    extracted_text: Optional[str]
    clinical_entities: Optional[dict[str, list[dict[str, Any]]]] = None
    uploaded_at: Optional[datetime]

    class Config:
        from_attributes = True


class ClinicalEntityApproval(BaseModel):
    condition: str
    diagnosis_date: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = "Active"


class ClinicalEntitiesUpdate(BaseModel):
    conditions: list[dict[str, Any]] = []
    medications: list[dict[str, Any]] = []
    lab_values: list[dict[str, Any]] = []
    clinical_events: list[dict[str, Any]] = []
    dates: list[dict[str, Any]] = []
