from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LabResultCreate(BaseModel):
    test_name: str = Field(min_length=1, max_length=100)
    value: float
    unit: Optional[str] = Field(default=None, max_length=40)
    test_date: Optional[str] = Field(default=None, max_length=50)


class LabResultOut(BaseModel):
    id: int
    patient_id: int
    test_name: str
    value: float
    unit: str
    reference_range: str
    status: str
    clinical_basis: str
    test_date: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
