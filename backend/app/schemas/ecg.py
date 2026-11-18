from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class ECGRecordBase(BaseModel):
    patient_id: int
    original_filename: str
    file_size_bytes: int
    mime_type: str
    processing_status: str
    analysis_status: str
    error_details: Optional[str] = None

class ECGRecordOut(ECGRecordBase):
    id: int
    doctor_id: int
    stored_filename: str
    file_path: str
    uploaded_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ECGUploadResponse(BaseModel):
    message: str
    record: ECGRecordOut
