"""
FastAPI router for 12-Lead ECG Analysis endpoints.

Phase 1 Scope:
- Validated ECG image uploads (PNG/JPEG)
- Safe storage in dedicated ECG upload directory
- Patient and doctor association with authorization checks
- Record retrieval by ID
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.session import get_db
from app.models.ecg import ECGRecord
from app.models.patient import Patient
from app.models.user import User
from app.schemas.ecg import ECGRecordOut, ECGUploadResponse
from app.schemas.ecg_explainability import ECGExplainabilityRequest, ECGExplainabilityResponse
from app.services.ecg.explainability_service import ECGExplainabilityService
from app.services.ecg.validation import save_ecg_file, validate_ecg_image

logger = logging.getLogger("ai_cdss.ecg")

router = APIRouter(prefix="/ecg", tags=["12-Lead ECG Analysis"])


def _verify_patient_access(patient_id: int, current_user: User, db: Session) -> Patient:
    """Ensure patient exists and belongs to the authenticated doctor."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )
    if patient.doctor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have authorization to access or modify records for this patient.",
        )
    return patient


@router.post(
    "/upload",
    response_model=ECGUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an ECG scan image",
    description="Validates and stores an ECG scan (PNG or JPEG, max 10MB) for a patient.",
)
async def upload_ecg_image(
    patient_id: int = Form(..., description="ID of the patient to associate the ECG recording with"),
    file: UploadFile = File(..., description="12-lead ECG image file (PNG or JPEG)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Authorize patient access
    _verify_patient_access(patient_id, current_user, db)

    # 2. Read bytes
    file_bytes = await file.read()
    raw_filename = file.filename or "uploaded_ecg.png"

    # 3. Validate image integrity, format, and size
    detected_format, dimensions = validate_ecg_image(
        file_bytes=file_bytes,
        filename=raw_filename,
        content_type=file.content_type,
    )

    # 4. Save file to dedicated ecg directory
    stored_filename, file_path = save_ecg_file(file_bytes, raw_filename)

    # 5. Persist database record
    ecg_record = ECGRecord(
        patient_id=patient_id,
        doctor_id=current_user.id,
        original_filename=raw_filename,
        stored_filename=stored_filename,
        file_path=file_path,
        file_size_bytes=len(file_bytes),
        mime_type=f"image/{detected_format.lower()}",
        processing_status="uploaded",
        analysis_status="pending",
        error_details=None,
    )
    db.add(ecg_record)
    db.commit()
    db.refresh(ecg_record)

    logger.info(
        "ECG image uploaded successfully: id=%s, patient_id=%s, size=%s, dims=%s",
        ecg_record.id, patient_id, len(file_bytes), dimensions,
    )

    return ECGUploadResponse(
        message="ECG image uploaded and validated successfully.",
        record=ECGRecordOut.model_validate(ecg_record),
    )


@router.get(
    "/{ecg_id}",
    response_model=ECGRecordOut,
    summary="Get ECG record by ID",
    description="Retrieve metadata and processing status for an ECG recording.",
)
def get_ecg_record(
    ecg_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = db.query(ECGRecord).filter(ECGRecord.id == ecg_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ECG record with ID {ecg_id} not found.",
        )

    # Check authorization via patient or doctor
    if record.doctor_id != current_user.id:
        patient = db.query(Patient).filter(Patient.id == record.patient_id).first()
        if not patient or patient.doctor_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have authorization to view this ECG record.",
            )

    return record


@router.get(
    "/patient/{patient_id}",
    response_model=List[ECGRecordOut],
    summary="List ECG records for a patient",
    description="Retrieve all ECG recordings associated with a patient in reverse chronological order.",
)
def list_patient_ecg_records(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify_patient_access(patient_id, current_user, db)
    records = (
        db.query(ECGRecord)
        .filter(ECGRecord.patient_id == patient_id)
        .order_by(ECGRecord.uploaded_at.desc())
        .all()
    )
    return records


@router.get(
    "/{ecg_id}/image",
    summary="Get raw ECG scan image file",
    description="Streams the original verified ECG scan image file for authenticated authorized doctors.",
)
def get_ecg_image(
    ecg_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = db.query(ECGRecord).filter(ECGRecord.id == ecg_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ECG record with ID {ecg_id} not found.",
        )

    if record.doctor_id != current_user.id:
        patient = db.query(Patient).filter(Patient.id == record.patient_id).first()
        if not patient or patient.doctor_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have authorization to view this ECG scan image.",
            )

    if not os.path.exists(record.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source ECG scan file not found on server storage.",
        )

    return FileResponse(
        path=record.file_path,
        media_type=record.mime_type or "image/png",
        filename=record.original_filename,
    )


@router.post(
    "/{ecg_id}/explainability",
    response_model=ECGExplainabilityResponse,
    summary="Generate ECG feature attribution and image alignment",
    description=(
        "Generates waveform-level Integrated Gradients lead attributions and dynamic "
        "lead-coordinate image alignment for an authorized ECG record. Results are confidence-gated."
    ),
)
def get_ecg_explainability(
    ecg_id: int,
    request: Optional[ECGExplainabilityRequest] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Authorize ECG Record
    record = db.query(ECGRecord).filter(ECGRecord.id == ecg_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ECG record with ID {ecg_id} not found.",
        )

    # Patient & doctor access check
    if record.doctor_id != current_user.id:
        patient = db.query(Patient).filter(Patient.id == record.patient_id).first()
        if not patient or patient.doctor_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have authorization to access explainability for this ECG record.",
            )

    # 2. Orchestrate explainability pipeline
    target_class = request.target_class if request else None
    steps = request.integration_steps if request and request.integration_steps else 50

    service = ECGExplainabilityService()
    try:
        response = service.generate_explainability(
            ecg_record=record,
            target_class=target_class,
            integration_steps=steps,
        )
        return response
    except FileNotFoundError as fnf:
        logger.error("ECG explainability file error for record %s: %s", ecg_id, fnf)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except (ValueError, IOError) as val_err:
        logger.warning("ECG explainability validation failure for record %s: %s", ecg_id, val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as ex:
        logger.exception("Unexpected error generating ECG explainability for record %s: %s", ecg_id, ex)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating explainability alignment for this record.",
        )

