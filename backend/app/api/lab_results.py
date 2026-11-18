from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.session import get_db
from app.models.lab_result import LabResult
from app.models.patient import Patient
from app.schemas.lab_result import LabResultCreate, LabResultOut
from app.services.lab_rules import analyse_lab_value

router = APIRouter(prefix="/patients/{patient_id}/lab-results", tags=["Laboratory Analysis"])


def _patient(patient_id: int, db: Session, doctor_id: int) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return patient


@router.get("", response_model=list[LabResultOut])
def list_lab_results(patient_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    _patient(patient_id, db, current_user.id)
    return db.query(LabResult).filter(LabResult.patient_id == patient_id).order_by(LabResult.created_at.desc()).all()


@router.post("", response_model=LabResultOut, status_code=status.HTTP_201_CREATED)
def record_and_analyse_lab_result(patient_id: int, payload: LabResultCreate, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    _patient(patient_id, db, current_user.id)
    try:
        analysis = analyse_lab_value(payload.test_name, payload.value, payload.unit)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
    result = LabResult(patient_id=patient_id, value=payload.value, test_date=payload.test_date, **analysis)
    db.add(result)
    db.commit()
    db.refresh(result)
    return result
