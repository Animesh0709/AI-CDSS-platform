from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc
from typing import List, Optional
from app.database.session import get_db
from app.models.user import User
from app.models.patient import Patient, MedicalHistory
from app.schemas.patient import (
    PatientCreate, PatientUpdate, PatientOut, PatientListOut,
    MedicalHistoryCreate, MedicalHistoryOut
)
from app.api.auth import get_current_user

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.get("/stats/summary")
def get_doctor_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    total_patients = db.query(Patient).filter(Patient.doctor_id == current_user.id).count()
    
    # Total chronic/active conditions tracked
    patient_ids = [p.id for p in db.query(Patient.id).filter(Patient.doctor_id == current_user.id).all()]
    total_conditions = 0
    if patient_ids:
        total_conditions = db.query(MedicalHistory).filter(MedicalHistory.patient_id.in_(patient_ids)).count()
    
    # Recent patients (up to 5)
    recent_patients = (
        db.query(Patient)
        .filter(Patient.doctor_id == current_user.id)
        .order_by(desc(Patient.created_at))
        .limit(5)
        .all()
    )
    
    recent_list = []
    for p in recent_patients:
        recent_list.append({
            "id": p.id,
            "full_name": f"{p.first_name} {p.last_name}",
            "gender": p.gender,
            "date_of_birth": p.date_of_birth,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "conditions_count": len(p.medical_history)
        })
    
    return {
        "total_patients": total_patients,
        "total_conditions": total_conditions,
        "recent_patients": recent_list,
        "system_status": "Operational (Doctor-in-the-loop)",
        "modules": {
            "auth": "Active",
            "patient_management": "Active",
            "xgboost_risk": "Ready (Week 1 Baseline)",
            "ocr_nlp": "Ready for Week 2-3",
            "rag_assistant": "Ready for Week 5-7",
            "xai_shap": "Ready for Week 10"
        }
    }

@router.get("", response_model=List[PatientListOut])
def list_patients(
    search: Optional[str] = Query(None, description="Search by name, phone or email"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Patient).filter(Patient.doctor_id == current_user.id)
    
    if search:
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Patient.first_name.ilike(search_pattern),
                Patient.last_name.ilike(search_pattern),
                Patient.phone.ilike(search_pattern),
                Patient.email.ilike(search_pattern)
            )
        )
    
    patients = query.order_by(desc(Patient.created_at)).offset(skip).limit(limit).all()
    
    result = []
    for p in patients:
        result.append(
            PatientListOut(
                id=p.id,
                doctor_id=p.doctor_id,
                first_name=p.first_name,
                last_name=p.last_name,
                date_of_birth=p.date_of_birth,
                gender=p.gender,
                phone=p.phone,
                email=p.email,
                blood_group=p.blood_group,
                created_at=p.created_at,
                history_count=len(p.medical_history)
            )
        )
    return result

@router.post("", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_patient = Patient(
        doctor_id=current_user.id,
        first_name=patient_in.first_name.strip(),
        last_name=patient_in.last_name.strip(),
        date_of_birth=patient_in.date_of_birth,
        gender=patient_in.gender,
        phone=patient_in.phone,
        email=patient_in.email,
        address=patient_in.address,
        blood_group=patient_in.blood_group
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    
    # If initial condition provided, record it in medical history
    if patient_in.initial_condition and patient_in.initial_condition.strip():
        history = MedicalHistory(
            patient_id=new_patient.id,
            condition=patient_in.initial_condition.strip(),
            description=patient_in.initial_condition_desc,
            status="Active"
        )
        db.add(history)
        db.commit()
        db.refresh(new_patient)
        
    return new_patient

@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
    return patient

@router.put("/{patient_id}", response_model=PatientOut)
def update_patient(
    patient_id: int,
    patient_update: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
        
    update_data = patient_update.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        if val is not None:
            setattr(patient, field, val)
            
    db.commit()
    db.refresh(patient)
    return patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
        
    db.delete(patient)
    db.commit()
    return None

@router.post("/{patient_id}/history", response_model=MedicalHistoryOut, status_code=status.HTTP_201_CREATED)
def add_patient_medical_history(
    patient_id: int,
    history_in: MedicalHistoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
        
    history = MedicalHistory(
        patient_id=patient.id,
        condition=history_in.condition.strip(),
        description=history_in.description,
        diagnosis_date=history_in.diagnosis_date,
        status=history_in.status or "Active"
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history
