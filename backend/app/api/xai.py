from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.session import get_db
from app.models.patient import Patient
from app.schemas.xai import PredictionInput, PredictionOut
from app.services.xai import global_explanation, local_explanation

router = APIRouter(prefix="/xai", tags=["Explainable AI"])


@router.post("/patients/{patient_id}/predict", response_model=PredictionOut)
def predict(patient_id: int, payload: PredictionInput, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_user.id).first()
    if not patient: raise HTTPException(status_code=404, detail="Patient record not found")
    try: return local_explanation(payload.model_dump(), patient)
    except FileNotFoundError: raise HTTPException(status_code=503, detail="The trained XGBoost model is unavailable. Run ml/scripts/train.py first.")


@router.get("/global")
def global_xai(current_user=Depends(get_current_user)):
    try: return global_explanation()
    except FileNotFoundError: raise HTTPException(status_code=503, detail="The trained XGBoost model is unavailable. Run ml/scripts/train.py first.")
