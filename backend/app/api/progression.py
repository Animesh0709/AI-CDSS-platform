"""
FastAPI Router for Tumor Progression Prediction Endpoints.
==========================================================

Endpoints:
  POST /progression/predict                — Multi-model progression prediction & record persistence
  GET  /progression/eval-report            — Held-out test evaluation report & bootstrap CIs
  GET  /progression/model-info             — Dataset provenance & model metadata
  GET  /progression/patients/{id}/history  — Chronological progression history for a patient
  POST /progression/studies/{id}/confirm   — Radiologist confirmation of RANO ground truth
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.session import get_db
from app.models.patient import Patient
from app.models.progression import ProgressionStudy
from app.models.user import User
from app.schemas.progression import (
    ConfirmRanoInput,
    ProgressionEvalReportOut,
    ProgressionHistoryItem,
    ProgressionImageEstimateOut,
    ProgressionPredictionInput,
    ProgressionPredictionOut,
    SingleProgressionPrediction,
)
from app.services import progression as progression_service

router = APIRouter(prefix="/progression", tags=["Tumor Progression Prediction"])


@router.post(
    "/predict",
    response_model=ProgressionPredictionOut,
    summary="Predict tumor progression from longitudinal MRI pair and save study",
)
def predict_progression(
    payload: ProgressionPredictionInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.query(Patient).filter(
        Patient.id == payload.patient_id,
        Patient.doctor_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {payload.patient_id} not found or access denied."
        )

    # Gather patient clinical context
    history_conditions = " ".join([h.condition for h in (patient.medical_history or [])])
    patient_context = {
        "gender": patient.gender,
        "medical_history_text": history_conditions
    }

    try:
        res = progression_service.predict_progression_transition(payload.model_dump(), patient_context)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}"
        )

    # Persist study in SQLite
    study = ProgressionStudy(
        patient_id=patient.id,
        doctor_id=current_user.id,
        baseline_date=payload.baseline_date,
        followup_date=payload.followup_date,
        interval_weeks=payload.interval_weeks,
        baseline_enhancing_vol_cm3=payload.baseline_enhancing_vol_cm3,
        followup_enhancing_vol_cm3=payload.followup_enhancing_vol_cm3,
        delta_enhancing_vol_cm3=payload.followup_enhancing_vol_cm3 - payload.baseline_enhancing_vol_cm3,
        rel_change_enhancing_vol=res["rel_change_enhancing_vol_pct"] / 100.0,
        baseline_edema_vol_cm3=payload.baseline_edema_vol_cm3,
        followup_edema_vol_cm3=payload.followup_edema_vol_cm3,
        steroid_dose_increased=payload.steroid_dose_increased,
        is_pseudo_progression_window=payload.is_pseudo_progression_window,
        clinical_notes=payload.clinical_notes,
        xgb_progression_probability=res["xgb_prob"],
        xgb_predicted_label=res["xgb_label"],
        svm_progression_probability=res["svm_prob"],
        svm_predicted_label=res["svm_label"],
        dl_progression_probability=res["dl_prob"],
        dl_predicted_label=res["dl_label"],
        consensus_label=res["consensus_label"],
        consensus_agreement=res["consensus_agreement"],
    )
    db.add(study)
    db.commit()
    db.refresh(study)

    predictions = [SingleProgressionPrediction(**p) for p in res["predictions"]]

    return ProgressionPredictionOut(
        study_id=study.id,
        patient_id=patient.id,
        baseline_date=payload.baseline_date,
        followup_date=payload.followup_date,
        interval_weeks=payload.interval_weeks,
        rel_change_enhancing_vol_pct=res["rel_change_enhancing_vol_pct"],
        consensus_label=res["consensus_label"],
        consensus_agreement=res["consensus_agreement"],
        predictions=predictions,
    )


_ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff", "image/tif"}


@router.post(
    "/analyze-image-pair",
    response_model=ProgressionImageEstimateOut,
    summary="Extract volumetric measurements and detect tumours directly from an MRI scan image pair",
)
async def analyze_image_pair(
    baseline_file: UploadFile = File(..., description="Baseline MRI scan image (T0)"),
    followup_file: UploadFile = File(..., description="Follow-up MRI scan image (T1)"),
    current_user: User = Depends(get_current_user),
):
    for f in (baseline_file, followup_file):
        ct = (f.content_type or "").lower()
        if ct not in _ALLOWED_MIME:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ct}'. Upload JPEG, PNG, BMP, or TIFF images."
            )

    baseline_bytes = await baseline_file.read()
    followup_bytes = await followup_file.read()

    try:
        estimates = progression_service.extract_volumetric_estimates_from_image_pair(
            baseline_bytes, followup_bytes
        )
        return ProgressionImageEstimateOut(**estimates)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image pair analysis error: {exc}"
        )


@router.get(
    "/eval-report",
    response_model=ProgressionEvalReportOut,
    summary="Progression model held-out evaluation report",
)
def get_progression_eval_report(current_user: User = Depends(get_current_user)):
    report = progression_service.load_eval_report()
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Progression evaluation report unavailable. Run ml/scripts/train_progression.py first."
        )
    return ProgressionEvalReportOut(**report)


@router.get(
    "/model-info",
    summary="Progression model metadata and provenance",
)
def get_progression_model_info(current_user: User = Depends(get_current_user)):
    return progression_service.load_metadata()


@router.get(
    "/patients/{patient_id}/history",
    response_model=List[ProgressionHistoryItem],
    summary="Retrieve progression studies history for a patient",
)
def get_patient_progression_history(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or unauthorized.")

    studies = (
        db.query(ProgressionStudy)
        .filter(ProgressionStudy.patient_id == patient_id)
        .order_by(ProgressionStudy.created_at.desc())
        .all()
    )
    return studies


@router.post(
    "/studies/{study_id}/confirm",
    summary="Doctor confirmation of RANO outcome category",
)
def confirm_study_rano(
    study_id: int,
    payload: ConfirmRanoInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study = db.query(ProgressionStudy).filter(
        ProgressionStudy.id == study_id,
        ProgressionStudy.doctor_id == current_user.id
    ).first()

    if not study:
        raise HTTPException(status_code=404, detail="Study not found or unauthorized.")

    study.confirmed_rano_category = payload.confirmed_rano_category
    if payload.clinical_notes:
        study.clinical_notes = (study.clinical_notes or "") + f"\n[RANO Confirmation]: {payload.clinical_notes}"

    db.commit()
    db.refresh(study)
    return {"status": "confirmed", "study_id": study.id, "confirmed_rano": study.confirmed_rano_category}
