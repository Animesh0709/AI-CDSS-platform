"""
FastAPI router — Early Warning System endpoints.

Endpoints
---------
POST   /early-warning/vitals                          – Record new vital signs
GET    /early-warning/patients/{id}/vitals            – Time-series vitals for patient
POST   /early-warning/patients/{id}/predict           – ML + NEWS2 risk assessment
GET    /early-warning/patients/{id}/history           – EWS assessment history
GET    /early-warning/alerts                          – High-risk alert queue
POST   /early-warning/assessments/{id}/acknowledge   – Doctor acknowledgement
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.session import get_db
from app.models.early_warning import EWSAssessment, VitalSign
from app.models.patient import Patient
from app.models.user import User
from app.schemas.early_warning import (
    AcknowledgeInput,
    AlertItem,
    CSVImportReport,
    EWSHistoryItem,
    EWSPredictionOut,
    SHAPEntry,
    VitalSignCreate,
    VitalSignOut,
)
from app.services import early_warning as ews_service

router = APIRouter(prefix="/early-warning", tags=["Early Warning System"])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: verify patient belongs to current doctor
# ─────────────────────────────────────────────────────────────────────────────

_CSV_MAX_BYTES = 2 * 1024 * 1024


def _get_patient_or_404(patient_id: int, db: Session, current_user: User) -> Patient:
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id,
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or access denied.")
    return patient


def _vital_out(vs: VitalSign) -> VitalSignOut:
    recs = ews_service.compute_news2({
        "heart_rate": vs.heart_rate,
        "respiratory_rate": vs.respiratory_rate,
        "spo2": vs.spo2,
        "supplemental_o2": vs.supplemental_o2,
        "temperature_c": vs.temperature_c,
        "systolic_bp": vs.systolic_bp,
        "consciousness_level": vs.consciousness_level,
    }).get("recommendations")
    out = VitalSignOut.model_validate(vs)
    return out.model_copy(update={"news2_recommendations": recs})


def _persist_vital(payload: VitalSignCreate, current_user: User, db: Session) -> VitalSign:
    vitals_dict = payload.model_dump()
    news2_result = ews_service.compute_news2(vitals_dict)
    vs = VitalSign(
        patient_id=payload.patient_id,
        doctor_id=current_user.id,
        recorded_at=payload.recorded_at or datetime.utcnow(),
        heart_rate=payload.heart_rate,
        respiratory_rate=payload.respiratory_rate,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        spo2=payload.spo2,
        temperature_c=payload.temperature_c,
        consciousness_level=(payload.consciousness_level or "A").upper(),
        supplemental_o2=payload.supplemental_o2,
        urine_output_ml_hr=payload.urine_output_ml_hr,
        notes=payload.notes,
        news2_score=news2_result["score"],
        news2_risk_level=news2_result["risk_level"],
    )
    db.add(vs)
    db.flush()
    return vs


# ─────────────────────────────────────────────────────────────────────────────
# POST /early-warning/vitals  — Record vital signs
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/vitals",
    response_model=VitalSignOut,
    status_code=status.HTTP_201_CREATED,
    summary="Record a new vital-sign observation for a patient",
)
def record_vitals(
    payload: VitalSignCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_patient_or_404(payload.patient_id, db, current_user)
    vs = _persist_vital(payload, current_user, db)
    db.commit()
    db.refresh(vs)
    return _vital_out(vs)


# ─────────────────────────────────────────────────────────────────────────────
# GET /early-warning/patients/{id}/vitals — Time-series vitals
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patients/{patient_id}/vitals",
    response_model=List[VitalSignOut],
    summary="Retrieve time-series vital signs for a patient",
)
def get_patient_vitals(
    patient_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_patient_or_404(patient_id, db, current_user)
    vitals = (
        db.query(VitalSign)
        .filter(VitalSign.patient_id == patient_id)
        .order_by(VitalSign.recorded_at.asc())
        .limit(limit)
        .all()
    )
    return [_vital_out(v) for v in vitals]


@router.post(
    "/vitals/csv",
    response_model=CSVImportReport,
    summary="Bulk-import vital signs from a CSV file",
)
async def import_vitals_csv(
    patient_id: int = Query(..., ge=1),
    file: UploadFile = File(..., description="UTF-8 CSV of time-series vitals (max 2 MB)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_patient_or_404(patient_id, db, current_user)

    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > _CSV_MAX_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large ({len(content) / 1024:.0f} KB). Maximum allowed is 2 MB.",
        )

    parsed = ews_service.parse_vitals_csv(content)
    errors = list(parsed["errors"])
    warnings = list(parsed["warnings"])
    clinical_flags: List[str] = []
    imported: List[VitalSignOut] = []

    existing_ts = {
        v.recorded_at
        for v in db.query(VitalSign).filter(VitalSign.patient_id == patient_id).all()
    }

    for row in parsed["rows"]:
        ts = row.get("recorded_at")
        if ts is not None and ts in existing_ts:
            errors.append(f"Skipped duplicate timestamp already stored for this patient: {ts.isoformat()}")
            continue
        payload = VitalSignCreate(patient_id=patient_id, **row)
        vs = _persist_vital(payload, current_user, db)
        out = _vital_out(vs)
        imported.append(out)
        if ts is not None:
            existing_ts.add(ts)
        if (vs.news2_score or 0) >= 5:
            clinical_flags.append(
                f"{(vs.recorded_at.isoformat() if vs.recorded_at else 'row')}: "
                f"NEWS2 {vs.news2_score} ({vs.news2_risk_level}) — possible clinical abnormality, not a data-quality error."
            )

    db.commit()
    return CSVImportReport(
        imported_count=len(imported),
        skipped_count=len(errors),
        errors=errors[:50],
        warnings=warnings[:50],
        clinical_flags=clinical_flags[:50],
        imported=imported[:50],
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /early-warning/patients/{id}/predict — ML + NEWS2 risk assessment
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/patients/{patient_id}/predict",
    response_model=EWSPredictionOut,
    summary="Run ML early-warning risk prediction for a patient",
)
def predict_ews(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = _get_patient_or_404(patient_id, db, current_user)

    # Latest 12 observations (not the first 12 in history)
    vitals_rows = list(
        reversed(
            db.query(VitalSign)
            .filter(VitalSign.patient_id == patient_id)
            .order_by(VitalSign.recorded_at.desc())
            .limit(12)
            .all()
        )
    )

    if not vitals_rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No vital signs recorded for this patient. Record at least one set of vitals first.",
        )

    # Convert ORM rows to dicts for service layer
    vitals_history = [
        {
            "heart_rate": v.heart_rate,
            "respiratory_rate": v.respiratory_rate,
            "systolic_bp": v.systolic_bp,
            "diastolic_bp": v.diastolic_bp,
            "spo2": v.spo2,
            "temperature_c": v.temperature_c,
            "news2_score": v.news2_score,
            "recorded_at": v.recorded_at,
        }
        for v in vitals_rows
    ]

    # V3: ML prediction
    pred = ews_service.predict_deterioration_risk(vitals_history)

    # V4: SHAP explanations
    shap_raw = ews_service.explain_ews_prediction(
        pred["feature_vector"], pred["feature_names"]
    )

    shap_entries = [SHAPEntry(**s) for s in shap_raw]
    trigger_features = [s["display_name"] for s in shap_raw[:3]]

    # Latest NEWS2
    latest = vitals_rows[-1]
    news2_latest = ews_service.compute_news2({
        "heart_rate": latest.heart_rate,
        "respiratory_rate": latest.respiratory_rate,
        "spo2": latest.spo2,
        "supplemental_o2": latest.supplemental_o2,
        "temperature_c": latest.temperature_c,
        "systolic_bp": latest.systolic_bp,
        "consciousness_level": latest.consciousness_level,
    })

    # Persist assessment
    assessment = EWSAssessment(
        patient_id=patient_id,
        doctor_id=current_user.id,
        assessment_dt=datetime.utcnow(),
        ml_risk_score=pred["ml_risk_score"],
        ml_risk_label=pred["ml_risk_label"],
        news2_score=latest.news2_score,
        shap_values_json=json.dumps(shap_raw),
        trigger_features_json=json.dumps(trigger_features),
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return EWSPredictionOut(
        assessment_id=assessment.id,
        patient_id=patient_id,
        assessment_dt=assessment.assessment_dt,
        ml_risk_score=pred["ml_risk_score"],
        ml_risk_label=pred["ml_risk_label"],
        news2_score=latest.news2_score,
        news2_risk_level=latest.news2_risk_level,
        news2_recommendations=news2_latest.get("recommendations"),
        shap_features=shap_entries,
        trigger_features=trigger_features,
        vitals_count=len(vitals_rows),
        model_used=pred.get("model_used", "unknown"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /early-warning/patients/{id}/history — EWS assessment history
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/patients/{patient_id}/history",
    response_model=List[EWSHistoryItem],
    summary="Retrieve EWS assessment history for a patient",
)
def get_patient_ews_history(
    patient_id: int,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_patient_or_404(patient_id, db, current_user)
    assessments = (
        db.query(EWSAssessment)
        .filter(EWSAssessment.patient_id == patient_id)
        .order_by(EWSAssessment.assessment_dt.desc())
        .limit(limit)
        .all()
    )
    return assessments


# ─────────────────────────────────────────────────────────────────────────────
# GET /early-warning/alerts — Alert queue for the doctor
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/alerts",
    response_model=List[AlertItem],
    summary="Get unacknowledged high-risk EWS alerts for the logged-in doctor",
)
def get_alerts(
    include_acknowledged: bool = Query(False),
    sort_by: str = Query("risk", regex="^(risk|time)$"),
    risk_level: Optional[str] = Query(None, regex="^(Low|Medium|High|Critical)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    labels = [risk_level] if risk_level else ["High", "Critical", "Medium"]
    order_col = (
        EWSAssessment.ml_risk_score.desc()
        if sort_by == "risk"
        else EWSAssessment.assessment_dt.desc()
    )
    query = (
        db.query(EWSAssessment)
        .filter(
            EWSAssessment.doctor_id == current_user.id,
            EWSAssessment.ml_risk_label.in_(labels),
        )
        .order_by(order_col)
    )

    if not include_acknowledged:
        query = query.filter(EWSAssessment.doctor_acknowledged == False)  # noqa: E712

    assessments = query.limit(50).all()

    # Enrich with patient name
    result = []
    for a in assessments:
        patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
        if patient:
            result.append(
                AlertItem(
                    assessment_id=a.id,
                    patient_id=a.patient_id,
                    patient_name=f"{patient.first_name} {patient.last_name}",
                    assessment_dt=a.assessment_dt,
                    ml_risk_score=a.ml_risk_score or 0.0,
                    ml_risk_label=a.ml_risk_label or "Unknown",
                    news2_score=a.news2_score,
                    news2_risk_level=None,
                    doctor_acknowledged=a.doctor_acknowledged,
                )
            )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /early-warning/assessments/{id}/acknowledge — Doctor acknowledgement
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/assessments/{assessment_id}/acknowledge",
    summary="Doctor acknowledges and dismisses an EWS alert",
)
def acknowledge_alert(
    assessment_id: int,
    payload: AcknowledgeInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(EWSAssessment).filter(
        EWSAssessment.id == assessment_id,
        EWSAssessment.doctor_id == current_user.id,
    ).first()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found or access denied.")

    assessment.doctor_acknowledged = True
    assessment.acknowledged_at = datetime.utcnow()
    assessment.acknowledge_notes = payload.acknowledge_notes
    db.commit()
    db.refresh(assessment)

    return {
        "status": "acknowledged",
        "assessment_id": assessment_id,
        "acknowledged_at": assessment.acknowledged_at,
    }
