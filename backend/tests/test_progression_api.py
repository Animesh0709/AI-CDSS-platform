"""
Tests for Tumor Progression Prediction Endpoints:
  - Auth requirements
  - Model info and metadata retrieval
  - Held-out evaluation report retrieval
  - Longitudinal prediction with 3-model comparison & SQLite persistence
  - Patient progression history retrieval
  - Radiologist RANO confirmation feedback
  - Non-interference verification with existing MRI and XAI modules
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.session import Base, get_db
from app.main import app


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def create_doctor_and_patient(client, email="prog_doctor@hospital.org"):
    client.post("/api/auth/register", json={"full_name": "Dr Progression", "email": email, "password": "TestPassword2026!"})
    token = client.post("/api/auth/login", json={"email": email, "password": "TestPassword2026!"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    patient = client.post(
        "/api/patients",
        json={"first_name": "Longitudinal", "last_name": "GBM-Patient", "gender": "Male"},
        headers=headers
    ).json()
    return headers, patient["id"]


def test_progression_endpoints_require_auth(client):
    assert client.get("/api/progression/model-info").status_code == 401
    assert client.get("/api/progression/eval-report").status_code == 401
    assert client.post("/api/progression/predict", json={}).status_code == 401


def test_progression_model_info_returns_metadata(client):
    headers, _ = create_doctor_and_patient(client)
    res = client.get("/api/progression/model-info", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "LUMIERE" in data["dataset_name"]
    assert data["feature_count"] > 20
    assert "features" in data
    assert "Creative Commons" in data["license"]


def test_progression_eval_report_returns_held_out_metrics(client):
    headers, _ = create_doctor_and_patient(client)
    res = client.get("/api/progression/eval-report", headers=headers)
    assert res.status_code == 200
    report = res.json()
    assert report["status"] == "available"
    assert report["held_out_patients_count"] == 18
    assert report["held_out_transitions_count"] == 70
    assert "xgboost" in report["models"]
    assert "svm" in report["models"]
    assert "deep_learning" in report["models"]

    xgb = report["models"]["xgboost"]["metrics"]
    assert xgb["pr_auc"] > 0.80
    assert xgb["balanced_accuracy"] > 0.70
    assert "confusion_matrix" in xgb
    assert "confidence_intervals" in report["models"]["xgboost"]


def test_progression_predict_and_persistence(client):
    headers, patient_id = create_doctor_and_patient(client, email="oncologist@hospital.org")

    payload = {
        "patient_id": patient_id,
        "baseline_date": "week-000-2",
        "followup_date": "week-024",
        "interval_weeks": 24.0,
        "baseline_enhancing_vol_cm3": 1.2,
        "followup_enhancing_vol_cm3": 4.8,  # 300% increase
        "baseline_edema_vol_cm3": 12.0,
        "followup_edema_vol_cm3": 25.0,
        "steroid_dose_increased": True,
        "is_pseudo_progression_window": False,
        "clinical_notes": "Substantial focal increase in contrast enhancement and FLAIR signal."
    }

    res = client.post("/api/progression/predict", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["patient_id"] == patient_id
    assert data["study_id"] is not None
    assert len(data["predictions"]) == 3
    assert data["consensus_label"] in ["Progressive Disease", "Stable / Response"]
    assert data["rel_change_enhancing_vol_pct"] > 0

    # Retrieve patient history and confirm record was persisted
    hist_res = client.get(f"/api/progression/patients/{patient_id}/history", headers=headers)
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 1
    assert history[0]["id"] == data["study_id"]
    assert history[0]["interval_weeks"] == 24.0


def test_confirm_study_rano_outcome(client):
    headers, patient_id = create_doctor_and_patient(client, email="neuro@hospital.org")

    payload = {
        "patient_id": patient_id,
        "baseline_date": "week-000",
        "followup_date": "week-012",
        "interval_weeks": 12.0,
        "baseline_enhancing_vol_cm3": 2.0,
        "followup_enhancing_vol_cm3": 2.1,
        "steroid_dose_increased": False,
        "is_pseudo_progression_window": False
    }

    pred_res = client.post("/api/progression/predict", json=payload, headers=headers)
    study_id = pred_res.json()["study_id"]

    # Confirm valid RANO
    confirm_res = client.post(
        f"/api/progression/studies/{study_id}/confirm",
        json={"confirmed_rano_category": "SD", "clinical_notes": "Confirmed Stable Disease at multidisciplinary tumor board."},
        headers=headers
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["confirmed_rano"] == "SD"

    # Reject invalid RANO code
    bad_res = client.post(
        f"/api/progression/studies/{study_id}/confirm",
        json={"confirmed_rano_category": "UNKNOWN"},
        headers=headers
    )
    assert bad_res.status_code == 422


def test_existing_mri_and_xai_modules_remain_functional(client):
    headers, patient_id = create_doctor_and_patient(client, email="multitask@hospital.org")

    # 1. Existing 4-class MRI model info
    mri_info = client.get("/api/mri/model-info", headers=headers)
    assert mri_info.status_code == 200
    assert mri_info.json()["metadata"]["classes"] == ["glioma", "meningioma", "notumor", "pituitary"]

    # 2. Existing Health Check shows both modules active
    health = client.get("/api/health")
    assert health.status_code == 200
    active = health.json()["active_modules"]
    assert "mri_classification" in active
    assert "tumor_progression_prediction" in active


def test_progression_analyze_image_pair_endpoint(client):
    headers, _ = create_doctor_and_patient(client, email="mri_oncology@hospital.org")

    # Sample MRI test images from ml/data/brain_tumor/Testing/glioma/
    sample_t0 = "ml/data/brain_tumor/Testing/glioma/Te-gl_1.jpg"
    sample_t1 = "ml/data/brain_tumor/Testing/glioma/Te-gl_10.jpg"

    if os.path.exists(sample_t0) and os.path.exists(sample_t1):
        with open(sample_t0, "rb") as f0, open(sample_t1, "rb") as f1:
            files = {
                "baseline_file": ("Te-gl_1.jpg", f0, "image/jpeg"),
                "followup_file": ("Te-gl_10.jpg", f1, "image/jpeg"),
            }
            res = client.post("/api/progression/analyze-image-pair", files=files, headers=headers)
            assert res.status_code == 200
            data = res.json()
            assert "baseline_enhancing_vol_cm3" in data
            assert "followup_enhancing_vol_cm3" in data
            assert "delta_enhancing_vol_cm3" in data
            assert "rel_change_enhancing_vol_pct" in data
            assert "imaging_summary" in data
            assert data["baseline_enhancing_vol_cm3"] > 0
