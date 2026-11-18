"""Tests for Early Warning System endpoints left unfinished by the prior agent."""
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
from app.services.early_warning import compute_news2, parse_vitals_csv


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


def doctor_and_patient(client, email="ews_doctor@hospital.org"):
    client.post("/api/auth/register", json={"full_name": "Dr EWS", "email": email, "password": "TestPassword2026!"})
    token = client.post("/api/auth/login", json={"email": email, "password": "TestPassword2026!"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    patient = client.post(
        "/api/patients",
        json={"first_name": "Early", "last_name": "Warning", "gender": "Female"},
        headers=headers,
    ).json()
    return headers, patient["id"]


def test_news2_normal_vitals_are_low_risk():
    result = compute_news2({
        "heart_rate": 78,
        "respiratory_rate": 16,
        "spo2": 98,
        "temperature_c": 37.0,
        "systolic_bp": 120,
        "consciousness_level": "A",
        "supplemental_o2": False,
    })
    assert result["score"] == 0
    assert result["risk_level"] == "Low"
    assert "Routine monitoring" in result["recommendations"]


def test_news2_critical_vitals_are_not_low():
    result = compute_news2({
        "heart_rate": 140,
        "respiratory_rate": 30,
        "spo2": 88,
        "temperature_c": 39.5,
        "systolic_bp": 80,
        "consciousness_level": "U",
        "supplemental_o2": True,
    })
    assert result["score"] >= 7
    assert result["risk_level"] in {"High", "Critical"}


def test_csv_parser_reports_missing_values_without_imputing():
    csv = (
        "recorded_at,heart_rate,spo2\n"
        "2026-10-01 08:00:00,80,\n"
        "not-a-date,90,98\n"
    ).encode("utf-8")
    parsed = parse_vitals_csv(csv)
    assert parsed["rows"][0]["heart_rate"] == 80
    assert "spo2" not in parsed["rows"][0]
    assert any("malformed timestamp" in e for e in parsed["errors"])
    assert any("missing spo2" in w for w in parsed["warnings"])


def test_ews_endpoints_require_auth(client):
    assert client.get("/api/early-warning/alerts").status_code == 401
    assert client.post("/api/early-warning/vitals", json={"patient_id": 1}).status_code == 401


def test_record_vitals_and_block_other_doctor(client):
    headers, patient_id = doctor_and_patient(client)
    res = client.post(
        "/api/early-warning/vitals",
        json={
            "patient_id": patient_id,
            "heart_rate": 78,
            "respiratory_rate": 16,
            "spo2": 98,
            "temperature_c": 37.0,
            "systolic_bp": 118,
            "consciousness_level": "A",
        },
        headers=headers,
    )
    assert res.status_code == 201
    body = res.json()
    assert body["news2_score"] == 0
    assert body["news2_recommendations"]

    other, _ = doctor_and_patient(client, "other_ews@hospital.org")
    denied = client.get(f"/api/early-warning/patients/{patient_id}/vitals", headers=other)
    assert denied.status_code == 404


def test_predict_requires_vitals_then_uses_latest_window(client):
    headers, patient_id = doctor_and_patient(client, "ews_predict@hospital.org")
    empty = client.post(f"/api/early-warning/patients/{patient_id}/predict", headers=headers)
    assert empty.status_code == 422

    for i, hr in enumerate([70, 72, 74, 140]):
        rec = client.post(
            "/api/early-warning/vitals",
            json={
                "patient_id": patient_id,
                "heart_rate": hr,
                "respiratory_rate": 16 if i < 3 else 28,
                "spo2": 98 if i < 3 else 90,
                "temperature_c": 37.0,
                "systolic_bp": 120,
                "recorded_at": f"2026-10-01T0{i}:00:00",
            },
            headers=headers,
        )
        assert rec.status_code == 201

    pred = client.post(f"/api/early-warning/patients/{patient_id}/predict", headers=headers)
    assert pred.status_code == 200
    data = pred.json()
    assert data["vitals_count"] == 4
    assert data["model_used"] in {"xgboost", "news2_fallback", "news2_fallback_error"}
    assert data["prediction_target"]
    assert data["news2_score"] is not None


def test_csv_import_and_acknowledge_alerts(client):
    headers, patient_id = doctor_and_patient(client, "ews_csv@hospital.org")
    csv_bytes = (
        "timestamp,hr,rr,spo2,temp,sbp\n"
        "2026-10-02 08:00:00,78,16,98,37.0,120\n"
        "2026-10-02 10:00:00,132,26,90,38.8,88\n"
    ).encode("utf-8")
    res = client.post(
        "/api/early-warning/vitals/csv",
        params={"patient_id": patient_id},
        files={"file": ("vitals.csv", csv_bytes, "text/csv")},
        headers=headers,
    )
    assert res.status_code == 200
    report = res.json()
    assert report["imported_count"] == 2
    assert report["clinical_flags"]

    pred = client.post(f"/api/early-warning/patients/{patient_id}/predict", headers=headers)
    assert pred.status_code == 200
    assessment_id = pred.json()["assessment_id"]

    alerts = client.get("/api/early-warning/alerts", params={"sort_by": "risk"}, headers=headers)
    assert alerts.status_code == 200

    ack = client.post(
        f"/api/early-warning/assessments/{assessment_id}/acknowledge",
        json={"acknowledge_notes": "Reviewed on ward round"},
        headers=headers,
    )
    assert ack.status_code == 200
    remaining = client.get("/api/early-warning/alerts", headers=headers).json()
    assert all(a["assessment_id"] != assessment_id for a in remaining)


def test_csv_rejects_non_csv(client):
    headers, patient_id = doctor_and_patient(client, "ews_badfile@hospital.org")
    res = client.post(
        "/api/early-warning/vitals/csv",
        params={"patient_id": patient_id},
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=headers,
    )
    assert res.status_code == 422
