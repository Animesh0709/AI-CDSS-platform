"""Week 12 regression tests for authenticated assistant and XAI routes."""
import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import assistant, xai  # noqa: E402
from app.database.session import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    def override_get_db():
        db = Session()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(assistant, "evidence_answer", lambda question, limit: {"query": question, "status": "evidence_summary", "answer": "[1] Curated evidence.", "message": None, "provider": "local", "model": None, "evidence": []})
    monkeypatch.setattr(xai, "local_explanation", lambda values, patient: {"probability": .24, "model_label": "test", "increasing_factors": [], "decreasing_factors": [], "data_quality": ["schema matched"], "ood_warning": None, "illustrative_counterfactual": None, "disclaimer": "Clinician review required."})
    yield TestClient(app)
    app.dependency_overrides.clear()


def doctor_and_patient(client, email="week12@example.org"):
    client.post("/api/auth/register", json={"full_name": "Dr Test", "email": email, "password": "TestPass2026!"})
    token = client.post("/api/auth/login", json={"email": email, "password": "TestPass2026!"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    patient = client.post("/api/patients", json={"first_name": "Test", "last_name": "Patient", "gender": "Other"}, headers=headers).json()
    return headers, patient["id"]


def test_assistant_response_uses_question_schema_and_owned_patient_context(client):
    headers, patient_id = doctor_and_patient(client)
    response = client.post("/api/assistant/ask", json={"question": "diabetes screening", "patient_id": patient_id}, headers=headers)
    assert response.status_code == 200
    assert response.json()["question"] == "diabetes screening"
    assert response.json()["patient_context"]["patient_id"] == patient_id


def test_xai_prediction_requires_owner_and_validated_inputs(client):
    headers, patient_id = doctor_and_patient(client)
    ok = client.post(f"/api/xai/patients/{patient_id}/predict", json={"number_inpatient": 2}, headers=headers)
    assert ok.status_code == 200 and ok.json()["probability"] == .24
    other_headers, _ = doctor_and_patient(client, "other-week12@example.org")
    denied = client.post(f"/api/xai/patients/{patient_id}/predict", json={}, headers=other_headers)
    assert denied.status_code == 404
    invalid = client.post(f"/api/xai/patients/{patient_id}/predict", json={"number_diagnoses": 0}, headers=headers)
    assert invalid.status_code == 422
