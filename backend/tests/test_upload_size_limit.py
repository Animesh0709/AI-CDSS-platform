"""
Minimal regression test for the report-upload size-limit bug:

    try:
        ...
        if size_bytes > max_bytes:
            raise HTTPException(413, ...)
    except Exception:
        pass   # <- this used to swallow the 413 too

Confirms an oversized upload now actually returns HTTP 413 through the
real endpoint (not just the size-check logic in isolation), and that a
normal-sized upload still succeeds (200) - i.e. no unrelated upload
behavior changed.

Uses an isolated in-memory SQLite DB via a get_db override so this does
NOT touch the real backend/ai_cdss.db dev database.

Run with:  cd backend && pytest tests/test_upload_size_limit.py -v
"""
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402
from app.database.session import Base, get_db  # noqa: E402
from app.core.config import settings  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolated, throwaway in-memory DB - never touches backend/ai_cdss.db.
    # StaticPool keeps a single connection alive so the in-memory DB
    # persists across requests within this test (otherwise each new
    # connection would see a fresh, empty :memory: database).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # Also isolate the upload directory to a throwaway tmp dir, so these
    # tests never write into the real backend/uploads/ tree.
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers_and_patient_id(client):
    """Register a doctor, log in, create a patient. Reuses the existing
    auth/patient endpoints exactly as they are - no new test scaffolding."""
    client.post("/api/auth/register", json={
        "full_name": "Dr. Test",
        "email": "upload_size_test@hospital.org",
        "password": "TestPass2026!",
    })
    login = client.post("/api/auth/login", json={
        "email": "upload_size_test@hospital.org",
        "password": "TestPass2026!",
    })
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    patient = client.post("/api/patients", json={
        "first_name": "Test",
        "last_name": "Patient",
        "gender": "Other",
    }, headers=headers)
    patient_id = patient.json()["id"]

    return headers, patient_id


def test_oversized_upload_returns_413(client, auth_headers_and_patient_id, monkeypatch):
    headers, patient_id = auth_headers_and_patient_id
    # Shrink the limit instead of uploading a real multi-MB file.
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.0001)  # ~100 bytes

    oversized_content = b"x" * 5000  # far above the shrunk limit
    files = {"report_file": ("too_big.txt", io.BytesIO(oversized_content), "text/plain")}

    resp = client.post(f"/api/patients/{patient_id}/reports", files=files, headers=headers)

    assert resp.status_code == 413, f"Expected 413, got {resp.status_code}: {resp.text}"


def test_normal_sized_upload_still_succeeds(client, auth_headers_and_patient_id):
    """Guards against the fix accidentally changing behavior for uploads
    that are within the limit."""
    headers, patient_id = auth_headers_and_patient_id

    files = {"report_file": ("fine.txt", io.BytesIO(b"small report contents"), "text/plain")}
    resp = client.post(f"/api/patients/{patient_id}/reports", files=files, headers=headers)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
