import io
import os
import sys
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database.session import Base, get_db
from app.models.user import User
from app.models.patient import Patient
from app.core.security import get_password_hash, create_access_token


@pytest.fixture
def ecg_test_context(tmp_path, monkeypatch):
    """
    Sets up an isolated SQLite in-memory DB and temporary upload dir for ECG tests.
    Creates 2 doctors and 2 patients to test access boundaries.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    # Override upload dir to isolated tmp_path
    monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("app.core.config.settings.MAX_UPLOAD_SIZE_MB", 2)  # 2MB for testing limit

    # Seed doctors and patients
    db = TestingSessionLocal()
    
    # Doctor 1 (Primary)
    doc1 = User(
        full_name="Dr. Jane Cardiologist",
        email="dr.jane@hospital.org",
        password_hash=get_password_hash("password123"),
        role="doctor",
        specialization="Cardiology",
    )
    # Doctor 2 (Unauthorized third party)
    doc2 = User(
        full_name="Dr. Bob Surgeon",
        email="dr.bob@hospital.org",
        password_hash=get_password_hash("password123"),
        role="doctor",
        specialization="Surgery",
    )
    db.add_all([doc1, doc2])
    db.commit()
    db.refresh(doc1)
    db.refresh(doc2)

    # Patient belonging to Doctor 1
    p1 = Patient(
        doctor_id=doc1.id,
        first_name="Alice",
        last_name="Smith",
        date_of_birth="1980-05-15",
        gender="Female",
    )
    # Patient belonging to Doctor 2
    p2 = Patient(
        doctor_id=doc2.id,
        first_name="Charlie",
        last_name="Brown",
        date_of_birth="1975-08-20",
        gender="Male",
    )
    db.add_all([p1, p2])
    db.commit()
    db.refresh(p1)
    db.refresh(p2)

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    token_doc1 = create_access_token(subject=doc1.id)
    token_doc2 = create_access_token(subject=doc2.id)

    client = TestClient(app)

    yield {
        "client": client,
        "db": db,
        "doc1": doc1,
        "doc2": doc2,
        "p1": p1,
        "p2": p2,
        "token_doc1": token_doc1,
        "token_doc2": token_doc2,
    }

    app.dependency_overrides.clear()
    db.close()


def create_synthetic_image(fmt: str = "PNG", size=(300, 300), color="white") -> bytes:
    """Helper to generate a valid PNG or JPEG image byte stream."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# 1. Valid PNG upload
def test_upload_valid_png(ecg_test_context):
    client = ecg_test_context["client"]
    token = ecg_test_context["token_doc1"]
    patient = ecg_test_context["p1"]

    img_bytes = create_synthetic_image("PNG")
    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"patient_id": patient.id},
        files={"file": ("ecg_lead_scan.png", img_bytes, "image/png")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "ECG image uploaded and validated successfully."
    assert data["record"]["patient_id"] == patient.id
    assert data["record"]["mime_type"] == "image/png"
    assert data["record"]["processing_status"] == "uploaded"
    assert os.path.exists(data["record"]["file_path"])


# 2. Valid JPEG upload
def test_upload_valid_jpeg(ecg_test_context):
    client = ecg_test_context["client"]
    token = ecg_test_context["token_doc1"]
    patient = ecg_test_context["p1"]

    img_bytes = create_synthetic_image("JPEG")
    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"patient_id": patient.id},
        files={"file": ("patient_scan.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["record"]["mime_type"] == "image/jpeg"
    assert data["record"]["original_filename"] == "patient_scan.jpg"


# 3. Invalid file type (e.g. PDF or plain text)
def test_upload_invalid_file_type(ecg_test_context):
    client = ecg_test_context["client"]
    token = ecg_test_context["token_doc1"]
    patient = ecg_test_context["p1"]

    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"patient_id": patient.id},
        files={"file": ("notes.txt", b"patient vitals text", "text/plain")},
    )
    assert response.status_code == 422
    assert "Unsupported" in response.json()["detail"]


# 4. Oversized upload
def test_upload_oversized(ecg_test_context):
    client = ecg_test_context["client"]
    token = ecg_test_context["token_doc1"]
    patient = ecg_test_context["p1"]

    # MAX_UPLOAD_SIZE_MB is set to 2 in fixture; upload 3MB
    oversized_bytes = b"0" * (3 * 1024 * 1024)
    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"patient_id": patient.id},
        files={"file": ("large_ecg.png", oversized_bytes, "image/png")},
    )
    assert response.status_code == 413
    assert "exceeds maximum upload size" in response.json()["detail"]


# 5. Corrupted image bytes or renamed non-image
def test_upload_corrupted_image(ecg_test_context):
    client = ecg_test_context["client"]
    token = ecg_test_context["token_doc1"]
    patient = ecg_test_context["p1"]

    fake_png = b"\x89PNG\r\n\x1a\n" + b"garbage corrupted content"
    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"patient_id": patient.id},
        files={"file": ("corrupted.png", fake_png, "image/png")},
    )
    assert response.status_code == 422
    assert "Invalid or corrupted image" in response.json()["detail"]


# 6. Unauthenticated request
def test_upload_unauthenticated(ecg_test_context):
    client = ecg_test_context["client"]
    patient = ecg_test_context["p1"]

    img_bytes = create_synthetic_image("PNG")
    response = client.post(
        "/api/ecg/upload",
        data={"patient_id": patient.id},
        files={"file": ("ecg.png", img_bytes, "image/png")},
    )
    assert response.status_code == 401


# 7. Missing or unauthorized patient access
def test_upload_unauthorized_patient(ecg_test_context):
    client = ecg_test_context["client"]
    token_doc1 = ecg_test_context["token_doc1"]
    patient_of_doc2 = ecg_test_context["p2"]

    img_bytes = create_synthetic_image("PNG")
    # Doctor 1 attempts to upload for Doctor 2's patient
    response = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token_doc1}"},
        data={"patient_id": patient_of_doc2.id},
        files={"file": ("ecg.png", img_bytes, "image/png")},
    )
    assert response.status_code == 403
    assert "not have authorization" in response.json()["detail"]

    # Non-existent patient
    response_missing = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token_doc1}"},
        data={"patient_id": 999999},
        files={"file": ("ecg.png", img_bytes, "image/png")},
    )
    assert response_missing.status_code == 404
    assert "not found" in response_missing.json()["detail"]


# 8. Successful ECG record retrieval and unauthorized access check
def test_get_ecg_record(ecg_test_context):
    client = ecg_test_context["client"]
    token_doc1 = ecg_test_context["token_doc1"]
    token_doc2 = ecg_test_context["token_doc2"]
    patient = ecg_test_context["p1"]

    # 1. Upload as Doctor 1
    img_bytes = create_synthetic_image("PNG")
    upload_res = client.post(
        "/api/ecg/upload",
        headers={"Authorization": f"Bearer {token_doc1}"},
        data={"patient_id": patient.id},
        files={"file": ("ecg.png", img_bytes, "image/png")},
    )
    ecg_id = upload_res.json()["record"]["id"]

    # 2. Retrieve as Doctor 1 (Success)
    get_res = client.get(
        f"/api/ecg/{ecg_id}",
        headers={"Authorization": f"Bearer {token_doc1}"},
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == ecg_id
    assert get_res.json()["patient_id"] == patient.id

    # 3. Retrieve as Doctor 2 (Forbidden)
    forbidden_res = client.get(
        f"/api/ecg/{ecg_id}",
        headers={"Authorization": f"Bearer {token_doc2}"},
    )
    assert forbidden_res.status_code == 403

    # 4. List records for patient
    list_res = client.get(
        f"/api/ecg/patient/{patient.id}",
        headers={"Authorization": f"Bearer {token_doc1}"},
    )
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1
    assert list_res.json()[0]["id"] == ecg_id
