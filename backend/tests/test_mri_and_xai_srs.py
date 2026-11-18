"""
Tests for new SRS features:
  - Feature A: MRI classification API endpoints (/api/mri/model-info, /api/mri/eval-report, /api/mri/predict)
  - Feature B: Explainability verification (four methods present, counterfactual absent)
"""
import io
import os
import sys
import pytest
from fastapi.testclient import TestClient
from PIL import Image
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
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def auth_headers(client, email="mri_tester@example.org"):
    client.post("/api/auth/register", json={"full_name": "Dr Radiologist", "email": email, "password": "TestPass2026!"})
    res = client.post("/api/auth/login", json={"email": email, "password": "TestPass2026!"})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_mri_endpoints_require_authentication(client):
    """Unauthenticated requests to MRI endpoints should return 401."""
    assert client.get("/api/mri/model-info").status_code == 401
    assert client.get("/api/mri/eval-report").status_code == 401
    assert client.post("/api/mri/predict").status_code == 401


def test_mri_model_info_returns_trained_metadata(client):
    """GET /api/mri/model-info returns metadata for all 3 models."""
    headers = auth_headers(client)
    res = client.get("/api/mri/model-info", headers=headers)
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["status"] == "trained"
    assert "xgboost" in meta["models"]
    assert "svm" in meta["models"]
    assert "vit" in meta["models"]
    assert meta["classes"] == ["glioma", "meningioma", "notumor", "pituitary"]


def test_mri_eval_report_returns_metrics(client):
    """GET /api/mri/eval-report returns accuracy, macro_f1, and confusion matrices."""
    headers = auth_headers(client)
    res = client.get("/api/mri/eval-report", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    report = data["report"]
    assert "xgboost" in report
    assert "svm" in report
    assert "vit" in report
    assert report["xgboost"]["accuracy"] > 0.85
    assert report["svm"]["accuracy"] > 0.85
    assert report["vit"]["accuracy"] > 0.85


def test_mri_predict_rejects_invalid_file_type(client):
    """Uploading a non-image file returns 422 Unprocessable Entity."""
    headers = auth_headers(client)
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    res = client.post("/api/mri/predict", files=files, headers=headers)
    assert res.status_code == 422
    assert "Unsupported file type" in res.json()["detail"]


def test_mri_predict_with_valid_image(client):
    """Uploading a valid MRI image runs inference and returns 3-model comparison."""
    headers = auth_headers(client)
    # Generate a dummy 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("synthetic_scan.jpg", buf.getvalue(), "image/jpeg")}
    res = client.post("/api/mri/predict", files=files, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data["predictions"]) == 3
    model_names = [p["model_name"] for p in data["predictions"]]
    assert "XGBoost" in model_names
    assert "SVM" in model_names
    assert "ViT (fine-tuned)" in model_names
    for p in data["predictions"]:
        assert p["status"] == "ok"
        assert p["predicted_class"] in ["glioma", "meningioma", "notumor", "pituitary"]
        assert 0.0 <= p["confidence"] <= 1.0


def test_counterfactual_strictly_absent_from_xai_service():
    """Verify counterfactual code is completely removed from xai service and schemas."""
    from app.services import xai as xai_service
    from app.schemas import xai as xai_schema
    import inspect

    service_src = inspect.getsource(xai_service)
    assert "illustrative_counterfactual" not in service_src
    assert "counterfactual" not in service_src.lower()

    schema_fields = xai_schema.PredictionOut.model_fields
    assert "illustrative_counterfactual" not in schema_fields
    assert "counterfactual" not in schema_fields
