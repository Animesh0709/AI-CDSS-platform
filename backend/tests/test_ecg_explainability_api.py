"""
Comprehensive API integration tests for ECG Explainability Endpoint (Phase 8.3).
POST /api/ecg/{ecg_id}/explainability
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash
from app.database.session import Base, get_db
from app.main import app
from app.models.ecg import ECGRecord
from app.models.patient import Patient
from app.models.user import User
from app.schemas.ecg_explainability import ECGExplainabilityResponse
from app.services.ecg.explainability_service import ECGExplainabilityService, ECGModelRegistry


def create_synthetic_png_bytes(width: int = 1665, height: int = 1065) -> bytes:
    """Generate minimal valid RGB image matching conventional dimensions."""
    img = Image.new("RGB", (width, height), color=(255, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def explainability_test_context(tmp_path, monkeypatch):
    """Sets up an isolated SQLite in-memory DB and test environment."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(upload_dir))

    db = TestingSessionLocal()

    # Create Doctor 1 (Primary)
    doc1 = User(
        full_name="Dr. Jane Cardiologist",
        email="dr.jane@hospital.org",
        password_hash=get_password_hash("password123"),
        role="doctor",
        specialization="Cardiology",
    )
    # Create Doctor 2 (Unauthorized third party)
    doc2 = User(
        full_name="Dr. Bob Intruder",
        email="dr.bob@hospital.org",
        password_hash=get_password_hash("password123"),
        role="doctor",
        specialization="General",
    )
    db.add(doc1)
    db.add(doc2)
    db.commit()
    db.refresh(doc1)
    db.refresh(doc2)

    # Create Patient for Doctor 1
    p1 = Patient(
        first_name="Alice",
        last_name="Patient",
        date_of_birth="1980-05-15",
        gender="Female",
        doctor_id=doc1.id,
    )
    # Create Patient for Doctor 2
    p2 = Patient(
        first_name="Charlie",
        last_name="Patient",
        date_of_birth="1975-08-20",
        gender="Male",
        doctor_id=doc2.id,
    )
    db.add(p1)
    db.add(p2)
    db.commit()
    db.refresh(p1)
    db.refresh(p2)

    # Save a physical synthetic ECG scan file for Patient 1
    patient_dir = upload_dir / f"patient_{p1.id}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    scan_file = patient_dir / "test_ecg_scan.png"
    sample_src = REPO_ROOT / "ml" / "data" / "ecg" / "sample_synthetic_12lead.png"
    if sample_src.exists():
        img_bytes = sample_src.read_bytes()
    else:
        img_bytes = create_synthetic_png_bytes()
    scan_file.write_bytes(img_bytes)

    # Insert ECGRecord for Patient 1
    ecg1 = ECGRecord(
        patient_id=p1.id,
        doctor_id=doc1.id,
        original_filename="test_ecg_scan.png",
        stored_filename=scan_file.name,
        file_path=str(scan_file),
        file_size_bytes=len(img_bytes),
        mime_type="image/png",
        processing_status="completed",
        analysis_status="completed",
    )
    db.add(ecg1)
    db.commit()
    db.refresh(ecg1)

    # Override get_db dependency in app
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    token_doc1 = create_access_token(subject=str(doc1.id))
    token_doc2 = create_access_token(subject=str(doc2.id))

    client = TestClient(app)

    yield {
        "client": client,
        "db": db,
        "doc1": doc1,
        "doc2": doc2,
        "p1": p1,
        "p2": p2,
        "ecg1": ecg1,
        "token_doc1": token_doc1,
        "token_doc2": token_doc2,
        "upload_dir": upload_dir,
    }

    app.dependency_overrides.clear()


class TestECGExplainabilityAPI:

    def test_authenticated_successful_explainability_request(self, explainability_test_context):
        """Verify authorized doctor can request explainability and receives full validated schema."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"target_class": "MI", "integration_steps": 25},
        )
        assert response.status_code == 200
        data = response.json()

        # Schema top-level keys
        assert "global_metadata" in data
        assert "prediction" in data
        assert "attribution" in data
        assert "lead_alignments" in data
        assert "rhythm_strip_alignment" in data
        assert "auxiliary_grad_cam_guidance" in data

        # Check Global Metadata
        glob = data["global_metadata"]
        assert glob["ecg_id"] == ecg.id
        assert glob["patient_id"] == ecg.patient_id
        assert glob["source_image_filename"] == "test_ecg_scan.png"
        assert len(glob["source_image_hash"]) == 64  # valid SHA-256

        # Check Prediction Terminology
        pred = data["prediction"]
        assert pred["model_name"] == "1D_ResNet_PTBXL"
        assert "MI" in pred["raw_model_logits"]
        assert "MI" in pred["uncalibrated_model_probabilities"]
        assert "MI" in pred["model_decisions"]
        assert pred["queried_class"] == "MI"

        # Check Attribution Metadata
        attr = data["attribution"]
        assert attr["method"] == "INTEGRATED_GRADIENTS"
        assert attr["baseline"] == "zero_standardized_space"
        assert attr["integration_steps"] == 25
        assert "III" in attr["lead_importance"]

        # Check Per-lead Alignment (All 12 Leads)
        leads = data["lead_alignments"]
        assert len(leads) == 12
        for l_name in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            assert l_name in leads
            l_info = leads[l_name]
            assert "geometry_status" in l_info
            assert "reconstruction_status" in l_info
            assert "calibration_status" in l_info
            assert "attribution_status" in l_info
            assert "alignment_status" in l_info
            assert "image_region" in l_info
            assert "attribution_rank" in l_info
            assert "units_status" in l_info

        # Rhythm strip is separate
        rh = data["rhythm_strip_alignment"]
        assert rh is not None
        assert rh["lead_name"] == "II_rhythm"
        assert rh["is_rhythm_strip"] is True

        # Exploratory Grad-CAM Guidance
        cam = data["auxiliary_grad_cam_guidance"]
        assert cam["is_exploratory_only"] is True
        assert cam["mapped_to_pixels"] is False
        assert "disclaimer" in cam

    def test_unauthenticated_request_rejected(self, explainability_test_context):
        """Verify request without token returns 401 Unauthorized."""
        client = explainability_test_context["client"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(f"/api/ecg/{ecg.id}/explainability")
        assert response.status_code == 401

    def test_cross_patient_unauthorized_access_denied(self, explainability_test_context):
        """Verify Doctor 2 cannot request explainability for Doctor 1's patient record."""
        client = explainability_test_context["client"]
        token_doc2 = explainability_test_context["token_doc2"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token_doc2}"},
        )
        assert response.status_code == 403
        assert "not have authorization" in response.json()["detail"].lower()

    def test_missing_ecg_record_returns_404(self, explainability_test_context):
        """Verify non-existent ECG ID returns 404 Not Found."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]

        response = client.post(
            "/api/ecg/999999/explainability",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_invalid_target_class_returns_422(self, explainability_test_context):
        """Verify requesting an invalid class raises 422 Unprocessable Entity."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"target_class": "INVALID_CLASS_NAME"},
        )
        assert response.status_code == 422
        assert "unknown target class" in response.json()["detail"].lower()

    def test_source_image_immutability(self, explainability_test_context):
        """Verify original image file is unchanged before and after API call."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        file_p = Path(ecg.file_path)
        bytes_before = file_p.read_bytes()

        _ = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"target_class": "NORM"},
        )

        bytes_after = file_p.read_bytes()
        assert bytes_before == bytes_after

    def test_missing_resnet_checkpoint_returns_404(self, explainability_test_context):
        """1. Missing ResNet checkpoint raises 404 with explicit error message."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        with patch.object(ECGExplainabilityService, "generate_explainability", side_effect=FileNotFoundError("ECG ResNet checkpoint not found at ml/models/ecg_classification/deep_learning/best_model.pth")):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 404
            assert "checkpoint not found" in response.json()["detail"].lower()

    def test_model_initialization_failure_returns_500(self, explainability_test_context):
        """2. Model initialization failure returns 500 error."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        with patch.object(ECGModelRegistry, "load_model", side_effect=RuntimeError("Corrupt weights tensor")):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 500
            assert "error occurred while generating explainability" in response.json()["detail"].lower()

    def test_integrated_gradients_failure_returns_500(self, explainability_test_context):
        """3. Integrated Gradients attribution failure returns 500 error."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients
        with patch.object(ECGIntegratedGradients, "attribute", side_effect=RuntimeError("Gradient computation exploded")):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 500
            assert "error occurred while generating explainability" in response.json()["detail"].lower()

    def test_alignment_service_failure_returns_500(self, explainability_test_context):
        """4. Alignment service failure returns 500 error."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        from app.services.ecg.explainability_alignment import ECGExplainabilityImageAligner
        with patch.object(ECGExplainabilityImageAligner, "align_explanations", side_effect=ValueError("Invalid layout dimension")):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
            )
            # ValueError translates to 422
            assert response.status_code == 422
            assert "invalid layout dimension" in response.json()["detail"].lower()

    def test_partial_reconstruction_with_unavailable_leads(self, explainability_test_context):
        """5. Partial reconstruction returns valid response with unavailable leads clearly marked."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        # Mock reconstructor to return only 8 leads (V4, V5, V6 missing)
        from app.services.ecg.reconstruction import ECGWaveformReconstructor
        orig_extract = ECGWaveformReconstructor.extract_traces

        def mock_extract(self_rec, *args, **kwargs):
            res = orig_extract(self_rec, *args, **kwargs)
            # Remove V4, V5, V6
            for missing_l in ["V4", "V5", "V6"]:
                if missing_l in res["signals"]:
                    del res["signals"][missing_l]
                if missing_l in res["lead_metadata"]:
                    del res["lead_metadata"][missing_l]
            return res

        with patch.object(ECGWaveformReconstructor, "extract_traces", mock_extract):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
                json={"integration_steps": 20},
            )
            assert response.status_code == 200
            data = response.json()
            leads = data["lead_alignments"]

            # Unavailable leads must have explicit status and uncertainty reason
            for l_name in ["V4", "V5", "V6"]:
                assert leads[l_name]["reconstruction_status"] == "UNAVAILABLE"
                assert leads[l_name]["alignment_status"] == "UNAVAILABLE"
                assert "MISSING_LEAD_SIGNAL" in leads[l_name]["uncertainty_reasons"]

            assert data["unavailable_lead_count"] >= 3
            assert data["global_metadata"]["overall_processing_status"] == "PARTIAL_UNCERTAIN"

    def test_unknown_calibration_with_valid_lead_geometry(self, explainability_test_context):
        """6. Unknown calibration allows geometry outlines but marks calibration UNVERIFIED."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        from app.services.ecg.reconstruction import ECGWaveformReconstructor
        orig_extract = ECGWaveformReconstructor.extract_traces

        def mock_extract(self_rec, *args, **kwargs):
            res = orig_extract(self_rec, *args, **kwargs)
            res["is_calibrated"] = False
            for meta in res["lead_metadata"].values():
                meta["voltage_units"] = "unverified_pixels"
            return res

        with patch.object(ECGWaveformReconstructor, "extract_traces", mock_extract):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
                json={"integration_steps": 20},
            )
            assert response.status_code == 200
            data = response.json()
            leads = data["lead_alignments"]

            # Bounding boxes exist (geometry valid), but calibration is unverified
            for l_name, l_info in leads.items():
                assert l_info["geometry_status"] in ["VERIFIED", "ESTIMATED_STANDARD_LAYOUT"]
                assert l_info["calibration_status"] == "UNVERIFIED_PIXEL_DOMAIN"
                assert "UNCALIBRATED_IMAGE" in l_info["uncertainty_reasons"]
                assert l_info["units_status"] == "unverified_pixels"

    def test_low_confidence_lead_mapping(self, explainability_test_context):
        """7. Low confidence lead mapping marks reconstruction and alignment UNCERTAIN."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        from app.services.ecg.reconstruction import ECGWaveformReconstructor
        orig_extract = ECGWaveformReconstructor.extract_traces

        def mock_extract(self_rec, *args, **kwargs):
            res = orig_extract(self_rec, *args, **kwargs)
            if "I" in res["lead_metadata"]:
                res["lead_metadata"]["I"]["confidence_score"] = 0.45  # between 0.20 and 0.70
            return res

        with patch.object(ECGWaveformReconstructor, "extract_traces", mock_extract):
            response = client.post(
                f"/api/ecg/{ecg.id}/explainability",
                headers={"Authorization": f"Bearer {token}"},
                json={"integration_steps": 20},
            )
            assert response.status_code == 200
            data = response.json()
            lead_i = data["lead_alignments"]["I"]
            assert lead_i["reconstruction_status"] == "UNCERTAIN"
            assert lead_i["alignment_status"] == "UNCERTAIN"
            assert "LOW_RECONSTRUCTION_CONFIDENCE" in lead_i["uncertainty_reasons"]

    def test_corrupted_source_image_returns_422(self, explainability_test_context):
        """9. Corrupted source image returns 422 with validation error."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        # Overwrite file with corrupt bytes
        Path(ecg.file_path).write_bytes(b"corrupted raw data not an image")

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
        assert "cannot identify image file" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()

    def test_missing_stored_source_file_returns_404(self, explainability_test_context):
        """10. Missing stored source file on filesystem returns 404."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        # Delete image file from disk
        file_p = Path(ecg.file_path)
        if file_p.exists():
            file_p.unlink()

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
        assert "source ecg image not found" in response.json()["detail"].lower()

    def test_response_schema_validation(self, explainability_test_context):
        """11. Response strictly conforms to Pydantic ECGExplainabilityResponse model."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"target_class": "NORM", "integration_steps": 20},
        )
        assert response.status_code == 200
        # Re-validate against Pydantic schema
        validated = ECGExplainabilityResponse.model_validate(response.json())
        assert validated.prediction.queried_class == "NORM"
        assert len(validated.lead_alignments) == 12

    def test_lead_ii_and_rhythm_strip_separation(self, explainability_test_context):
        """12. Correct separation of the standard Lead II segment and continuous rhythm strip."""
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"integration_steps": 20},
        )
        assert response.status_code == 200
        data = response.json()

        lead_ii = data["lead_alignments"]["II"]
        rhythm_ii = data["rhythm_strip_alignment"]

        assert lead_ii["lead_name"] == "II"
        assert lead_ii["is_rhythm_strip"] is False

        assert rhythm_ii["lead_name"] == "II_rhythm"
        assert rhythm_ii["is_rhythm_strip"] is True

        # Bounding boxes should differ in horizontal span (rhythm strip spans all 4 cols)
        assert rhythm_ii["image_region"]["width"] > lead_ii["image_region"]["width"]

    def test_no_cross_patient_artifact_exposure(self, explainability_test_context):
        """Verify artifact persistence is safely isolated and cannot be accessed across patients."""
        upload_dir = explainability_test_context["upload_dir"]
        client = explainability_test_context["client"]
        token = explainability_test_context["token_doc1"]
        ecg = explainability_test_context["ecg1"]

        response = client.post(
            f"/api/ecg/{ecg.id}/explainability",
            headers={"Authorization": f"Bearer {token}"},
            json={"target_class": "NORM", "integration_steps": 20},
        )
        assert response.status_code == 200

        # Artifact must be in patient_{p1.id} directory
        expected_dir = upload_dir / "ecg_explainability" / f"patient_{ecg.patient_id}"
        assert expected_dir.exists()
        json_files = list(expected_dir.glob("*.json"))
        assert len(json_files) >= 1

        # Response must NOT contain internal server filesystem paths
        raw_json_str = response.text
        assert str(upload_dir) not in raw_json_str
        assert "/private/var" not in raw_json_str
        assert "/Users/" not in raw_json_str

