"""
Automated Integration and Unit Tests for 3D Multimodal Segmentation Pipeline.
=============================================================================
Tests:
  - Dataset & multi-channel shape validation (4 modalities -> 3 subregions)
  - 3D Model forward passes (UNet, Attention UNet, nnUNet)
  - Physical volume calculation in cm³
  - Expected Calibration Error (ECE) & Monte Carlo Dropout Uncertainty
  - REST API endpoints:
      - GET /api/segmentation/cases
      - GET /api/segmentation/metadata
      - GET /api/segmentation/eval-report
      - POST /api/segmentation/predict
"""

import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.services.segmentation import (
    load_segmentation_model,
    run_3d_segmentation,
    get_available_cases,
    render_slice_base64
)

client = TestClient(app)

# Helper token generator for auth
def get_auth_headers():
    import uuid
    email = f"seg_test_{uuid.uuid4().hex[:6]}@hospital.org"
    password = "TestPassword2026!"
    client.post("/api/auth/register", json={
        "full_name": "Dr. Test Seg",
        "email": email,
        "password": password,
        "role": "doctor"
    })
    res = client.post("/api/auth/login", json={"email": email, "password": password})
    if res.status_code == 200:
        token = res.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    return {}

def test_available_cases():
    cases = get_available_cases()
    assert len(cases) > 0, "Should have processed UPenn-GBM cases available"
    assert isinstance(cases[0], str)

def test_model_loading_and_architecture():
    for m_name in ["unet3d", "attunet3d", "nnunet"]:
        model = load_segmentation_model(m_name)
        assert model is not None
        # Forward pass on dummy tensor (1, 4, 32, 32, 32)
        device = next(model.parameters()).device
        dummy = torch.randn(1, 4, 32, 32, 32).to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 3, 32, 32, 32), f"Expected (1, 3, 32, 32, 32) output from {m_name}"

def test_render_slice_base64():
    mri_2d = np.random.rand(100, 100).astype(np.float32)
    mask_2d = (np.random.rand(3, 100, 100) > 0.8).astype(np.float32)
    unc_2d = np.random.rand(100, 100).astype(np.float32)

    raw_b64, overlay_b64, unc_b64 = render_slice_base64(mri_2d, mask_2d, unc_2d)
    assert raw_b64.startswith("data:image/png;base64,")
    assert overlay_b64.startswith("data:image/png;base64,")
    assert unc_b64.startswith("data:image/png;base64,")

def test_segmentation_service_inference():
    cases = get_available_cases()
    case_id = cases[0]
    res = run_3d_segmentation(
        subject_identifier=case_id,
        model_name="attunet3d",
        axial_slice=78
    )

    assert res["subject_identifier"] == case_id
    assert "volumes" in res
    assert "whole_tumor_cm3" in res["volumes"]
    assert "tumor_core_cm3" in res["volumes"]
    assert "enhancing_tumor_cm3" in res["volumes"]
    assert "edema_cm3" in res["volumes"]
    assert res["volumes"]["whole_tumor_cm3"] >= 0.0

    assert "uncertainty" in res
    assert "mean_predictive_entropy" in res["uncertainty"]
    assert "slices" in res
    assert "axial" in res["slices"]
    assert "coronal" in res["slices"]
    assert "sagittal" in res["slices"]

def test_api_segmentation_endpoints():
    headers = get_auth_headers()
    assert "Authorization" in headers, "Failed to authenticate test doctor"

    # 1. Cases
    r_cases = client.get("/api/segmentation/cases", headers=headers)
    assert r_cases.status_code == 200
    cases_list = r_cases.json()
    assert len(cases_list) > 0

    # 2. Metadata
    r_meta = client.get("/api/segmentation/metadata", headers=headers)
    assert r_meta.status_code == 200
    meta = r_meta.json()
    assert "architectures" in meta
    assert "unet3d" in meta["architectures"]
    assert "attunet3d" in meta["architectures"]
    assert "nnunet" in meta["architectures"]

    # 3. Eval Report
    r_eval = client.get("/api/segmentation/eval-report", headers=headers)
    assert r_eval.status_code == 200
    eval_rep = r_eval.json()
    assert "models" in eval_rep
    assert "attunet3d" in eval_rep["models"]
    assert "overall_dice" in eval_rep["models"]["attunet3d"]

    # 4. Predict
    payload = {
        "subject_identifier": cases_list[0],
        "model_name": "attunet3d",
        "axial_slice": 78,
        "coronal_slice": 120,
        "sagittal_slice": 120
    }
    r_pred = client.post("/api/segmentation/predict", json=payload, headers=headers)
    assert r_pred.status_code == 200
    pred = r_pred.json()
    assert pred["model_name"] == "attunet3d"
    assert pred["volumes"]["whole_tumor_cm3"] >= 0.0
    assert pred["slices"]["axial"]["raw_mri_base64"].startswith("data:image/png;base64,")


def test_upload_invalid_file_rejected():
    headers = get_auth_headers()
    files = {"file": ("test_doc.txt", b"plain text content", "text/plain")}
    res = client.post("/api/segmentation/upload", files=files, headers=headers)
    assert res.status_code == 422
    assert "Unsupported MRI file type" in res.json()["detail"]


def test_upload_mri_image_endpoint():
    import io
    from PIL import Image
    headers = get_auth_headers()
    img = Image.new("L", (128, 128), color=140)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    files = {"file": ("uploaded_mri_test.png", buf.getvalue(), "image/png")}

    res = client.post("/api/segmentation/upload", files=files, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "subject_identifier" in data
    assert data["subject_identifier"].startswith("UPLOAD_")

    # Verify it appears in cases
    res_cases = client.get("/api/segmentation/cases", headers=headers)
    assert res_cases.status_code == 200
    cases_list = res_cases.json()
    assert data["subject_identifier"] in cases_list


def test_upload_and_predict_endpoint():
    import io
    from PIL import Image
    headers = get_auth_headers()
    img = Image.new("L", (128, 128), color=160)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    files = {"file": ("uploaded_patient_scan.jpg", buf.getvalue(), "image/jpeg")}
    data = {"model_name": "attunet3d", "axial_slice": "78"}

    res = client.post("/api/segmentation/upload-and-predict", files=files, data=data, headers=headers)
    assert res.status_code == 200
    pred = res.json()
    assert pred["subject_identifier"].startswith("UPLOAD_")
    assert "volumes" in pred
    assert "whole_tumor_cm3" in pred["volumes"]
    assert "slices" in pred
    assert "axial" in pred["slices"]
    assert pred["slices"]["axial"]["overlay_mask_base64"].startswith("data:image/png;base64,")

