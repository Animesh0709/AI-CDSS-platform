"""
FastAPI router for MRI Brain Tumor Classification endpoints.

Endpoints:
  POST /mri/predict        — Upload a brain MRI image; get three-model comparison
  GET  /mri/eval-report    — Evaluation metrics from training (accuracy, F1, CM)
  GET  /mri/model-info     — Model metadata, dataset provenance, limitations

All endpoints require a valid JWT (doctor authentication).
File validation: JPEG/PNG/BMP/TIFF, max 10 MB.
Missing model artifacts return HTTP 503 with actionable instructions.
Runtime errors are caught and returned as HTTP 500 — they do NOT crash the process.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.auth import get_current_user
from app.schemas.mri import MRIEvalReport, MRIModelInfo, MRIPredictionOut, SingleModelPrediction
from app.services import mri as mri_service

router = APIRouter(prefix="/mri", tags=["MRI Classification"])

# Accepted MIME types and file size limit
_ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff", "image/tif"}
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_INFERENCE_TIMEOUT = 60.0       # seconds


def _validate_upload(file: UploadFile) -> None:
    """Raise HTTP 422 for invalid content type."""
    ct = (file.content_type or "").lower()
    if ct not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '{ct}'. "
                "Please upload a JPEG, PNG, BMP, or TIFF MRI image."
            ),
        )


@router.post(
    "/predict",
    response_model=MRIPredictionOut,
    summary="Upload an MRI image and get three-model classification",
    description=(
        "Accepts a brain MRI image (JPEG/PNG/BMP/TIFF, max 10 MB) and returns "
        "AI-assisted tumour classification from XGBoost, SVM, and ViT models. "
        "All outputs require clinician review before any clinical action is taken."
    ),
)
async def predict_mri(
    file: UploadFile = File(..., description="Brain MRI image (JPEG/PNG/BMP/TIFF, max 10 MB)"),
    current_user=Depends(get_current_user),
):
    # ── Validate content type ─────────────────────────────────────────────────
    _validate_upload(file)

    # ── Read and size-check ───────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). Maximum allowed is 10 MB.",
        )

    # ── Run inference with timeout ────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        raw_results = await asyncio.wait_for(
            loop.run_in_executor(None, mri_service.predict_all, image_bytes),
            timeout=_INFERENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"MRI inference timed out after {_INFERENCE_TIMEOUT:.0f} s. "
                   "Try again or check server resources.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected inference error: {exc}",
        )

    # ── Map raw dicts to schema objects ───────────────────────────────────────
    predictions = [SingleModelPrediction(**r) for r in raw_results]

    return MRIPredictionOut(
        filename=file.filename or "uploaded_image",
        file_size_bytes=len(image_bytes),
        predictions=predictions,
    )


@router.get(
    "/eval-report",
    response_model=MRIEvalReport,
    summary="MRI model evaluation metrics from training",
    description=(
        "Returns accuracy, precision, recall, F1-score, and confusion matrices "
        "for each model on the held-out test set. Returns HTTP 503 if training "
        "has not been run yet."
    ),
)
def get_eval_report(current_user=Depends(get_current_user)):
    report = mri_service.load_eval_report()
    if report is None:
        return MRIEvalReport(
            status="not_trained",
            message=(
                "Evaluation report not available. "
                "Run ml/scripts/train_mri.py after placing the dataset at ml/data/brain_tumor/."
            ),
        )
    return MRIEvalReport(status="available", report=report)


@router.get(
    "/model-info",
    response_model=MRIModelInfo,
    summary="MRI model metadata, dataset provenance, and limitations",
)
def get_model_info(current_user=Depends(get_current_user)):
    metadata = mri_service.load_model_metadata()
    return MRIModelInfo(metadata=metadata)
