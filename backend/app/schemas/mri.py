"""Pydantic schemas for MRI brain tumor classification endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SingleModelPrediction(BaseModel):
    """Prediction result from one model."""
    model_name: str
    status: str                        # "ok" | "unavailable" | "error"
    predicted_class: str | None = None # e.g. "glioma"
    confidence: float | None = None    # 0.0 – 1.0 (probability of predicted class)
    probabilities: dict[str, float] | None = None  # per-class probabilities
    message: str | None = None         # human-readable status / error detail


class MRIPredictionOut(BaseModel):
    """Response returned by POST /mri/predict."""
    filename: str
    file_size_bytes: int
    predictions: list[SingleModelPrediction]
    disclaimer: str = (
        "AI-assisted classification — Predicted class. "
        "These outputs are NOT a clinical diagnosis and require review by a qualified clinician."
    )


class MRIEvalReport(BaseModel):
    """Response returned by GET /mri/eval-report."""
    status: str          # "available" | "not_trained"
    report: dict[str, Any] | None = None
    message: str | None = None


class MRIModelInfo(BaseModel):
    """Response returned by GET /mri/model-info."""
    metadata: dict[str, Any]
