"""
Pydantic schemas for ECG Explainability Backend API (Phase 8.3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ImageRegionCoordinatesOut(BaseModel):
    """Bounding box coordinates for a lead's region in the ECG scan image."""
    x_min: float = Field(..., description="Left pixel coordinate of the lead bounding region")
    y_min: float = Field(..., description="Top pixel coordinate of the lead bounding region")
    x_max: float = Field(..., description="Right pixel coordinate of the lead bounding region")
    y_max: float = Field(..., description="Bottom pixel coordinate of the lead bounding region")
    width: float = Field(..., description="Width of the bounding box in pixels")
    height: float = Field(..., description="Height of the bounding box in pixels")

    model_config = ConfigDict(from_attributes=True)


class LeadAlignmentOut(BaseModel):
    """Per-lead explainability alignment and confidence metadata."""
    lead_name: str = Field(..., description="Standard lead name (e.g., 'I', 'V1', 'II_rhythm')")
    geometry_status: str = Field(..., description="Status of lead bounding box layout: VERIFIED, ESTIMATED_STANDARD_LAYOUT, or INVALID_OR_UNKNOWN")
    reconstruction_status: str = Field(..., description="Status of waveform extraction: CONFIDENT, UNCERTAIN, or UNAVAILABLE")
    calibration_status: str = Field(..., description="Status of physical units: VERIFIED or UNVERIFIED_PIXEL_DOMAIN")
    attribution_status: str = Field(..., description="Status of model attribution: AVAILABLE or UNAVAILABLE")
    alignment_status: str = Field(..., description="Overall mapping synthesis: CONFIDENT, UNCERTAIN, or UNAVAILABLE")
    image_region: Optional[ImageRegionCoordinatesOut] = Field(None, description="Image bounding box in pixel coordinates")
    reconstruction_confidence: float = Field(..., description="Quality/continuity score of the reconstructed lead trace in [0.0, 1.0]")
    attribution_importance: float = Field(..., description="Absolute Integrated Gradients feature importance for this lead")
    attribution_rank: int = Field(..., description="Attribution importance rank among leads (1 = highest contribution)")
    uncertainty_reasons: List[str] = Field(default_factory=list, description="Reason codes for uncertainty or unavailability")
    units_status: str = Field(..., description="Voltage and time units representation (e.g., 'millivolts', 'unverified_pixels')")
    is_rhythm_strip: bool = Field(False, description="Whether this alignment represents the continuous Lead II rhythm strip")
    notes: Optional[str] = Field(None, description="Diagnostic notes or reason details")

    model_config = ConfigDict(from_attributes=True)


class PredictionMetadataOut(BaseModel):
    """Structured model prediction output with strict non-diagnostic terminology."""
    model_name: str = Field(..., description="Name of the evaluated model architecture (e.g., '1D_ResNet_PTBXL')")
    model_version: str = Field(..., description="Model checkpoint or version identifier")
    target_classes: List[str] = Field(..., description="List of target classes evaluated by the multi-label classifier")
    raw_model_logits: Dict[str, float] = Field(..., description="Raw unbounded linear logit activations per class")
    uncalibrated_model_probabilities: Dict[str, float] = Field(..., description="Raw sigmoid outputs per class (uncalibrated model probability)")
    model_decisions: Dict[str, str] = Field(..., description="Threshold-based decision string per class (e.g., 'POSITIVE' or 'NEGATIVE')")
    thresholds_used: Dict[str, float] = Field(..., description="Operating threshold values used for decision determination")
    queried_class: str = Field(..., description="Primary class for which feature attribution and explainability was computed")
    prediction_timestamp: datetime = Field(..., description="Timestamp when model inference occurred")

    model_config = ConfigDict(from_attributes=True)


class AttributionMetadataOut(BaseModel):
    """Integrated Gradients attribution metadata and completeness proof."""
    method: str = Field("INTEGRATED_GRADIENTS", description="Explainability method used")
    baseline: str = Field("zero_standardized_space", description="Baseline input representation")
    integration_steps: int = Field(..., description="Number of interpolation steps along the straight-line path")
    completeness_error: float = Field(..., description="Absolute difference between path integral sum and target logit difference")
    lead_importance: Dict[str, float] = Field(..., description="Computed feature importance score per lead")
    attribution_status: str = Field(..., description="Status of attribution computation: AVAILABLE or UNAVAILABLE")
    artifact_id: Optional[str] = Field(None, description="Unique artifact identifier for stored attribution arrays")

    model_config = ConfigDict(from_attributes=True)


class AuxiliaryGradCamGuidanceOut(BaseModel):
    """Exploratory-only 1D temporal attention guidance."""
    is_exploratory_only: bool = Field(True, description="Always true: indicates Grad-CAM is strictly exploratory")
    mapped_to_pixels: bool = Field(False, description="Always false: Grad-CAM is never projected onto image pixels")
    disclaimer: str = Field(..., description="Mandatory clinical and methodological disclaimer")
    temporal_points_available: int = Field(..., description="Number of temporal points along the reconstructed waveform")
    peak_temporal_sec: Optional[float] = Field(None, description="Time in seconds where peak 1D activation occurred")

    model_config = ConfigDict(from_attributes=True)


class GlobalMetadataOut(BaseModel):
    """Provenance and image acquisition metadata."""
    ecg_id: int = Field(..., description="Database ID of the analyzed ECG record")
    patient_id: int = Field(..., description="ID of the patient who owns the record")
    source_image_filename: str = Field(..., description="Original uploaded filename")
    source_image_hash: str = Field(..., description="SHA-256 hash of the original image bytes for provenance verification")
    layout_type: str = Field(..., description="Detected layout geometry (e.g., 'Conventional 3x4 + Lead II Rhythm Strip')")
    calibration_verified: bool = Field(..., description="Whether physical calibration (mm/mV, mm/s) was confirmed")
    processing_timestamp: datetime = Field(..., description="Timestamp when explainability alignment was generated")
    overall_processing_status: str = Field(..., description="Overall pipeline status: COMPLETED, PARTIAL_UNCERTAIN, or FAILED")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings regarding uncalibrated units or gating")

    model_config = ConfigDict(from_attributes=True)


class ECGExplainabilityResponse(BaseModel):
    """Complete backend API response contract for ECG explainability."""
    global_metadata: GlobalMetadataOut
    prediction: PredictionMetadataOut
    attribution: AttributionMetadataOut
    lead_alignments: Dict[str, LeadAlignmentOut] = Field(..., description="Alignments for all 12 standard leads")
    rhythm_strip_alignment: Optional[LeadAlignmentOut] = Field(None, description="Separate alignment for continuous rhythm strip")
    auxiliary_grad_cam_guidance: AuxiliaryGradCamGuidanceOut
    confident_lead_count: int
    uncertain_lead_count: int
    unavailable_lead_count: int
    overall_alignment_coverage_pct: float
    clinical_measurements: Optional[Dict[str, Any]] = Field(None, description="Clinical intervals and HR metrics if available")

    model_config = ConfigDict(from_attributes=True)


class ECGExplainabilityRequest(BaseModel):
    """Optional parameters for the explainability endpoint."""
    target_class: Optional[str] = Field(None, description="Optional target superclass to explain (NORM, MI, STTC, CD, HYP). Defaults to model's top predicted class.")
    integration_steps: Optional[int] = Field(50, ge=10, le=200, description="Interpolation steps for Integrated Gradients (default: 50)")

    model_config = ConfigDict(from_attributes=True)
