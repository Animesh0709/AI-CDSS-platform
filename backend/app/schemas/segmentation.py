"""
Pydantic Schemas for 3D Brain Tumor Segmentation & Physical Volumetrics.
========================================================================
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VolumeQuantification(BaseModel):
    whole_tumor_cm3: float = Field(..., description="Whole Tumor (WT = ED + NET + ET) volume in cm³")
    tumor_core_cm3: float = Field(..., description="Tumor Core (TC = NET + ET) volume in cm³")
    enhancing_tumor_cm3: float = Field(..., description="Enhancing Tumor (ET) active volume in cm³")
    edema_cm3: float = Field(..., description="Peritumoral Edema (ED = WT - TC) volume in cm³")


class UncertaintyMetrics(BaseModel):
    mean_predictive_entropy: float
    max_predictive_entropy: float
    epistemic_variance: float
    calibration_ece: float
    confidence_level: str  # "High Confidence", "Moderate", "High Ambiguity"


class SliceOverlayResponse(BaseModel):
    plane: str  # "axial", "coronal", "sagittal"
    slice_index: int
    total_slices: int
    raw_mri_base64: str
    overlay_mask_base64: str
    uncertainty_map_base64: str


class SegmentationPredictionResponse(BaseModel):
    study_id: Optional[int] = None
    subject_identifier: str
    model_name: str
    inference_latency_ms: float
    volumes: VolumeQuantification
    uncertainty: UncertaintyMetrics
    slices: Dict[str, SliceOverlayResponse]  # axial, coronal, sagittal
    comparative_summary: Dict[str, Any]
    clinical_interpretation: str


class SegmentationPredictionRequest(BaseModel):
    subject_identifier: Optional[str] = "UPENN-GBM-00030_11"
    model_name: Optional[str] = "attunet3d"  # unet3d, attunet3d, nnunet
    axial_slice: Optional[int] = 78
    coronal_slice: Optional[int] = 120
    sagittal_slice: Optional[int] = 120
    patient_id: Optional[int] = None
