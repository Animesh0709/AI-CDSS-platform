"""
FastAPI Router for Multimodal 3D Brain Tumor Segmentation & Uncertainty.
========================================================================
Endpoints:
  POST /api/segmentation/predict
  GET  /api/segmentation/eval-report
  GET  /api/segmentation/cases
  GET  /api/segmentation/studies/{study_id}
"""
from typing import List, Dict, Any, Optional
import os
import json
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.models.segmentation import SegmentationStudy
from app.schemas.segmentation import (
    SegmentationPredictionRequest,
    SegmentationPredictionResponse,
    VolumeQuantification,
    UncertaintyMetrics,
    SliceOverlayResponse
)
from app.services.segmentation import (
    run_3d_segmentation,
    get_available_cases,
    process_uploaded_mri,
    MODELS_DIR
)
from app.api.auth import get_current_user

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".nii", ".nii.gz"}
_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB



@router.get("/cases", response_model=List[str])
def list_available_cases(
    current_user: User = Depends(get_current_user)
):
    """Lists preprocessed multimodal 3D brain MRI scan subjects available for segmentation."""
    cases = get_available_cases()
    return cases


@router.get("/eval-report", response_model=Dict[str, Any])
def get_evaluation_report(
    current_user: User = Depends(get_current_user)
):
    """
    Returns the rigorous three-model benchmark evaluation report
    (3D U-Net, 3D Attention U-Net, 3D nnU-Net v2) on held-out test patients with
    Dice, HD95, sensitivity, specificity, bootstrap 95% CIs, and ECE calibration.
    """
    report_file = os.path.join(MODELS_DIR, "seg_eval_report.json")
    if not os.path.exists(report_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Segmentation evaluation report not generated yet."
        )
    try:
        with open(report_file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation report: {str(e)}"
        )


@router.get("/metadata", response_model=Dict[str, Any])
def get_model_metadata(
    current_user: User = Depends(get_current_user)
):
    """Returns segmentation model architectural metadata, input channels, and target subregions."""
    meta_file = os.path.join(MODELS_DIR, "seg_metadata.json")
    if not os.path.exists(meta_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Segmentation metadata not found."
        )
    with open(meta_file) as f:
        return json.load(f)


@router.post("/predict", response_model=SegmentationPredictionResponse)
def predict_3d_segmentation(
    request: SegmentationPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Runs 3D multimodal brain tumor segmentation using raw MRI voxel volumes:
    - 4 Modalities (FLAIR, T1, T1ce, T2)
    - Chosen architecture (3D U-Net, Attention U-Net, or nnU-Net v2)
    - Sub-region quantification in cm³ (Whole Tumor, Tumor Core, Enhancing Tumor)
    - Spatial uncertainty heatmap (predictive entropy) & calibration analysis
    - Multi-planar slice rendering (Axial, Coronal, Sagittal)
    """
    try:
        result = run_3d_segmentation(
            subject_identifier=request.subject_identifier or "UPENN-GBM-00030_11",
            model_name=request.model_name or "attunet3d",
            axial_slice=request.axial_slice or 78,
            coronal_slice=request.coronal_slice or 120,
            sagittal_slice=request.sagittal_slice or 120
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D Segmentation inference failed: {str(e)}"
        )

    # Persist study in database
    try:
        study = SegmentationStudy(
            patient_id=request.patient_id,
            doctor_id=current_user.id,
            subject_identifier=result["subject_identifier"],
            model_name=result["model_name"],
            whole_tumor_volume_cm3=result["volumes"]["whole_tumor_cm3"],
            tumor_core_volume_cm3=result["volumes"]["tumor_core_cm3"],
            enhancing_tumor_volume_cm3=result["volumes"]["enhancing_tumor_cm3"],
            edema_volume_cm3=result["volumes"]["edema_cm3"],
            mean_predictive_entropy=result["uncertainty"]["mean_predictive_entropy"],
            max_predictive_entropy=result["uncertainty"]["max_predictive_entropy"],
            uncertainty_level=result["uncertainty"]["confidence_level"],
            representative_slice_idx=request.axial_slice or 78,
            clinical_summary=result["clinical_interpretation"]
        )
        db.add(study)
        db.commit()
        db.refresh(study)
        result["study_id"] = study.id
    except Exception:
        db.rollback()
        # Non-blocking if persistence fails for demo/unassigned scans
        result["study_id"] = None

    return result


def _validate_segmentation_file(filename: str, file_size: int):
    lower = filename.lower()
    is_valid = any(lower.endswith(ext) for ext in _ALLOWED_EXTENSIONS)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported MRI file type for '{filename}'. Please upload a PNG, JPEG, BMP, TIFF, or NIfTI (.nii/.nii.gz) MRI scan."
        )
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded MRI scan file is empty."
        )
    if file_size > _MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file exceeds maximum limit of {_MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )


@router.post("/upload", response_model=Dict[str, Any])
async def upload_mri_scan(
    file: UploadFile = File(..., description="Brain MRI scan (PNG/JPEG/TIFF/BMP or NIfTI .nii/.nii.gz)"),
    current_user: User = Depends(get_current_user)
):
    """
    Uploads an MRI scan image or volumetric NIfTI file, standardizes it into a
    multimodal 3D volume, and registers it in available cases for segmentation.
    """
    content = await file.read()
    _validate_segmentation_file(file.filename or "uploaded_scan", len(content))

    try:
        subject_id = process_uploaded_mri(content, file.filename or "uploaded_scan")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and reconstruct uploaded MRI volume: {str(e)}"
        )

    return {
        "subject_identifier": subject_id,
        "filename": file.filename,
        "message": "MRI scan successfully uploaded and reconstructed into 3D volume."
    }


@router.post("/upload-and-predict", response_model=SegmentationPredictionResponse)
async def upload_and_predict_segmentation(
    file: UploadFile = File(..., description="Brain MRI scan (PNG/JPEG/TIFF/BMP or NIfTI .nii/.nii.gz)"),
    model_name: str = Form("attunet3d"),
    axial_slice: int = Form(78),
    coronal_slice: int = Form(120),
    sagittal_slice: int = Form(120),
    patient_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    One-step endpoint: Uploads an MRI scan image, generates pseudo-3D or extracts
    4D volume, runs chosen 3D deep segmentation model, quantifies sub-region volumes,
    and returns multi-planar visualization overlays and spatial uncertainty.
    """
    content = await file.read()
    _validate_segmentation_file(file.filename or "uploaded_scan", len(content))

    try:
        subject_id = process_uploaded_mri(content, file.filename or "uploaded_scan")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and reconstruct uploaded MRI volume: {str(e)}"
        )

    try:
        result = run_3d_segmentation(
            subject_identifier=subject_id,
            model_name=model_name or "attunet3d",
            axial_slice=axial_slice or 78,
            coronal_slice=coronal_slice or 120,
            sagittal_slice=sagittal_slice or 120
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D Segmentation inference failed on uploaded scan: {str(e)}"
        )

    # Persist study in database
    try:
        study = SegmentationStudy(
            patient_id=patient_id,
            doctor_id=current_user.id,
            subject_identifier=result["subject_identifier"],
            model_name=result["model_name"],
            whole_tumor_volume_cm3=result["volumes"]["whole_tumor_cm3"],
            tumor_core_volume_cm3=result["volumes"]["tumor_core_cm3"],
            enhancing_tumor_volume_cm3=result["volumes"]["enhancing_tumor_cm3"],
            edema_volume_cm3=result["volumes"]["edema_cm3"],
            mean_predictive_entropy=result["uncertainty"]["mean_predictive_entropy"],
            max_predictive_entropy=result["uncertainty"]["max_predictive_entropy"],
            uncertainty_level=result["uncertainty"]["confidence_level"],
            representative_slice_idx=axial_slice or 78,
            clinical_summary=result["clinical_interpretation"]
        )
        db.add(study)
        db.commit()
        db.refresh(study)
        result["study_id"] = study.id
    except Exception:
        db.rollback()
        result["study_id"] = None

    return result

