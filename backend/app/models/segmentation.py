"""
SQLAlchemy Model for Multimodal 3D Brain Tumor Segmentation Studies.
===================================================================
Maintains records of 3D MRI segmentation runs, including:
- 4-Modality inputs (FLAIR, T1, T1ce, T2)
- Multi-model segmentations (3D U-Net, Attention U-Net, nnU-Net v2)
- Sub-tumor region volumes in cm³ (WT, TC, ET)
- Spatial uncertainty statistics and calibration metrics
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text

from app.database.session import Base


class SegmentationStudy(Base):
    __tablename__ = "segmentation_studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Patient scan metadata
    subject_identifier = Column(String(128), nullable=False, default="ANONYMOUS_SCAN")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Active model used
    model_name = Column(String(64), nullable=False, default="attunet3d")  # unet3d, attunet3d, nnunet

    # Physical Volume Quantification (cm³)
    whole_tumor_volume_cm3 = Column(Float, nullable=False, default=0.0)
    tumor_core_volume_cm3 = Column(Float, nullable=False, default=0.0)
    enhancing_tumor_volume_cm3 = Column(Float, nullable=False, default=0.0)
    edema_volume_cm3 = Column(Float, nullable=False, default=0.0)

    # Spatial Uncertainty Quantification
    mean_predictive_entropy = Column(Float, nullable=True)
    max_predictive_entropy = Column(Float, nullable=True)
    uncertainty_level = Column(String(32), nullable=True)  # Low, Moderate, High

    # Comparative Consensus & Metrics across the 3 models
    unet_wt_vol_cm3 = Column(Float, nullable=True)
    attunet_wt_vol_cm3 = Column(Float, nullable=True)
    nnunet_wt_vol_cm3 = Column(Float, nullable=True)

    # Slice Index representative view
    representative_slice_idx = Column(Integer, nullable=True, default=78)

    # JSON or serialized summary notes
    clinical_summary = Column(Text, nullable=True)
