"""
ECG Explainability Orchestrator and Service Layer (Phase 8.3).

Coordinates:
- ECG image loading and SHA-256 provenance calculation.
- Waveform reconstruction and metadata caching.
- Shared 1D ResNet model loading and caching.
- Integrated Gradients attribution and completeness auditing.
- Exploratory 1D Grad-CAM profile generation.
- Dynamic lead-coordinate alignment and multi-dimensional confidence gating.
- Safe persistence of explainability JSON artifacts in dedicated patient directories.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.models.ecg import ECGRecord
from app.schemas.ecg_explainability import (
    AttributionMetadataOut,
    AuxiliaryGradCamGuidanceOut,
    ECGExplainabilityResponse,
    GlobalMetadataOut,
    ImageRegionCoordinatesOut,
    LeadAlignmentOut,
    PredictionMetadataOut,
)
from app.services.ecg.delineation import ECGWaveformDelineator
from app.services.ecg.explainability_alignment import (
    AlignmentStatus,
    AttributionStatus,
    CalibrationStatus,
    ECGExplainabilityImageAligner,
    ExplainabilityImageAlignmentBundle,
    GeometryStatus,
    ReconstructionStatus,
)
from app.services.ecg.image_processing import ALL_12_LEADS, ECGImageProcessor
from app.services.ecg.measurements import ECGMeasurementEngine
from app.services.ecg.reconstruction import ECGWaveformReconstructor
from ml.ecg_classification.explainability.grad_cam import ECGGradCAM1D
from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients
from ml.ecg_classification.resnet1d import ECGResNet1D
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES, STANDARD_12_LEADS

logger = logging.getLogger("ai_cdss.ecg.explainability_service")


class ECGModelRegistry:
    """Thread-safe singleton registry for shared 1D ResNet ECG classification model."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model: Optional[ECGResNet1D] = None
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self.thresholds: Dict[str, float] = {
            "NORM": 0.50,
            "MI": 0.50,
            "STTC": 0.50,
            "CD": 0.50,
            "HYP": 0.50,
        }
        self.checkpoint_version = "v1.0.3_deep_learning_resnet1d"
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> ECGModelRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        preprocessing_path: Optional[str] = None,
    ) -> None:
        """Load and cache the trained ECGResNet1D and normalization stats."""
        with self._lock:
            if self._is_loaded:
                return

            # Determine checkpoint paths.
            # Default paths are relative to the repo root (parent of backend/).
            # Using an anchor derived from __file__ makes the lookup CWD-independent.
            _repo_root = Path(__file__).resolve().parents[4]  # …/backend/app/services/ecg/ → repo root
            _default_ckpt = _repo_root / "ml" / "models" / "ecg_classification" / "deep_learning" / "best_model.pth"
            _default_prep = _repo_root / "ml" / "models" / "ecg_classification" / "deep_learning" / "preprocessing.json"
            ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_ckpt
            prep_path = Path(preprocessing_path) if preprocessing_path else _default_prep

            if not ckpt_path.exists():
                raise FileNotFoundError(f"ECG ResNet checkpoint not found at {ckpt_path}")
            if not prep_path.exists():
                raise FileNotFoundError(f"ECG preprocessing stats not found at {prep_path}")

            # Load checkpoint
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.model = model

            # Load normalization stats
            with open(prep_path) as f:
                prep = json.load(f)
            self.channel_mean = np.array(prep["channel_mean"], dtype=np.float32)
            self.channel_std = np.array(prep["channel_std"], dtype=np.float32)

            # Check for optimal operating thresholds in checkpoint if saved
            if "selected_thresholds" in ckpt:
                self.thresholds.update(ckpt["selected_thresholds"])

            self._is_loaded = True
            logger.info("ECGModelRegistry: Successfully loaded ECGResNet1D checkpoint from %s", ckpt_path)


class ECGExplainabilityService:
    """
    Orchestration service for generating and retrieving ECG explainability alignments.
    """

    def __init__(
        self,
        storage_base_dir: Optional[str] = None,
        aligner: Optional[ECGExplainabilityImageAligner] = None,
    ):
        base = storage_base_dir or getattr(settings, "UPLOAD_DIR", "uploads")
        self.storage_dir = Path(base) / "ecg_explainability"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.aligner = aligner or ECGExplainabilityImageAligner()
        self.processor = ECGImageProcessor()
        self.reconstructor = ECGWaveformReconstructor()
        self.delineator = ECGWaveformDelineator()
        self.measurement_engine = ECGMeasurementEngine()
        self.registry = ECGModelRegistry.get_instance()

    def _get_patient_storage_dir(self, patient_id: int) -> Path:
        """Secure isolated directory per patient to prevent cross-patient data exposure."""
        p_dir = self.storage_dir / f"patient_{patient_id}"
        p_dir.mkdir(parents=True, exist_ok=True)
        return p_dir

    def _compute_image_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of original image bytes for provenance."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def generate_explainability(
        self,
        ecg_record: ECGRecord,
        target_class: Optional[str] = None,
        integration_steps: int = 50,
    ) -> ECGExplainabilityResponse:
        """
        Orchestrate the full explainability pipeline for an authorized ECGRecord.
        """
        # Validate target class parameter first if provided
        if target_class is not None:
            if target_class not in DIAGNOSTIC_SUPERCLASSES:
                raise ValueError(
                    f"Unknown target class '{target_class}'. Valid classes: {DIAGNOSTIC_SUPERCLASSES}"
                )

        image_path = Path(ecg_record.file_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Source ECG image not found for record {ecg_record.id}")

        # 1. Provenance: Compute image hash without modifying source image
        image_hash = self._compute_image_sha256(image_path)
        img = Image.open(image_path)
        w, h = img.size

        # 2. Image Processing Audit & Reconstruction
        audit = self.processor.process_and_audit(image_path)
        rec = self.reconstructor.extract_traces(image_path, validation_audit=audit)

        if rec.get("reconstruction_status") == "FAILED":
            raise ValueError(
                f"ECG waveform reconstruction failed: {rec.get('failure_reasons', ['Unknown error'])}"
            )

        # 3. Model Loading & Inference
        self.registry.load_model()
        model = self.registry.model
        mean = self.registry.channel_mean
        std = self.registry.channel_std
        thresholds = self.registry.thresholds

        # Build 12-lead signal array of shape (12, 1000)
        sig_12lead = np.zeros((12, 1000), dtype=np.float32)
        signals = rec.get("signals", {})
        rhythm = rec.get("rhythm_strip")

        for l_idx, lead in enumerate(STANDARD_12_LEADS):
            if lead in signals and len(signals[lead]) > 0:
                lead_sig = signals[lead]
                if len(lead_sig) == 250:
                    sig_12lead[l_idx, :] = np.tile(lead_sig, 4)
                else:
                    sig_12lead[l_idx, :] = np.interp(
                        np.linspace(0, 1, 1000), np.linspace(0, 1, len(lead_sig)), lead_sig
                    )
            elif lead == "II" and rhythm is not None:
                sig_12lead[l_idx, :] = rhythm

        # Standardize input
        norm_sig = (sig_12lead - mean[:, None]) / std[:, None]
        x_tensor = torch.from_numpy(norm_sig).float()

        with torch.no_grad():
            out_logits = model(x_tensor.unsqueeze(0))[0]

        # Extract predictions for all 5 classes
        logits_dict = {}
        probs_dict = {}
        decisions_dict = {}
        for c_idx, c_name in enumerate(DIAGNOSTIC_SUPERCLASSES):
            l_val = float(out_logits[c_idx].item())
            p_val = float(torch.sigmoid(torch.tensor(l_val)).item())
            th = thresholds.get(c_name, 0.50)
            logits_dict[c_name] = round(l_val, 4)
            probs_dict[c_name] = round(p_val, 4)
            decisions_dict[c_name] = (
                f"POSITIVE (ABOVE THRESHOLD {th:.2f})" if p_val >= th else f"NEGATIVE (BELOW THRESHOLD {th:.2f})"
            )

        # Target class selection: use requested target or top-predicted class
        if target_class:
            if target_class not in DIAGNOSTIC_SUPERCLASSES:
                raise ValueError(
                    f"Unknown target class '{target_class}'. Valid classes: {DIAGNOSTIC_SUPERCLASSES}"
                )
            selected_target = target_class
        else:
            top_idx = int(torch.argmax(out_logits).item())
            selected_target = DIAGNOSTIC_SUPERCLASSES[top_idx]

        # 4. Integrated Gradients Lead Attribution
        ig = ECGIntegratedGradients(model)
        ig_res = ig.attribute(x_tensor, selected_target, n_steps=integration_steps)
        lead_imp_tensor = ig_res["lead_importance"].numpy()
        completeness_err = float(ig_res["completeness_error"])

        ig_scores: Dict[str, float] = {}
        for l_idx, lead in enumerate(STANDARD_12_LEADS):
            ig_scores[lead] = float(lead_imp_tensor[l_idx])

        # 5. Exploratory 1D Grad-CAM
        cam_engine = ECGGradCAM1D(model)
        cam_1d = cam_engine.generate_cam(x_tensor, class_target=selected_target).numpy()

        # 6. Alignment Engine
        bundle = self.aligner.align_explanations(
            image_shape=(h, w),
            reconstruction_result=rec,
            ig_lead_importance=ig_scores,
            target_class=selected_target,
            target_logit=logits_dict[selected_target],
            target_probability=probs_dict[selected_target],
            classification_threshold=thresholds.get(selected_target, 0.50),
            grad_cam_1d=cam_1d,
            validation_audit=audit,
        )

        # 6.5. Optional Clinical Measurements Extraction (Non-blocking)
        clinical_meas = None
        try:
            delineation_res = self.delineator.delineate_bundle(rec)
            meas_bundle = self.measurement_engine.measure_bundle(delineation_res)
            # Serialize for JSON schema compliance
            clinical_meas = {
                "measurement_status": meas_bundle.get("measurement_status"),
                "is_calibrated": meas_bundle.get("is_calibrated"),
                "global_rhythm": meas_bundle.get("global_rhythm"),
                "rhythm_strip_summary": meas_bundle.get("rhythm_strip", {}).get("summary_metrics") if meas_bundle.get("rhythm_strip") else None,
                "leads_summary": {
                    lead_k: lead_v.get("summary_metrics")
                    for lead_k, lead_v in meas_bundle.get("leads", {}).items()
                    if isinstance(lead_v, dict) and "summary_metrics" in lead_v
                },
            }
        except Exception as meas_ex:
            logger.warning("Clinical measurements generation encountered error (non-fatal): %s", meas_ex)
            clinical_meas = None

        now = datetime.utcnow()
        artifact_id = f"exp_ecg_{ecg_record.id}_{selected_target}_{now.strftime('%Y%m%d_%H%M%S')}"

        # 7. Build Pydantic Response Contract
        pred_meta = PredictionMetadataOut(
            model_name="1D_ResNet_PTBXL",
            model_version=self.registry.checkpoint_version,
            target_classes=list(DIAGNOSTIC_SUPERCLASSES),
            raw_model_logits=logits_dict,
            uncalibrated_model_probabilities=probs_dict,
            model_decisions=decisions_dict,
            thresholds_used=thresholds,
            queried_class=selected_target,
            prediction_timestamp=now,
        )

        attr_meta = AttributionMetadataOut(
            method="INTEGRATED_GRADIENTS",
            baseline="zero_standardized_space",
            integration_steps=integration_steps,
            completeness_error=round(completeness_err, 4),
            lead_importance={k: round(v, 4) for k, v in ig_scores.items()},
            attribution_status="AVAILABLE",
            artifact_id=artifact_id,
        )

        grad_cam_out = AuxiliaryGradCamGuidanceOut(
            is_exploratory_only=True,
            mapped_to_pixels=False,
            disclaimer=bundle.grad_cam_guidance["disclaimer"],
            temporal_points_available=bundle.grad_cam_guidance.get("temporal_points_available", 1000),
            peak_temporal_sec=bundle.grad_cam_guidance.get("peak_temporal_sec"),
        )

        global_meta = GlobalMetadataOut(
            ecg_id=ecg_record.id,
            patient_id=ecg_record.patient_id,
            source_image_filename=ecg_record.original_filename,
            source_image_hash=image_hash,
            layout_type="Conventional 3x4 + Lead II Rhythm Strip",
            calibration_verified=bundle.calibration_verified,
            processing_timestamp=now,
            overall_processing_status=(
                "COMPLETED" if bundle.confident_lead_count == 12 else "PARTIAL_UNCERTAIN"
            ),
            warnings=bundle.warnings,
        )

        lead_align_out: Dict[str, LeadAlignmentOut] = {}
        for l_name, l_data in bundle.lead_alignments.items():
            region = None
            if l_data.image_bounding_box:
                b = l_data.image_bounding_box
                region = ImageRegionCoordinatesOut(
                    x_min=b.x_min,
                    y_min=b.y_min,
                    x_max=b.x_max,
                    y_max=b.y_max,
                    width=b.width,
                    height=b.height,
                )

            lead_align_out[l_name] = LeadAlignmentOut(
                lead_name=l_name,
                geometry_status=l_data.geometry_status.value,
                reconstruction_status=l_data.reconstruction_status.value,
                calibration_status=l_data.calibration_status.value,
                attribution_status=l_data.attribution_status.value,
                alignment_status=l_data.alignment_status.value,
                image_region=region,
                reconstruction_confidence=l_data.reconstruction_confidence,
                attribution_importance=l_data.attribution_importance,
                attribution_rank=l_data.attribution_rank,
                uncertainty_reasons=[r.value for r in l_data.uncertainty_reasons],
                units_status=l_data.voltage_units,
                is_rhythm_strip=False,
                notes=l_data.notes,
            )

        rhythm_out = None
        if bundle.rhythm_strip_alignment:
            rh = bundle.rhythm_strip_alignment
            rh_region = None
            if rh.image_bounding_box:
                rb = rh.image_bounding_box
                rh_region = ImageRegionCoordinatesOut(
                    x_min=rb.x_min,
                    y_min=rb.y_min,
                    x_max=rb.x_max,
                    y_max=rb.y_max,
                    width=rb.width,
                    height=rb.height,
                )
            rhythm_out = LeadAlignmentOut(
                lead_name="II_rhythm",
                geometry_status=rh.geometry_status.value,
                reconstruction_status=rh.reconstruction_status.value,
                calibration_status=rh.calibration_status.value,
                attribution_status=rh.attribution_status.value,
                alignment_status=rh.alignment_status.value,
                image_region=rh_region,
                reconstruction_confidence=rh.reconstruction_confidence,
                attribution_importance=rh.attribution_importance,
                attribution_rank=rh.attribution_rank,
                uncertainty_reasons=[r.value for r in rh.uncertainty_reasons],
                units_status=rh.voltage_units,
                is_rhythm_strip=True,
                notes=rh.notes,
            )

        response = ECGExplainabilityResponse(
            global_metadata=global_meta,
            prediction=pred_meta,
            attribution=attr_meta,
            lead_alignments=lead_align_out,
            rhythm_strip_alignment=rhythm_out,
            auxiliary_grad_cam_guidance=grad_cam_out,
            confident_lead_count=bundle.confident_lead_count,
            uncertain_lead_count=bundle.uncertain_lead_count,
            unavailable_lead_count=bundle.unavailable_lead_count,
            overall_alignment_coverage_pct=bundle.overall_alignment_coverage_pct,
            clinical_measurements=clinical_meas,
        )

        # 8. Safe Persistence in Isolated Patient Directory
        patient_dir = self._get_patient_storage_dir(ecg_record.patient_id)
        artifact_path = patient_dir / f"{artifact_id}.json"
        with open(artifact_path, "w") as f:
            f.write(response.model_dump_json(indent=2))

        logger.info(
            "ECGExplainabilityService: Successfully generated explainability for record=%s, patient=%s, artifact=%s",
            ecg_record.id, ecg_record.patient_id, artifact_id
        )

        return response
