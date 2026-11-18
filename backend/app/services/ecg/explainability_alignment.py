"""
ECG Explainability-to-Image Alignment and Confidence-Gated Mapping Service.

Bridges waveform-level Integrated Gradients lead attribution and clinical reconstruction
confidence with original 12-lead ECG image coordinates.

Strict architectural, semantic, and clinical constraints (Refined Phase 8.2.1):
1. Independent confidence dimensions:
   - `geometry_status`: whether the lead's image quadrant/bounding-box is reliably identified.
   - `reconstruction_status`: whether the lead waveform reconstruction is reliable.
   - `calibration_status`: whether physical voltage (mV) and time (mm/s) calibration is verified.
   - `attribution_status`: whether valid model attribution score/ranking exists.
   - `alignment_status`: overall mapping status synthesizing the above dimensions.
2. Uncalibrated images allow cautious geometry outlines when layout is verified, but
   voltage/time physical claims are strictly marked UNVERIFIED.
3. Explicit heuristic threshold documentation:
   - >= 0.70: CONFIDENT reconstruction quality.
   - 0.20 <= c < 0.70: UNCERTAIN reconstruction quality.
   - < 0.20 or missing: UNAVAILABLE reconstruction quality.
4. Lead importance remains strictly a LEAD-LEVEL score; no pixel-level saliency is hallucinated.
5. Model prediction scores are explicitly termed `uncalibrated_model_probability`,
   `raw_model_logit`, and `model_decision` (never clinical probability or certainty).
6. Grad-CAM is strictly kept exploratory on 1D waveforms with explicit methodological disclaimers.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.services.ecg.image_processing import (
    ALL_12_LEADS,
    CONVENTIONAL_3X4_LEADS,
    ECGImageProcessor,
)

logger = logging.getLogger("ai_cdss.ecg.explainability_alignment")


class GeometryStatus(str, enum.Enum):
    """Status of image lead quadrant and bounding box identification."""
    VERIFIED = "VERIFIED"
    ESTIMATED_STANDARD_LAYOUT = "ESTIMATED_STANDARD_LAYOUT"
    INVALID_OR_UNKNOWN = "INVALID_OR_UNKNOWN"


class ReconstructionStatus(str, enum.Enum):
    """Status of numerical waveform extraction quality."""
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"


class CalibrationStatus(str, enum.Enum):
    """Status of physical unit calibration (gain in mm/mV, paper speed in mm/s)."""
    VERIFIED = "VERIFIED"
    UNVERIFIED_PIXEL_DOMAIN = "UNVERIFIED_PIXEL_DOMAIN"


class AttributionStatus(str, enum.Enum):
    """Status of model attribution scores."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class AlignmentStatus(str, enum.Enum):
    """Resulting overall status of the explainability lead-to-image mapping."""
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"


# Maintain backward compatibility alias
MappingStatus = AlignmentStatus


class UncertaintyReason(str, enum.Enum):
    """Explicit reason codes for uncertainty or unavailability."""
    NONE = "NONE"
    LOW_RECONSTRUCTION_CONFIDENCE = "LOW_RECONSTRUCTION_CONFIDENCE"
    MISSING_LEAD_SIGNAL = "MISSING_LEAD_SIGNAL"
    INVALID_LAYOUT_GEOMETRY = "INVALID_LAYOUT_GEOMETRY"
    UNCALIBRATED_IMAGE = "UNCALIBRATED_IMAGE"
    BOUNDARY_CLIPPING = "BOUNDARY_CLIPPING"
    EXCESSIVE_MISSING_PIXELS = "EXCESSIVE_MISSING_PIXELS"
    AMBIGUOUS_WAVEFORM_OWNERSHIP = "AMBIGUOUS_WAVEFORM_OWNERSHIP"


@dataclass
class ImageBoundingBox:
    """Bounding box in image pixel coordinates (x_min, y_min, x_max, y_max)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x_min": round(self.x_min, 1),
            "y_min": round(self.y_min, 1),
            "x_max": round(self.x_max, 1),
            "y_max": round(self.y_max, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
        }


@dataclass
class LeadExplanationAlignment:
    """Structured attribution, geometry, reconstruction, and calibration mapping for a single ECG lead."""
    lead_name: str
    attribution_importance: float
    attribution_rank: int
    reconstruction_confidence: float
    image_bounding_box: Optional[ImageBoundingBox]
    
    # 5 Independent Status Dimensions
    geometry_status: GeometryStatus
    reconstruction_status: ReconstructionStatus
    calibration_status: CalibrationStatus
    attribution_status: AttributionStatus
    alignment_status: AlignmentStatus

    # Backward compatibility field
    @property
    def mapping_status(self) -> MappingStatus:
        return self.alignment_status

    uncertainty_reasons: List[UncertaintyReason] = field(default_factory=list)
    lead_row: Optional[int] = None
    lead_col: Optional[int] = None
    is_rhythm_strip: bool = False
    voltage_units: str = "millivolts"
    time_units: str = "seconds"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_name": self.lead_name,
            "attribution_importance": round(self.attribution_importance, 4),
            "attribution_rank": self.attribution_rank,
            "reconstruction_confidence": round(self.reconstruction_confidence, 2),
            "image_bounding_box": self.image_bounding_box.to_dict() if self.image_bounding_box else None,
            "geometry_status": self.geometry_status.value,
            "reconstruction_status": self.reconstruction_status.value,
            "calibration_status": self.calibration_status.value,
            "attribution_status": self.attribution_status.value,
            "alignment_status": self.alignment_status.value,
            "mapping_status": self.alignment_status.value,  # backward compatibility
            "uncertainty_reasons": [r.value for r in self.uncertainty_reasons],
            "lead_row": self.lead_row,
            "lead_col": self.lead_col,
            "is_rhythm_strip": self.is_rhythm_strip,
            "voltage_units": self.voltage_units,
            "time_units": self.time_units,
            "notes": self.notes,
        }


@dataclass
class ExplainabilityImageAlignmentBundle:
    """Complete alignment bundle for an ECG image with explicit semantic dimensions."""
    target_class: str
    raw_model_logit: float
    uncalibrated_model_probability: float
    classification_threshold: float
    model_decision: str  # "POSITIVE (ABOVE THRESHOLD)" or "NEGATIVE"

    # Backward compatibility properties
    @property
    def target_logit(self) -> float:
        return self.raw_model_logit

    @property
    def target_probability(self) -> float:
        return self.uncalibrated_model_probability

    @property
    def is_target_predicted(self) -> bool:
        return "POSITIVE" in self.model_decision

    lead_alignments: Dict[str, LeadExplanationAlignment]
    rhythm_strip_alignment: Optional[LeadExplanationAlignment]
    total_leads_mapped: int
    confident_lead_count: int
    uncertain_lead_count: int
    unavailable_lead_count: int
    overall_alignment_coverage_pct: float
    grad_cam_guidance: Dict[str, Any]
    calibration_verified: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_class": self.target_class,
            "raw_model_logit": round(self.raw_model_logit, 4),
            "uncalibrated_model_probability": round(self.uncalibrated_model_probability, 4),
            "classification_threshold": round(self.classification_threshold, 4),
            "model_decision": self.model_decision,
            "target_logit": round(self.raw_model_logit, 4),  # compatibility
            "target_probability": round(self.uncalibrated_model_probability, 4),  # compatibility
            "is_target_predicted": self.is_target_predicted,  # compatibility
            "lead_alignments": {k: v.to_dict() for k, v in self.lead_alignments.items()},
            "rhythm_strip_alignment": self.rhythm_strip_alignment.to_dict() if self.rhythm_strip_alignment else None,
            "total_leads_mapped": self.total_leads_mapped,
            "confident_lead_count": self.confident_lead_count,
            "uncertain_lead_count": self.uncertain_lead_count,
            "unavailable_lead_count": self.unavailable_lead_count,
            "overall_alignment_coverage_pct": round(self.overall_alignment_coverage_pct, 1),
            "grad_cam_guidance": self.grad_cam_guidance,
            "calibration_verified": self.calibration_verified,
            "warnings": self.warnings,
        }


class ECGExplainabilityImageAligner:
    """
    Service for aligning waveform-level explanations with 12-lead ECG image regions.

    Threshold Documentation:
    - CONFIDENCE_THRESHOLD_RELIABLE = 0.70:
      Origin: Engineering heuristic calibrated during Phase 3 & 4 reconstruction benchmarking.
      Applies to: `reconstruction_confidence` score returned by ECGWaveformReconstructor.
      Validation: Heuristic threshold reflecting continuous trace morphology, minimal clipping, and plausible dynamic range.
      Not clinically validated.
    - CONFIDENCE_THRESHOLD_UNAVAILABLE = 0.20:
      Origin: Lower bound below which traces are severely degraded (excessive missing samples, catastrophic clipping).
      Applies to: `reconstruction_confidence` score.
      Not clinically validated.
    """

    CONFIDENCE_THRESHOLD_RELIABLE = 0.70
    CONFIDENCE_THRESHOLD_UNAVAILABLE = 0.20
    MAX_MISSING_PIXEL_FRACTION = 0.25
    MAX_CLIPPED_BOUNDARY_FRACTION = 0.15

    def __init__(
        self,
        confidence_threshold_reliable: float = CONFIDENCE_THRESHOLD_RELIABLE,
        confidence_threshold_unavailable: float = CONFIDENCE_THRESHOLD_UNAVAILABLE,
    ):
        self.confidence_threshold_reliable = confidence_threshold_reliable
        self.confidence_threshold_unavailable = confidence_threshold_unavailable

    def compute_lead_bounding_boxes(
        self,
        image_width: int,
        image_height: int,
        grid_detection: Optional[Dict[str, Any]] = None,
        layout_audit: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, ImageBoundingBox], GeometryStatus]:
        """
        Dynamically calculate standard 3x4 lead bounding boxes and Lead II rhythm strip box.
        Returns bounding boxes and explicit GeometryStatus.
        """
        w, h = float(image_width), float(image_height)
        geom_status = GeometryStatus.ESTIMATED_STANDARD_LAYOUT

        # Detect grid bounds
        if grid_detection and grid_detection.get("grid_detected", False):
            x_grid_start = float(grid_detection.get("x_grid_start", 0.005 * w))
            x_grid_end = float(grid_detection.get("x_grid_end", 0.995 * w))
            y_grid_top = float(grid_detection.get("y_grid_top", 0.008 * h))
            y_grid_bottom = float(grid_detection.get("y_grid_bottom", 0.992 * h))
            if layout_audit and layout_audit.get("is_layout_valid", False):
                geom_status = GeometryStatus.VERIFIED
        else:
            x_grid_start = 0.005 * w
            x_grid_end = 0.995 * w
            y_grid_top = 0.008 * h
            y_grid_bottom = 0.992 * h

        total_grid_width = max(100.0, x_grid_end - x_grid_start)
        total_grid_height = max(100.0, y_grid_bottom - y_grid_top)
        col_width = total_grid_width / 4.0

        # Row midpoints
        baseline_rows_y = [
            y_grid_top + (1.5 / 6.0) * total_grid_height,
            y_grid_top + (2.5 / 6.0) * total_grid_height,
            y_grid_top + (3.5 / 6.0) * total_grid_height,
        ]
        rhythm_baseline_y = y_grid_top + (4.5 / 6.0) * total_grid_height

        midpoint_0_1 = (baseline_rows_y[0] + baseline_rows_y[1]) / 2.0
        midpoint_1_2 = (baseline_rows_y[1] + baseline_rows_y[2]) / 2.0
        midpoint_2_rh = (baseline_rows_y[2] + rhythm_baseline_y) / 2.0

        boxes: Dict[str, ImageBoundingBox] = {}

        for row_idx in range(3):
            if row_idx == 0:
                y_min = max(0.0, baseline_rows_y[0] - 0.8 * (total_grid_height / 6.0))
                y_max = midpoint_0_1
            elif row_idx == 1:
                y_min = midpoint_0_1
                y_max = midpoint_1_2
            else:
                y_min = midpoint_1_2
                y_max = midpoint_2_rh

            for col_idx in range(4):
                lead_name = CONVENTIONAL_3X4_LEADS[row_idx][col_idx]
                x_min = x_grid_start + col_idx * col_width
                x_max = x_grid_start + (col_idx + 1) * col_width

                boxes[lead_name] = ImageBoundingBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                )

        # Rhythm strip
        boxes["II_rhythm"] = ImageBoundingBox(
            x_min=float(x_grid_start),
            y_min=float(midpoint_2_rh),
            x_max=float(x_grid_end),
            y_max=float(min(h, y_grid_bottom)),
        )

        return boxes, geom_status

    def align_explanations(
        self,
        image_shape: Tuple[int, int],  # (height, width)
        reconstruction_result: Dict[str, Any],
        ig_lead_importance: Dict[str, float],
        target_class: str,
        target_logit: float,
        target_probability: float,
        classification_threshold: float = 0.50,
        grad_cam_1d: Optional[np.ndarray] = None,
        validation_audit: Optional[Dict[str, Any]] = None,
    ) -> ExplainabilityImageAlignmentBundle:
        """
        Build the confidence-gated lead explanation alignment bundle with 5 distinct status dimensions.
        """
        h, w = image_shape
        warnings: List[str] = []

        is_calibrated = bool(reconstruction_result.get("is_calibrated", False))
        cal_status = CalibrationStatus.VERIFIED if is_calibrated else CalibrationStatus.UNVERIFIED_PIXEL_DOMAIN
        
        lead_meta = reconstruction_result.get("lead_metadata", {})
        signals = reconstruction_result.get("signals", {})

        grid_meta = validation_audit.get("grid_detection") if validation_audit else None
        layout_meta = validation_audit.get("layout_detection") if validation_audit else None

        # 1. Compute dynamic lead bounding boxes and geometry status
        bounding_boxes, default_geom_status = self.compute_lead_bounding_boxes(
            image_width=w,
            image_height=h,
            grid_detection=grid_meta,
            layout_audit=layout_meta,
        )

        # 2. Rank leads by attribution importance
        valid_ig_scores = {k: float(v) for k, v in ig_lead_importance.items() if k in ALL_12_LEADS}
        sorted_leads = sorted(valid_ig_scores.items(), key=lambda kv: kv[1], reverse=True)
        lead_ranks = {lead: rank for rank, (lead, _) in enumerate(sorted_leads, start=1)}

        lead_alignments: Dict[str, LeadExplanationAlignment] = {}
        confident_count = 0
        uncertain_count = 0
        unavailable_count = 0

        for r_idx in range(3):
            for c_idx in range(4):
                lead = CONVENTIONAL_3X4_LEADS[r_idx][c_idx]
                imp = valid_ig_scores.get(lead, 0.0)
                rank = lead_ranks.get(lead, 12)
                box = bounding_boxes.get(lead)

                meta = lead_meta.get(lead, {})
                conf = float(meta.get("confidence_score", 0.0))
                missing_frac = float(meta.get("missing_pixel_fraction", 0.0))
                clipped_frac = float(meta.get("clipped_boundary_fraction", 0.0))

                reasons: List[UncertaintyReason] = []
                
                # 1. Geometry Status
                geom_stat = default_geom_status
                if box is None or box.width <= 0 or box.height <= 0:
                    geom_stat = GeometryStatus.INVALID_OR_UNKNOWN
                    reasons.append(UncertaintyReason.INVALID_LAYOUT_GEOMETRY)

                # 2. Attribution Status
                attr_stat = AttributionStatus.AVAILABLE if lead in valid_ig_scores else AttributionStatus.UNAVAILABLE

                # 3. Reconstruction Status
                if lead not in signals or len(signals[lead]) == 0 or conf <= 0.0:
                    rec_stat = ReconstructionStatus.UNAVAILABLE
                    reasons.append(UncertaintyReason.MISSING_LEAD_SIGNAL)
                elif conf < self.confidence_threshold_unavailable:
                    rec_stat = ReconstructionStatus.UNAVAILABLE
                    reasons.append(UncertaintyReason.LOW_RECONSTRUCTION_CONFIDENCE)
                elif conf < self.confidence_threshold_reliable:
                    rec_stat = ReconstructionStatus.UNCERTAIN
                    reasons.append(UncertaintyReason.LOW_RECONSTRUCTION_CONFIDENCE)
                else:
                    rec_stat = ReconstructionStatus.CONFIDENT

                if missing_frac > self.MAX_MISSING_PIXEL_FRACTION:
                    if rec_stat == ReconstructionStatus.CONFIDENT:
                        rec_stat = ReconstructionStatus.UNCERTAIN
                    reasons.append(UncertaintyReason.EXCESSIVE_MISSING_PIXELS)

                if clipped_frac > self.MAX_CLIPPED_BOUNDARY_FRACTION:
                    if rec_stat == ReconstructionStatus.CONFIDENT:
                        rec_stat = ReconstructionStatus.UNCERTAIN
                    reasons.append(UncertaintyReason.BOUNDARY_CLIPPING)

                # 4. Calibration Status impact:
                # If calibration is unverified, geometry remains valid, but alignment is cautious (UNCERTAIN)
                # and physical claims are disallowed.
                if not is_calibrated:
                    reasons.append(UncertaintyReason.UNCALIBRATED_IMAGE)

                # 5. Overall Alignment Status Synthesis:
                if rec_stat == ReconstructionStatus.UNAVAILABLE or geom_stat == GeometryStatus.INVALID_OR_UNKNOWN:
                    align_stat = AlignmentStatus.UNAVAILABLE
                elif rec_stat == ReconstructionStatus.UNCERTAIN or cal_status == CalibrationStatus.UNVERIFIED_PIXEL_DOMAIN:
                    align_stat = AlignmentStatus.UNCERTAIN
                else:
                    align_stat = AlignmentStatus.CONFIDENT

                # Count tallies
                if align_stat == AlignmentStatus.CONFIDENT:
                    confident_count += 1
                elif align_stat == AlignmentStatus.UNCERTAIN:
                    uncertain_count += 1
                else:
                    unavailable_count += 1

                voltage_u = meta.get("voltage_units", "millivolts" if is_calibrated else "unverified_pixels")
                time_u = meta.get("time_units", "seconds" if is_calibrated else "seconds_approx_layout")

                lead_alignments[lead] = LeadExplanationAlignment(
                    lead_name=lead,
                    attribution_importance=imp,
                    attribution_rank=rank,
                    reconstruction_confidence=conf,
                    image_bounding_box=box,
                    geometry_status=geom_stat,
                    reconstruction_status=rec_stat,
                    calibration_status=cal_status,
                    attribution_status=attr_stat,
                    alignment_status=align_stat,
                    uncertainty_reasons=reasons if reasons else [UncertaintyReason.NONE],
                    lead_row=r_idx,
                    lead_col=c_idx,
                    is_rhythm_strip=False,
                    voltage_units=voltage_u,
                    time_units=time_u,
                    notes=(
                        f"Rank #{rank} for {target_class}. Alignment: {align_stat.value}. "
                        f"Geometry: {geom_stat.value}, Rec: {rec_stat.value}, Cal: {cal_status.value}."
                    ),
                )

        # 3. Handle Rhythm Strip
        rhythm_meta = reconstruction_result.get("rhythm_metadata", {})
        rhythm_box = bounding_boxes.get("II_rhythm")
        rhythm_conf = float(rhythm_meta.get("confidence_score", 0.0))
        rhythm_reasons: List[UncertaintyReason] = []

        rh_geom_stat = default_geom_status
        if rhythm_box is None:
            rh_geom_stat = GeometryStatus.INVALID_OR_UNKNOWN
            rhythm_reasons.append(UncertaintyReason.INVALID_LAYOUT_GEOMETRY)

        if "rhythm_strip" not in reconstruction_result or rhythm_conf <= 0.0:
            rh_rec_stat = ReconstructionStatus.UNAVAILABLE
            rhythm_reasons.append(UncertaintyReason.MISSING_LEAD_SIGNAL)
        elif rhythm_conf < self.confidence_threshold_reliable:
            rh_rec_stat = ReconstructionStatus.UNCERTAIN
            rhythm_reasons.append(UncertaintyReason.LOW_RECONSTRUCTION_CONFIDENCE)
        else:
            rh_rec_stat = ReconstructionStatus.CONFIDENT

        if not is_calibrated:
            rhythm_reasons.append(UncertaintyReason.UNCALIBRATED_IMAGE)

        if rh_rec_stat == ReconstructionStatus.UNAVAILABLE or rh_geom_stat == GeometryStatus.INVALID_OR_UNKNOWN:
            rh_align_stat = AlignmentStatus.UNAVAILABLE
        elif rh_rec_stat == ReconstructionStatus.UNCERTAIN or cal_status == CalibrationStatus.UNVERIFIED_PIXEL_DOMAIN:
            rh_align_stat = AlignmentStatus.UNCERTAIN
        else:
            rh_align_stat = AlignmentStatus.CONFIDENT

        lead_ii_imp = valid_ig_scores.get("II", 0.0)
        lead_ii_rank = lead_ranks.get("II", 12)

        rhythm_alignment = LeadExplanationAlignment(
            lead_name="II_rhythm",
            attribution_importance=lead_ii_imp,
            attribution_rank=lead_ii_rank,
            reconstruction_confidence=rhythm_conf,
            image_bounding_box=rhythm_box,
            geometry_status=rh_geom_stat,
            reconstruction_status=rh_rec_stat,
            calibration_status=cal_status,
            attribution_status=AttributionStatus.AVAILABLE if "II" in valid_ig_scores else AttributionStatus.UNAVAILABLE,
            alignment_status=rh_align_stat,
            uncertainty_reasons=rhythm_reasons if rhythm_reasons else [UncertaintyReason.NONE],
            lead_row=3,
            lead_col=0,
            is_rhythm_strip=True,
            voltage_units="millivolts" if is_calibrated else "unverified_pixels",
            time_units="seconds" if is_calibrated else "seconds_approx_layout",
            notes="Continuous 10.0s Lead II rhythm strip. Shares model attribution with Lead II segment.",
        )

        total_mapped = len(lead_alignments)
        coverage_pct = (confident_count / max(1, total_mapped)) * 100.0

        if not is_calibrated:
            warnings.append(
                "Image calibration is unverified. While lead geometry is mapped, voltage amplitude and time "
                "measurements are strictly pixel-domain offsets and must NOT be interpreted as physical mV or mm/s."
            )
        if uncertain_count > 0:
            warnings.append(f"{uncertain_count} leads exhibit low confidence, clipping, or unverified calibration and are flagged UNCERTAIN.")

        # 4. Grad-CAM strictly isolated
        grad_cam_guidance: Dict[str, Any] = {
            "is_exploratory_only": True,
            "mapped_to_pixels": False,
            "disclaimer": (
                "1D Grad-CAM failed temporal progressive deletion and model-randomization sanity checks in Phase 8.1. "
                "It is strictly provided as an exploratory temporal attention guide along the reconstructed 1D waveform, "
                "and MUST NOT be interpreted as confirmed localized lesion boundaries or pixel-level image evidence."
            ),
            "temporal_points_available": int(len(grad_cam_1d)) if grad_cam_1d is not None else 0,
            "peak_temporal_window_sec": None,
        }
        if grad_cam_1d is not None and len(grad_cam_1d) == 1000:
            peak_idx = int(np.argmax(grad_cam_1d))
            grad_cam_guidance["peak_temporal_sample"] = peak_idx
            grad_cam_guidance["peak_temporal_sec"] = round(peak_idx / 100.0, 3)

        decision_str = (
            f"POSITIVE (ABOVE THRESHOLD {classification_threshold:.2f})"
            if target_probability >= classification_threshold
            else f"NEGATIVE (BELOW THRESHOLD {classification_threshold:.2f})"
        )

        return ExplainabilityImageAlignmentBundle(
            target_class=target_class,
            raw_model_logit=target_logit,
            uncalibrated_model_probability=target_probability,
            classification_threshold=classification_threshold,
            model_decision=decision_str,
            lead_alignments=lead_alignments,
            rhythm_strip_alignment=rhythm_alignment,
            total_leads_mapped=total_mapped,
            confident_lead_count=confident_count,
            uncertain_lead_count=uncertain_count,
            unavailable_lead_count=unavailable_count,
            overall_alignment_coverage_pct=coverage_pct,
            grad_cam_guidance=grad_cam_guidance,
            calibration_verified=is_calibrated,
            warnings=warnings,
        )
