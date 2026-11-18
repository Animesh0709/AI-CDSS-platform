"""
Unit tests for ECG Explainability-to-Image Alignment and Confidence-Gating (Refined Phase 8.2.1).
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.services.ecg.explainability_alignment import (
    AlignmentStatus,
    AttributionStatus,
    CalibrationStatus,
    ECGExplainabilityImageAligner,
    ExplainabilityImageAlignmentBundle,
    GeometryStatus,
    ImageBoundingBox,
    ReconstructionStatus,
    UncertaintyReason,
)
from app.services.ecg.image_processing import ALL_12_LEADS, CONVENTIONAL_3X4_LEADS
from ml.ecg_classification.explainability.alignment.alignment_visualization import (
    plot_explainability_image_alignment,
)


@pytest.fixture
def aligner():
    return ECGExplainabilityImageAligner()


@pytest.fixture
def synthetic_sample_image_path():
    p = Path("ml/data/ecg/sample_synthetic_12lead.png")
    if not p.exists():
        pytest.skip(f"Synthetic image not found at {p}")
    return p


class TestExplainabilityAlignmentSemantics:

    def test_independent_confidence_dimensions(self, aligner):
        """Verify the 5 status fields are maintained independently."""
        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": True,
            "signals": {l: np.zeros(250) for l in ALL_12_LEADS},
            "lead_metadata": {
                l: {"confidence_score": 0.95, "missing_pixel_fraction": 0.0, "clipped_boundary_fraction": 0.0}
                for l in ALL_12_LEADS
            },
            "rhythm_strip": np.zeros(1000),
            "rhythm_metadata": {"confidence_score": 0.94},
        }
        dummy_ig = {l: 1.0 for l in ALL_12_LEADS}

        bundle = aligner.align_explanations(
            image_shape=(1065, 1665),
            reconstruction_result=dummy_rec,
            ig_lead_importance=dummy_ig,
            target_class="MI",
            target_logit=1.8,
            target_probability=0.86,
        )

        l_align = bundle.lead_alignments["I"]
        assert hasattr(l_align, "geometry_status")
        assert hasattr(l_align, "reconstruction_status")
        assert hasattr(l_align, "calibration_status")
        assert hasattr(l_align, "attribution_status")
        assert hasattr(l_align, "alignment_status")

        assert l_align.geometry_status in [GeometryStatus.VERIFIED, GeometryStatus.ESTIMATED_STANDARD_LAYOUT]
        assert l_align.reconstruction_status == ReconstructionStatus.CONFIDENT
        assert l_align.calibration_status == CalibrationStatus.VERIFIED
        assert l_align.attribution_status == AttributionStatus.AVAILABLE
        assert l_align.alignment_status == AlignmentStatus.CONFIDENT

    def test_reliable_geometry_with_unknown_calibration(self, aligner):
        """Verify uncalibrated image maintains valid geometry but flags calibration and alignment UNCERTAIN."""
        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": False,  # UNVERIFIED
            "signals": {l: np.zeros(250) for l in ALL_12_LEADS},
            "lead_metadata": {
                l: {"confidence_score": 0.95, "missing_pixel_fraction": 0.0, "clipped_boundary_fraction": 0.0}
                for l in ALL_12_LEADS
            },
            "rhythm_strip": np.zeros(1000),
            "rhythm_metadata": {"confidence_score": 0.95},
        }
        dummy_ig = {l: 1.0 for l in ALL_12_LEADS}

        bundle = aligner.align_explanations(
            image_shape=(1065, 1665),
            reconstruction_result=dummy_rec,
            ig_lead_importance=dummy_ig,
            target_class="NORM",
            target_logit=1.0,
            target_probability=0.73,
        )

        l_align = bundle.lead_alignments["II"]
        # Geometry is NOT invalidated merely because calibration pulse is missing
        assert l_align.geometry_status != GeometryStatus.INVALID_OR_UNKNOWN
        assert l_align.image_bounding_box is not None
        assert l_align.image_bounding_box.width > 300

        # Calibration is explicitly unverified pixel domain
        assert l_align.calibration_status == CalibrationStatus.UNVERIFIED_PIXEL_DOMAIN
        assert l_align.voltage_units == "unverified_pixels"

        # Overall alignment status is cautiously UNCERTAIN
        assert l_align.alignment_status == AlignmentStatus.UNCERTAIN
        assert UncertaintyReason.UNCALIBRATED_IMAGE in l_align.uncertainty_reasons

    def test_missing_signal_vs_low_confidence(self, aligner):
        """Verify missing lead is UNAVAILABLE while low-confidence lead is UNCERTAIN."""
        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": True,
            "signals": {
                "I": np.zeros(250),
                "II": np.zeros(250),  # low confidence
                # V1 is completely missing
            },
            "lead_metadata": {
                "I": {"confidence_score": 0.95, "missing_pixel_fraction": 0.0, "clipped_boundary_fraction": 0.0},
                "II": {"confidence_score": 0.50, "missing_pixel_fraction": 0.0, "clipped_boundary_fraction": 0.0},
            },
            "rhythm_strip": np.zeros(1000),
            "rhythm_metadata": {"confidence_score": 0.90},
        }
        dummy_ig = {l: 1.0 for l in ALL_12_LEADS}

        bundle = aligner.align_explanations(
            image_shape=(1065, 1665),
            reconstruction_result=dummy_rec,
            ig_lead_importance=dummy_ig,
            target_class="MI",
            target_logit=1.2,
            target_probability=0.77,
        )

        # Lead I: Confident
        assert bundle.lead_alignments["I"].reconstruction_status == ReconstructionStatus.CONFIDENT
        assert bundle.lead_alignments["I"].alignment_status == AlignmentStatus.CONFIDENT

        # Lead II: Low confidence -> UNCERTAIN
        assert bundle.lead_alignments["II"].reconstruction_status == ReconstructionStatus.UNCERTAIN
        assert bundle.lead_alignments["II"].alignment_status == AlignmentStatus.UNCERTAIN
        assert UncertaintyReason.LOW_RECONSTRUCTION_CONFIDENCE in bundle.lead_alignments["II"].uncertainty_reasons

        # Lead V1: Missing -> UNAVAILABLE
        assert bundle.lead_alignments["V1"].reconstruction_status == ReconstructionStatus.UNAVAILABLE
        assert bundle.lead_alignments["V1"].alignment_status == AlignmentStatus.UNAVAILABLE
        assert UncertaintyReason.MISSING_LEAD_SIGNAL in bundle.lead_alignments["V1"].uncertainty_reasons

    def test_uncalibrated_measurements_disallow_physical_units(self, aligner):
        """Verify uncalibrated leads explicitly use pixel-domain unit labels."""
        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": False,
            "signals": {"I": np.zeros(250)},
            "lead_metadata": {
                "I": {"confidence_score": 0.85, "voltage_units": "unverified_pixels", "time_units": "seconds_approx_layout"}
            },
        }
        bundle = aligner.align_explanations(
            image_shape=(1065, 1665),
            reconstruction_result=dummy_rec,
            ig_lead_importance={"I": 1.0},
            target_class="MI",
            target_logit=1.0,
            target_probability=0.73,
        )
        l_i = bundle.lead_alignments["I"]
        assert l_i.voltage_units != "millivolts"
        assert "pixel" in l_i.voltage_units.lower()

    def test_semantic_model_prediction_fields(self, aligner):
        """Verify model fields use precise semantic terminology (never clinical certainty)."""
        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": True,
            "signals": {l: np.zeros(250) for l in ALL_12_LEADS},
            "lead_metadata": {l: {"confidence_score": 0.95} for l in ALL_12_LEADS},
            "rhythm_strip": np.zeros(1000),
            "rhythm_metadata": {"confidence_score": 0.95},
        }
        bundle = aligner.align_explanations(
            image_shape=(1065, 1665),
            reconstruction_result=dummy_rec,
            ig_lead_importance={l: 1.0 for l in ALL_12_LEADS},
            target_class="MI",
            target_logit=2.45,
            target_probability=0.92,
            classification_threshold=0.50,
        )

        assert hasattr(bundle, "raw_model_logit")
        assert hasattr(bundle, "uncalibrated_model_probability")
        assert hasattr(bundle, "model_decision")
        assert bundle.raw_model_logit == 2.45
        assert bundle.uncalibrated_model_probability == 0.92
        assert "ABOVE THRESHOLD" in bundle.model_decision

    def test_original_image_immutability(self, aligner, synthetic_sample_image_path):
        """Verify visualization and alignment never mutate the source image bytes."""
        img = Image.open(synthetic_sample_image_path).convert("RGB")
        orig_bytes = img.tobytes()

        dummy_rec = {
            "reconstruction_status": "SUCCESS",
            "is_calibrated": True,
            "signals": {l: np.zeros(250) for l in ALL_12_LEADS},
            "lead_metadata": {l: {"confidence_score": 0.95} for l in ALL_12_LEADS},
            "rhythm_strip": np.zeros(1000),
            "rhythm_metadata": {"confidence_score": 0.95},
        }
        bundle = aligner.align_explanations(
            image_shape=(img.height, img.width),
            reconstruction_result=dummy_rec,
            ig_lead_importance={l: 1.0 for l in ALL_12_LEADS},
            target_class="MI",
            target_logit=1.5,
            target_probability=0.82,
        )

        _ = plot_explainability_image_alignment(img, bundle, save_path=None)
        assert img.tobytes() == orig_bytes
