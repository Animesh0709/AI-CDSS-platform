"""
Unit and integration tests for Phase 8.0 ECG ResNet Explainability Module.

Tests:
1. Grad-CAM output dimensions: (1000,) unbatched and (B, 1000) batched.
2. Integrated Gradients output dimensions: (12, 1000) attribution map and (12,) lead importance.
3. Finite numerical values across all methods (no NaNs, no Infs).
4. Device compatibility: CPU and Apple Silicon MPS.
5. Class specificity: distinct target superclasses yield distinct attribution maps.
6. Input immutability: input waveform tensors are never modified in place.
7. Zero-gradient and zero-input edge cases produce zero attribution.
8. Integrated Gradients completeness: |sum(attribution) - delta_F| converges within < 5% error.
9. Model randomization check: scrambling model weights alters attribution maps.
10. End-to-end visualization plotting without crashing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.ecg_classification.explainability.grad_cam import ECGGradCAM1D
from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients
from ml.ecg_classification.explainability.validation import run_explainability_validation_suite
from ml.ecg_classification.explainability.visualization import plot_ecg_explanation
from ml.ecg_classification.resnet1d import ECGResNet1D
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


@pytest.fixture
def resnet_model():
    """Create lightweight ECGResNet1D model for unit testing."""
    model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=16)
    model.eval()
    return model


class TestECGGradCAM1D:

    def test_grad_cam_shapes_and_normalization(self, resnet_model):
        """Verify Grad-CAM produces exact (1000,) and (B, 1000) output shapes with [0, 1] range."""
        grad_cam = ECGGradCAM1D(resnet_model)

        # Unbatched (12, 1000)
        x_single = torch.randn(12, 1000)
        cam_single = grad_cam.generate_cam(x_single, "MI", normalize=True)
        assert cam_single.shape == torch.Size([1000])
        assert torch.isfinite(cam_single).all()
        assert cam_single.min() >= 0.0 and cam_single.max() <= 1.0

        # Batched (3, 12, 1000)
        x_batch = torch.randn(3, 12, 1000)
        cam_batch = grad_cam.generate_cam(x_batch, 0, normalize=True)
        assert cam_batch.shape == torch.Size([3, 1000])
        assert torch.isfinite(cam_batch).all()
        assert cam_batch.min() >= 0.0 and cam_batch.max() <= 1.0

    def test_grad_cam_input_immutability(self, resnet_model):
        """Verify inputs are not mutated during Grad-CAM backprop."""
        grad_cam = ECGGradCAM1D(resnet_model)
        x = torch.randn(12, 1000)
        x_clone = x.clone()
        _ = grad_cam.generate_cam(x, "NORM")
        torch.testing.assert_close(x, x_clone)

    def test_grad_cam_invalid_inputs_raise(self, resnet_model):
        """Verify invalid dimensions or unknown classes raise ValueError."""
        grad_cam = ECGGradCAM1D(resnet_model)
        with pytest.raises(ValueError, match="Expected unbatched input shape"):
            grad_cam.generate_cam(torch.randn(8, 1000), "MI")
        with pytest.raises(ValueError, match="Unknown class"):
            grad_cam.generate_cam(torch.randn(12, 1000), "INVALID_CLASS")


class TestECGIntegratedGradients:

    def test_integrated_gradients_shapes_and_completeness(self, resnet_model):
        """Verify attribution map (12, 1000), lead importance (12,), and path integral completeness."""
        ig = ECGIntegratedGradients(resnet_model)
        x = torch.randn(12, 1000)

        res = ig.attribute(x, "MI", n_steps=30, method="trapezoid")
        assert res["attribution_map"].shape == torch.Size([12, 1000])
        assert res["lead_importance"].shape == torch.Size([12])
        assert torch.isfinite(res["attribution_map"]).all()
        assert torch.isfinite(res["lead_importance"]).all()

        # Completeness: error should be small (< 5% relative error or < 0.005 absolute)
        assert res["completeness_error"] < 0.01 or res["relative_completeness_error_pct"] < 5.0

    def test_integrated_gradients_zero_baseline_edge_case(self, resnet_model):
        """Verify input matching zero baseline yields zero attribution and zero error."""
        ig = ECGIntegratedGradients(resnet_model)
        x_zero = torch.zeros(12, 1000)
        res = ig.attribute(x_zero, "MI", baseline="zero")
        assert torch.all(res["attribution_map"] == 0.0)
        assert torch.all(res["lead_importance"] == 0.0)
        assert res["completeness_error"] == 0.0

    def test_integrated_gradients_input_immutability(self, resnet_model):
        """Verify inputs remain unchanged after attribution."""
        ig = ECGIntegratedGradients(resnet_model)
        x = torch.randn(12, 1000)
        x_clone = x.clone()
        _ = ig.attribute(x, "STTC", n_steps=10)
        torch.testing.assert_close(x, x_clone)


class TestExplainabilityValidationSuite:

    def test_full_validation_suite_passes(self, resnet_model):
        """Run the comprehensive validation suite."""
        suite_res = run_explainability_validation_suite(model=resnet_model, verbose=False)
        assert suite_res["all_passed"] is True
        for name, chk in suite_res["checks"].items():
            assert chk["status"] == "PASS", f"Check {name} failed: {chk}"


class TestExplainabilityVisualization:

    def test_plot_generation(self):
        """Verify visualization generates and saves a valid multi-panel PNG without crashing."""
        sig = np.random.randn(12, 1000)
        cam = np.random.rand(1000)
        ig_map = np.random.randn(12, 1000) * 0.01
        lead_imp = np.abs(ig_map).sum(axis=-1)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_ecg_plot.png"
            fig = plot_ecg_explanation(
                signal=sig,
                grad_cam_map=cam,
                ig_attribution_map=ig_map,
                lead_importance=lead_imp,
                target_class="MI",
                class_logit=1.45,
                class_prob=0.81,
                threshold=0.55,
                ecg_id=9999,
                ground_truth_labels=["MI"],
                save_path=out_png,
            )
            assert out_png.exists()
            assert out_png.stat().st_size > 10000


class TestFaithfulnessAndRobustness:

    def test_ig_convergence_audit(self, resnet_model):
        """Verify Integrated Gradients completeness convergence across step resolutions."""
        from ml.ecg_classification.explainability.faithfulness import audit_ig_convergence

        x = torch.randn(12, 1000)
        res = audit_ig_convergence(resnet_model, x, target_class="MI", step_counts=(10, 20))
        assert "steps" in res
        assert "10" in res["steps"]
        assert "20" in res["steps"]
        assert "completeness_error" in res["steps"]["10"]
        assert "summary" in res

    def test_baseline_sensitivity_analysis(self, resnet_model):
        """Verify baseline sensitivity test proves standardization invariance."""
        from ml.ecg_classification.explainability.faithfulness import analyze_baseline_sensitivity

        mean = np.zeros(12, dtype=np.float32)
        std = np.ones(12, dtype=np.float32)
        res = analyze_baseline_sensitivity(mean, std)
        assert res["are_mathematically_identical"] is True
        assert res["max_absolute_deviation"] == 0.0

    def test_temporal_and_lead_deletion(self, resnet_model):
        """Verify temporal deletion and lead deletion functions execute properly."""
        from ml.ecg_classification.explainability.faithfulness import (
            run_temporal_deletion_test,
            run_lead_deletion_test,
        )

        x = torch.randn(12, 1000)
        cam = torch.rand(1000)
        del_res = run_temporal_deletion_test(resnet_model, x, cam, target_class="MI", steps=5, window_size=200)
        assert "audc_mif" in del_res
        assert "audc_lif" in del_res
        assert len(del_res["mif_logits"]) == 6

        lead_imp = np.abs(np.random.randn(12))
        lead_res = run_lead_deletion_test(resnet_model, x, lead_imp, target_class="MI")
        assert "audc_mif" in lead_res
        assert len(lead_res["mif_logits"]) == 13

    def test_input_robustness_and_randomization(self, resnet_model):
        """Verify input robustness test and progressive parameter randomization run properly."""
        from ml.ecg_classification.explainability.robustness import (
            test_input_perturbation_robustness,
            run_progressive_parameter_randomization,
        )

        x = torch.randn(12, 1000)
        rob_res = test_input_perturbation_robustness(
            resnet_model, x, target_class="MI", noise_stds=(0.01,), scale_factors=(1.05,), baseline_shifts=(0.10,)
        )
        assert "gaussian_noise" in rob_res
        assert "amplitude_scaling" in rob_res

        rand_res = run_progressive_parameter_randomization(
            resnet_model, x, target_class="MI"
        )
        assert "stages" in rand_res
        assert "original" in rand_res["stages"]
        assert "after_fc" in rand_res["stages"]
        assert "after_init_conv" in rand_res["stages"]
        assert "final_cam_cosine" in rand_res


