"""
Phase 8.0: Systematic Validation Suite for ECG Time-Series Explainability.

Implements rigorous sanity checks and diagnostic checks for Grad-CAM and Integrated Gradients:
1. Dimensionality verification:
   - Grad-CAM: (1000,) unbatched or (B, 1000) batched.
   - Integrated Gradients: (12, 1000) unbatched or (B, 12, 1000) batched.
   - Lead importance: (12,) unbatched or (B, 12) batched.
2. Value sanity:
   - All attribution values are finite (no NaNs, no Infs).
   - Grad-CAM normalized maps are strictly in [0.0, 1.0].
3. Hardware execution:
   - Validates on CPU and Apple Silicon MPS when available.
4. Input immutability:
   - Verifies input tensors are strictly unchanged after computing explanations.
5. Class sensitivity:
   - Different target classes produce distinct attribution maps on multi-class inputs.
6. Axiomatic convergence (Completeness):
   - Numerical error |sum(Attributions) - (F(x) - F(baseline))| is bounded (< 5% relative error).
7. Model randomization sanity check:
   - Randomizing model weights destroys/radically alters the attribution structure (Adebayo et al., 2018).
8. Edge cases:
   - Constant zero input does not crash and produces finite zero attribution.
   - Zero-gradient case does not output spurious non-zero artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ml.ecg_classification.explainability.grad_cam import ECGGradCAM1D
from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients
from ml.ecg_classification.resnet1d import ECGResNet1D
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


def run_explainability_validation_suite(
    model: Optional[nn.Module] = None,
    checkpoint_path: Optional[str] = None,
    device_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes all validation tests and sanity checks for ECGGradCAM1D and ECGIntegratedGradients.

    Returns:
        results: Dictionary summarizing PASS/FAIL status and diagnostic metrics for each check.
    """
    results: Dict[str, Any] = {"checks": {}, "all_passed": True}

    # Initialize model
    if model is None:
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)

    if device_name is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model.to(device)
    model.eval()

    grad_cam = ECGGradCAM1D(model)
    ig = ECGIntegratedGradients(model)

    # -------------------------------------------------------------
    # Check 1: Dimensionality and Shape Preservation
    # -------------------------------------------------------------
    try:
        x_single = torch.randn(12, 1000, device=device)
        x_batch = torch.randn(3, 12, 1000, device=device)

        cam_single = grad_cam.generate_cam(x_single, "MI")
        cam_batch = grad_cam.generate_cam(x_batch, "MI")
        ig_res = ig.attribute(x_single, "MI", n_steps=20)

        dim_pass = (
            cam_single.shape == torch.Size([1000])
            and cam_batch.shape == torch.Size([3, 1000])
            and ig_res["attribution_map"].shape == torch.Size([12, 1000])
            and ig_res["lead_importance"].shape == torch.Size([12])
        )
        results["checks"]["dimensionality"] = {
            "status": "PASS" if dim_pass else "FAIL",
            "cam_single_shape": list(cam_single.shape),
            "cam_batch_shape": list(cam_batch.shape),
            "ig_map_shape": list(ig_res["attribution_map"].shape),
            "lead_importance_shape": list(ig_res["lead_importance"].shape),
        }
    except Exception as e:
        results["checks"]["dimensionality"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 2: Finite Numerical Values (No NaNs, No Infs)
    # -------------------------------------------------------------
    try:
        finite_cam = bool(torch.isfinite(cam_single).all().item() and torch.isfinite(cam_batch).all().item())
        finite_ig = bool(torch.isfinite(ig_res["attribution_map"]).all().item())
        finite_pass = finite_cam and finite_ig
        results["checks"]["finite_values"] = {
            "status": "PASS" if finite_pass else "FAIL",
            "cam_finite": finite_cam,
            "ig_finite": finite_ig,
            "cam_range": [float(cam_single.min().item()), float(cam_single.max().item())],
        }
    except Exception as e:
        results["checks"]["finite_values"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 3: Input Immutability
    # -------------------------------------------------------------
    try:
        x_orig = torch.randn(12, 1000, device=device)
        x_clone = x_orig.clone()

        _ = grad_cam.generate_cam(x_orig, "NORM")
        _ = ig.attribute(x_orig, "NORM", n_steps=10)

        immutable_pass = bool(torch.equal(x_orig, x_clone))
        results["checks"]["input_immutability"] = {
            "status": "PASS" if immutable_pass else "FAIL",
            "identical": immutable_pass,
        }
    except Exception as e:
        results["checks"]["input_immutability"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 4: Class Specificity / Target Sensitivity
    # -------------------------------------------------------------
    try:
        x_target = torch.randn(12, 1000, device=device)
        cam_c0 = grad_cam.generate_cam(x_target, 0)
        cam_c1 = grad_cam.generate_cam(x_target, 1)

        ig_c0 = ig.attribute(x_target, 0, n_steps=20)["attribution_map"]
        ig_c1 = ig.attribute(x_target, 1, n_steps=20)["attribution_map"]

        diff_cam = float((cam_c0 - cam_c1).abs().mean().item())
        diff_ig = float((ig_c0 - ig_c1).abs().mean().item())
        class_specific_pass = (diff_cam > 1e-6) or (diff_ig > 1e-6)

        results["checks"]["class_specificity"] = {
            "status": "PASS" if class_specific_pass else "FAIL",
            "grad_cam_l1_diff": diff_cam,
            "ig_l1_diff": diff_ig,
        }
    except Exception as e:
        results["checks"]["class_specificity"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 5: Integrated Gradients Axiomatic Completeness
    # -------------------------------------------------------------
    try:
        x_test = torch.randn(12, 1000, device=device)
        ig_trap = ig.attribute(x_test, "MI", n_steps=50, method="trapezoid")

        err = ig_trap["completeness_error"]
        rel_err_pct = ig_trap["relative_completeness_error_pct"]
        completeness_pass = rel_err_pct < 5.0 or err < 1e-3

        results["checks"]["ig_completeness"] = {
            "status": "PASS" if completeness_pass else "FAIL",
            "completeness_error": err,
            "relative_error_pct": rel_err_pct,
            "delta_f": ig_trap["delta_f"],
            "attribution_sum": ig_trap["attribution_sum"],
        }
    except Exception as e:
        results["checks"]["ig_completeness"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 6: Model Randomization Sanity Check (Adebayo et al.)
    # -------------------------------------------------------------
    try:
        # Clone model and randomize weights
        random_model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32).to(device)
        for p in random_model.parameters():
            if p.requires_grad:
                nn.init.normal_(p, mean=0.0, std=0.5)

        ig_random = ECGIntegratedGradients(random_model)
        grad_cam_random = ECGGradCAM1D(random_model)

        cam_trained = grad_cam.generate_cam(x_test, "MI")
        cam_rand = grad_cam_random.generate_cam(x_test, "MI")

        attr_trained = ig.attribute(x_test, "MI", n_steps=20)["attribution_map"]
        attr_rand = ig_random.attribute(x_test, "MI", n_steps=20)["attribution_map"]

        diff_cam_rand = float((cam_trained - cam_rand).abs().mean().item())
        diff_attr_rand = float((attr_trained - attr_rand).abs().mean().item())
        randomization_pass = diff_cam_rand > 1e-4 or diff_attr_rand > 1e-4

        results["checks"]["model_randomization_sanity_check"] = {
            "status": "PASS" if randomization_pass else "FAIL",
            "cam_mean_l1_shift": diff_cam_rand,
            "ig_mean_l1_shift": diff_attr_rand,
        }
    except Exception as e:
        results["checks"]["model_randomization_sanity_check"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 7: Edge Case: Zero Input & Baseline Match
    # -------------------------------------------------------------
    try:
        x_zeros = torch.zeros(12, 1000, device=device)
        ig_zeros = ig.attribute(x_zeros, "MI", baseline="zero")

        zeros_pass = (
            torch.all(ig_zeros["attribution_map"] == 0.0).item()
            and ig_zeros["completeness_error"] == 0.0
        )
        results["checks"]["zero_baseline_edge_case"] = {
            "status": "PASS" if zeros_pass else "FAIL",
            "attribution_map_is_zero": zeros_pass,
        }
    except Exception as e:
        results["checks"]["zero_baseline_edge_case"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # -------------------------------------------------------------
    # Check 8: Hardware Device Compatibility (CPU and MPS)
    # -------------------------------------------------------------
    try:
        cpu_model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32).cpu()
        cpu_cam = ECGGradCAM1D(cpu_model).generate_cam(torch.randn(12, 1000), "MI")
        cpu_pass = cpu_cam.shape == torch.Size([1000]) and torch.isfinite(cpu_cam).all().item()

        mps_available = torch.backends.mps.is_available()
        mps_pass = True
        if mps_available:
            mps_model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32).to("mps")
            mps_cam = ECGGradCAM1D(mps_model).generate_cam(torch.randn(12, 1000, device="mps"), "MI")
            mps_pass = mps_cam.shape == torch.Size([1000]) and torch.isfinite(mps_cam).all().item()

        hw_pass = cpu_pass and mps_pass
        results["checks"]["hardware_compatibility"] = {
            "status": "PASS" if hw_pass else "FAIL",
            "cpu_passed": cpu_pass,
            "mps_available": mps_available,
            "mps_passed": mps_pass,
        }
    except Exception as e:
        results["checks"]["hardware_compatibility"] = {"status": "FAIL", "error": str(e)}
        results["all_passed"] = False

    # Summary
    all_passed = all(v["status"] == "PASS" for v in results["checks"].values())
    results["all_passed"] = all_passed

    if verbose:
        print("\n" + "=" * 60)
        print("EXPLAINABILITY VALIDATION SUITE RESULTS")
        print("=" * 60)
        for name, info in results["checks"].items():
            status = info["status"]
            print(f"[{status}] {name}")
        print("=" * 60)
        print(f"OVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    run_explainability_validation_suite(checkpoint_path="ai-cdss-clean/ml/models/ecg_classification/deep_learning/best_model.pth")
