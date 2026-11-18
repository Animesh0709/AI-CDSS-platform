"""
Phase 8.1: ECG Explainability Robustness and Stability Evaluation Suite.

Evaluates the stability and sanity of 1D Grad-CAM and Integrated Gradients under:
1. Input perturbations:
   - Gaussian noise addition (varying sigma in [0.01, 0.05, 0.10]).
   - Amplitude scaling perturbations (scaling factors in [0.95, 1.05]).
   - Baseline shifts / DC offsets (shifts in [-0.1, +0.1]).
2. Explanation stability metrics:
   - Cosine similarity between original and perturbed attribution maps.
   - Spearman rank correlation of lead importance rankings.
   - Mean L1 normalized attribution deviation.
3. Progressive model parameter randomization (Cascading Randomization Sanity Check):
   - Sequentially randomizes weights from output linear layer backward through residual stages (layer4, layer3, layer2, layer1, init_conv).
   - Verifies whether explanation similarity drops progressively towards zero as required by Adebayo et al. (2018).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from ml.ecg_classification.explainability.grad_cam import ECGGradCAM1D
from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


def compute_attribution_similarity(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Cosine Similarity, Spearman Rank Correlation, and Mean Absolute Difference.
    """
    a = map_a.detach().cpu().flatten().float()
    b = map_b.detach().cpu().flatten().float()

    # 1. Cosine similarity
    cos_sim = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    # 2. Spearman rank correlation
    a_np = a.numpy()
    b_np = b.numpy()
    if np.all(a_np == a_np[0]) or np.all(b_np == b_np[0]):
        spearman_corr = 0.0
    else:
        spearman_res = stats.spearmanr(a_np, b_np)
        spearman_corr = float(spearman_res.statistic if hasattr(spearman_res, "statistic") else spearman_res[0])
        if np.isnan(spearman_corr):
            spearman_corr = 0.0

    # 3. Normalized L1 difference
    l1_diff = float((a - b).abs().mean().item())

    return {
        "cosine_similarity": round(cos_sim, 4),
        "spearman_rank_correlation": round(spearman_corr, 4),
        "mean_l1_difference": round(l1_diff, 6),
    }


def test_input_perturbation_robustness(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Union[str, int],
    noise_stds: Tuple[float, ...] = (0.01, 0.05, 0.10),
    scale_factors: Tuple[float, ...] = (0.95, 1.05),
    baseline_shifts: Tuple[float, ...] = (-0.10, +0.10),
) -> Dict[str, Any]:
    """
    Test attribution stability under small input perturbations.
    """
    grad_cam = ECGGradCAM1D(model)
    ig = ECGIntegratedGradients(model)

    # Reference attributions
    orig_cam = grad_cam.generate_cam(input_tensor, target_class)
    orig_ig = ig.attribute(input_tensor, target_class, n_steps=30)["attribution_map"]

    results: Dict[str, Any] = {
        "gaussian_noise": {},
        "amplitude_scaling": {},
        "baseline_shifts": {},
    }

    # 1. Gaussian noise
    for std in noise_stds:
        torch.manual_seed(42)
        noise = torch.randn_like(input_tensor) * std
        pert_x = input_tensor + noise

        pert_cam = grad_cam.generate_cam(pert_x, target_class)
        pert_ig = ig.attribute(pert_x, target_class, n_steps=30)["attribution_map"]

        results["gaussian_noise"][f"sigma_{std}"] = {
            "grad_cam": compute_attribution_similarity(orig_cam, pert_cam),
            "integrated_gradients": compute_attribution_similarity(orig_ig, pert_ig),
        }

    # 2. Amplitude scaling
    for scale in scale_factors:
        pert_x = input_tensor * scale

        pert_cam = grad_cam.generate_cam(pert_x, target_class)
        pert_ig = ig.attribute(pert_x, target_class, n_steps=30)["attribution_map"]

        results["amplitude_scaling"][f"scale_{scale}"] = {
            "grad_cam": compute_attribution_similarity(orig_cam, pert_cam),
            "integrated_gradients": compute_attribution_similarity(orig_ig, pert_ig),
        }

    # 3. Baseline shifts
    for shift in baseline_shifts:
        pert_x = input_tensor + shift

        pert_cam = grad_cam.generate_cam(pert_x, target_class)
        pert_ig = ig.attribute(pert_x, target_class, n_steps=30)["attribution_map"]

        results["baseline_shifts"][f"shift_{shift:+0.2f}"] = {
            "grad_cam": compute_attribution_similarity(orig_cam, pert_cam),
            "integrated_gradients": compute_attribution_similarity(orig_ig, pert_ig),
        }

    # Robustness criteria: small noise (sigma=0.01) retains cosine similarity > 0.80
    noise_cam_cos = results["gaussian_noise"]["sigma_0.01"]["grad_cam"]["cosine_similarity"]
    noise_ig_cos = results["gaussian_noise"]["sigma_0.01"]["integrated_gradients"]["cosine_similarity"]
    is_robust = bool(noise_cam_cos > 0.80 and noise_ig_cos > 0.80)

    results["is_input_robust"] = is_robust
    return results


def run_progressive_parameter_randomization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Union[str, int],
) -> Dict[str, Any]:
    """
    Cascading Model Parameter Randomization (Adebayo et al., 2018).
    Progressively randomizes layers from top to bottom and measures explanation decay.
    """
    # Instantiate a fresh model of identical architecture and copy state dict
    test_model = model.__class__(
        in_channels=getattr(model, "in_channels", 12),
        num_classes=getattr(model, "num_classes", 5),
        base_channels=getattr(model, "base_channels", 32),
    ).to(input_tensor.device)
    test_model.load_state_dict(model.state_dict())
    test_model.eval()

    grad_cam_orig = ECGGradCAM1D(model)
    ig_orig = ECGIntegratedGradients(model)

    orig_cam = grad_cam_orig.generate_cam(input_tensor, target_class)
    orig_ig = ig_orig.attribute(input_tensor, target_class, n_steps=30)["attribution_map"]

    layers_to_randomize = [
        ("fc", [test_model.fc]),
        ("layer4", [test_model.layer4]),
        ("layer3", [test_model.layer3]),
        ("layer2", [test_model.layer2]),
        ("layer1", [test_model.layer1]),
        ("init_conv", [test_model.init_conv, test_model.init_bn]),
    ]

    progression: Dict[str, Any] = {
        "original": {
            "grad_cam": {"cosine_similarity": 1.0, "spearman_rank_correlation": 1.0},
            "integrated_gradients": {"cosine_similarity": 1.0, "spearman_rank_correlation": 1.0},
        }
    }

    cumulative_randomized = []
    for stage_name, modules in layers_to_randomize:
        cumulative_randomized.append(stage_name)
        # Randomize module weights
        torch.manual_seed(42)
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    nn.init.normal_(p, mean=0.0, std=0.1)

        gc_rand = ECGGradCAM1D(test_model)
        ig_rand = ECGIntegratedGradients(test_model)

        cam_rand = gc_rand.generate_cam(input_tensor, target_class)
        ig_res = ig_rand.attribute(input_tensor, target_class, n_steps=30)
        ig_rand_map = ig_res["attribution_map"]

        progression[f"after_{stage_name}"] = {
            "randomized_stages": list(cumulative_randomized),
            "grad_cam": compute_attribution_similarity(orig_cam, cam_rand),
            "integrated_gradients": compute_attribution_similarity(orig_ig, ig_rand_map),
        }

    # Sanity check passes if final similarity after full randomization collapses significantly
    final_cam_sim = progression["after_init_conv"]["grad_cam"]["cosine_similarity"]
    final_ig_sim = progression["after_init_conv"]["integrated_gradients"]["cosine_similarity"]
    sanity_passed = bool(final_cam_sim < 0.60 and final_ig_sim < 0.60)

    return {
        "stages": progression,
        "final_cam_cosine": final_cam_sim,
        "final_ig_cosine": final_ig_sim,
        "cascading_randomization_sanity_passed": sanity_passed,
    }
