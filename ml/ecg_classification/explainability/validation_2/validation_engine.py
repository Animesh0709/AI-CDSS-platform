#!/usr/bin/env python3
"""
Phase 8.1.1 Validation Engine:
Explainability Metric Reconciliation, Lead Attribution Validation,
Perturbation Strategy Comparison, and Lead Redundancy Analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.scripts.ecg.ptbxl_loader import (
    DIAGNOSTIC_SUPERCLASSES,
    LEAD_NAME_MAP,
    PTBXLDatasetLoader,
    STANDARD_12_LEADS,
)
from ml.ecg_classification.resnet1d import ECGResNet1D
from ml.ecg_classification.explainability.grad_cam import ECGGradCAM1D
from ml.ecg_classification.explainability.integrated_gradients import ECGIntegratedGradients


# Standard clinical ordering (standard 12-lead layout: Limb I, II, III, aVR, aVL, aVF; Precordial V1-V6)
CLINICAL_LEAD_ORDERING = list(STANDARD_12_LEADS)


def get_trapezoid_fn():
    return getattr(np, "trapezoid", getattr(np, "trapz", None))


def audit_temporal_deletion_metrics(
    model: nn.Module,
    input_tensor: torch.Tensor,
    cam_map: torch.Tensor,
    target_class: str,
    window_size: int = 50,
    steps: int = 10,
) -> Dict[str, Any]:
    """
    Detailed mathematical audit of temporal deletion under various metric formulations:
    - Raw logit AUDC
    - Sigmoid probability AUDC
    - Normalized logit degradation AUDC: (logit - logit_final) / (logit_init - logit_final)
    - Relative logit difference AUDC: (logit - logit_init)
    """
    target_idx = DIAGNOSTIC_SUPERCLASSES.index(target_class)
    x = input_tensor.unsqueeze(0) if input_tensor.ndim == 2 else input_tensor.clone()
    model.eval()

    with torch.no_grad():
        initial_logit = float(model(x)[0, target_idx].item())
        initial_prob = float(torch.sigmoid(torch.tensor(initial_logit)).item())

    n_samples = x.shape[-1]
    n_windows = n_samples // window_size
    cam_np = cam_map.detach().cpu().numpy()
    window_scores = [float(np.mean(cam_np[w * window_size : (w + 1) * window_size])) for w in range(n_windows)]

    ranked_mif = np.argsort(window_scores)[::-1].copy()
    ranked_lif = np.argsort(window_scores).copy()

    # 5 random seeds for robust random control
    random_rankings = []
    for s in [42, 101, 202, 303, 404]:
        rng = np.random.RandomState(s)
        random_rankings.append(rng.permutation(n_windows).copy())

    fractions = np.linspace(0.0, 1.0, steps + 1)

    mif_logits = [initial_logit]
    lif_logits = [initial_logit]
    rand_logits_seeds = [[initial_logit] for _ in random_rankings]

    for f in fractions[1:]:
        n_mask = int(round(f * n_windows))

        # MIF
        x_mif = x.clone()
        for idx in ranked_mif[:n_mask]:
            x_mif[:, :, idx * window_size : (idx + 1) * window_size] = 0.0
        with torch.no_grad():
            mif_logits.append(float(model(x_mif)[0, target_idx].item()))

        # LIF
        x_lif = x.clone()
        for idx in ranked_lif[:n_mask]:
            x_lif[:, :, idx * window_size : (idx + 1) * window_size] = 0.0
        with torch.no_grad():
            lif_logits.append(float(model(x_lif)[0, target_idx].item()))

        # Random control across seeds
        for s_idx, r_rank in enumerate(random_rankings):
            x_rnd = x.clone()
            for idx in r_rank[:n_mask]:
                x_rnd[:, :, idx * window_size : (idx + 1) * window_size] = 0.0
            with torch.no_grad():
                rand_logits_seeds[s_idx].append(float(model(x_rnd)[0, target_idx].item()))

    avg_rand_logits = np.mean(rand_logits_seeds, axis=0).tolist()
    trapz = get_trapezoid_fn()

    # 1. Raw logit AUDC
    audc_mif_raw = float(trapz(mif_logits, fractions))
    audc_lif_raw = float(trapz(lif_logits, fractions))
    audc_rand_raw = float(trapz(avg_rand_logits, fractions))

    # 2. Sigmoid probability AUDC
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.array(z)))
    mif_probs = sig(mif_logits).tolist()
    lif_probs = sig(lif_logits).tolist()
    rand_probs = sig(avg_rand_logits).tolist()
    audc_mif_prob = float(trapz(mif_probs, fractions))
    audc_lif_prob = float(trapz(lif_probs, fractions))
    audc_rand_prob = float(trapz(rand_probs, fractions))

    # 3. Normalized logit degradation curve AUDC (range 0 to 1 if monotonic)
    denom = (mif_logits[0] - mif_logits[-1])
    if abs(denom) < 1e-6:
        denom = 1.0
    norm_mif = ((np.array(mif_logits) - mif_logits[-1]) / denom).tolist()
    norm_lif = ((np.array(lif_logits) - lif_logits[-1]) / denom).tolist()
    norm_rand = ((np.array(avg_rand_logits) - avg_rand_logits[-1]) / denom).tolist()
    audc_mif_norm = float(trapz(norm_mif, fractions))
    audc_lif_norm = float(trapz(norm_lif, fractions))
    audc_rand_norm = float(trapz(norm_rand, fractions))

    # Check monotonicity
    diffs_mif = np.diff(mif_logits)
    is_mif_monotonic = bool(np.all(diffs_mif <= 1e-4))
    non_monotonic_steps_mif = int(np.sum(diffs_mif > 0.05))

    return {
        "target_class": target_class,
        "initial_logit": initial_logit,
        "initial_prob": initial_prob,
        "final_zero_input_logit": mif_logits[-1],
        "fractions": fractions.tolist(),
        "trajectories": {
            "mif_logits": mif_logits,
            "lif_logits": lif_logits,
            "random_logits_mean": avg_rand_logits,
            "mif_probs": mif_probs,
            "lif_probs": lif_probs,
            "random_probs_mean": rand_probs,
        },
        "audc_metrics": {
            "raw_logits": {
                "audc_mif": round(audc_mif_raw, 4),
                "audc_lif": round(audc_lif_raw, 4),
                "audc_random_mean": round(audc_rand_raw, 4),
                "is_faithful_mif_lower_than_random": bool(audc_mif_raw < audc_rand_raw),
                "is_faithful_mif_lower_than_lif": bool(audc_mif_raw < audc_lif_raw),
            },
            "probabilities": {
                "audc_mif": round(audc_mif_prob, 4),
                "audc_lif": round(audc_lif_prob, 4),
                "audc_random_mean": round(audc_rand_prob, 4),
                "is_faithful_mif_lower_than_random": bool(audc_mif_prob < audc_rand_prob),
                "is_faithful_mif_lower_than_lif": bool(audc_mif_prob < audc_lif_prob),
            },
            "normalized_logits": {
                "audc_mif": round(audc_mif_norm, 4),
                "audc_lif": round(audc_lif_norm, 4),
                "audc_random_mean": round(audc_rand_norm, 4),
                "is_faithful_mif_lower_than_random": bool(audc_mif_norm < audc_rand_norm),
                "is_faithful_mif_lower_than_lif": bool(audc_mif_norm < audc_lif_norm),
            },
        },
        "monotonicity_audit": {
            "is_mif_monotonic": is_mif_monotonic,
            "non_monotonic_increase_steps": non_monotonic_steps_mif,
            "max_non_monotonic_spike": float(np.max(diffs_mif)) if len(diffs_mif) > 0 else 0.0,
        },
    }


def evaluate_lead_deletion_orderings(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: str,
    ig_lead_importance: np.ndarray,
    n_random_seeds: int = 5,
) -> Dict[str, Any]:
    """
    Compare 4 lead deletion orderings:
    1. IG Most-Important-First (MIF)
    2. IG Least-Important-First (LIF)
    3. Random Lead Orderings (averaged across multiple seeds)
    4. Clinical Lead Ordering (I, II, III, aVR, aVL, aVF, V1-V6)
    """
    target_idx = DIAGNOSTIC_SUPERCLASSES.index(target_class)
    x = input_tensor.unsqueeze(0) if input_tensor.ndim == 2 else input_tensor.clone()
    model.eval()

    with torch.no_grad():
        initial_logit = float(model(x)[0, target_idx].item())

    ranked_mif = np.argsort(ig_lead_importance)[::-1].copy()
    ranked_lif = np.argsort(ig_lead_importance).copy()
    clinical_order = list(range(12))  # Canonical STANDARD_12_LEADS layout

    random_rankings = []
    for s in [42, 101, 202, 303, 404][:n_random_seeds]:
        rng = np.random.RandomState(s)
        random_rankings.append(rng.permutation(12).copy())

    lead_counts = list(range(13))
    mif_logits = [initial_logit]
    lif_logits = [initial_logit]
    clinical_logits = [initial_logit]
    rand_logits_seeds = [[initial_logit] for _ in random_rankings]

    for k in range(1, 13):
        # MIF
        x_mif = x.clone()
        x_mif[:, ranked_mif[:k], :] = 0.0
        with torch.no_grad():
            mif_logits.append(float(model(x_mif)[0, target_idx].item()))

        # LIF
        x_lif = x.clone()
        x_lif[:, ranked_lif[:k], :] = 0.0
        with torch.no_grad():
            lif_logits.append(float(model(x_lif)[0, target_idx].item()))

        # Clinical
        x_clin = x.clone()
        x_clin[:, clinical_order[:k], :] = 0.0
        with torch.no_grad():
            clinical_logits.append(float(model(x_clin)[0, target_idx].item()))

        # Random across seeds
        for s_idx, r_rank in enumerate(random_rankings):
            x_rnd = x.clone()
            x_rnd[:, r_rank[:k], :] = 0.0
            with torch.no_grad():
                rand_logits_seeds[s_idx].append(float(model(x_rnd)[0, target_idx].item()))

    avg_rand_logits = np.mean(rand_logits_seeds, axis=0).tolist()
    fractions = [k / 12.0 for k in lead_counts]
    trapz = get_trapezoid_fn()

    audc_mif = float(trapz(mif_logits, fractions))
    audc_lif = float(trapz(lif_logits, fractions))
    audc_clin = float(trapz(clinical_logits, fractions))
    audc_rand = float(trapz(avg_rand_logits, fractions))

    return {
        "lead_counts": lead_counts,
        "fractions": fractions,
        "initial_logit": initial_logit,
        "ranked_mif_lead_names": [STANDARD_12_LEADS[i] for i in ranked_mif],
        "ranked_lif_lead_names": [STANDARD_12_LEADS[i] for i in ranked_lif],
        "mif_logits": mif_logits,
        "lif_logits": lif_logits,
        "clinical_logits": clinical_logits,
        "random_logits_mean": avg_rand_logits,
        "audc_mif": round(audc_mif, 4),
        "audc_lif": round(audc_lif, 4),
        "audc_clinical": round(audc_clin, 4),
        "audc_random_mean": round(audc_rand, 4),
        "mif_beats_random": bool(audc_mif <= audc_rand),
        "mif_beats_clinical": bool(audc_mif <= audc_clin),
        "mif_beats_lif": bool(audc_mif <= audc_lif),
    }


def compare_perturbation_strategies(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: str,
    top_lead_indices: List[int],
) -> Dict[str, Any]:
    """
    Compare 3 perturbation strategies when replacing important leads:
    1. Standardized zero-baseline replacement (constant 0.0 in normalized space)
    2. Training-mean replacement (explicit channel mean, which maps to 0.0)
    3. Smooth local replacement: linear baseline interpolation between first and last sample
       t=0 and t=T-1 to preserve baseline voltage continuity without introducing step artifacts.
    """
    target_idx = DIAGNOSTIC_SUPERCLASSES.index(target_class)
    x = input_tensor.unsqueeze(0) if input_tensor.ndim == 2 else input_tensor.clone()
    model.eval()

    with torch.no_grad():
        initial_logit = float(model(x)[0, target_idx].item())

    # Strategy 1: Standardized zero replacement
    x_zero = x.clone()
    x_zero[:, top_lead_indices, :] = 0.0
    with torch.no_grad():
        logit_zero = float(model(x_zero)[0, target_idx].item())

    # Strategy 2: Training-mean replacement (exact mathematical zero in normalized space)
    x_mean = x.clone()
    for l_idx in top_lead_indices:
        x_mean[:, l_idx, :] = 0.0  # Training mean normalized = (mean - mean)/std = 0.0
    with torch.no_grad():
        logit_mean = float(model(x_mean)[0, target_idx].item())

    # Strategy 3: Smooth local linear interpolation between endpoints
    x_smooth = x.clone()
    L = x.shape[-1]
    for l_idx in top_lead_indices:
        start_v = x[0, l_idx, 0].item()
        end_v = x[0, l_idx, -1].item()
        x_smooth[:, l_idx, :] = torch.linspace(start_v, end_v, L).unsqueeze(0)
    with torch.no_grad():
        logit_smooth = float(model(x_smooth)[0, target_idx].item())

    # Endpoint boundary step artifact analysis:
    # Measure maximum step gradient introduced at t=0 and t=T-1 for zero vs smooth
    step_artifact_zero = []
    step_artifact_smooth = []
    for l_idx in top_lead_indices:
        # Step artifact: magnitude difference from original endpoint
        step_artifact_zero.append(float(abs(x[0, l_idx, 0].item() - 0.0) + abs(x[0, l_idx, -1].item() - 0.0)))
        step_artifact_smooth.append(0.0)  # Continuous at endpoints by construction

    return {
        "initial_logit": initial_logit,
        "masked_leads": [STANDARD_12_LEADS[i] for i in top_lead_indices],
        "zero_baseline": {
            "perturbed_logit": logit_zero,
            "delta_logit": round(logit_zero - initial_logit, 4),
            "step_artifact_endpoint_error": round(float(np.mean(step_artifact_zero)), 4),
        },
        "training_mean_baseline": {
            "perturbed_logit": logit_mean,
            "delta_logit": round(logit_mean - initial_logit, 4),
            "is_identical_to_zero": bool(abs(logit_zero - logit_mean) < 1e-6),
        },
        "smooth_interpolation_baseline": {
            "perturbed_logit": logit_smooth,
            "delta_logit": round(logit_smooth - initial_logit, 4),
            "step_artifact_endpoint_error": round(float(np.mean(step_artifact_smooth)), 4),
            "delta_from_zero_baseline": round(logit_smooth - logit_zero, 4),
        },
    }


def analyze_lead_redundancy(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: str,
    ig_lead_importance: np.ndarray,
    n_random_combinations: int = 10,
) -> Dict[str, Any]:
    """
    Test lead redundancy and cross-lead compensation:
    - Individual lead deletion: does removing any single top lead collapse the prediction?
    - Top-2 lead deletion vs Random-2 combinations.
    - Top-3 lead deletion vs Random-3 combinations.
    - Re-compute IG on perturbed inputs to check whether attribution ranking shifts to correlated leads.
    """
    target_idx = DIAGNOSTIC_SUPERCLASSES.index(target_class)
    x = input_tensor.unsqueeze(0) if input_tensor.ndim == 2 else input_tensor.clone()
    model.eval()

    with torch.no_grad():
        initial_logit = float(model(x)[0, target_idx].item())

    ranked_leads = np.argsort(ig_lead_importance)[::-1].copy()
    top1 = ranked_leads[0]
    top2 = ranked_leads[:2].tolist()
    top3 = ranked_leads[:3].tolist()

    # 1. Single lead deletions
    single_lead_effects = {}
    for i in range(12):
        x_single = x.clone()
        x_single[:, i, :] = 0.0
        with torch.no_grad():
            l_out = float(model(x_single)[0, target_idx].item())
        single_lead_effects[STANDARD_12_LEADS[i]] = {
            "perturbed_logit": round(l_out, 4),
            "delta_logit": round(l_out - initial_logit, 4),
            "relative_drop": round((initial_logit - l_out) / (abs(initial_logit) + 1e-6), 4),
        }

    # 2. Top-2 vs Random-2
    x_top2 = x.clone()
    x_top2[:, top2, :] = 0.0
    with torch.no_grad():
        l_top2 = float(model(x_top2)[0, target_idx].item())

    rand2_deltas = []
    rng = np.random.RandomState(42)
    for _ in range(n_random_combinations):
        combo = rng.choice(12, size=2, replace=False)
        x_r = x.clone()
        x_r[:, combo, :] = 0.0
        with torch.no_grad():
            rand2_deltas.append(float(model(x_r)[0, target_idx].item()) - initial_logit)

    # 3. Top-3 vs Random-3
    x_top3 = x.clone()
    x_top3[:, top3, :] = 0.0
    with torch.no_grad():
        l_top3 = float(model(x_top3)[0, target_idx].item())

    rand3_deltas = []
    for _ in range(n_random_combinations):
        combo = rng.choice(12, size=3, replace=False)
        x_r = x.clone()
        x_r[:, combo, :] = 0.0
        with torch.no_grad():
            rand3_deltas.append(float(model(x_r)[0, target_idx].item()) - initial_logit)

    # 4. Attribution ranking stability under top-lead deletion
    # Compute new IG after zeroing top-1 lead
    ig_engine = ECGIntegratedGradients(model)
    res_after_top1 = ig_engine.attribute(x_top2, target_class, n_steps=25)
    new_imp = res_after_top1["lead_importance"].numpy()
    # Mask out the deleted lead from ranking
    new_imp_masked = new_imp.copy()
    new_imp_masked[top2] = -1.0
    surviving_ranked = [STANDARD_12_LEADS[i] for i in np.argsort(new_imp_masked)[::-1] if i not in top2]

    # Check if a correlated lead steps up
    return {
        "initial_logit": initial_logit,
        "single_lead_effects": single_lead_effects,
        "top_1_lead": STANDARD_12_LEADS[top1],
        "top_1_delta_logit": single_lead_effects[STANDARD_12_LEADS[top1]]["delta_logit"],
        "top_2_leads": [STANDARD_12_LEADS[i] for i in top2],
        "top_2_delta_logit": round(l_top2 - initial_logit, 4),
        "random_2_mean_delta_logit": round(float(np.mean(rand2_deltas)), 4),
        "top_2_more_impactful_than_random_2": bool((l_top2 - initial_logit) < np.mean(rand2_deltas)),
        "top_3_leads": [STANDARD_12_LEADS[i] for i in top3],
        "top_3_delta_logit": round(l_top3 - initial_logit, 4),
        "random_3_mean_delta_logit": round(float(np.mean(rand3_deltas)), 4),
        "top_3_more_impactful_than_random_3": bool((l_top3 - initial_logit) < np.mean(rand3_deltas)),
        "surviving_leads_ranked_after_top2_mask": surviving_ranked[:5],
        "redundancy_observed": bool(abs(single_lead_effects[STANDARD_12_LEADS[top1]]["delta_logit"]) < 1.0),
    }
