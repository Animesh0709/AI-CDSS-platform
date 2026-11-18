"""
Phase 8.0: Explainability Visualization for 12-Lead ECG Abnormality Classification.

Produces clear, publication-quality multi-panel diagnostic explanation figures containing:
1. Original 12-lead ECG waveform signals with physiological lead labels and time scale (seconds).
2. Temporal Grad-CAM attribution overlay highlighting regions critical to the target diagnosis.
3. Lead-wise importance ranking bar chart (from Integrated Gradients absolute energy).
4. Lead-by-time attribution 2D heatmap showing spatio-temporal contribution dynamics.
5. Target superclass label, model logit, calibrated sigmoid probability score, and decision threshold.

Methodological safeguards:
- Waveform samples are plotted authentically; no smoothing or distortion is applied to input data.
- Explicit clinical warning labels: "RESEARCH EXPLANATION ONLY — NOT CLINICAL EVIDENCE".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES

# Standard 12-lead ECG layout and color palette
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_ecg_explanation(
    signal: Union[np.ndarray, torch.Tensor],
    grad_cam_map: Union[np.ndarray, torch.Tensor],
    ig_attribution_map: Union[np.ndarray, torch.Tensor],
    lead_importance: Union[np.ndarray, torch.Tensor],
    target_class: str,
    class_logit: float,
    class_prob: float,
    threshold: float,
    ecg_id: Optional[Union[int, str]] = None,
    ground_truth_labels: Optional[List[str]] = None,
    sampling_rate: int = 100,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Generate an explainability figure for a single 12-lead ECG record.

    Args:
        signal: Array of shape (12, samples) or (samples, 12).
        grad_cam_map: 1D array of shape (samples,) in [0, 1].
        ig_attribution_map: 2D array of shape (12, samples).
        lead_importance: 1D array of shape (12,).
        target_class: Explaining superclass (e.g. 'MI').
        class_logit: Raw model logit.
        class_prob: Sigmoid probability score.
        threshold: Decision threshold.
        ecg_id: Optional PTB-XL record identifier.
        ground_truth_labels: List of true diagnostic superclasses.
        sampling_rate: ECG sampling rate in Hz (default: 100).
        save_path: Destination path for saving PNG/PDF.
        dpi: Output resolution.

    Returns:
        fig: Matplotlib Figure instance.
    """
    # Convert tensors to numpy if needed
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(grad_cam_map, torch.Tensor):
        grad_cam_map = grad_cam_map.detach().cpu().numpy()
    if isinstance(ig_attribution_map, torch.Tensor):
        ig_attribution_map = ig_attribution_map.detach().cpu().numpy()
    if isinstance(lead_importance, torch.Tensor):
        lead_importance = lead_importance.detach().cpu().numpy()

    # Ensure signal shape is (12, n_samples)
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    n_samples = signal.shape[1]
    time_sec = np.arange(n_samples) / float(sampling_rate)

    # Normalize grad_cam to [0, 1] if not already
    cam = np.array(grad_cam_map, dtype=np.float32).flatten()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1.0, 0.9], width_ratios=[3.0, 1.2], hspace=0.35, wspace=0.25)

    # Header / Title annotation
    is_positive = class_prob >= threshold
    status_str = "POSITIVE (Above Threshold)" if is_positive else "NEGATIVE (Below Threshold)"
    gt_str = f" | True Labels: {', '.join(ground_truth_labels)}" if ground_truth_labels else ""
    id_str = f"Record ID: {ecg_id} | " if ecg_id is not None else ""

    fig.suptitle(
        f"{id_str}1D ResNet Explanation for Superclass: {target_class}\n"
        f"Model Logit: {class_logit:+.3f} | Probability: {class_prob:.3f} (Threshold: {threshold:.2f}) -> {status_str}{gt_str}\n"
        r"$\bf{RESEARCH\ EXPLANATION\ ONLY\ —\ NOT\ FOR\ STANDALONE\ CLINICAL\ DIAGNOSIS}$",
        fontsize=12,
        fontweight="bold",
        color="#1A202C",
        y=0.98,
    )

    # -------------------------------------------------------------
    # Panel 1: Original 12-Lead ECG Signals with Grad-CAM Color Highlights
    # -------------------------------------------------------------
    ax_ecg = fig.add_subplot(gs[0, 0])
    lead_offsets = np.arange(12)[::-1] * 3.0  # Vertical spacing in mV units

    for i, lead_name in enumerate(CANONICAL_LEADS):
        lead_sig = signal[i] + lead_offsets[i]
        ax_ecg.plot(time_sec, lead_sig, color="#2B6CB0", linewidth=0.9, alpha=0.9)
        ax_ecg.text(
            -0.2,
            lead_offsets[i],
            lead_name,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="right",
            color="#2D3748",
        )

    # Overlay Grad-CAM temporal focus as semi-transparent vertical spans
    # Threshold at 0.5 to highlight top active intervals
    top_intervals = cam >= 0.5
    if np.any(top_intervals):
        ax_ecg.imshow(
            cam[np.newaxis, :],
            aspect="auto",
            cmap="YlOrRd",
            alpha=0.22,
            extent=[time_sec[0], time_sec[-1], lead_offsets.min() - 2.0, lead_offsets.max() + 2.0],
            origin="lower",
        )

    ax_ecg.set_xlim(time_sec[0] - 0.3, time_sec[-1] + 0.1)
    ax_ecg.set_ylim(lead_offsets.min() - 2.5, lead_offsets.max() + 2.5)
    ax_ecg.set_yticks([])
    ax_ecg.set_xlabel("Time (seconds)", fontsize=10)
    ax_ecg.set_title("12-Lead Waveform with Temporal Grad-CAM Salience Overlay", fontsize=11, fontweight="semibold")
    ax_ecg.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 2: Lead-wise Importance Ranking (Integrated Gradients)
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    lead_imp_norm = lead_importance / (lead_importance.sum() + 1e-8)
    sorted_indices = np.argsort(lead_imp_norm)
    sorted_leads = [CANONICAL_LEADS[idx] for idx in sorted_indices]
    sorted_vals = lead_imp_norm[sorted_indices]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 12))
    bars = ax_bar.barh(sorted_leads, sorted_vals, color=colors, edgecolor="#2C5282", height=0.65)
    ax_bar.set_xlabel("Relative Attribution Weight", fontsize=10)
    ax_bar.set_title("Lead-Wise Importance", fontsize=11, fontweight="semibold")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.5)

    # Annotate percentage on bars
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            color="#2D3748",
        )
    ax_bar.set_xlim(0, max(sorted_vals.max() * 1.25, 0.15))

    # -------------------------------------------------------------
    # Panel 3: 1D Grad-CAM Activation Profile Across Time
    # -------------------------------------------------------------
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_cam.plot(time_sec, cam, color="#DD6B20", linewidth=1.8, label="Grad-CAM Activation")
    ax_cam.fill_between(time_sec, 0, cam, color="#ED8936", alpha=0.35)
    ax_cam.axhline(0.5, color="#C53030", linestyle="--", linewidth=1.0, label="High-Saliency Cutoff (0.50)")
    ax_cam.set_xlim(time_sec[0], time_sec[-1])
    ax_cam.set_ylim(-0.02, 1.05)
    ax_cam.set_xlabel("Time (seconds)", fontsize=10)
    ax_cam.set_ylabel("Grad-CAM Score", fontsize=10)
    ax_cam.set_title("Temporal Attention Profile (Grad-CAM)", fontsize=11, fontweight="semibold")
    ax_cam.legend(loc="upper right", fontsize=8)
    ax_cam.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 4: Integrated Gradients Spatio-Temporal Attribution Heatmap
    # -------------------------------------------------------------
    ax_heat = fig.add_subplot(gs[2, :])
    # Symmetric colormap centered at zero (diverging Coolwarm or Seismic)
    vmax = np.percentile(np.abs(ig_attribution_map), 99.5) + 1e-8
    im = ax_heat.imshow(
        ig_attribution_map,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=+vmax,
        extent=[time_sec[0], time_sec[-1], -0.5, 11.5],
    )
    ax_heat.set_yticks(range(12))
    ax_heat.set_yticklabels(CANONICAL_LEADS, fontsize=9)
    ax_heat.set_xlabel("Time (seconds)", fontsize=10)
    ax_heat.set_ylabel("ECG Lead", fontsize=10)
    ax_heat.set_title(
        "Lead-by-Time Attribution Matrix (Integrated Gradients: Blue = Negative / Attenuating, Red = Positive / Contributing)",
        fontsize=11,
        fontweight="semibold",
    )
    cbar = fig.colorbar(im, ax=ax_heat, orientation="horizontal", pad=0.25, shrink=0.5)
    cbar.set_label(f"Attribution towards '{target_class}' logit", fontsize=9)

    # Panel 5 (small text box at gs[1, 1]): Safeguards & Methodology Notice
    ax_notice = fig.add_subplot(gs[1, 1])
    ax_notice.axis("off")
    notice_text = (
        "METHODOLOGICAL NOTES:\n"
        "• Grad-CAM: Class-logit gradient pooled\n"
        "  across final conv layer (128 channels, 32 steps).\n"
        "• Integrated Gradients: Path integral against\n"
        "  zero isoelectric baseline (50 steps).\n"
        "• Saliency reflects model attention, not\n"
        "  anatomical lesion localization.\n"
        "• Do not use for unsupervised therapy triage."
    )
    ax_notice.text(
        0.05,
        0.5,
        notice_text,
        fontsize=8.5,
        family="monospace",
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EDF2F7", edgecolor="#CBD5E0"),
    )

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig
