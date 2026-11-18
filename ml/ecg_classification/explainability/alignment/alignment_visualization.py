"""
Alignment Visualization Service.

Generates research-grade, non-invasive visualizations aligning lead attributions
and reconstruction confidence with original ECG images.

Visual Safeguards (Refined Phase 8.2.1):
1. Original image is completely preserved without pixel modification.
2. Neutral, non-alarmist color palettes (no bright warning red on healthy traces).
3. Lead regions are indicated by clean bounding borders and lead rank badges.
4. Uncertain regions are highlighted with neutral dashed/shaded overlays and explicit warning labels.
5. Model scores are strictly labeled: `uncalibrated_model_probability`, `raw_model_logit`, and `model_decision`.
6. Grad-CAM temporal guide is plotted strictly as an auxiliary 1D trace with prominent disclaimer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app.services.ecg.explainability_alignment import (
    AlignmentStatus,
    CalibrationStatus,
    ExplainabilityImageAlignmentBundle,
    GeometryStatus,
    LeadExplanationAlignment,
    ReconstructionStatus,
)

logger = logging.getLogger("ai_cdss.ecg.alignment_visualization")


def plot_explainability_image_alignment(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    alignment_bundle: ExplainabilityImageAlignmentBundle,
    reconstructed_signals: Optional[Dict[str, np.ndarray]] = None,
    grad_cam_1d: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Render static research visualization connecting explainability to ECG image.
    """
    # 1. Load image
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    img_np = np.array(img)
    h, w, _ = img_np.shape

    # Setup figure with 2 rows: Top = ECG image overlay, Bottom = Metadata & Grad-CAM
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 0.8, 0.8], hspace=0.35, wspace=0.25)

    # Panel 1: Original ECG Image with Lead-Region Overlay
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_np)
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)
    ax_img.axis("off")

    target_cls = alignment_bundle.target_class
    prob = alignment_bundle.uncalibrated_model_probability
    thresh = alignment_bundle.classification_threshold
    decision = alignment_bundle.model_decision

    title_text = (
        f"ECG Explainability-to-Image Alignment — Class: {target_cls} | "
        f"Uncalibrated Probability: {prob:.3f} (Threshold: {thresh:.2f}) -> {decision}\n"
        f"Confidence Gating: {alignment_bundle.confident_lead_count}/12 Confident, "
        f"{alignment_bundle.uncertain_lead_count} Uncertain, {alignment_bundle.unavailable_lead_count} Unavailable "
        f"(Coverage: {alignment_bundle.overall_alignment_coverage_pct:.1f}%)"
    )
    ax_img.set_title(title_text, fontsize=11.5, fontweight="bold", pad=12)

    # Draw Lead Bounding Boxes
    for lead_name, lead_align in alignment_bundle.lead_alignments.items():
        box = lead_align.image_bounding_box
        if not box:
            continue

        status = lead_align.alignment_status
        rank = lead_align.attribution_rank
        imp = lead_align.attribution_importance
        conf = lead_align.reconstruction_confidence

        if status == AlignmentStatus.CONFIDENT:
            # Color intensity scaled by rank
            if rank <= 3:
                edge_color = "#004d80"
                bg_color = (0.0, 0.3, 0.5, 0.08)
                lw = 2.2
            elif rank <= 8:
                edge_color = "#336699"
                bg_color = (0.2, 0.4, 0.6, 0.05)
                lw = 1.4
            else:
                edge_color = "#778899"
                bg_color = (0.5, 0.5, 0.5, 0.03)
                lw = 1.0

            rect = patches.Rectangle(
                (box.x_min, box.y_min),
                box.width,
                box.height,
                linewidth=lw,
                edgecolor=edge_color,
                facecolor=bg_color,
            )
            ax_img.add_patch(rect)

            badge_text = f"{lead_name} (Rank #{rank} | Attr: {imp:.2f} | Conf: {conf:.2f})"
            ax_img.text(
                box.x_min + 8,
                box.y_min + 22,
                badge_text,
                fontsize=8.5,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=edge_color, alpha=0.85, edgecolor="none"),
            )
        elif status == AlignmentStatus.UNCERTAIN:
            edge_color = "#d97706"
            rect = patches.Rectangle(
                (box.x_min, box.y_min),
                box.width,
                box.height,
                linewidth=1.8,
                linestyle="--",
                edgecolor=edge_color,
                facecolor=(0.85, 0.55, 0.1, 0.08),
                hatch="//",
            )
            ax_img.add_patch(rect)

            cal_tag = "Uncalibrated" if lead_align.calibration_status == CalibrationStatus.UNVERIFIED_PIXEL_DOMAIN else f"Conf {conf:.2f}"
            badge_text = f"{lead_name} [UNCERTAIN: {cal_tag}]"
            ax_img.text(
                box.x_min + 8,
                box.y_min + 22,
                badge_text,
                fontsize=8.0,
                fontweight="bold",
                color="#78350f",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef3c7", alpha=0.90, edgecolor=edge_color),
            )
        else:  # UNAVAILABLE
            edge_color = "#6b7280"
            rect = patches.Rectangle(
                (box.x_min, box.y_min),
                box.width,
                box.height,
                linewidth=1.2,
                linestyle=":",
                edgecolor=edge_color,
                facecolor=(0.5, 0.5, 0.5, 0.12),
                hatch="xx",
            )
            ax_img.add_patch(rect)

            badge_text = f"{lead_name} [UNAVAILABLE]"
            ax_img.text(
                box.x_min + 8,
                box.y_min + 22,
                badge_text,
                fontsize=8.0,
                fontweight="bold",
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5e7eb", alpha=0.90, edgecolor=edge_color),
            )

    # Lead II Rhythm Strip Box
    if alignment_bundle.rhythm_strip_alignment and alignment_bundle.rhythm_strip_alignment.image_bounding_box:
        rh_box = alignment_bundle.rhythm_strip_alignment.image_bounding_box
        rh_status = alignment_bundle.rhythm_strip_alignment.alignment_status
        rh_conf = alignment_bundle.rhythm_strip_alignment.reconstruction_confidence

        rh_color = "#475569" if rh_status == AlignmentStatus.CONFIDENT else "#d97706"
        rh_rect = patches.Rectangle(
            (rh_box.x_min, rh_box.y_min),
            rh_box.width,
            rh_box.height,
            linewidth=1.5,
            linestyle="-." if rh_status == AlignmentStatus.CONFIDENT else "--",
            edgecolor=rh_color,
            facecolor="none",
        )
        ax_img.add_patch(rh_rect)
        ax_img.text(
            rh_box.x_min + 8,
            rh_box.y_min + 20,
            f"Lead II Rhythm Strip (10.0s continuous | Conf: {rh_conf:.2f})",
            fontsize=8.0,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f1f5f9", alpha=0.85, edgecolor=rh_color),
        )

    # Panel 2: Lead Attribution Ranking Bar Chart
    ax_bar = fig.add_subplot(gs[1, 0])
    sorted_leads = sorted(
        alignment_bundle.lead_alignments.values(),
        key=lambda x: x.attribution_importance,
        reverse=True,
    )
    names = [x.lead_name for x in sorted_leads]
    scores = [x.attribution_importance for x in sorted_leads]
    bar_colors = [
        "#004d80" if x.alignment_status == AlignmentStatus.CONFIDENT else "#d97706"
        for x in sorted_leads
    ]
    y_pos = np.arange(len(names))
    ax_bar.barh(y_pos, scores, color=bar_colors, align="center")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Integrated Gradients Importance", fontsize=8)
    ax_bar.set_title("Lead Attribution Ranking (IG)", fontsize=9, fontweight="bold")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel 3: Reconstruction Quality Scores
    ax_conf = fig.add_subplot(gs[1, 1])
    conf_scores = [x.reconstruction_confidence for x in sorted_leads]
    conf_colors = [
        "#10b981" if c >= 0.70 else ("#f59e0b" if c >= 0.20 else "#ef4444")
        for c in conf_scores
    ]
    ax_conf.barh(y_pos, conf_scores, color=conf_colors, align="center")
    ax_conf.axvline(0.70, color="#059669", linestyle="--", linewidth=1.2, label="Reliable (>=0.70)")
    ax_conf.set_yticks(y_pos)
    ax_conf.set_yticklabels(["" for _ in names])
    ax_conf.set_xlim(0.0, 1.05)
    ax_conf.invert_yaxis()
    ax_conf.set_xlabel("Reconstruction Quality Score", fontsize=8)
    ax_conf.set_title("Reconstruction Quality", fontsize=9, fontweight="bold")
    ax_conf.legend(loc="lower right", fontsize=7.5)
    ax_conf.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel 4: Diagnostic Output & Semantic Metadata Card
    ax_card = fig.add_subplot(gs[1, 2])
    ax_card.axis("off")
    card_text = (
        f"MODEL PREDICTION SUMMARY\n"
        f"-----------------------------------------\n"
        f"• Target Superclass: {target_cls}\n"
        f"• Raw Model Logit: {alignment_bundle.raw_model_logit:+.4f}\n"
        f"• Uncalibrated Probability: {alignment_bundle.uncalibrated_model_probability:.4f}\n"
        f"• Decision Threshold: {alignment_bundle.classification_threshold:.4f}\n"
        f"• Model Decision: {alignment_bundle.model_decision}\n\n"
        f"INDEPENDENT STATUS DIMENSIONS\n"
        f"-----------------------------------------\n"
        f"• Physical Calibration: {'VERIFIED' if alignment_bundle.calibration_verified else 'UNVERIFIED (PIXEL OFFSET)'}\n"
        f"• Confident Leads: {alignment_bundle.confident_lead_count} / 12\n"
        f"• Uncertain Leads: {alignment_bundle.uncertain_lead_count} / 12\n"
        f"• Unavailable Leads: {alignment_bundle.unavailable_lead_count} / 12\n"
        f"• Alignment Coverage: {alignment_bundle.overall_alignment_coverage_pct:.1f}%\n"
    )
    ax_card.text(
        0.05,
        0.95,
        card_text,
        fontsize=8.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2),
    )

    # Panel 5: Exploratory 1D Grad-CAM Temporal Guidance
    ax_cam = fig.add_subplot(gs[2, :])
    if grad_cam_1d is not None and len(grad_cam_1d) == 1000:
        time_axis = np.linspace(0.0, 10.0, 1000)
        ax_cam.plot(time_axis, grad_cam_1d, color="#7c3aed", linewidth=1.4, label="1D Grad-CAM Heatmap")
        ax_cam.fill_between(time_axis, 0, grad_cam_1d, color="#7c3aed", alpha=0.15)
        ax_cam.set_ylabel("Grad-CAM Activation", fontsize=8)
        ax_cam.set_ylim(-0.05, 1.05)
    else:
        ax_cam.text(
            0.5,
            0.5,
            "1D Grad-CAM temporal attribution array not provided or unavailable.",
            ha="center",
            va="center",
            fontsize=9,
            color="#6b7280",
        )

    ax_cam.set_xlim(0.0, 10.0)
    ax_cam.set_xlabel("ECG Recording Time (seconds)", fontsize=8)
    ax_cam.set_title(
        "Auxiliary 1D Temporal Attention Guide (Grad-CAM) — Exploratory Only",
        fontsize=9,
        fontweight="bold",
        color="#5b21b6",
    )
    ax_cam.grid(True, linestyle=":", alpha=0.5)

    # Prominent Disclaimer Footer
    disclaimer = (
        "METHODOLOGICAL & CLINICAL DISCLAIMER: Lead attributions reflect mathematical sensitivity of the 1D ResNet "
        "and do NOT prove localized cardiac infarction or diagnostic truth. Grad-CAM temporal maps are exploratory guides "
        "and are NOT mapped to image pixels. Bounding boxes represent analytical layout boundaries only."
    )
    fig.text(
        0.5,
        0.01,
        disclaimer,
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#475569",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f8fafc", edgecolor="#e2e8f0"),
    )

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        logger.info("Saved explainability alignment visualization to %s", out_p)

    return fig
