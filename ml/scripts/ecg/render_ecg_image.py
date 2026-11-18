"""
Reproducible 12-Lead ECG Paper Image Renderer.

Simulates standard clinical ECG recording paper with the conventional 3x4 lead arrangement:
- Layout (Conventional 3 rows x 4 columns):
    Row 1: I,   aVR, V1, V4
    Row 2: II,  aVL, V2, V5
    Row 3: III, aVF, V3, V6
  Each column displays an exact 2.5-second time window:
    Col 0 (t = 0.0s to 2.5s):   I,   II,  III
    Col 1 (t = 2.5s to 5.0s):   aVR, aVL, aVF
    Col 2 (t = 5.0s to 7.5s):   V1,  V2,  V3
    Col 3 (t = 7.5s to 10.0s):  V4,  V5,  V6
  Bottom Rhythm Strip (full 10.0s): Lead II

- Calibrated Paper Grid:
    Paper speed: 25 mm/s (0.04 s = 1 mm small box, 0.20 s = 5 mm large box)
    Voltage gain: 10 mm/mV (0.1 mV = 1 mm small box, 0.5 mV = 5 mm large box)
    Grid colors: Light pink/salmon (#ffccd2) for minor, darker pink (#ff8594) for major.
- Explicit watermark / banner indicating synthetic simulation.
- Preserves full source signal in accompanying .json metadata for reconstruction benchmarking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Standard 12 leads list and conventional 3x4 matrix
STANDARD_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Conventional 3x4 Grid Mapping:
# Row 0: I   (col 0, t=0.0-2.5s), aVR (col 1, t=2.5-5.0s), V1 (col 2, t=5.0-7.5s), V4 (col 3, t=7.5-10.0s)
# Row 1: II  (col 0, t=0.0-2.5s), aVL (col 1, t=2.5-5.0s), V2 (col 2, t=5.0-7.5s), V5 (col 3, t=7.5-10.0s)
# Row 2: III (col 0, t=0.0-2.5s), aVF (col 1, t=2.5-5.0s), V3 (col 2, t=5.0-7.5s), V6 (col 3, t=7.5-10.0s)
CONVENTIONAL_3X4_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def render_12lead_ecg_image(
    signal: np.ndarray,
    sampling_rate: int,
    output_image_path: str | Path,
    lead_names: Optional[List[str]] = None,
    paper_speed_mm_s: float = 25.0,
    voltage_gain_mm_mv: float = 10.0,
    dpi: int = 150,
    include_rhythm_strip: bool = True,
    rhythm_lead: str = "II",
    save_metadata: bool = True,
) -> Dict:
    """
    Render a standard 12-lead ECG strip onto calibrated paper grid in conventional 3x4 format.

    signal: np.ndarray of shape (num_samples, 12)
    sampling_rate: Sampling frequency in Hz (e.g. 100 or 500)
    output_image_path: Destination file path for PNG or JPEG image
    """
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if lead_names is None:
        lead_names = STANDARD_LEAD_NAMES

    num_samples, num_leads = signal.shape
    if num_leads != 12:
        raise ValueError(f"Signal must contain exactly 12 channels; got {num_leads}")

    duration_sec = num_samples / sampling_rate
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Lead name to column index lookup in source signal
    lead_idx_map = {name: idx for idx, name in enumerate(lead_names)}
    for row in CONVENTIONAL_3X4_GRID:
        for l_name in row:
            if l_name not in lead_idx_map:
                raise ValueError(f"Required lead '{l_name}' not found in provided lead_names: {lead_names}")

    fig_width_in = 11.0
    fig_height_in = 7.0 if include_rhythm_strip else 5.5

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#ffffff")

    # Disable spines and tick marks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Total display space:
    # X-axis: 0 to 10.0 seconds
    # Y-axis: -4.0 to 1.5 (accommodates Rows 0, 1, 2 and bottom Rhythm strip at -3.0)
    num_rows = 4 if include_rhythm_strip else 3
    y_min = -1.0 * (num_rows) - 0.5
    y_max = 1.5

    ax.set_xlim(0, 10.0)
    ax.set_ylim(y_min, y_max)

    # Standard ECG Grid lines
    # Minor: 0.04s, 0.1mV (1 mm at 25 mm/s, 10 mm/mV)
    # Major: 0.20s, 0.5mV (5 mm at 25 mm/s, 10 mm/mV)
    x_minor = np.arange(0, 10.0 + 0.04, 0.04)
    x_major = np.arange(0, 10.0 + 0.20, 0.20)
    y_minor = np.arange(y_min, y_max + 0.1, 0.1)
    y_major = np.arange(y_min, y_max + 0.5, 0.5)

    for xm in x_minor:
        ax.axvline(xm, color="#ffccd2", linewidth=0.5, alpha=0.7, zorder=1)
    for xM in x_major:
        ax.axvline(xM, color="#ff8594", linewidth=1.0, alpha=0.8, zorder=2)

    for ym in y_minor:
        ax.axhline(ym, color="#ffccd2", linewidth=0.5, alpha=0.7, zorder=1)
    for yM in y_major:
        ax.axhline(yM, color="#ff8594", linewidth=1.0, alpha=0.8, zorder=2)

    samples_per_col = int(2.5 * sampling_rate)

    # 1. Plot conventional 3x4 layout
    # Row 0: I,   aVR, V1, V4  (baseline y = 0.0)
    # Row 1: II,  aVL, V2, V5  (baseline y = -1.0)
    # Row 2: III, aVF, V3, V6  (baseline y = -2.0)
    rendered_segments = {}

    for row_idx, row_leads in enumerate(CONVENTIONAL_3X4_GRID):
        y_baseline = -row_idx * 1.0

        for col_idx, lead_name in enumerate(row_leads):
            t_start = col_idx * 2.5
            t_end = t_start + 2.5
            s_start = col_idx * samples_per_col
            s_end = min(s_start + samples_per_col, num_samples)

            channel_idx = lead_idx_map[lead_name]
            lead_segment = signal[s_start:s_end, channel_idx]
            sub_t = time_axis[s_start:s_end]

            ax.plot(
                sub_t,
                y_baseline + lead_segment,
                color="#001428",
                linewidth=1.2,
                zorder=5,
            )

            ax.text(
                t_start + 0.08,
                y_baseline + 0.35,
                lead_name,
                fontsize=9,
                fontweight="bold",
                color="#0a2540",
                zorder=6,
            )

            rendered_segments[lead_name] = {
                "row": row_idx,
                "col": col_idx,
                "t_start_sec": float(t_start),
                "t_end_sec": float(t_end),
                "sample_start": int(s_start),
                "sample_end": int(s_end),
            }

    # 2. Continuous rhythm strip (Lead II full 10 seconds)
    if include_rhythm_strip:
        rhythm_channel_idx = lead_idx_map[rhythm_lead]
        rhythm_y_baseline = -3.0
        rhythm_signal = signal[:num_samples, rhythm_channel_idx]

        ax.plot(
            time_axis[:num_samples],
            rhythm_y_baseline + rhythm_signal,
            color="#001428",
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            0.08,
            rhythm_y_baseline + 0.35,
            f"{rhythm_lead} (Continuous Rhythm Strip)",
            fontsize=9,
            fontweight="bold",
            color="#0a2540",
            zorder=6,
        )

    # 3. Calibration bar and banner
    # Standard calibration pulse: 0.20s wide, 1.0 mV high (from y=0.2 to y=1.2 mV, within y_max=1.5)
    cal_x = np.array([0.05, 0.05, 0.25, 0.25])
    cal_y = np.array([0.2, 1.2, 1.2, 0.2])
    ax.plot(cal_x, cal_y, color="#001428", linewidth=1.2, zorder=5)
    ax.text(0.30, 0.70, "1.0 mV (10 mm)", fontsize=7.5, color="#2d3748", zorder=6)

    # Synthetic banner
    ax.text(
        0.05, y_max - 0.25,
        f"[SYNTHETIC 12-LEAD SIMULATION] Speed: {paper_speed_mm_s:.0f} mm/s | Gain: {paper_speed_mm_s/2.5:.0f} mm/mV ({voltage_gain_mm_mv:.0f} mm/mV) | fs: {sampling_rate} Hz",
        fontsize=8,
        color="#c53030",
        fontweight="semibold",
        zorder=6,
    )

    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    metadata = {
        "image_file": str(output_path.name),
        "is_synthetic": True,
        "layout_format": "Conventional 3x4 + Lead II Rhythm Strip",
        "row_layout": {
            "row_1": ["I", "aVR", "V1", "V4"],
            "row_2": ["II", "aVL", "V2", "V5"],
            "row_3": ["III", "aVF", "V3", "V6"],
            "rhythm_strip": f"{rhythm_lead} (10.0s continuous)",
        },
        "sampling_rate_hz": sampling_rate,
        "duration_seconds": duration_sec,
        "paper_speed_mm_s": paper_speed_mm_s,
        "voltage_gain_mm_mv": voltage_gain_mm_mv,
        "calibration_grid": {
            "small_box": "0.04 s x 0.1 mV (1 mm x 1 mm)",
            "large_box": "0.20 s x 0.5 mV (5 mm x 5 mm)",
        },
        "dpi": dpi,
        "rendered_segments": rendered_segments,
        "source_signal_stats": {
            lead: {
                "min_mv": float(np.min(signal[:, idx])),
                "max_mv": float(np.max(signal[:, idx])),
                "mean_mv": float(np.mean(signal[:, idx])),
                "std_mv": float(np.std(signal[:, idx])),
            }
            for lead, idx in lead_idx_map.items()
        },
    }

    if save_metadata:
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata
