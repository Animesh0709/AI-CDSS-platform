# ECG Explainability-to-Image Alignment and Confidence-Gated Mapping

This module provides a research-grade alignment layer that connects waveform-level explainability (Integrated Gradients lead attributions and exploratory 1D Grad-CAM temporal profiles) with original 12-lead ECG images without hallucinating visual evidence or inventing pixel-level masks.

---

## 1. Independent Status Dimensions (Refined Phase 8.2.1)

To prevent conflating geometric layout recognition with physical voltage calibration or waveform continuity, the alignment layer tracks 5 distinct, independent semantic dimensions for each lead:

1. **`geometry_status`**:
   - `VERIFIED`: Lead quadrant and bounding box corroborated by layout validation and grid detection.
   - `ESTIMATED_STANDARD_LAYOUT`: Conventional 3×4 coordinates derived from standard proportions.
   - `INVALID_OR_UNKNOWN`: Bounding box degenerate or layout geometry unverified.
2. **`reconstruction_status`**:
   - `CONFIDENT`: Reconstruction quality score $\ge 0.70$, missing pixels $\le 25\%$, boundary clipping $\le 15\%$.
   - `UNCERTAIN`: Reconstruction quality score in $[0.20, 0.70)$, or moderate clipping.
   - `UNAVAILABLE`: Missing lead signal or reconstruction quality score $< 0.20$.
3. **`calibration_status`**:
   - `VERIFIED`: Calibration pulse detected with verified paper speed ($\text{mm/s}$) and gain ($\text{mm/mV}$).
   - `UNVERIFIED_PIXEL_DOMAIN`: Calibration pulse absent or non-standard. Waveform signals remain in pixel offset domain; physical $\text{mV}$ or $\text{mm/s}$ claims are strictly prohibited.
4. **`attribution_status`**:
   - `AVAILABLE`: Valid model attribution score and rank computed from the classifier.
   - `UNAVAILABLE`: Lead attribution score absent or outside model target vocabulary.
5. **`alignment_status`**:
   - Resulting synthesized mapping status: `CONFIDENT`, `UNCERTAIN`, or `UNAVAILABLE`.

---

## 2. Confidence-Gating Threshold Documentation

The thresholds used for confidence gating originate from the Phase 3 & 4 reconstruction benchmarking and are documented as follows:

| Threshold | Value | Origin & Validation Status | Field Governed | Rationale & Gating Behavior |
| :--- | :---: | :--- | :--- | :--- |
| **`CONFIDENCE_THRESHOLD_RELIABLE`** | **0.70** | Engineering heuristic calibrated on synthetic and benchmark ECGs. *Not clinically validated.* | `reconstruction_confidence` | Ensures high trace fidelity based on continuity, low boundary clipping, and plausible dynamic range. Scores below $0.70$ downgrade alignment status to `UNCERTAIN`. |
| **`CONFIDENCE_THRESHOLD_UNAVAILABLE`** | **0.20** | Engineering heuristic lower bound. *Not clinically validated.* | `reconstruction_confidence` | Signals with confidence $< 0.20$ have catastrophic gaps or noise, making any attribution overlay clinically hazardous; marked `UNAVAILABLE`. |
| **`MAX_MISSING_PIXEL_FRACTION`** | **0.25** | Engineering heuristic. | `missing_pixel_fraction` | If $> 25\%$ of columns along the lead have missing trace pixels, the lead is marked `UNCERTAIN`. |
| **`MAX_CLIPPED_BOUNDARY_FRACTION`** | **0.15** | Engineering heuristic. | `clipped_boundary_fraction` | If $> 15\%$ of trace excursions hit row midpoints (e.g. S-wave truncation), marked `UNCERTAIN`. |

> [!IMPORTANT]
> **Unknown Calibration vs. Lead Geometry:** Unknown calibration does **not** invalidate lead geometry. If the layout is valid conventional 3×4, lead bounding boxes are drawn, but voltages are explicitly designated as unverified pixel-domain offsets, and the overall alignment status is conservatively marked `UNCERTAIN`.

---

## 3. Model Prediction Terminology Standards

To eliminate overclaiming, model prediction scores are strictly labeled with precise technical descriptors:
- `uncalibrated_model_probability`: Raw sigmoid output $\sigma(z_{\text{target}})$, reflecting model output without clinical calibration.
- `raw_model_logit`: Unbounded linear activation $z_{\text{target}}$ prior to the sigmoid function.
- `model_decision`: Classification threshold comparison result (e.g. `POSITIVE (ABOVE THRESHOLD 0.50)`).
- Model outputs are **never** described as "diagnostic certainty", "clinical probability", or "alignment confidence".

---

## 4. Grad-CAM Handling & Safeguards

- 1D Grad-CAM failed temporal progressive deletion and model-randomization sanity checks in Phase 8.1.
- In Phase 8.2 & 8.2.1, Grad-CAM is **strictly decoupled** from image coordinates.
- Grad-CAM is displayed only as an auxiliary 1D temporal profile along the 10.0-second time axis with a visible methodological disclaimer.
- Mapping Grad-CAM directly onto ECG image pixels is strictly prohibited.
