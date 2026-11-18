# Phase 8.2 — ECG Explainability-to-Image Alignment and Confidence-Gated Mapping Report

**Evaluation Date:** 2026-11-01  
**Lead Engineer:** Senior ML Engineer specializing in ECG Signal Processing and Explainable CDSS  
**Module Deliverables:**
- Backend Service: [`backend/app/services/ecg/explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/app/services/ecg/explainability_alignment.py)
- Visualization Engine: [`ml/ecg_classification/explainability/alignment/alignment_visualization.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/alignment_visualization.py)
- Validation Suite: [`ml/ecg_classification/explainability/alignment/alignment_validation.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/alignment_validation.py)
- Unit Tests: [`backend/tests/test_ecg_explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/tests/test_ecg_explainability_alignment.py)
- Documentation: [`ml/ecg_classification/explainability/alignment/README.md`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/README.md)

---

## 1. Executive Summary & Verification Matrix

Phase 8.2 establishes a research-grade alignment layer connecting waveform-level explanations (Integrated Gradients lead attributions and exploratory 1D Grad-CAM temporal profiles) directly with original 12-lead ECG images. 

Crucially, the alignment layer enforces strict clinical safeguards:
1. **No Hallucinated Visual Evidence:** Attribution remains strictly a **lead-level** metric. It is represented as analytical bounding regions and lead rank badges rather than continuous pixel-level saliency masks.
2. **Confidence-Gated Mapping:** Reconstructions that are uncalibrated, clipped, or low-confidence are explicitly gated and flagged as `UNCERTAIN` or `UNAVAILABLE`.
3. **Decoupled Grad-CAM:** Because 1D Grad-CAM failed temporal progressive deletion and model randomization in Phase 8.1, it is quarantined as an auxiliary 1D waveform guide with a prominent clinical disclaimer and is **never** projected onto image pixels.

### Verification Matrix & Verdicts

| Dimension | Evaluation Protocol | Empirical Findings | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Lead-Coordinate Mapping** | Dynamic bounding-box derivation for conventional 3×4 layout + Lead II rhythm strip | All 12 leads correctly mapped to their quadrants (Rows: I..V4, II..V5, III..V6). Continuous rhythm strip accurately spans the full width of the 4th row. | **PASS** |
| **2. Missing Lead Handling** | Reconstructing partial signals with missing leads | Missing leads are reliably marked `UNAVAILABLE` with `MISSING_LEAD_SIGNAL` uncertainty code; no false overlays are generated. | **PASS** |
| **3. Low-Confidence Gating** | Leads with confidence $< 0.70$ | Reliably downgraded to `UNCERTAIN` with `LOW_RECONSTRUCTION_CONFIDENCE` code; neutral hatching applied. | **PASS** |
| **4. Uncalibrated Image Gating** | Inputs with missing or unverified calibration pulse | All lead regions automatically flagged `UNCERTAIN` with `UNCALIBRATED_IMAGE` code; voltages labeled unverified. | **PASS** |
| **5. Attribution Preservation** | Rank and magnitude fidelity through mapping pipeline | Exact numerical attributions and relative rankings preserved ($0.0$ deviation). | **PASS** |
| **6. Original-Image Immutability** | Byte-level image hashing before and after visualization | Source image bytes remain $100.0\%$ identical. | **PASS** |
| **7. Synthetic vs. Real Validation** | End-to-end evaluation on synthetic renders vs. real photographed scans | Synthetic render: **100.0%** confident coverage (12/12). Real photograph: **0.0%** confident coverage (12/12 flagged `UNCERTAIN` due to unverified calibration), demonstrating proper clinical gating. | **PASS** |

---

## 2. Lead-Coordinate Mapping Architecture

The alignment service dynamically computes bounding boxes for each lead quadrant based on detected grid boundaries:
- **Row 0:** I ($[0.0, 2.5]\text{s}$), aVR ($[2.5, 5.0]\text{s}$), V1 ($[5.0, 7.5]\text{s}$), V4 ($[7.5, 10.0]\text{s}$)
- **Row 1:** II ($[0.0, 2.5]\text{s}$), aVL ($[2.5, 5.0]\text{s}$), V2 ($[5.0, 7.5]\text{s}$), V5 ($[7.5, 10.0]\text{s}$)
- **Row 2:** III ($[0.0, 2.5]\text{s}$), aVF ($[2.5, 5.0]\text{s}$), V3 ($[5.0, 7.5]\text{s}$), V6 ($[7.5, 10.0]\text{s}$)
- **Row 3 (Rhythm):** Lead II Continuous Rhythm Strip ($[0.0, 10.0]\text{s}$, spanning the full grid width)

Vertical row partitions are established via natural Voronoi midpoints between calibrated baselines to prevent lead intrusion across rows while accommodating tall R-waves and deep S-waves.

---

## 3. Confidence-Gating Rules

```
                      Lead Signal & Metadata
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       Signal Missing?               Confidence < 0.20?
       ├── YES: UNAVAILABLE          ├── YES: UNAVAILABLE
       └── NO                        └── NO
                │
                ├─────────────────────────────┐
                ▼                             ▼
      Confidence < 0.70?             Calibration Unverified?
      ├── YES: UNCERTAIN             ├── YES: UNCERTAIN
      └── NO                         └── NO
                │
                ├─────────────────────────────┐
                ▼                             ▼
     Missing Pixels > 25%?           Boundary Clipping > 15%?
     ├── YES: UNCERTAIN             ├── YES: UNCERTAIN
     └── NO                         └── NO
                │
                ▼
            CONFIDENT
```

1. **`CONFIDENT`:** Rendered with solid borders, rank badge, and subtle steel-blue background tint proportional to attribution rank.
2. **`UNCERTAIN`:** Rendered with amber dashed borders, diagonal warning hatch (`//`), and caution badge indicating the specific reason (e.g. `[UNCERTAIN: Conf 0.55]`).
3. **`UNAVAILABLE`:** Rendered with neutral gray dotted borders, cross-hatching (`xx`), and an explicit `[UNAVAILABLE]` tag.

---

## 4. Synthetic vs. Real Photographed ECG Validation

End-to-end extraction, classification, Integrated Gradients attribution, alignment, and visualization were executed on representative samples:

### 4.1 Synthetic Render Validation (`sample_synthetic_12lead.png`)
- **Image Dimensions:** $1665 \times 1065$ pixels
- **Calibration Status:** Verified ($25.0\text{ mm/s}$, $10.0\text{ mm/mV}$)
- **Reconstruction Status:** `SUCCESS` (Mean confidence $= 0.98$)
- **Model Output:** Class `MI` (Logit $= +1.8054$, Probability $= 0.8588$, Decision: POSITIVE)
- **Confident Leads:** **12 / 12 (100.0% coverage)**
- **Uncertain / Unavailable Leads:** 0 / 0
- **Artifact:** [`ml/outputs/ecg_classification/explainability/alignment/synthetic_alignment_prototype.png`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/outputs/ecg_classification/explainability/alignment/synthetic_alignment_prototype.png)

### 4.2 Real Photographed ECG Validation (`uploads/ecg/processed_test/ecg_scan_processed.png`)
- **Image Dimensions:** $1665 \times 1065$ pixels
- **Calibration Status:** `UNVERIFIED` (Calibration pulse absent or non-standard)
- **Reconstruction Status:** `WARNING`
- **Model Output:** Class `HYP` (Logit $= +21.2764$, Probability $= 1.0000$)
- **Confident Leads:** **0 / 12 (0.0% coverage)**
- **Uncertain Leads:** **12 / 12 (100.0% flagged UNCERTAIN)**
- **Gating Behavior:** The system properly suppressed all high-confidence overlays, preventing unverified physical interpretations on an uncalibrated photograph.
- **Artifact:** [`ml/outputs/ecg_classification/explainability/alignment/real_photograph_alignment_prototype.png`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/outputs/ecg_classification/explainability/alignment/real_photograph_alignment_prototype.png)

---

## 5. Regression Testing & Stability

1. **New Alignment Unit Tests:**
   - [`backend/tests/test_ecg_explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/tests/test_ecg_explainability_alignment.py): **7 passed in 2.34s**.
2. **Full ECG Regression Suite (All 12 Modules):**
   - `test_ecg_classification_data.py`: PASSED (6 tests)
   - `test_ecg_classification_models.py`: PASSED (9 tests)
   - `test_ecg_classification_svm.py`: PASSED (4 tests)
   - `test_ecg_classification_resnet.py`: PASSED (7 tests)
   - `test_ecg_classification_explainability.py`: PASSED (12 tests)
   - `test_ecg_explainability_alignment.py`: PASSED (7 tests)
   - `test_ecg_dataset_pipeline.py`: PASSED (8 tests)
   - `test_ecg_measurements.py`: PASSED (10 tests)
   - `test_ecg_delineation.py`: PASSED (8 tests)
   - `test_ecg_reconstruction.py`: PASSED (11 tests)
   - `test_ecg_image_processing.py`: PASSED (11 tests)
   - `test_ecg_api.py`: PASSED (8 tests)

**Total Regression Results:** **101 passed, 0 failed in 78.50s (100% pass rate)**.

---

## 6. Known Limitations & Architectural Boundaries

1. **Non-Standard ECG Layouts:** Layout mapping currently assumes the conventional 3×4 + Lead II rhythm strip format. Cabrero layouts or 6×2 layouts require explicit layout detection flags before bounding box instantiation.
2. **Perspective Distortion:** Photographed mobile phone scans with severe trapezoidal perspective skew must pass projective deskewing in Phase 3 before lead-coordinate mapping can be considered reliable.
3. **No Doctor-Facing Integration:** In accordance with constraints, this alignment layer remains strictly a research module and has **not** been integrated into the doctor-facing production frontend.

Phase 8.2 execution is complete.
