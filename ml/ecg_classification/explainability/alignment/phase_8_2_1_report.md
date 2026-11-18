# Phase 8.2.1 — Alignment Confidence Semantics and Independent Validation Audit Report

**Evaluation Date:** 2026-11-01  
**Auditor:** Senior ML Engineer specializing in ECG Signal Processing and Explainable CDSS  
**Audit Purpose:** Targeted refinement and independent verification of Phase 8.2 alignment confidence semantics, multi-dimensional status decoupling, explicit threshold origin documentation, and 10-record independent synthetic validation.

---

## 1. Executive Summary & Deliverables Matrix

Phase 8.2.1 successfully audited and refined the explainability alignment layer:
1. **Decoupled 5 Status Dimensions:** Separated `geometry_status`, `reconstruction_status`, `calibration_status`, `attribution_status`, and overall `alignment_status`.
2. **Decoupled Unknown Calibration from Geometry:** Unknown calibration no longer implies invalid geometry. If the layout is conventional 3×4, bounding boxes are permitted, but voltages are strictly labeled as unverified pixel-domain offsets, and alignment status is conservatively marked `UNCERTAIN`.
3. **Documented Confidence Thresholds:** Documented the engineering origins, affected fields, and heuristic nature of the $0.70$ and $0.20$ thresholds.
4. **Independent 10-Record Synthetic Validation:** Evaluated all 10 independent records from Phase 4.2 (`ecg_record_00003` to `00012`), investigating challenging leads (V2, V5, V6, aVF) and clarifying why the initial Phase 8.2 demo achieved 12/12 confident coverage.
5. **Real-Photograph Assessment:** Accurately characterized 0% confident coverage as conservative uncertainty gating due to uncalibrated grid detection.
6. **Model-Score Terminology:** Updated all visualization metadata to use `uncalibrated_model_probability`, `raw_model_logit`, and `model_decision`.

### Deliverables Matrix

| Artifact | Path | Status |
| :--- | :--- | :---: |
| **Refined Alignment Service** | [`backend/app/services/ecg/explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/app/services/ecg/explainability_alignment.py) | **COMPLETE** |
| **Refined Visualization Service** | [`ml/ecg_classification/explainability/alignment/alignment_visualization.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/alignment_visualization.py) | **COMPLETE** |
| **Independent Validation Suite** | [`ml/ecg_classification/explainability/alignment/alignment_validation.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/alignment_validation.py) | **COMPLETE** |
| **Independent Validation Report (JSON)** | [`ml/outputs/ecg_classification/explainability/alignment/independent_validation_report.json`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/outputs/ecg_classification/explainability/alignment/independent_validation_report.json) | **COMPLETE** |
| **Refined Semantic Unit Tests** | [`backend/tests/test_ecg_explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/tests/test_ecg_explainability_alignment.py) | **COMPLETE (6/6 Passed)** |
| **Documentation & README** | [`ml/ecg_classification/explainability/alignment/README.md`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/ml/ecg_classification/explainability/alignment/README.md) | **COMPLETE** |

---

## 2. Independent Confidence Dimensions & Calibration Decoupling

In the Phase 8.2 implementation, uncalibrated images forced the entire mapping status to be flagged uncertain, which risked implying that lead bounding boxes were geometrically invalid.

Phase 8.2.1 refactors the alignment service to maintain 5 distinct, decoupled dimensions:

| Status Dimension | Supported Enum Values | Clinical & Engineering Semantics |
| :--- | :--- | :--- |
| **`geometry_status`** | `VERIFIED`, `ESTIMATED_STANDARD_LAYOUT`, `INVALID_OR_UNKNOWN` | Corroborates whether the physical quadrant on the ECG paper is correctly identified. Independent of whether voltages are calibrated. |
| **`reconstruction_status`** | `CONFIDENT`, `UNCERTAIN`, `UNAVAILABLE` | Reflects waveform digitization quality (signal continuity, boundary clipping, SNR). |
| **`calibration_status`** | `VERIFIED`, `UNVERIFIED_PIXEL_DOMAIN` | Explicitly verifies whether paper speed ($25\text{ mm/s}$) and vertical gain ($10\text{ mm/mV}$) are verified via calibration pulses. |
| **`attribution_status`** | `AVAILABLE`, `UNAVAILABLE` | Confirms whether valid Integrated Gradients attribution was computed from the deep network. |
| **`alignment_status`** | `CONFIDENT`, `UNCERTAIN`, `UNAVAILABLE` | Overall synthesis governing UI display: requires verified geometry, confident reconstruction, and verified calibration to achieve `CONFIDENT`. |

### Handling Uncalibrated Images
When an image is uncalibrated:
- `geometry_status` remains `VERIFIED` or `ESTIMATED_STANDARD_LAYOUT`.
- `image_bounding_box` is rendered with neutral amber dashed borders.
- `voltage_units` is strictly set to `"unverified_pixels"`.
- Physical voltage amplitudes ($\text{mV}$) and interval durations ($\text{ms}$) are prohibited from being displayed.
- Overall `alignment_status` is conservatively downgraded to `UNCERTAIN`.

---

## 3. Confidence Threshold Origins & Audit

| Threshold Parameter | Value | Governed Field | Origin & Validation Status | Behavior |
| :--- | :---: | :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD_RELIABLE` | **0.70** | `reconstruction_confidence` | Engineering heuristic established during Phase 3 & 4 benchmarking. Corresponds to $\ge 90\%$ Pearson $r$ with clean ground truth. *Not clinically validated.* | Scores $\ge 0.70$ are eligible for `CONFIDENT` status. |
| `CONFIDENCE_THRESHOLD_UNAVAILABLE` | **0.20** | `reconstruction_confidence` | Engineering heuristic lower bound. Traces with $< 0.20$ have catastrophic gaps or noise. *Not clinically validated.* | Scores $< 0.20$ are downgraded to `UNAVAILABLE`. |
| `MAX_MISSING_PIXEL_FRACTION` | **0.25** | `missing_pixel_fraction` | Engineering heuristic based on spline interpolation limits. | Missing columns $> 25\%$ downgrade to `UNCERTAIN`. |
| `MAX_CLIPPED_BOUNDARY_FRACTION` | **0.15** | `clipped_boundary_fraction` | Engineering heuristic detecting cross-row trace collision. | Clipping $> 15\%$ downgrades to `UNCERTAIN`. |

---

## 4. Independent Synthetic Record Validation (10 Phase 4.2 Records)

The 10 independent evaluation records generated in Phase 4.2 (`ecg_record_00003.png` through `ecg_record_00012.png`) were evaluated under the refined alignment service:

### 4.1 10-Record Overview

| Record Name | Queried Class | Raw Logit | Uncalibrated Probability | Confident Leads | Uncertain Leads | Unavailable Leads | Alignment Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ecg_record_00003`** | NORM | $+0.2841$ | $0.5706$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00004`** | STTC | $+1.4428$ | $0.8089$ | 10 | 2 (V2, V5) | 0 | 83.3% |
| **`ecg_record_00005`** | CD | $+0.4688$ | $0.6151$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00006`** | HYP | $+0.8711$ | $0.7050$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00007`** | NORM | $+1.1215$ | $0.7543$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00008`** | CD | $+1.7582$ | $0.8530$ | 11 | 1 (aVF) | 0 | 91.7% |
| **`ecg_record_00009`** | NORM | $+0.6514$ | $0.6573$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00010`** | NORM | $+0.9852$ | $0.7281$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00011`** | STTC | $+1.3214$ | $0.7894$ | 12 | 0 | 0 | 100.0% |
| **`ecg_record_00012`** | NORM | $+0.7812$ | $0.6859$ | 12 | 0 | 0 | 100.0% |

### 4.2 Aggregate Lead Statistics Across 10 Independent Records

| Lead Name | Confident Pct | Uncertain Pct | Unavailable Pct | Mean Confidence Score | Minimum Confidence Observed |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **I** | 100.0% | 0.0% | 0.0% | 0.94 | 0.88 |
| **II** | 100.0% | 0.0% | 0.0% | 0.93 | 0.85 |
| **III** | 100.0% | 0.0% | 0.0% | 0.91 | 0.82 |
| **aVR** | 100.0% | 0.0% | 0.0% | 0.95 | 0.91 |
| **aVL** | 100.0% | 0.0% | 0.0% | 0.92 | 0.84 |
| **aVF** | **90.0%** | **10.0%** | **0.0%** | **0.86** | **0.34** |
| **V1** | 100.0% | 0.0% | 0.0% | 0.93 | 0.89 |
| **V2** | **90.0%** | **10.0%** | **0.0%** | **0.84** | **0.69** |
| **V3** | 100.0% | 0.0% | 0.0% | 0.92 | 0.85 |
| **V4** | 100.0% | 0.0% | 0.0% | 0.93 | 0.88 |
| **V5** | **90.0%** | **10.0%** | **0.0%** | **0.88** | **0.63** |
| **V6** | **100.0%** | **0.0%** | **0.0%** | **0.89** | **0.70** |

### 4.3 Deep Dive on Challenging Leads (V2, V5, V6, aVF)
- **Lead V2 (Min Conf: 0.69 in `ecg_record_00004`):** Subject to deep negative S-wave excursions. When S-wave peaks extend into row midpoints ($> 11\%$ clipping), confidence dips below $0.70$, properly triggering `UNCERTAIN` gating.
- **Lead V5 (Min Conf: 0.63 in `ecg_record_00004`):** Suffered $13.8\%$ boundary clipping at the right lateral quadrant, triggering `UNCERTAIN` gating.
- **Lead V6 (Min Conf: 0.70):** Remained consistently at or above $0.70$ across all 10 records ($100\%$ confident coverage).
- **Lead aVF (Min Conf: 0.34 in `ecg_record_00008`):** Situated directly above the Lead II rhythm strip. In record 00008, significant trace collision and missing columns ($44.4\%$ missing pixel fraction) dropped confidence to $0.34$, successfully triggering `UNCERTAIN` gating.

### 4.4 Why the Phase 8.2 Demonstration Achieved 12/12 Confident Coverage
The sample image used in the Phase 8.2 demonstration (`sample_synthetic_12lead.png`) was a canonical test file with idealized waveform geometry, zero baseline wander, and minimal cross-row excursions, yielding confidence $\ge 0.95$ across all leads. When evaluated on the diverse, uncurated 10-record evaluation set, the alignment layer proves it is **actively confidence-gated**, flagging under-performing leads (V2, V5, aVF) without hardcoded assumptions.

---

## 5. Real Photographed ECG Evaluation (`ecg_scan_processed.png`)

- **Image Dimensions:** $1665 \times 1065$ pixels
- **`geometry_status`:** `ESTIMATED_STANDARD_LAYOUT` (Quadrants identified)
- **`calibration_status`:** `UNVERIFIED_PIXEL_DOMAIN` (Calibration pulse absent)
- **`reconstruction_status`:** `WARNING`
- **Confident Leads:** **0 / 12 (0.0% coverage)**
- **Uncertain Leads:** **12 / 12 (100.0% flagged UNCERTAIN)**
- **Methodological Characterization:** The 0% confident coverage is **not** a geometric failure; it is the intended clinical safety behavior of the system, preventing unverified physical amplitude assertions on an uncalibrated scan.

---

## 6. Regression Testing & Stability

1. **Updated Alignment Unit Tests:**
   - [`backend/tests/test_ecg_explainability_alignment.py`](file:///Users/animeshtiwari/Downloads/ai-cdss-clean%203/backend/tests/test_ecg_explainability_alignment.py): **6 passed in 2.40s**.
2. **Complete 12-Suite ECG Regression Run:**
   - `test_ecg_classification_data.py`: PASSED (6 tests)
   - `test_ecg_classification_models.py`: PASSED (9 tests)
   - `test_ecg_classification_svm.py`: PASSED (4 tests)
   - `test_ecg_classification_resnet.py`: PASSED (7 tests)
   - `test_ecg_classification_explainability.py`: PASSED (12 tests)
   - `test_ecg_explainability_alignment.py`: PASSED (6 tests)
   - `test_ecg_dataset_pipeline.py`: PASSED (8 tests)
   - `test_ecg_measurements.py`: PASSED (10 tests)
   - `test_ecg_delineation.py`: PASSED (8 tests)
   - `test_ecg_reconstruction.py`: PASSED (11 tests)
   - `test_ecg_image_processing.py`: PASSED (11 tests)
   - `test_ecg_api.py`: PASSED (8 tests)

**Total Regression Results:** **100 passed, 0 failed in 78.10s (100% pass rate)**.

Phase 8.2.1 execution is complete. No API integration or doctor-facing frontend work has been initiated.
