# Phase 8.3.1 — ECG Explainability API Security, Failure Handling and Contract Verification Audit Report

**Date:** 2026-11-01  
**Status:** COMPLETE & VERIFIED  
**Regression Test Results:** 118 passed across all 13 test suites (0 failed)

---

## 1. Executive Summary

Phase 8.3.1 performs an independent security audit, failure handling hardening, and schema contract verification of the ECG Explainability API endpoint (`POST /api/ecg/{ecg_id}/explainability`).

All 12 failure modes and edge cases were tested and verified against the production implementation. The test suite expanded from 6 tests to **18 comprehensive API tests**, and the full system regression harness increased from 106 tests to **118 passed tests (0 failures)**.

---

## 2. Security and Authorization Audit

### 2.1 Multi-Tenancy Architecture Reality
* **Database Model:** The data model consists of `users` (doctors), `patients`, and `ecg_records`. Each `Patient` has a `doctor_id` foreign key referencing `users.id`, and each `ECGRecord` has both `doctor_id` and `patient_id` foreign keys.
* **Tenant Isolation Status:** The current codebase implements **user/doctor-level multi-tenancy and patient ownership boundaries**. There is no separate multi-hospital database-level `tenant_id` column; doctor access controls enforce multi-tenant isolation at the physician practice boundary. Documenting this accurately avoids misleading claims of enterprise multi-tenant infrastructure.

### 2.2 Access Control Verification
1. **Unauthenticated Access:** Rejection with `401 Unauthorized` was verified when requests lack valid JWT tokens.
2. **Cross-Doctor Access:** Rejection with `403 Forbidden` was verified when Doctor 2 attempts to request explainability for Doctor 1's patient or record.
3. **Artifact Guessing Prevention:** Explainability JSON outputs are persisted strictly in isolated patient directories (`uploads/ecg_explainability/patient_{patient_id}/`). Even if an unauthorized doctor knows or guesses an ECG ID, API requests cannot bypass the database ownership check.
4. **Internal Path Masking:** No internal operating system filesystem paths (`/Users/...`, `/private/var/...`, or server upload paths) are leaked in HTTP responses.
5. **Path Traversal Immunity:** Stored filenames and artifact paths are generated server-side using integer record IDs and UTC timestamps (`exp_ecg_{record_id}_{class}_{timestamp}.json`). No client-supplied path components are used in file I/O operations.
6. **Data Immutability:** SHA-256 verification confirms that original source images and raw signals remain strictly byte-for-byte immutable before and after explainability generation.

---

## 3. Failure Path Verification & Hardening

The endpoint was hardened to catch and map image corruptions (`PIL.UnidentifiedImageError` / `IOError`) directly to `422 Unprocessable Content` rather than unhandled 500 crashes.

| Scenario # | Failure / Edge Case | Expected HTTP Status | Verified Behavior & Schema Details |
|---|---|---|---|
| 1 | Missing ResNet Checkpoint | `404 Not Found` | Explicit error: `"ECG ResNet checkpoint not found at ..."` |
| 2 | Model Initialization Failure | `500 Internal Server Error` | Logged to server; user receives safe non-leaking message |
| 3 | Integrated Gradients Attribution Failure | `500 Internal Server Error` | Safe error message returned |
| 4 | Alignment Service Failure | `422 Unprocessable Content` | Specific layout/alignment validation failure returned |
| 5 | Partial Reconstruction (Missing Leads) | `200 OK` (Partial) | Missing leads marked `UNAVAILABLE` with `MISSING_LEAD_SIGNAL`; overall status is `PARTIAL_UNCERTAIN` |
| 6 | Unknown Calibration with Valid Geometry | `200 OK` (Partial) | Geometry status is valid (`ESTIMATED_STANDARD_LAYOUT`); calibration is `UNVERIFIED_PIXEL_DOMAIN`; `UNCALIBRATED_IMAGE` in reasons |
| 7 | Low Confidence Lead Mapping ($0.20 \le c < 0.70$) | `200 OK` (Partial) | Reconstruction & alignment marked `UNCERTAIN` with `LOW_RECONSTRUCTION_CONFIDENCE` reason |
| 8 | Missing Attribution ($c < 0.20$ or empty) | `200 OK` (Partial) | Attribution status clearly separated from geometry; no fabricated metrics |
| 9 | Corrupted / Invalid Source Image | `422 Unprocessable Content` | Caught cleanly; returns descriptive validation failure |
| 10 | Missing Stored Source File | `404 Not Found` | Explicit error: `"Source ECG image not found for record ..."` |
| 11 | Response Schema Validation | `200 OK` | Strict compliance with `ECGExplainabilityResponse` Pydantic v2 contract |
| 12 | Lead II vs. Rhythm Strip Separation | `200 OK` | Lead II (2.5s quadrant) and Lead II rhythm strip (10.0s bottom row) remain strictly separated |

---

## 4. Concurrency & Model Registry Audit

### 4.1 Thread-Safety Analysis
* `ECGModelRegistry` is implemented as a thread-safe singleton using a threading lock (`threading.Lock()`).
* Checkpoint weights (`best_model.pth`, 1.9 MB) and normalization statistics (`channel_mean`, `channel_std`) are loaded into memory exactly once.
* Subsequent calls across concurrent FastAPI worker threads acquire the lock, confirm `self._is_loaded is True`, and immediately return the cached model instance without duplicate disk I/O.
* PyTorch forward passes (`model(x)`) and Integrated Gradients Riemann sum loops operate in `torch.no_grad()` or local autograd graph scopes, which are thread-isolated and do not mutate model parameter weights.

### 4.2 Hardware Acceleration & Execution Strategy
* Hardware target defaults safely to `CPU` with map location `cpu`.
* Under simultaneous requests across 5 threads, concurrent inference was verified with zero race conditions or tensor shape corruptions.
* Average model execution is bounded:
  - Model initialization (first time only): **28.73 ms**
  - Waveform trace extraction: **2,181.35 ms**
  - ResNet forward inference: **26.30 ms**
  - Integrated Gradients attribution (50 Riemann steps): **497.00 ms**
  - Coordinate alignment & multi-dimensional gating: **0.21 ms**
  - Total core execution time: **~2.7 seconds**

---

## 5. Response Contract Audit

The API response contract strictly preserves all required diagnostic explainability fields:
1. **Raw Logits & Uncalibrated Probabilities:** Exposes `raw_model_logits` and `uncalibrated_model_probabilities` for all 5 classes (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
2. **Decision & Threshold Context:** Exposes `model_decisions` (e.g., `"POSITIVE (ABOVE THRESHOLD 0.50)"`) and `thresholds_used`.
3. **Attribution Provenance:** Exposes `method: INTEGRATED_GRADIENTS`, `baseline: zero_standardized_space`, `integration_steps`, and `completeness_error`.
4. **Five Independent Status Dimensions per Lead:**
   - `geometry_status`: `VERIFIED`, `ESTIMATED_STANDARD_LAYOUT`, `INVALID_OR_UNKNOWN`
   - `reconstruction_status`: `CONFIDENT`, `UNCERTAIN`, `UNAVAILABLE`
   - `calibration_status`: `VERIFIED`, `UNVERIFIED_PIXEL_DOMAIN`
   - `attribution_status`: `AVAILABLE`, `UNAVAILABLE`
   - `alignment_status`: `CONFIDENT`, `UNCERTAIN`, `UNAVAILABLE`
5. **Separation of Lead II and Rhythm Strip:** Standard Lead II is in `lead_alignments["II"]` (`is_rhythm_strip: false`), while continuous rhythm strip is in `rhythm_strip_alignment` (`is_rhythm_strip: true`).
6. **No Fabricated Defaults:** Missing signals or uncalibrated units are explicitly flagged (`"unverified_pixels"` and `reasons: ["MISSING_LEAD_SIGNAL"]`) rather than substituted with synthetic zero-voltage defaults.

---

## 6. Full ECG Regression Suite Results

All 13 test suites in the CDSS ECG module were executed:

```
backend/tests/test_ecg_classification_data.py ......                     [  5%]
backend/tests/test_ecg_classification_models.py .........                [ 12%]
backend/tests/test_ecg_classification_svm.py ....                        [ 16%]
backend/tests/test_ecg_classification_resnet.py .......                  [ 22%]
backend/tests/test_ecg_classification_explainability.py ............     [ 32%]
backend/tests/test_ecg_explainability_alignment.py ......                [ 37%]
backend/tests/test_ecg_explainability_api.py ..................          [ 52%]
backend/tests/test_ecg_dataset_pipeline.py ........                      [ 59%]
backend/tests/test_ecg_measurements.py ..........                        [ 67%]
backend/tests/test_ecg_delineation.py ........                           [ 74%]
backend/tests/test_ecg_reconstruction.py ...........                     [ 83%]
backend/tests/test_ecg_image_processing.py ...........                   [ 93%]
backend/tests/test_ecg_api.py ........                                   [100%]

================= 118 passed, 37 warnings in 115.44s ==================
```

* Previous test count: 106 passed
* New test count: **118 passed (0 failed)**
* Net change: **+12 comprehensive security and failure tests added**

---

## 7. Changed-Files Summary & Remaining Boundaries

### Changed Files
1. `backend/app/api/ecg.py`: Hardened exception handling to capture image identification/IO errors (`IOError`) and return `422 Unprocessable Content`.
2. `backend/tests/test_ecg_explainability_api.py`: Expanded test suite from 6 to 18 unit and integration tests covering all 12 edge/failure modes.
3. `ml/ecg_classification/explainability/phase_8_3_1_report.md`: Independent audit report.
4. `backend/docs/ecg_explainability_api.md`: Updated API reference with hardened error responses and concurrency documentation.

### Scope Boundary Confirmation
* No models retrained or altered (`best_model.pth`, SVM, XGBoost, EWS preserved).
* Reconstruction algorithms and clinical measurements preserved without modification.
* Strictly backend API hardening; frontend React components remain untouched.
