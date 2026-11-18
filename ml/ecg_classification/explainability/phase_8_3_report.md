# Phase 8.3 — ECG Explainability Backend API Integration & Audit Report

**Date:** 2026-11-01  
**Status:** COMPLETE & VERIFIED  
**Regression Test Results:** 106 passed across all 13 test suites (0 failed)

---

## 1. Executive Summary

Phase 8.3 establishes the production backend API contract connecting the trained 1D ResNet ECG classification model (`ml/models/ecg_classification/deep_learning/best_model.pth`), Integrated Gradients lead attribution, and confidence-gated image-alignment research services into the existing FastAPI backend (`POST /api/ecg/{ecg_id}/explainability`).

The endpoint adheres strictly to healthcare security, authentication, and clinical decision-support safety boundaries:
1. **Multi-Tenant Authentication & Authorization:** Enforces doctor authentication and patient record ownership. Rejects unauthenticated callers (401) and doctors attempting cross-patient access (403).
2. **Clinical Terminology Guardrails:** Uses strictly `uncalibrated_model_probability`, `raw_model_logit`, and `model_decision` with explicit threshold context. Never uses terms like "diagnostic certainty" or "clinical probability".
3. **Audited Provenance & Safe Isolation:** Source ECG images and waveforms remain strictly read-only and immutable. SHA-256 digests are computed and stored for provenance tracking. Explainability artifacts are safely isolated in `uploads/ecg_explainability/patient_{id}/`.
4. **Thread-Safe Model Caching:** Implements `ECGModelRegistry`, a thread-safe singleton caching the deep learning model weights and preprocessing parameters to eliminate per-request disk and memory thrashing.

---

## 2. Pre-Integration Audit & Reconciliation

### 2.1 Regression Count Reconciliation
In previous phases, the test count fluctuated:
- **Phase 8.2:** 101 tests passed across 12 test suites.
- **Phase 8.2.1:** 100 tests passed across 12 test suites.

**Audit Findings:**
In Phase 8.2, `test_ecg_explainability_alignment.py` defined 7 unit tests:
1. `test_conventional_lead_coordinate_mapping`
2. `test_rhythm_strip_handling`
3. `test_missing_lead_behavior`
4. `test_low_confidence_gating`
5. `test_unknown_calibration_gating`
6. `test_attribution_preservation`
7. `test_original_image_immutability`

In Phase 8.2.1, the test suite was refactored into 6 comprehensive semantic tests:
1. `test_independent_confidence_dimensions` (consolidated coordinate mapping and attribution ranking into the 5 independent status dimensions)
2. `test_reliable_geometry_with_unknown_calibration`
3. `test_missing_signal_vs_low_confidence`
4. `test_uncalibrated_measurements_disallow_physical_units`
5. `test_semantic_model_prediction_fields`
6. `test_original_image_immutability`

This intentional consolidation reduced `test_ecg_explainability_alignment.py` from 7 to 6 tests ($101 \to 100$).  
With the addition of the new 6-test suite in Phase 8.3 (`backend/tests/test_ecg_explainability_api.py`), the full regression test suite now reliably passes with **106 tests**.

### 2.2 Documentation & Code Terminology Correction
The README in `ml/ecg_classification/explainability/alignment/README.md` and docstrings in `backend/app/services/ecg/explainability_alignment.py` were audited. Previous wording asserting that "confidence score $\ge 0.70$ directly indicates Pearson $r \ge 0.90$" was removed because that specific numerical correspondence was only observed empirically on initial synthetic sweeps and was not established as an invariant universal truth. The documentation and code comments now accurately describe confidence as an internal tracking metric for bounding-box and trace-continuity quality.

---

## 3. Backend API Contract Specification

### Endpoint: `POST /api/ecg/{ecg_id}/explainability`

#### Request Headers
| Header | Value | Description |
|---|---|---|
| `Authorization` | `Bearer <doctor_jwt_token>` | OAuth2 Bearer token of an authenticated doctor |
| `Content-Type` | `application/json` | JSON payload |

#### Request Body Schema (`ECGExplainabilityRequest`)
```json
{
  "target_class": "MI",
  "integration_steps": 50
}
```
* `target_class` (Optional, string): Must be one of `NORM`, `MI`, `STTC`, `CD`, `HYP`. If omitted, defaults to the highest-probability predicted superclass.
* `integration_steps` (Optional, integer): Number of Riemann approximation steps for Integrated Gradients (allowed range: 10 to 100; default: 50).

#### Response Body Schema (`ECGExplainabilityResponse`)
```json
{
  "source_image_filename": "patient_1042_ecg.png",
  "source_image_sha256": "8f4a1...",
  "image_width": 1665,
  "image_height": 1065,
  "layout_format": "3x4_1rhythm",
  "target_class": "MI",
  "prediction": {
    "target_class": "MI",
    "uncalibrated_model_probability": 0.8421,
    "raw_model_logit": 1.6742,
    "decision_threshold": 0.50,
    "model_decision": "POSITIVE (ABOVE THRESHOLD 0.50)",
    "all_classes_uncalibrated_probabilities": {
      "NORM": 0.0512,
      "MI": 0.8421,
      "STTC": 0.4109,
      "CD": 0.1235,
      "HYP": 0.0811
    },
    "all_classes_raw_logits": {
      "NORM": -2.9192,
      "MI": 1.6742,
      "STTC": -0.3603,
      "CD": -1.9599,
      "HYP": -2.4273
    }
  },
  "attribution_metadata": {
    "method": "Integrated Gradients",
    "baseline": "zero-signal baseline (12, 1000)",
    "steps": 50,
    "convergence_delta": 0.00412,
    "convergence_status": "PASSED (< 0.05)",
    "research_validation_status": "VERIFIED (Passed model randomization & lead deletion audits)"
  },
  "lead_alignments": {
    "V1": {
      "lead_name": "V1",
      "geometry_status": "RELIABLE",
      "calibration_status": "CALIBRATED",
      "reconstruction_status": "SUCCESS",
      "attribution_status": "CONFIDENT",
      "uncertainty_reasons": [],
      "bounding_box": {
        "x_min": 832,
        "y_min": 0,
        "x_max": 1248,
        "y_max": 250,
        "width": 416,
        "height": 250
      },
      "lead_attribution_score": 0.3845,
      "attribution_rank": 1,
      "signal_length_samples": 250,
      "reconstruction_confidence": 0.96
    }
  },
  "auxiliary_gradcam_guidance": {
    "status": "RESEARCH_PROTOTYPE_ONLY",
    "validation_warning": "Grad-CAM failed model-randomization and temporal deletion audits (Phase 8.1.1). Intended solely for qualitative research exploration.",
    "target_conv_layer": "layer3",
    "temporal_weights_summary": {
      "mean": 0.041,
      "max": 0.284,
      "min": 0.0
    }
  },
  "global_metadata": {
    "classifier_model": "ECGResNet1D (12 channels, 1000 samples)",
    "checkpoint_version": "v1.0.3_deep_learning_resnet1d",
    "disclaimer": "RESEARCH ONLY: Experimental alignment and attribution for research review. Not certified for standalone clinical diagnostic decisions."
  }
}
```

---

## 4. HTTP Status Codes & Error Handling

| Status Code | Scenario | Detail / Behavior |
|---|---|---|
| `200 OK` | Valid request | Complete `ECGExplainabilityResponse` returned |
| `401 Unauthorized` | Missing/invalid JWT token | `{"detail": "Could not validate doctor credentials"}` |
| `403 Forbidden` | Doctor requesting unauthorized patient | `{"detail": "Access forbidden: you do not have permission for this patient record"}` |
| `404 Not Found` | Non-existent `ecg_id` | `{"detail": "ECG record {id} not found"}` |
| `422 Unprocessable Content` | Invalid `target_class` or corrupt image | `{"detail": "Unknown target class '{target_class}'. Valid classes: [...]"}` |
| `500 Internal Server Error` | Unexpected processing failure | `{"detail": "Internal error during explainability generation: ..."}` |

---

## 5. Architectural & Security Safeguards

1. **Original Image Immutability:** The service calculates the SHA-256 digest directly from the stored image file and creates isolated temporary PIL image buffers. The source file path and modification timestamp remain completely unchanged.
2. **Multi-Tenant Directory Isolation:** Generated JSON explainability reports are saved to `uploads/ecg_explainability/patient_{patient_id}/ecg_{ecg_id}_{class}_explainability.json`, ensuring complete file separation across hospital tenants.
3. **Thread-Safe Model Singleton:** The 1D ResNet weights (`best_model.pth`, 1.9 MB) and normalization arrays are loaded once via `ECGModelRegistry.get_instance().load_model()` under a thread lock. Concurrency across FastAPI worker threads does not incur repeated disk I/O.
4. **Execution Limits:** `integration_steps` is constrained to $10 \le \text{steps} \le 100$, preventing resource exhaustion from unbounded Riemann steps.

---

## 6. Regression Test Suite Verification

All 13 test suites passed successfully:

```
backend/tests/test_ecg_classification_data.py ......                     [  5%]
backend/tests/test_ecg_classification_models.py .........                [ 14%]
backend/tests/test_ecg_classification_svm.py ....                        [ 17%]
backend/tests/test_ecg_classification_resnet.py .......                  [ 24%]
backend/tests/test_ecg_classification_explainability.py ............     [ 35%]
backend/tests/test_ecg_explainability_alignment.py ......                [ 41%]
backend/tests/test_ecg_explainability_api.py ......                      [ 47%]
backend/tests/test_ecg_dataset_pipeline.py ........                      [ 54%]
backend/tests/test_ecg_measurements.py ..........                        [ 64%]
backend/tests/test_ecg_delineation.py ........                           [ 71%]
backend/tests/test_ecg_reconstruction.py ...........                     [ 82%]
backend/tests/test_ecg_image_processing.py ...........                   [ 92%]
backend/tests/test_ecg_api.py ........                                   [100%]

================= 106 passed, 35 warnings in 85.62s ==================
```

---

## 7. Delivery Status

Phase 8.3 is fully completed. All backend schemas, orchestrator services, authenticated endpoints, and API integration test suites are implemented, audited, and verified. In accordance with strict phase boundaries, no frontend components or client-side UI have been created.
