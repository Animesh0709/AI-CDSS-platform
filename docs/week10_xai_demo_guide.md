# Week 10 explainable-AI demo guide

Open a patient profile and use **Readmission prediction & explainability**.

1. Enter the encounter inputs and select **Run prediction & SHAP**.
2. Show the XGBoost model score, local SHAP factors increasing and decreasing the score, the input-schema check, and any out-of-distribution warning.
3. Select **Load global XAI**. The first run computes global mean absolute SHAP, permutation importance, PDP, and ICE from a reproducible held-out dataset sample; later calls reuse it for the running server.
4. Explain that the green PDP line is an average model response while gray ICE lines are individual encounter response curves.
5. If displayed, the counterfactual is explicitly labelled illustrative model behavior, not treatment advice.

The endpoint is `POST /api/xai/patients/{patient_id}/predict`; it enforces patient ownership. Global analysis is available at `GET /api/xai/global`. Both outputs are clinical decision support only and require clinician review.
