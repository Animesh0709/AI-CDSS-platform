# Weeks 7–9 delivery guide

## Week 7 — AI clinical assistant

1. Log in and open **AI Clinical Assistant** from the sidebar.
2. Ask a guideline question to view a response grounded in curated RAG excerpts and its sources.
3. Optionally select a patient. The assistant then shows matching stored-report passages and documented history so the clinician can navigate the record.
4. Select **Create record draft** for a transparent, deterministic summary of stored records. It does not infer diagnoses or treatments.

## Week 8 — end-to-end MVP

From a patient profile, demonstrate: upload PDF/image report → OCR → review extracted findings → add/review lab result → retrieve medical evidence → use the assistant with that patient selected. All patient and report access remains scoped to the logged-in doctor.

## Week 9 — improved ML experiment

From `ml/scripts`, run:

```powershell
python train_week9.py --quick
```

Remove `--quick` for the full dataset experiment. The script compares a class-balanced logistic-regression baseline with three XGBoost configurations under stratified three-fold ROC-AUC validation. It saves the selected calibrated model to `ml/models/xgboost_readmission_week9_calibrated.joblib` and a full report to `ml/outputs/week9_metrics.json`.

The report contains validation comparisons, accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and calibration-curve points. Scores remain decision support only and require clinician review and external validation.
