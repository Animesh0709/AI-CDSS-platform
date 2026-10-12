# Week 12 testing and evaluation

## Automated checks

Run from `backend`:

```powershell
python -m pytest tests -q
```

Coverage includes authentication-protected patient access, upload-size validation, OCR extraction/failure handling, RAG grounding guardrails, assistant response schema and patient ownership, plus XAI ownership and input validation. The PDF OCR test skips only when the optional `reportlab` test fixture package is absent; this does not mask an OCR failure.

Run the frontend production verification from `frontend`:

```powershell
npm run build
```

## Manual frontend and responsive flow

Test at 1440px, 768px, and 375px widths:

1. Register/login, then verify protected routes redirect after logout.
2. Add, search, edit, and open a patient.
3. Upload a report, inspect OCR text, and review NLP matches before confirming a condition.
4. Add a lab result and verify its configured rule basis/status.
5. Run the XAI prediction, global XAI, patient-aware evidence answer, and stored-record summary draft.
6. Confirm controls remain visible, forms do not overflow, and the sidebar collapses on the mobile layout.

## ML evaluation

`ml/outputs/week9_metrics.json` records the reproducible three-fold comparison. On the stored quick experiment, tuned shallow XGBoost had validation ROC-AUC 0.6557, above the logistic baseline (0.6241). Its held-out ROC-AUC was 0.6795, precision 0.1835, recall 0.5746, and F1 0.2782. These modest values are reported to avoid misleading accuracy-only evaluation.

The sigmoid-calibrated model retained ROC-AUC 0.6794, but produced no positive predictions at a 0.5 threshold in that saved run. This demonstrates why calibration must be inspected together with a clinically justified operating threshold; calibrated probabilities must not be presented as validated clinical probabilities without external validation.

## XAI and RAG evaluation

The XAI route validates numeric ranges, patient ownership, feature schema alignment, and warns for numeric inputs more than three standard deviations from training reference data. Local SHAP, global mean absolute SHAP, permutation importance, PDP, and ICE describe model behavior—not causal or treatment effects.

RAG questions are answered only from retrieved local excerpts. The automated suite verifies a known diabetes question returns cited evidence and an unrelated bicycle question returns `insufficient_evidence`, so it is not sent to an LLM.

## Limitations

- The UCI diabetes encounter dataset is historical and not a deployment-ready local clinical population.
- Patient-level model input currently uses a limited encounter form; missing categorical encounter features use training-reference baselines.
- OCR and rule-based NLP can miss or misread report content; clinician review is mandatory.
- The curated RAG corpus is intentionally small and cannot answer all medical questions.
- This prototype does not diagnose, prescribe, or replace clinician judgment.
