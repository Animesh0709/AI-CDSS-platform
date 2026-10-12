# Week 11 unified patient dashboard demo

Open a patient profile after logging in. The page presents separate, clinician-reviewable sections rather than one combined AI result:

1. **Patient overview and medical history** — demographics plus doctor-recorded conditions.
2. **Reports and OCR/NLP review** — upload a PDF/image, inspect extracted text, then review possible clinical entities before confirming any condition.
3. **Lab analysis** — add a laboratory value and inspect the configured rule basis and status.
4. **Readmission prediction and XAI** — enter encounter inputs, run the XGBoost score, inspect local SHAP contributors, then load global SHAP, permutation importance, PDP, and ICE.
5. **Medical evidence** — ask a guideline question. On a patient profile, the answer is grounded in curated sources and may show matching stored-report passages only for record navigation.
6. **AI clinical summary** — generate a transparent draft compiled only from this patient’s stored history and extracted reports.

Every AI-related panel is labelled clinical decision support and requires clinician review. No output diagnoses or prescribes.
