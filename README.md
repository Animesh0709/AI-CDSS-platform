# Explainable AI-Based Clinical Decision Support System for Doctors (AI-CDSS)

An end-to-end, doctor-in-the-loop web application combining:
1. Patient Electronic Medical Records (EMR) & Demographics Management
2. Medical Report OCR Text Extraction (PyTesseract & pdf2image)
3. Deterministic Rules-Based Laboratory Value Interpretation
4. 30-Day Hospital Readmission Risk Prediction (XGBoost binary classifier)
5. Explainable AI Engine (SHAP force/summary plots, LIME surrogates, Interactive What-If analysis, Model Overview)
6. Grounded Clinical Guideline Retrieval-Augmented Generation (FAISS vector DB + Extractive evidence)
7. Brain Tumour Multi-Class MRI Classification (ViT, XGBoost, and SVM)
8. **Longitudinal Brain Tumour Progression Prediction (RANO criteria, LUMIERE benchmark, 3-model comparison: Longitudinal XGBoost, SVM, and Deep Neural Net)**

---

> [!IMPORTANT]
> ### Clinical Decision Support Disclaimer
> This system is designed solely as a **decision support prototype** for certified healthcare practitioners and **does not** claim to autonomously diagnose diseases, stage tumours, or prescribe medications. All generated summaries, volumetric deltas, statistical probabilities, and risk assessments require final physician review and validation.

---

## 1. System Architecture & Component Separation

```
                  +-------------------------------------------------------------+
                  |                     Doctor / Clinician                      |
                  +------------------------------+------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |               React 18 + Vite Frontend (JS)                 |
                  |          Medical Green Healthcare Design System             |
                  |  Dashboard · Patients · EMR · OCR · Labs · Readmission      |
                  |  Explainability (SHAP/LIME/What-If) · Guidelines · Diagnosis|
                  |  Tumor Progression Workstation (Serial Scans T0 -> T1)     |
                  +------------------------------+------------------------------+
                                                 | REST APIs (Axios + JWT)
                                                 v
                  +-------------------------------------------------------------+
                  |                  FastAPI Python 3.11 Backend                |
                  |             Authentication & Service Integrator             |
                  +-------+--------------------+---------------------+----------+
                          |                    |                     |
        +-----------------+                    |                     +-----------------+
        v                                      v                                       v
+-----------------------+   +------------------------------------+   +------------------------------------+
|    SQLite Database    |   |     Clinical & Imaging AI Units    |   |     Longitudinal Progression Unit  |
| • users (doctors)     |   | • Readmission: XGBoost + SHAP/LIME |   | • LUMIERE Dataset Benchmark        |
| • patients            |   | • Brain MRI 4-Class: ViT, XGB, SVM |   | • Temporal Volumetric Radiomics    |
| • medical_history     |   | • OCR: PyTesseract Report Parser   |   | • Models: XGBoost, SVM, Deep Net   |
| • reports             |   | • Labs: Reference Range Engine     |   | • Multimodel Consensus Logic       |
| • lab_results         |   | • RAG: FAISS + Extractive Evidence |   | • RANO Outcome Audit & Log         |
| • progression_studies |   +------------------------------------+   +------------------------------------+
+-----------------------+
```

---

## 2. Technology Stack

| Layer | Technologies | Description |
| :--- | :--- | :--- |
| **Frontend** | React 18, Vite, Tailwind CSS, Lucide React, Axios, React Router v6 | Healthcare UI with medical green aesthetic (`#0f4c3a`, `#0e7c5b`), interactive metrics, and modal dialogs |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2.0, Pydantic v2, Uvicorn | Modular RESTful architecture with JWT token auth and route guards |
| **Database** | SQLite (`ai_cdss.db`) with PostgreSQL compatibility | Stores doctors, patients, medical conditions, reports, lab results, and progression studies |
| **Readmission & XAI** | XGBoost, Scikit-learn, SHAP, LIME | 30-day diabetes readmission prediction, patient-level local force values, What-If simulation, and global population overview |
| **Brain MRI 4-Class** | Vision Transformer (ViT), XGBoost, SVM | Four-class tumour classification: Glioma, Meningioma, Pituitary tumour, and No Tumour |
| **Tumor Progression** | XGBoost, Calibrated SVM, PyTorch Deep Net, Joblib | Longitudinal RANO evaluation: Progressive Disease (PD) vs Stable / Response (SD/PR/CR) across serial scans $(T_0 \to T_1)$ |
| **OCR & Guidelines** | PyTesseract, pdf2image, FAISS, Sentence Transformers | Document extraction and source-cited clinical guideline retrieval |

---

## 3. Tumor Progression Prediction Module

### 3.1 Dataset Provenance & Patient-Level Leakage Prevention
- **Dataset**: Published **LUMIERE** longitudinal glioblastoma MRI dataset (*Scientific Data / Nature*, 2022).
- **Expert Ratings**: 616 expert RANO evaluated clinical visits across 91 glioblastoma patients.
- **Pairwise Transitions**: 358 valid temporal pairs $(T_{prev} \to T_{curr})$ across 76 unique patients.
- **Strict Leakage Prevention**:
  - Partitioning is performed strictly by **`patient_id`** before generating pairwise transitions.
  - Scans and radiomic features from a patient never appear across more than one split (Zero patient leakage).
  - Pre-Op / Post-Op baseline scans lacking prior comparator visits are strictly excluded from prediction targets.

### 3.2 Held-Out Test Evaluation Benchmark (70 transitions across 18 patients)

| Model Architecture | PR-AUC (95% CI) | Balanced Acc | Sensitivity (Recall) | Specificity | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Longitudinal XGBoost** | **91.00%** [82.0% – 95.7%] | **80.07%** | **90.91%** (40/44) | **69.23%** (18/26) | **84.97%** | **0.1514** |
| **Longitudinal SVM (Calibrated)** | **81.62%** [65.2% – 90.9%] | **72.38%** | **90.91%** (40/44) | **53.85%** (14/26) | **75.17%** | **0.1858** |
| **Longitudinal Deep Net** | **81.90%** [58.8% – 96.0%] | **76.66%** | **84.09%** (37/44) | **69.23%** (18/26) | **79.90%** | **0.1889** |

*Confidence intervals calculated via 1,000 patient-level stratified bootstrap iterations.*

### 3.3 Target Label Mapping (RANO Criteria)
- **Positive (1)**: Progressive Disease (`PD`, `Post-Op/PD`) requiring treatment modification.
- **Negative (0)**: Stable / Response (`SD`, `PR`, `CR`) where current therapy is maintained.
- **Pseudo-Progression Covariate**: Flags scans within 12 weeks of chemoradiotherapy completion (`is_pseudo_progression_window`) to account for radiation necrosis flares.

---

## 4. API Endpoints Reference

### Authentication & Patients
- `POST /api/auth/register` — Register doctor account
- `POST /api/auth/login` — Authenticate and receive JWT access token
- `GET /api/auth/me` — Current authenticated doctor profile
- `GET /api/patients` — List patients assigned to current doctor
- `POST /api/patients` — Register new patient record
- `GET /api/patients/{id}` — Retrieve complete patient profile and history

### Readmission & Explainable AI
- `POST /api/xai/patients/{id}/predict` — Predict encounter readmission score with OOD detection
- `POST /api/xai/patients/{id}/explain` — Compute local SHAP and LIME explanations
- `POST /api/xai/patients/{id}/whatif` — Interactive What-If simulation with feature overrides
- `GET /api/xai/global-summary` — Pre-computed global population feature importance

### Brain MRI 4-Class Diagnosis
- `POST /api/mri/predict` — Upload single MRI scan image; runs ViT, XGBoost, and SVM
- `GET /api/mri/eval-report` — Four-class test set metrics and confusion matrices
- `GET /api/mri/model-info` — Dataset metadata and limitations

### Tumor Progression Prediction
- `POST /api/progression/predict` — Evaluate serial scan observation $(T_0 \to T_1)$, run all 3 models, determine consensus, and persist study
- `GET /api/progression/eval-report` — Held-out test evaluation report with bootstrap 95% CIs
- `GET /api/progression/model-info` — Dataset provenance and feature definitions
- `GET /api/progression/patients/{id}/history` — Chronological trajectory of previous studies for patient
- `POST /api/progression/studies/{id}/confirm` — Radiologist RANO outcome confirmation (`PD`, `SD`, `PR`, `CR`) and audit notes

---

## 5. Local Setup & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11.16)
- Node.js 18+ & npm
- macOS / Linux / Windows WSL

### Step 1: Environment Setup
```bash
# Clone repository and enter directory
cd ai-cdss-clean

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Step 2: Running Backend Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
*API Swagger Documentation is accessible at `http://localhost:8000/docs`.*

### Step 3: Running Frontend Application
```bash
cd frontend
npm install
npm run dev
```
*Doctor Portal is accessible at `http://localhost:5173`.*

### Step 4: Model Training & Evaluation Commands
```bash
# Train XGBoost Readmission Model
python ml/scripts/train.py

# Train 4-Class Brain MRI Models (XGBoost, SVM, ViT)
python ml/scripts/train_mri.py

# Preprocess LUMIERE Longitudinal Dataset (Zero-Leakage Patient Split)
python ml/scripts/progression_preprocess.py

# Train & Benchmark 3 Longitudinal Progression Models
python ml/scripts/train_progression.py
```

### Step 5: Running Automated Test Suites
```bash
cd backend
pytest -v
```
*Expected test outcome: 23 passed, 1 skipped in ~17 seconds.*

---

## 6. Known Limitations & Safe Use Boundary

1. **Research & Educational Decision Support Only**: This platform is not a certified Medical Device (SaMD) and has not received FDA 510(k) or CE-mark approval.
2. **Pseudo-Progression Confounding**: Contrast enhancement enlargement within 12 weeks of radiation therapy cannot be distinguished from radiation necrosis with 100% certainty via structural MRI alone; advanced perfusion MRI (rCBV) or biopsy remains required.
3. **Out-of-Distribution Inputs**: Models trained on high-grade adult glioblastoma should not be applied to pediatric brain tumours or secondary metastases.
4. **Physician-in-the-Loop Requirement**: All predictions require multidisciplinary review before initiating or altering oncological therapy.
