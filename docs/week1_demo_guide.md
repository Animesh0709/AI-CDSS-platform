# Week 1 Mentor Demonstration Script

This document details the step-by-step procedure to demonstrate the completed **Week 1 Foundation** to mentors and evaluators.

---

## Demonstration Workflow

### 1. Doctor Registration & JWT Authentication
- Open the web application at `http://localhost:5173`.
- Navigate to **Register Doctor Account**.
- Register a doctor profile (e.g. `Dr. Sarah Jenkins`, `s.jenkins@metrohospital.org`, specialization: `Cardiology / Internal Medicine`).
- Observe instant JWT token generation and storage in `localStorage`.

### 2. Clinical Dashboard Overview
- Observe the **Doctor Portal** banner and doctor-in-the-loop clinical disclaimer:
  *"This system organizes patient metrics and provides statistical risk estimates to assist certified doctors. It does not autonomously diagnose medical conditions or issue prescriptions."*
- View initial statistics: Assigned Patients count, Active Diagnoses, and Readmission Baseline module status.
- Observe the 12-Week Roadmap sidebar with Week 1 active and Week 2–12 labeled.

### 3. Add New Patient Record
- Click **Add New Patient** from the dashboard or sidebar.
- Enter patient demographics:
  - **Name**: `Robert Davis`
  - **DOB**: `1965-08-14` (Age: ~61)
  - **Gender**: `Male`
  - **Blood Group**: `O+`
  - **Phone**: `+1 (555) 349-2180`
  - **Primary Diagnosis**: `Type 2 Diabetes Mellitus with Peripheral Neuropathy`
- Click **Save & Open Profile**.

### 4. Database Persistence Verification
- Verify patient profile loaded with ID `#PT-0001`.
- Verify record persisted in the relational database (`patients` and `medical_history` tables).

### 5. Medical History Management
- On the patient profile, click **Add Medical Condition**.
- Enter: `Hypertension (Stage 2)` | Status: `Chronic` | Notes: `Prescribed Lisinopril 20mg daily.`
- Click **Save Diagnosis** and observe real-time timeline update.

### 6. Patient Search & Directory
- Click **Patients** in the sidebar.
- Test the search bar by typing `Robert` or `Davis` or `555`.
- Filter by gender (`Male` / `Female`).
- Open the profile, then click **Edit Patient**, change phone or blood group, and save. Confirm the profile shows the updated values from PostgreSQL.

### 7. Machine Learning Readmission Pipeline Demonstration
- Open a terminal and run the ML pipeline:
  ```bash
  python ml/scripts/train.py
  ```
- Demonstrate:
  - Dataset schema loading (UCI Diabetes 130-US Hospitals dataset, 100k+ encounters / representative sample).
  - Target definition: 30-day hospital readmission (`<30` as 1, else 0).
  - Categorical feature encoding & numeric standard scaling.
  - XGBoost training with `scale_pos_weight` handling class imbalance.
  - Evaluation metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
  - Feature importance output (top drivers of readmission).
- Run sample inference:
  ```bash
  python ml/scripts/predict_sample.py
  ```
  Demonstrates a sample encounter probability from the trained artifacts. The patient profile does **not** show a fake risk score in Week 1.
