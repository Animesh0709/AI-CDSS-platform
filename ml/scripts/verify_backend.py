"""
Week 1 End-to-End Verification Test Script
Tests:
1. Database table creation
2. Doctor registration
3. Doctor login & JWT token extraction
4. Authenticated /me endpoint
5. Patient creation
6. Medical history creation
7. Patient search & statistics endpoint
"""

import sys
import os

# Add backend to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def run_e2e_verification():
    print("=" * 60)
    print("AI-CDSS: Running Week 1 Backend Integration Verification")
    print("=" * 60)
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASS] 1. Health check endpoint OK.")

    # 2. Register Doctor
    doctor_email = "mentor_eval@hospital.org"
    register_payload = {
        "full_name": "Dr. Christopher House",
        "email": doctor_email,
        "password": "ClinicalPass2026!",
        "specialization": "Diagnostic & Internal Medicine",
        "hospital_affiliation": "Princeton Plainsboro Teaching Hospital"
    }
    res = client.post("/api/auth/register", json=register_payload)
    if res.status_code == 400: # Already exists from previous run
        print("[INFO] Doctor already registered, testing login...")
    else:
        assert res.status_code == 201, f"Registration failed: {res.text}"
        print(f"[PASS] 2. Doctor Registration OK. User ID: {res.json()['user']['id']}")

    # 3. Login Doctor
    login_payload = {
        "email": doctor_email,
        "password": "ClinicalPass2026!"
    }
    res = client.post("/api/auth/login", json=login_payload)
    assert res.status_code == 200, f"Login failed: {res.text}"
    token_data = res.json()
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[PASS] 3. Doctor Login & JWT Issuance OK. Token: {token[:20]}...")

    # 4. Get Doctor Profile
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200
    assert res.json()["email"] == doctor_email
    print(f"[PASS] 4. Authenticated Doctor Profile OK: Dr. {res.json()['full_name']}")

    # 5. Create Patient Record
    patient_payload = {
        "first_name": "Eleanor",
        "last_name": "Vance",
        "date_of_birth": "1968-04-12",
        "gender": "Female",
        "phone": "+1 (555) 789-0123",
        "email": "eleanor.v@example.com",
        "address": "452 Maple Ridge, Boston, MA",
        "blood_group": "A+",
        "initial_condition": "Type 2 Diabetes Mellitus with Microalbuminuria",
        "initial_condition_desc": "Patient diagnosed with elevated fasting glucose and early renal markers."
    }
    res = client.post("/api/patients", json=patient_payload, headers=headers)
    assert res.status_code == 201, f"Patient creation failed: {res.text}"
    patient = res.json()
    patient_id = patient["id"]
    print(f"[PASS] 5. Patient Created in DB OK: ID #{patient_id} ({patient['first_name']} {patient['last_name']})")

    # 6. Add Medical History
    history_payload = {
        "condition": "Essential Hypertension",
        "description": "Blood pressure 145/92 mmHg on presentation. Started on Amlodipine 5mg.",
        "diagnosis_date": "2026-02-15",
        "status": "Active"
    }
    res = client.post(f"/api/patients/{patient_id}/history", json=history_payload, headers=headers)
    assert res.status_code == 201, f"Adding history failed: {res.text}"
    print(f"[PASS] 6. Medical History Added OK: {res.json()['condition']}")

    # 7. Query Patient Profile with History
    res = client.get(f"/api/patients/{patient_id}", headers=headers)
    assert res.status_code == 200
    assert len(res.json()["medical_history"]) >= 2
    print(f"[PASS] 7. Patient Profile Retrieved with {len(res.json()['medical_history'])} verified medical conditions.")

    # 8. Test Search & Dashboard Statistics
    res = client.get("/api/patients?search=Eleanor", headers=headers)
    assert res.status_code == 200
    assert len(res.json()) >= 1
    print(f"[PASS] 8. Patient Search Filter OK. Found {len(res.json())} matching records.")

    res = client.get("/api/patients/stats/summary", headers=headers)
    assert res.status_code == 200
    stats = res.json()
    print(f"[PASS] 9. Dashboard Stats OK: Total Patients = {stats['total_patients']}, Tracked Conditions = {stats['total_conditions']}")

    # 10. Edit patient
    res = client.put(
        f"/api/patients/{patient_id}",
        json={"phone": "+1 (555) 000-1111", "blood_group": "O-"},
        headers=headers,
    )
    assert res.status_code == 200, f"Patient update failed: {res.text}"
    assert res.json()["phone"] == "+1 (555) 000-1111"
    print("[PASS] 10. Patient Update (PUT) OK.")

    print("=" * 60)
    print("ALL WEEK 1 BACKEND & DATABASE INTEGRATION CHECKS PASSED (100% SUCCESS)")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_verification()
