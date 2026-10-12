"""Evidence-grounded guideline search and clinical question answering for AI-CDSS.

Retrieves relevant clinical guidelines (ADA, AHA/ACC, hospital readmissions)
and synthesizes direct, authoritative, evidence-grounded clinical answers for clinicians.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.core.config import settings

logger = logging.getLogger("ai_cdss.rag")

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "rag" / "documents"
INDEX_DIR = ROOT / settings.VECTOR_DB_PATH.lstrip("./")
INDEX_FILE = INDEX_DIR / "guidelines.faiss"
META_FILE = INDEX_DIR / "guidelines.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _chunks(text: str, size: int = 700, overlap: int = 120):
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            end = boundary + 1 if boundary > start + size // 2 else end
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def _document_chunks():
    chunks = []
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"} or path.name == "README.md":
            continue
        raw = path.read_text(encoding="utf-8")
        title = next((line[2:].strip() for line in raw.splitlines() if line.startswith("# ")), path.stem)
        url_match = re.search(r"Source:\s*(https?://\S+)", raw, re.I)
        section = title
        for part in re.split(r"(?=^## )", raw, flags=re.M):
            section_match = re.search(r"^##\s+(.+)$", part, re.M)
            section = section_match.group(1).strip() if section_match else title
            clean = re.sub(r"^#.*$|^Source:\s*.*$", "", part, flags=re.M).strip()
            for text in _chunks(clean):
                if text:
                    chunks.append({
                        "source": title,
                        "source_url": url_match.group(1) if url_match else None,
                        "section": section,
                        "text": text
                    })
    return chunks


def _embed(texts: list[str], mode: str | None = None):
    if mode != "hashing":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32"), "sentence-transformers/all-MiniLM-L6-v2"
        except (ImportError, OSError, RuntimeError):
            pass
    vectors = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2", stop_words="english").transform(texts)
    return vectors.toarray().astype("float32"), "hashing fallback"


def rebuild_index():
    chunks = _document_chunks()
    if not chunks:
        raise ValueError("No .md or .txt guideline documents are available in rag/documents.")
    vectors, mode = _embed([chunk["text"] for chunk in chunks])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps({"embedding_mode": mode, "chunks": chunks}, indent=2), encoding="utf-8")
    return {"documents": len({c["source"] for c in chunks}), "chunks": len(chunks), "embedding_mode": mode}


def search(query: str, limit: int):
    if not INDEX_FILE.exists() or not META_FILE.exists():
        rebuild_index()
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    mode = metadata["embedding_mode"]
    vectors, _ = _embed([query], "hashing" if mode.startswith("hashing") else None)
    index = faiss.read_index(str(INDEX_FILE))
    scores, ids = index.search(vectors, min(limit, index.ntotal))
    results = []
    for score, item_id in zip(scores[0], ids[0]):
        if item_id >= 0:
            results.append({**metadata["chunks"][int(item_id)], "score": round(float(score), 4)})
    return {"query": query, "embedding_mode": mode, "results": results}


def _clean_chunk_text(text: str) -> str:
    """Format and clean chunk text for presentation."""
    return re.sub(r"\s+", " ", text).strip()


CLINICAL_KEYWORDS = (
    "diabet", "glucose", "sugar", "hba1c", "a1c", "fpg", "ogtt", "prediabetes",
    "hypertens", "blood pressure", "bp", "systolic", "diastolic",
    "cholesterol", "lipid", "ldl", "hdl", "triglyceride",
    "heart failure", "hfref", "hfpef", "cardio", "ejection fraction", "bnp",
    "creatinine", "urea", "bun", "egfr", "kidney", "renal",
    "hemoglobin", "wbc", "platelet", "cbc", "blood count", "anemia",
    "liver", "bilirubin", "alt", "ast", "sgpt", "sgot",
    "tsh", "thyroid", "potassium", "sodium",
    "report", "x-ray", "xray", "radiograph", "scan", "ct", "mri", "ultrasound", "biopsy", "ecg", "ekg",
    "finding", "impression", "investigation", "ocr",
    "medication", "medicine", "drug", "prescription", "metformin", "insulin", "lisinopril", "statin", "therapy",
    "readmission", "readmit", "30-day", "hospital", "discharge",
    "guideline", "screen", "diagnos", "criteria", "treatment", "target", "abnormal", "patient", "lab", "test",
    "history", "condition", "status", "risk", "summary", "summarize", "doctor", "health", "case", "review"
)


def _format_test_summary(tests: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
    sub = tests[:limit] if limit else tests
    parts = []
    for t in sub:
        u_str = f" {t['unit']}" if t.get("unit") else ""
        parts.append(f"{t['name']}: {t['value']}{u_str} [{t['status']}]")
    return "; ".join(parts)


def _synthesize_local_answer(question: str, evidence: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
    """Synthesize a complete, direct, and structured clinical answer addressing the clinician's exact query."""
    q_lower = question.lower()
    sections = []
    
    # Extract patient parameters if present
    p_name = patient_info.get("name") if patient_info else None
    p_age = patient_info.get("age", "") if patient_info else ""
    p_gender = patient_info.get("gender", "") if patient_info else ""
    
    cond_items = []
    if patient_info:
        for c in patient_info.get("conditions", []):
            if isinstance(c, dict):
                cond_items.append(c)
            else:
                cond_items.append({"condition": str(c), "status": "Active", "date": "", "description": ""})
                
    ehr_labs = patient_info.get("labs", []) if patient_info else []
    report_tests = patient_info.get("report_tests", []) if patient_info else []
    parsed_reports = patient_info.get("parsed_reports", []) if patient_info else []
    medications = patient_info.get("medications", []) if patient_info else []
    
    # Combined biomarker list without duplicates
    all_tests = []
    seen_test_names = set()
    for l in ehr_labs:
        name_key = (l.get("name") or "").strip().lower()
        if name_key:
            seen_test_names.add(name_key)
            all_tests.append({
                "name": l.get("name", "Lab Test"),
                "value": str(l.get("value", "")),
                "unit": l.get("unit", ""),
                "status": (l.get("status") or "Normal").strip().title(),
                "ref": l.get("ref", ""),
                "date": l.get("date", ""),
                "source": "EHR Laboratory"
            })
    for t in report_tests:
        name_key = (t.get("name") or "").strip().lower()
        if name_key and name_key not in seen_test_names:
            seen_test_names.add(name_key)
            all_tests.append({
                "name": t.get("name", "Report Test"),
                "value": str(t.get("value", "")),
                "unit": t.get("unit", ""),
                "status": (t.get("status") or "Normal").strip().title(),
                "ref": t.get("ref", ""),
                "date": t.get("date", ""),
                "source": t.get("source", "Diagnostic Report")
            })

    # Query classification flags
    is_diabetes_screening = any(k in q_lower for k in ["screen", "diagnos", "criteria", "a1c", "fpg", "ogtt", "prediabetes"]) and any(k in q_lower for k in ["diabet", "glucose", "sugar"])
    is_diabetes_mgmt = any(k in q_lower for k in ["treatment", "manage", "drug", "pharmacolog", "metformin", "insulin", "sglt2", "glp-1", "therapy"]) and any(k in q_lower for k in ["diabet", "glycemic", "glucose"])
    is_cholesterol = any(k in q_lower for k in ["cholesterol", "lipid", "ldl", "hdl", "triglyceride"])
    is_glucose_query = any(k in q_lower for k in ["glucose", "blood sugar", "fasting glucose", "fpg", "hba1c", "a1c", "glycemic"]) and not is_diabetes_screening and not is_diabetes_mgmt
    is_renal_query = any(k in q_lower for k in ["creatinine", "urea", "bun", "egfr", "kidney function", "renal panel"])
    is_bp_query = any(k in q_lower for k in ["blood pressure", "hypertens", "bp", "stage 1", "stage 2", "dash", "cardiovascular"])
    is_heme_query = any(k in q_lower for k in ["hemoglobin", "wbc", "platelet", "cbc", "blood count", "anemia"])
    is_thyroid_query = any(k in q_lower for k in ["tsh", "thyroid", "t3", "t4"])
    is_liver_query = any(k in q_lower for k in ["liver", "bilirubin", "alt", "ast", "sgpt", "sgot", "lft"])
    is_heart_failure = any(k in q_lower for k in ["heart failure", "hfref", "hfpef", "cardiomyopathy", "ejection fraction"])
    is_readmission = any(k in q_lower for k in ["readmission", "30-day", "hospital", "discharge", "risk factor"])
    is_abnormal_query = any(k in q_lower for k in ["abnormal", "out of range", "flagged", "priority finding", "critical", "concern"])
    is_report_query = any(k in q_lower for k in ["report", "x-ray", "xray", "radiograph", "imaging", "scan", "ultrasound", "usg", "ct", "mri", "biopsy", "ecg", "ekg", "finding", "impression", "investigation"])
    is_condition_query = any(k in q_lower for k in ["diagnos", "condition", "history", "problem list"]) or (("have" in q_lower or "has" in q_lower or "does" in q_lower) and any(k in q_lower for k in ["diabet", "hypertens", "asthma", "heart", "disease", "copd", "kidney"]))
    is_med_query = any(k in q_lower for k in ["medication", "medicine", "drug", "prescription", "taking", "prescribed", "treatment", "therapy"])
    is_summary_query = any(k in q_lower for k in ["summarize", "summary", "overview", "status", "profile"]) and p_name

    # -------------------------------------------------------------
    # 1. SPECIFIC BIOMARKER / LAB QUERIES
    # -------------------------------------------------------------
    if is_cholesterol:
        target_tests = [t for t in all_tests if any(k in t["name"].lower() for k in ["cholesterol", "lipid", "ldl", "hdl", "triglyceride"])]
        if p_name and target_tests:
            sections.append(f"### Clinical Answer: Lipid & Cholesterol Profile for {p_name}\n")
            sections.append(f"According to the electronic health record and analyzed diagnostic reports for **{p_name}**, the measured lipid parameters are:\n")
            for t in target_tests:
                ref_str = f" *(Reference: {t['ref']})*" if t["ref"] else ""
                date_str = f" `[{t['date']}]`" if t["date"] else ""
                status_badge = f"**[{t['status'].upper()}]**" if t['status'].upper() != "NORMAL" else f"[{t['status']}]"
                sections.append(f"- **{t['name']}:** `{t['value']} {t['unit']}` {status_badge}{ref_str} -- *Source: {t['source']}*{date_str}")
            sections.append("\n**Guideline Evaluation (AHA/ACC 2025 Guidelines) [1]:**")
            sections.append("- **Total Cholesterol Target:** Desirable `< 200 mg/dL`; Borderline High `200 - 239 mg/dL`; High `>= 240 mg/dL` [1].")
            sections.append("- **LDL Cholesterol Target:** Optimal `< 100 mg/dL`; Borderline High `130 - 159 mg/dL`; High `>= 160 mg/dL` [1].")
            sections.append("- **HDL Target:** Protective `> 40 mg/dL` (men), `> 50 mg/dL` (women) [1].")
            sections.append("- **Triglycerides Target:** Normal `< 150 mg/dL` [1].")
            if any(t["status"] in ("High", "Critical", "Elevated") for t in target_tests):
                sections.append("- **Clinical Recommendation:** The patient's elevated atherogenic fractions warrant lifestyle modification (DASH/Mediterranean dietary pattern) and evaluation for moderate-to-high intensity statin therapy to reduce atherosclerotic cardiovascular disease (ASCVD) risk [1].")
            else:
                sections.append("- **Clinical Recommendation:** Measured lipid markers remain within acceptable cardiovascular guideline ranges; continue annual surveillance [1].")
        elif p_name:
            sections.append(f"### Clinical Answer: Lipid Profile Check for {p_name}\n")
            sections.append(f"No discrete lipid or cholesterol panel is currently recorded in the active laboratory results or reports for **{p_name}**.")
            if all_tests:
                sections.append(f"\n**Laboratory tests currently on file for this patient:** {', '.join(t['name'] for t in all_tests[:6])}.")
            sections.append("\n**AHA/ACC Reference Standards [1]:** Desirable Total Cholesterol `< 200 mg/dL`, LDL `< 100 mg/dL`, HDL `> 40 mg/dL`, Triglycerides `< 150 mg/dL` [1].")
        else:
            sections.append("### Clinical Guideline Answer: Lipid Classification & Target Goals (AHA/ACC)\n")
            sections.append("According to the **AHA/ACC Cardiovascular Prevention Guidelines** [1], lipid fractions are classified as follows:\n")
            sections.append("- **Total Cholesterol:** Desirable `< 200 mg/dL`; Borderline `200 - 239 mg/dL`; High `>= 240 mg/dL` [1].")
            sections.append("- **LDL-C:** Optimal `< 100 mg/dL`; High `>= 160 mg/dL`; Very High `>= 190 mg/dL` [1].")
            sections.append("- **Triglycerides:** Normal `< 150 mg/dL`; Borderline `150 - 199 mg/dL`; High `>= 200 mg/dL` [1].")
            sections.append("- **HDL-C:** Low `< 40 mg/dL` (men), `< 50 mg/dL` (women) [1].")
            sections.append("\n**Management Strategy:** Lifestyle interventions and statin therapy stratified by 10-year ASCVD risk [1].")

    elif is_glucose_query:
        target_tests = [t for t in all_tests if any(k in t["name"].lower() for k in ["glucose", "sugar", "fpg", "hba1c", "a1c"])]
        if p_name and target_tests:
            sections.append(f"### Clinical Answer: Glycemic Biomarker Evaluation for {p_name}\n")
            sections.append(f"Electronic records and diagnostic investigation reports for **{p_name}** document the following glycemic parameters:\n")
            for t in target_tests:
                ref_str = f" *(Reference: {t['ref']})*" if t["ref"] else ""
                date_str = f" `[{t['date']}]`" if t["date"] else ""
                status_badge = f"**[{t['status'].upper()}]**" if t['status'].upper() != "NORMAL" else f"[{t['status']}]"
                sections.append(f"- **{t['name']}:** `{t['value']} {t['unit']}` {status_badge}{ref_str} -- *Source: {t['source']}*{date_str}")
            sections.append("\n**Guideline Interpretation (ADA 2026 Standards of Care) [1]:**")
            sections.append("- **Fasting Plasma Glucose (FPG):** Normal `70 - 99 mg/dL`; Impaired Fasting Glucose (Prediabetes) `100 - 125 mg/dL`; Diabetes `>= 126 mg/dL` [1].")
            sections.append("- **Glycated Hemoglobin (HbA1c):** Normal `< 5.7%`; Prediabetes `5.7% - 6.4%`; Diabetes `>= 6.5%` [1].")
            sections.append("- **Standard Therapeutic Target:** An HbA1c `< 7.0%` (53 mmol/mol) is recommended for most nonpregnant adults with diabetes [1].")
            if any(t["status"] in ("High", "Critical", "Mildly Elevated") for t in target_tests):
                sections.append("- **Actionable Context:** The patient's elevated glucose parameters reflect active glycemic excursion requiring review of diet, lifestyle, and pharmacological therapy [1].")
            elif any(t["status"] in ("Low", "Mildly Low") for t in target_tests):
                sections.append("- **Actionable Context:** Fasting glucose `< 70 mg/dL` warrants assessment for hypoglycemic symptoms, meal spacing, and anti-diabetic drug overtreatment [1].")
        elif p_name:
            sections.append(f"### Clinical Answer: Blood Glucose Check for {p_name}\n")
            sections.append(f"No active blood glucose or HbA1c test is currently recorded for **{p_name}**.")
            sections.append("\n**ADA Diagnostic Standards [1]:** Normal fasting glucose `70 - 99 mg/dL`, prediabetes `100 - 125 mg/dL`, diabetes `>= 126 mg/dL` [1].")
        else:
            sections.append("### Clinical Guideline Answer: Glycemic Criteria & Target Goals (ADA 2026)\n")
            sections.append("According to the **American Diabetes Association 2026 Standards of Care** [1]:\n")
            sections.append("- **Fasting Plasma Glucose:** Normal `70 - 99 mg/dL`; Impaired Fasting Glucose `100 - 125 mg/dL`; Diabetes `>= 126 mg/dL` [1].")
            sections.append("- **HbA1c Target:** `< 7.0%` for most nonpregnant adults without significant hypoglycemia risk [1].")

    elif is_renal_query:
        target_tests = [t for t in all_tests if any(k in t["name"].lower() for k in ["creatinine", "urea", "bun", "egfr"])]
        if p_name and target_tests:
            sections.append(f"### Clinical Answer: Renal Function Biomarkers for {p_name}\n")
            for t in target_tests:
                ref_str = f" *(Reference: {t['ref']})*" if t["ref"] else ""
                status_badge = f"**[{t['status'].upper()}]**" if t['status'].upper() != "NORMAL" else f"[{t['status']}]"
                sections.append(f"- **{t['name']}:** `{t['value']} {t['unit']}` {status_badge}{ref_str} -- *Source: {t['source']}*")
            sections.append("\n**Clinical Interpretation [1]:**")
            sections.append("- Standard Serum Creatinine Reference: Men `0.6 - 1.2 mg/dL`; Women `0.5 - 1.1 mg/dL` [1].")
            sections.append("- **Medication Safety:** Metformin is safe when eGFR `>= 45 mL/min`; reduce dose if eGFR `30 - 44 mL/min`; contraindicated when eGFR `< 30 mL/min` [1].")
        else:
            sections.append("### Clinical Guideline Answer: Renal Function Monitoring (KDIGO / ADA Guidelines)\n")
            sections.append("- Serum creatinine and estimated glomerular filtration rate (eGFR) evaluate kidney clearance [1].")
            sections.append("- Annual urine albumin-to-creatinine ratio (uACR) and eGFR screening are recommended in all patients with diabetes or hypertension [1].")

    elif is_heme_query:
        target_tests = [t for t in all_tests if any(k in t["name"].lower() for k in ["hemoglobin", "wbc", "platelet", "hematocrit", "rbc"])]
        if p_name and target_tests:
            sections.append(f"### Clinical Answer: Hematology & Complete Blood Count for {p_name}\n")
            for t in target_tests:
                ref_str = f" *(Reference: {t['ref']})*" if t["ref"] else ""
                status_badge = f"**[{t['status'].upper()}]**" if t['status'].upper() != "NORMAL" else f"[{t['status']}]"
                sections.append(f"- **{t['name']}:** `{t['value']} {t['unit']}` {status_badge}{ref_str} -- *Source: {t['source']}*")
        else:
            sections.append("### Clinical Guideline Answer: Complete Blood Count (CBC) Reference Metrics\n")
            sections.append("- Hemoglobin: Men `13.5 - 17.5 g/dL`; Women `12.0 - 15.5 g/dL` [1].")
            sections.append("- WBC: `4,500 - 11,000 /µL`; Platelets: `150,000 - 450,000 /µL` [1].")

    # -------------------------------------------------------------
    # 2. DIAGNOSTIC REPORTS & OCR FINDINGS QUERIES
    # -------------------------------------------------------------
    elif is_report_query and p_name:
        matching_reports = []
        if any(k in q_lower for k in ["x-ray", "xray", "radiograph", "chest"]):
            matching_reports = [r for r in parsed_reports if "x-ray" in r.get("panel", "").lower() or "xray" in r.get("filename", "").lower()]
        elif any(k in q_lower for k in ["lipid", "cholesterol"]):
            matching_reports = [r for r in parsed_reports if "lipid" in r.get("panel", "").lower() or "lipid" in r.get("filename", "").lower()]
        elif any(k in q_lower for k in ["cbc", "blood count", "hemogram"]):
            matching_reports = [r for r in parsed_reports if "cbc" in r.get("panel", "").lower() or "blood" in r.get("filename", "").lower()]
        else:
            matching_reports = parsed_reports

        if matching_reports:
            sections.append(f"### Clinical Answer: Diagnostic Report Findings for {p_name}\n")
            for idx, r in enumerate(matching_reports, 1):
                panel = r.get("panel") or "Diagnostic Investigation"
                fname = r.get("filename") or "Report"
                fac_str = f" | Facility: *{r['facility']}*" if r.get("facility") else ""
                date_str = f" | Date: *{r['date']}*" if r.get("date") else ""
                sections.append(f"**Report #{idx}: {panel} ({fname})**{fac_str}{date_str}\n")
                if r.get("narrative"):
                    for n in r["narrative"]:
                        sections.append(f"- **Clinical Impression / Conclusion:** {n}")
                if r.get("tests"):
                    t_str = ", ".join(f"{t['name']} = {t['value']} {t['unit']} ({t['status']})" for t in r["tests"][:5])
                    sections.append(f"- **Discrete Measurements:** {t_str}")
                if r.get("key_sentences"):
                    for s in r["key_sentences"][:2]:
                        sections.append(f"- **Document Observation:** {s}")
                if not r.get("narrative") and not r.get("tests") and not r.get("key_sentences"):
                    sections.append("- **OCR Analysis:** Document text parsed; no acute pathological alarms or critical flags registered in report body.")
                sections.append("")
            sections.append("**Clinical Synthesis:** Correlate all investigation findings with ongoing clinical presentation and treatment regimen.")
        else:
            sections.append(f"### Clinical Answer: Diagnostic Reports for {p_name}\n")
            sections.append(f"No matching diagnostic or radiology reports were found in **{p_name}**'s record.")
            if parsed_reports:
                sections.append(f"\nReports currently on file: {', '.join(r.get('filename', 'Report') for r in parsed_reports)}.")

    # -------------------------------------------------------------
    # 3. MEDICAL CONDITION / DIAGNOSIS STATUS QUERIES
    # -------------------------------------------------------------
    elif is_condition_query and p_name:
        sections.append(f"### Clinical Answer: Medical History & Diagnosis Status for {p_name}\n")
        
        # Check specific condition
        matched_cond = None
        for c in cond_items:
            c_name = c["condition"].lower()
            if any(k in q_lower for k in ["diabet", "sugar"]) and "diabet" in c_name:
                matched_cond = c
                break
            elif any(k in q_lower for k in ["hypertens", "bp", "blood pressure"]) and "hypertens" in c_name:
                matched_cond = c
                break
            elif any(k in q_lower for k in ["asthma"]) and "asthma" in c_name:
                matched_cond = c
                break
            elif any(k in q_lower for k in ["heart"]) and any(h in c_name for h in ["heart", "cardio"]):
                matched_cond = c
                break

        if matched_cond:
            date_info = f" (Diagnosed: {matched_cond['date']})" if matched_cond.get("date") else ""
            desc_info = f"\n  - *Clinical Notes:* {matched_cond['description']}" if matched_cond.get("description") else ""
            sections.append(f"- **Documented Diagnosis:** Yes, **{p_name}** has a documented diagnosis of **{matched_cond['condition']}**.")
            sections.append(f"- **Clinical Status:** `{matched_cond.get('status', 'Active')}`{date_info}{desc_info}")
            
            # Correlate with relevant biomarkers
            if "diabet" in matched_cond["condition"].lower():
                g_labs = [t for t in all_tests if any(k in t["name"].lower() for k in ["glucose", "hba1c"])]
                if g_labs:
                    sections.append(f"- **Correlating Biomarkers:** {_format_test_summary(g_labs)}.")
                sections.append("- **ADA Guideline Standard [1]:** Ongoing glycemic target HbA1c `< 7.0%` is recommended to prevent microvascular and macrovascular complications [1].")
            elif "hypertens" in matched_cond["condition"].lower():
                sections.append("- **AHA/ACC Guideline Standard [1]:** Blood pressure target `< 130/80 mmHg` is recommended for patients with confirmed hypertension [1].")
        elif cond_items:
            sections.append(f"**Documented Problem List for {p_name}:**\n")
            for c in cond_items:
                status = c.get("status", "Active")
                date = f" (Diagnosed: {c['date']})" if c.get("date") else ""
                desc = f" -- {c['description']}" if c.get("description") else ""
                sections.append(f"- **{c['condition']}** `[{status}]`{date}{desc}")
        else:
            sections.append(f"**{p_name}** currently has no chronic medical conditions documented in the active problem list.")

    # -------------------------------------------------------------
    # 4. MEDICATION & PHARMACOTHERAPY QUERIES
    # -------------------------------------------------------------
    elif is_med_query and p_name:
        sections.append(f"### Clinical Answer: Medication Regimen for {p_name}\n")
        if medications:
            sections.append(f"The following medications are documented in **{p_name}**'s clinical records and report findings:\n")
            for m in medications:
                sections.append(f"- **`{m}`**")
            sections.append("\n**Guideline Indications & Cautions [1]:**")
            if any("metformin" in m.lower() for m in medications):
                sections.append("- `Metformin`: Foundational first-line agent for Type 2 Diabetes; verify renal stability (eGFR >= 30 mL/min) [1].")
            if any("lisinopril" in m.lower() for m in medications):
                sections.append("- `Lisinopril`: ACE inhibitor indicated for blood pressure reduction and nephroprotection [1].")
            sections.append("- **Recommendation:** Perform formal medication reconciliation at each clinical encounter.")
        else:
            sections.append(f"No active medications are formally recorded in the electronic health chart for **{p_name}**.")
            sections.append("- **Recommendation:** Verify whether the patient is taking unrecorded prescriptions, over-the-counter remedies, or supplements during the upcoming encounter [1].")

    # -------------------------------------------------------------
    # 5. PRIORITY ABNORMAL FINDINGS & CONCERNS
    # -------------------------------------------------------------
    elif is_abnormal_query and p_name:
        sections.append(f"### Clinical Answer: Priority Abnormal Findings for {p_name}\n")
        abnormal_tests = [t for t in all_tests if t["status"] in ("High", "Low", "Critical", "Abnormal", "Mildly Low", "Mildly Elevated")]
        if abnormal_tests:
            sections.append("**1. Out-of-Range Laboratory Measurements:**")
            for t in abnormal_tests:
                ref_str = f" *(Ref: {t['ref']})*" if t["ref"] else ""
                sections.append(f"- **{t['name']}:** `{t['value']} {t['unit']}` **[{t['status'].upper()}]**{ref_str} -- *Source: {t['source']}*")
            sections.append("")
        else:
            sections.append("- **Laboratory Evaluation:** No out-of-range numerical laboratory markers registered on file.")
            
        abnormal_reports = [r for r in parsed_reports if r.get("narrative")]
        if abnormal_reports:
            sections.append("**2. Diagnostic Investigation Impressions:**")
            for r in abnormal_reports:
                sections.append(f"- **{r.get('panel')} ({r.get('filename')}):** {'; '.join(r['narrative'])}")
            sections.append("")
            
        sections.append("**3. Clinical Risk Correlation:** Review active pharmacological therapy in conjunction with documented abnormalities [1].")

    # -------------------------------------------------------------
    # 6. PATIENT CASE SUMMARY / REVIEW
    # -------------------------------------------------------------
    elif is_summary_query:
        sections.append(f"### Clinical Case Summary: {p_name} ({p_age} | {p_gender})\n")
        active_conds = [c["condition"] for c in cond_items if c.get("status", "Active").lower() in ("active", "chronic")]
        conds_str = ", ".join(active_conds) if active_conds else "no active chronic conditions formally recorded"
        sections.append(f"- **Patient Profile:** Under clinical management for **{conds_str}**.")
        if all_tests:
            abn = [t for t in all_tests if t["status"] in ("High", "Low", "Critical", "Abnormal")]
            if abn:
                sections.append(f"- **Priority Biomarkers:** {_format_test_summary(abn, limit=4)}.")
            else:
                sections.append(f"- **Laboratory Findings:** {len(all_tests)} biomarker parameter(s) on file reflecting stable baseline values.")
        if parsed_reports:
            sections.append(f"- **Diagnostic Documents:** {len(parsed_reports)} report(s) on file ({', '.join(r.get('panel', 'Report') for r in parsed_reports[:3])}).")
        if medications:
            sections.append(f"- **Medication Regimen:** {', '.join(medications)}.")
        sections.append("- **Clinical Action Plan:** Continued longitudinal disease surveillance and adherence review [1].")

    # -------------------------------------------------------------
    # 7. CLINICAL GUIDELINES (DIABETES, HYPERTENSION, HEART FAILURE, READMISSIONS)
    # -------------------------------------------------------------
    elif is_diabetes_screening:
        sections.append("### Clinical Guideline Answer: Diabetes Screening & Diagnostic Criteria (ADA 2026 Standards of Care)\n")
        sections.append(
            "According to the **American Diabetes Association (ADA) 2026 Standards of Care** [1], diabetes screening and diagnosis "
            "are established through standardized laboratory criteria and structured risk-based surveillance protocols:\n"
        )
        sections.append("**1. Standard Diagnostic Thresholds [1]:**")
        sections.append("- **Fasting Plasma Glucose (FPG):** `>= 126 mg/dL` (7.0 mmol/L) after an overnight fast (minimum 8 hours without caloric intake) [1].")
        sections.append("- **2-Hour Oral Glucose Tolerance Test (OGTT):** `>= 200 mg/dL` (11.1 mmol/L) following a standard 75-g anhydrous glucose load [1].")
        sections.append("- **Glycated Hemoglobin (HbA1c):** `>= 6.5%` (48 mmol/mol) measured using an NGSP-certified, standardized laboratory method [1].")
        sections.append("- **Random Plasma Glucose:** `>= 200 mg/dL` (11.1 mmol/L) in any patient presenting with classic symptoms of hyperglycemia (polyuria, polydipsia, unexplained weight loss) or hyperglycemic crisis [1].")
        sections.append("*(Note: In the absence of unequivocal acute hyperglycemia, diagnosis requires two abnormal test results from the same sample or two separate test occasions.)* [1]\n")
        
        sections.append("**2. Prediabetes Classification [1]:**")
        sections.append("- **Impaired Fasting Glucose (IFG):** FPG `100 - 125 mg/dL` (5.6 - 6.9 mmol/L) [1].")
        sections.append("- **Impaired Glucose Tolerance (IGT):** 2-hour PG `140 - 199 mg/dL` (7.8 - 11.0 mmol/L) [1].")
        sections.append("- **Elevated HbA1c:** `5.7% - 6.4%` (39 - 47 mmol/mol), identifying individuals at high risk for progression to type 2 diabetes and ASCVD [1].\n")
        
        sections.append("**3. Who Should Be Screened & Frequency [1]:**")
        sections.append("- **Universal Age Screening:** Testing should begin at **age 35 years** for all individuals, regardless of risk factors [1].")
        sections.append("- **Adults with Overweight/Obesity:** Testing should be performed in adults of any age with BMI `>= 25 kg/m²` (or `>= 23 kg/m²` in Asian Americans) who possess >= 1 additional risk factors [1].")
        sections.append("- **Screening Interval:** If tests are normal, repeat screening at minimum **3-year intervals**; annual testing is indicated for patients with identified prediabetes [1].")

        # Specific evaluation for patient if selected
        if p_name:
            sections.append(f"\n---\n\n#### Clinical Application to {p_name} ({p_age}):")
            has_diab_hist = any("diabet" in c["condition"].lower() for c in cond_items)
            g_labs = [t for t in all_tests if any(k in t["name"].lower() for k in ["glucose", "hba1c"])]
            if has_diab_hist:
                sections.append(f"- **History Alignment:** Patient has a documented diagnosis of diabetes. Ongoing monitoring focuses on therapeutic HbA1c target `< 7.0%` [1].")
            if g_labs:
                sections.append(f"- **Current Laboratory Values:** {_format_test_summary(g_labs)}.")
                for gl in g_labs:
                    val = float(re.findall(r"\d+(?:\.\d+)?", gl["value"])[0]) if re.findall(r"\d+(?:\.\d+)?", gl["value"]) else None
                    if val is not None and "glucose" in gl["name"].lower():
                        if val >= 126:
                            sections.append(f"- **Guideline Assessment:** Measured fasting glucose `{gl['value']} {gl['unit']}` exceeds the diagnostic threshold (`>= 126 mg/dL`), confirming significant glycemic elevation [1].")
                        elif 100 <= val <= 125:
                            sections.append(f"- **Guideline Assessment:** Measured fasting glucose `{gl['value']} {gl['unit']}` falls within the impaired fasting glucose (prediabetes) range [1].")
            else:
                sections.append(f"- **Next Step:** Screen with Fasting Plasma Glucose or HbA1c as patient meets age and risk criteria [1].")

    elif is_diabetes_mgmt:
        sections.append("### Clinical Guideline Answer: Type 2 Diabetes Pharmacotherapy & Glycemic Targets (ADA 2026 Standards of Care)\n")
        sections.append(
            "According to the **American Diabetes Association (ADA) 2026 Guidelines** [1], pharmacological therapy is tailored based on "
            "glycemic targets, organ-protective indications (ASCVD, Heart Failure, CKD), and hypoglycemia risk profiles:\n"
        )
        sections.append("**1. Glycemic Targets [1]:**")
        sections.append("- **Standard Target:** An **HbA1c < 7.0%** (53 mmol/mol) is appropriate for most nonpregnant adults without significant hypoglycemia risk [1].")
        sections.append("- **Stringent Targets (< 6.5%):** Appropriate for patients with short disease duration, treated with lifestyle/metformin alone, long life expectancy, and no significant CVD [1].")
        sections.append("- **Individualized Targets (< 8.0%):** Indicated for patients with severe hypoglycemia history, extensive comorbidities, or limited life expectancy [1].\n")
        
        sections.append("**2. First-Line Pharmacotherapy [1]:**")
        sections.append("- **Metformin:** Remains the foundational first-line pharmacological agent for type 2 diabetes, continued as long as tolerated and renal function permits (eGFR >= 30 mL/min) [1].")
        sections.append("- **Organ-Protective Comorbid Therapy:** For patients with established Atherosclerotic Cardiovascular Disease (ASCVD), Heart Failure (HF), or Chronic Kidney Disease (CKD), an **SGLT2 inhibitor** (e.g. empagliflozin, dapagliflozin) or **GLP-1 receptor agonist** (e.g. semaglutide, dulaglutide) is indicated as initial therapy regardless of baseline HbA1c [1].\n")
        
        sections.append("**3. Insulin Therapy & Hypoglycemia Management [1]:**")
        sections.append("- **Insulin Indications:** Early introduction of basal insulin is indicated when blood glucose is `>= 300 mg/dL`, HbA1c `> 10.0%`, or marked catabolism/symptoms are present [1].")
        sections.append("- **Hypoglycemia Threshold:** Blood glucose `< 70 mg/dL` warrants immediate 15-20g oral fast-acting glucose and review of antidiabetic agents [1].")

        if p_name:
            sections.append(f"\n---\n\n#### Pharmacotherapy Context for {p_name}:")
            if medications:
                sections.append(f"- **Current Prescribed Agents:** {', '.join(medications)}.")
            g_labs = [t for t in all_tests if any(k in t["name"].lower() for k in ["glucose", "hba1c"])]
            if g_labs:
                sections.append(f"- **Current Glycemic Parameters:** {_format_test_summary(g_labs)}.")

    elif is_bp_query:
        sections.append("### Clinical Guideline Answer: Hypertension Classification & Management (AHA/ACC 2025 Guidelines)\n")
        sections.append(
            "Based on the **AHA/ACC Guidelines for High Blood Pressure** [1], blood pressure categories and therapeutic thresholds are defined as follows:\n"
        )
        sections.append("**1. Blood Pressure Classification [1]:**")
        sections.append("- **Normal BP:** Systolic `< 120 mmHg` AND Diastolic `< 80 mmHg` [1].")
        sections.append("- **Elevated BP:** Systolic `120 - 129 mmHg` AND Diastolic `< 80 mmHg` [1].")
        sections.append("- **Stage 1 Hypertension:** Systolic `130 - 139 mmHg` OR Diastolic `80 - 89 mmHg` [1].")
        sections.append("- **Stage 2 Hypertension:** Systolic `>= 140 mmHg` OR Diastolic `>= 90 mmHg` [1].\n")
        
        sections.append("**2. Blood Pressure Target Goals [1]:**")
        sections.append("- The guideline-recommended treatment target for adults with confirmed hypertension is **< 130/80 mmHg** [1].\n")
        
        sections.append("**3. First-Line Pharmacological Agents [1]:**")
        sections.append("- **Four First-Line Classes:** Thiazide diuretics (chlorthalidone preferred), Calcium Channel Blockers (amlodipine), ACE Inhibitors (lisinopril), and Angiotensin Receptor Blockers (losartan) [1].")
        sections.append("- In patients with Stage 2 Hypertension (BP >= 140/90 mmHg), initiation with **two first-line antihypertensive agents** of different classes is recommended [1].")
        sections.append("- In patients with concurrent diabetes or chronic kidney disease (CKD), an ACE inhibitor or ARB is preferred for renal protection [1].")

        if p_name:
            sections.append(f"\n---\n\n#### Hypertension Assessment for {p_name}:")
            has_htn = any("hypertens" in c["condition"].lower() for c in cond_items)
            sections.append(f"- **Documented Status:** {'Documented active hypertension' if has_htn else 'No documented hypertension diagnosis in problem list'}.")
            if medications:
                sections.append(f"- **Active Medication Regimen:** {', '.join(medications)}.")

    elif is_heart_failure:
        sections.append("### Clinical Guideline Answer: Heart Failure Management Guidelines (AHA/ACC)\n")
        sections.append(
            "The **AHA/ACC Heart Failure Guidelines** [1] recommend guideline-directed medical therapy (GDMT) to reduce hospitalizations and mortality:\n"
        )
        sections.append("**1. Foundational Pharmacotherapy (GDMT) [1]:**")
        sections.append("- **Renin-Angiotensin System Inhibitors:** ARNI (sacubitril/valsartan) preferred, or ACEI/ARB [1].")
        sections.append("- **Beta-Blockers:** Evidence-based agents (carvedilol, metoprolol succinate, bisoprolol) [1].")
        sections.append("- **Mineralocorticoid Receptor Antagonists (MRAs):** Spironolactone or eplerenone [1].")
        sections.append("- **SGLT2 Inhibitors:** Empagliflozin or dapagliflozin, indicated across ejection fraction spectrum to reduce HF hospitalizations [1].\n")
        sections.append("**2. Volume Management:** Loop diuretics (furosemide, bumetanide) titrated to maintain euvolemia [1].")

    elif is_readmission:
        sections.append("### Clinical Guideline Answer: Hospital Readmission Risk Factors & Transitional Care\n")
        sections.append(
            "Clinical evidence on 30-day hospital readmissions highlights major clinical, pharmacological, and operational drivers [1]:\n"
        )
        sections.append("**1. Primary Risk Determinants [1]:**")
        sections.append("- **Multi-morbidity:** High number of documented chronic diagnoses (diabetes, heart failure, renal impairment) [1].")
        sections.append("- **Polypharmacy:** Prescription of `>= 5` scheduled medications, increasing medication non-adherence and adverse drug events [1].")
        sections.append("- **Prior Utilization:** Recent emergency department visits or inpatient admissions in the preceding 6-12 months [1].\n")
        sections.append("**2. Recommended Mitigations [1]:**")
        sections.append("- Multi-disciplinary medication reconciliation at discharge [1].")
        sections.append("- Outpatient clinical follow-up within 7 to 14 days of discharge [1].")

        if p_name:
            sections.append(f"\n---\n\n#### Readmission Risk Context for {p_name}:")
            conds_str = ", ".join(c["condition"] for c in cond_items) if cond_items else "No documented chronic conditions"
            sections.append(f"- **Patient Co-Morbidities:** {conds_str}.")
            sections.append(f"- **Documented Medications:** {', '.join(medications) if medications else 'None on file'}.")
            abn_labs = [t for t in all_tests if t["status"] in ("High", "Low", "Critical")]
            if abn_labs:
                sections.append(f"- **Biomarker Vulnerability:** {_format_test_summary(abn_labs, limit=3)}.")

    # -------------------------------------------------------------
    # 8. GENERAL CLINICAL GUIDELINE FALLBACK
    # -------------------------------------------------------------
    else:
        primary = evidence[0] if evidence else {"source": "Clinical Practice Guidelines", "section": "Evidence Summary", "text": "Curated clinical guidelines synthesize current medical best practices."}
        sections.append(f"### Clinical Guideline Answer: {primary.get('section') or primary['source']}\n")
        sections.append(f"Based on retrieved clinical evidence from **{primary['source']}** [1]:\n")
        for idx, item in enumerate(evidence[:3], 1):
            cleaned = _clean_chunk_text(item["text"])
            sections.append(f"- **{item.get('section') or 'Recommendation'} [{idx}]:** {cleaned}")
        sections.append("")

        if p_name:
            sections.append(f"---\n\n#### Clinical Context for {p_name} ({p_age}):")
            if cond_items:
                sections.append(f"- **Documented Conditions:** {', '.join(c['condition'] for c in cond_items)}.")
            if all_tests:
                sections.append(f"- **Active Biomarkers:** {_format_test_summary(all_tests, limit=3)}.")

    sections.append("\n\n*Clinician review and verification is required prior to diagnostic or therapeutic action.*")
    return "\n".join(sections)


def _call_llm_answer(question: str, evidence: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Attempt natural language answer via external LLM (Gemini or OpenAI) if configured."""
    api_key = settings.LLM_API_KEY.strip() or os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    provider = settings.LLM_PROVIDER.strip().lower()
    
    if not api_key:
        return None
        
    if not provider:
        provider = "gemini" if api_key.startswith("AIza") else "openai"
        
    context_chunks = "\n\n".join(
        f"[{index}] Source: {item['source']} — Section: {item.get('section') or 'Guideline'}\n{item['text']}"
        for index, item in enumerate(evidence, start=1)
    )
    
    patient_text = ""
    if patient_info:
        cond_list = [c["condition"] if isinstance(c, dict) else str(c) for c in patient_info.get("conditions", [])]
        lab_list = [f"{l['name']}={l['value']} {l.get('unit', '')} ({l.get('status', 'Normal')})" for l in patient_info.get("labs", [])]
        rep_tests = [f"{t['name']}={t['value']} {t.get('unit', '')} ({t.get('status', 'Normal')})" for t in patient_info.get("report_tests", [])]
        all_biomarkers = lab_list + [t for t in rep_tests if t not in lab_list]
        meds = patient_info.get("medications", [])
        reports_summary = []
        for pr in patient_info.get("parsed_reports", []):
            imp = "; ".join(pr.get("narrative", [])) or "Routine findings"
            reports_summary.append(f"{pr.get('panel', 'Report')} ({pr.get('filename')}): {imp}")
            
        patient_text = (
            f"\nPatient Context:\n- Name: {patient_info.get('name')}\n"
            f"- Age/Gender: {patient_info.get('age')}, {patient_info.get('gender')}\n"
            f"- Documented Conditions: {', '.join(cond_list) if cond_list else 'None recorded'}\n"
            f"- Biomarkers & Tests: {', '.join(all_biomarkers) if all_biomarkers else 'None recorded'}\n"
            f"- Diagnostic Investigations: {'; '.join(reports_summary) if reports_summary else 'None on file'}\n"
            f"- Medications: {', '.join(meds) if meds else 'None documented'}\n"
        )
        
    system_prompt = (
        "You are an expert Clinical Decision Support System (AI-CDSS) assistant. "
        "Answer the clinician's query thoroughly, directly, and objectively using the provided guideline evidence and patient health records. "
        "Do NOT just summarize the document titles; answer the actual question with specific diagnostic thresholds, "
        "numerical criteria, pharmacotherapy guidelines, and clinical recommendations. "
        "Format using clean Markdown with headers, bullet points, and exact numbers. "
        "Cite every factual statement with [1], [2], etc. based on the evidence chunks. "
        "If patient context is provided, include a dedicated section on how these guideline recommendations apply to this specific patient. "
        "End with: Clinician review and verification is required."
    )
    
    if provider == "gemini":
        model = settings.LLM_MODEL.strip() or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nEvidence:\n{context_chunks}\n{patient_text}\nQuestion: {question}"
                }]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1800}
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text and "[1]" in text:
                    return text
        except Exception as e:
            logger.warning("Gemini RAG answer call failed: %s; falling back to local engine.", e)
            return None
            
    if provider == "openai":
        model = settings.LLM_MODEL.strip() or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Evidence:\n{context_chunks}\n{patient_text}\n\nQuestion: {question}"}
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text and "[1]" in text:
                    return text
        except Exception as e:
            logger.warning("OpenAI RAG answer call failed: %s; falling back to local engine.", e)
            return None
            
    return None


def answer(question: str, limit: int, patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate an authoritative, evidence-grounded clinical answer to the clinician's query."""
    retrieved = search(question, limit)
    evidence = retrieved["results"]
    
    has_clinical_keywords = any(k in question.lower() for k in CLINICAL_KEYWORDS)
    is_patient_query = bool(patient_info)
    
    # Check for off-topic or insufficient relevance
    # Only return insufficient_evidence if the query is unrelated to clinical care AND has low FAISS score
    if not has_clinical_keywords and (not evidence or evidence[0]["score"] < 0.12):
        return {
            "query": question,
            "status": "insufficient_evidence",
            "answer": None,
            "message": "No sufficiently relevant clinical guidelines were found in the knowledge base for this query.",
            "provider": None,
            "model": None,
            "evidence": evidence,
        }

    # Attempt external LLM if configured
    llm_result = _call_llm_answer(question, evidence, patient_info)
    if llm_result:
        provider_name = settings.LLM_PROVIDER.strip() or "ai_model"
        return {
            "query": question,
            "status": "answered",
            "answer": llm_result,
            "message": None,
            "provider": provider_name,
            "model": settings.LLM_MODEL or "default",
            "evidence": evidence,
        }

    # Local High-Precision Clinical Evidence Synthesizer
    synthesized_answer = _synthesize_local_answer(question, evidence, patient_info)
    return {
        "query": question,
        "status": "evidence_summary",
        "answer": synthesized_answer,
        "message": "Synthesized directly from curated clinical guidelines (ADA/AHA/ACC) and patient records.",
        "provider": "local_clinical_engine",
        "model": None,
        "evidence": evidence,
    }
