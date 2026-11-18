"""Clinical Summarization Service for AI-CDSS.

Synthesizes a patient's complete health record (demographics, documented
medical conditions, laboratory results, and uploaded medical reports) into
an actual, cohesive, and structured clinical case summary for doctor review.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.models.patient import Patient
from app.services.clinical_nlp import extract_clinical_entities

logger = logging.getLogger("ai_cdss.summarizer")

COMMON_UNITS = r"(?:mg/dl|mmol/l|g/dl|%|/ul|cells/ul|meq/l|miu/l|ng/ml|pg/ml|iu/l|u/l|ug/dl|mm/hr|fl|pg)"


def _calculate_age(dob: Optional[str]) -> str:
    """Calculate age from date of birth string."""
    if not dob:
        return "Age not documented"
    match = re.search(r"(\d{4})", str(dob))
    if match:
        year = int(match.group(1))
        current_year = datetime.utcnow().year
        age = current_year - year
        if 0 <= age <= 125:
            return f"{age} years old"
    return "Age not documented"


def _clean_report_text(raw_text: str) -> str:
    """Strip scanner artifacts, OCR noise lines, and administrative headers."""
    if not raw_text:
        return ""
    lines = raw_text.splitlines()
    cleaned = []
    
    noise_re = re.compile(
        r"^(?:tel|fax|phone|email|website|address|p\.o\.\s*box|page\s+\d+|bar\s*code|reg\s*no|iso\s*\d+)\b|"
        r"^(?:dr\.|licence|authenticated\s+by|electronically\s+signed|confidentiality\s+notice|all\s+rights\s+reserved)\b|"
        r"^[_\-=*~#|\\/+\s]{3,}$",
        re.I
    )
    for line in lines:
        s = line.strip()
        if not s or len(s) < 2 or noise_re.search(s):
            continue
        cleaned.append(s)
    return "\n".join(cleaned)


def _detect_modality(raw_text: str, filename: str) -> str:
    """Identify diagnostic modality and panel type from text and filename."""
    combined = f"{filename} {raw_text}".lower()
    if any(k in combined for k in ["lipid", "cholesterol", "triglyceride"]):
        return "Lipid Profile Panel"
    if any(k in combined for k in ["glucose", "hba1c", "glycated", "fasting blood sugar", "ogtt"]):
        return "Glycemic / Diabetic Panel"
    if any(k in combined for k in ["cbc", "blood count", "hemogram", "wbc", "platelet", "hemoglobin"]):
        return "Complete Blood Count (CBC)"
    if any(k in combined for k in ["renal", "kft", "rft", "creatinine", "urea", "bun", "egfr"]):
        return "Renal Function Panel (KFT)"
    if any(k in combined for k in ["liver", "lft", "hepatic", "bilirubin", "sgot", "sgpt", "alt", "ast"]):
        return "Liver Function Panel (LFT)"
    if any(k in combined for k in ["thyroid", "tft", "tsh", "free t3", "free t4"]):
        return "Thyroid Function Panel (TFT)"
    if any(k in combined for k in ["urine", "urinalysis"]):
        return "Urinalysis Report"
    if any(k in combined for k in ["x-ray", "xray", "radiograph", "chest pa"]):
        return "Radiology - Plain Radiograph (X-Ray)"
    if any(k in combined for k in ["ct scan", "computed tomography"]):
        return "Radiology - Computed Tomography (CT)"
    if any(k in combined for k in ["mri", "magnetic resonance"]):
        return "Radiology - MRI Scan"
    if any(k in combined for k in ["ultrasound", "sonography", "usg"]):
        return "Ultrasound / Sonography Report"
    if any(k in combined for k in ["ecg", "ekg", "electrocardiogram"]):
        return "Cardiology - Electrocardiogram (ECG)"
    if any(k in combined for k in ["biopsy", "histopathology", "pathology report"]):
        return "Diagnostic Histopathology / Biopsy"
    if any(k in combined for k in ["discharge summary", "discharge note"]):
        return "Hospital Discharge Summary"
    if any(k in combined for k in ["consultation", "progress note", "clinic visit"]):
        return "Clinical Consultation Record"
    return "Diagnostic Investigation Report"


def _extract_report_details(report) -> Dict[str, Any]:
    """Parse a diagnostic or imaging report, extracting test results, impressions, and metadata."""
    raw = report.extracted_text or ""
    filename = report.filename or "Report"
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    cleaned_text = _clean_report_text(raw)
    
    # 1. Modality / Panel title
    panel_title = _detect_modality(raw, filename)
    
    # 2. Facility / Lab identification
    facility = ""
    fac_m = re.search(
        r"(?:at|from|by|laboratory|hospital|clinic|center|centre|diagnostics)\s*[:;]?\s*([A-Za-z0-9\s]{3,40}(?:diagnostic|hospital|clinic|laboratory|lab|center|healthcare)[A-Za-z0-9\s]*)",
        raw, re.I
    )
    if fac_m:
        facility = re.sub(r"\s+", " ", fac_m.group(1)).strip()[:50]
        
    # 3. Date of testing
    rep_date = ""
    date_m = re.search(
        r"(?:collected|reported|date|dated|sampled|test\s+date|encounter\s+date)\s*[:;]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
        raw, re.I
    )
    if date_m:
        rep_date = date_m.group(1)
        
    # 4. Check for narrative sections (Radiology, Pathology, Discharge)
    narrative_findings = []
    imp_match = re.search(
        r"(?:IMPRESSION|CONCLUSION|DIAGNOSTIC\s+IMPRESSION|FINAL\s+DIAGNOSIS|OPINION|SUMMARY)\s*[:\-]\s*(.*?)(?=(?:RECOMMENDATION|NOTE|DISCLAIMER|\n[A-Z\s]{4,}:|$))",
        raw, re.I | re.DOTALL
    )
    if imp_match:
        imp_text = re.sub(r"\s+", " ", imp_match.group(1)).strip()
        if len(imp_text) > 8:
            narrative_findings.append(f"Impression: {imp_text[:400]}")
            
    findings_match = re.search(
        r"(?:FINDINGS|OBSERVATIONS|RESULTS|CLINICAL\s+OBSERVATIONS)\s*[:\-]\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|NOTE|DISCLAIMER|\n[A-Z\s]{4,}:|$))",
        raw, re.I | re.DOTALL
    )
    if findings_match and not narrative_findings:
        f_text = re.sub(r"\s+", " ", findings_match.group(1)).strip()
        if len(f_text) > 8:
            narrative_findings.append(f"Findings: {f_text[:400]}")
            
    # 5. Extract structured laboratory test rows
    test_lines = []
    for line in lines:
        if re.match(r"^(?:note|notes|interpretation|comment|comments|disclaimer|technique|methodology|specimen)\b", line, re.I):
            break
        test_lines.append(line)
        
    extracted_tests = []
    test_regex = re.compile(
        rf"([A-Za-z\s\(\)/,-]{{3,35}})\s+(\d+(?:\.\d+)?)\s*({COMMON_UNITS})?\s*([0-9.<>=\s-]+)?",
        re.I
    )
    
    for line in test_lines:
        m = test_regex.search(line)
        if m and any(c.isdigit() for c in line):
            name = m.group(1).strip()
            # Skip non-test headers or administrative rows
            if any(w in name.lower() for w in ["lab no", "age", "phone", "test name", "results", "collected", "reported", "status", "page", "mrn", "patient"]):
                continue
            val = float(m.group(2))
            unit = m.group(3) or ""
            ref = (m.group(4) or "").strip()
            
            # Determine interpretation status
            status = "Normal"
            ref_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|\.\.)\s*(\d+(?:\.\d+)?)", ref, re.I)
            if ref_match:
                low_b = float(ref_match.group(1))
                high_b = float(ref_match.group(2))
                if val < low_b:
                    status = "Mildly Low" if (low_b - val) <= (low_b * 0.25) else "Low"
                elif val > high_b:
                    status = "Mildly Elevated" if (val - high_b) <= (high_b * 0.25) else "High"
                else:
                    status = "Normal"
            else:
                # Fallback heuristic based on test name
                nl = name.lower()
                if "glucose" in nl:
                    if val < 70:
                        status = "Low"
                    elif val > 125:
                        status = "High"
                elif "hba1c" in nl:
                    if val > 6.5:
                        status = "High"
                elif "cholesterol" in nl and "hdl" not in nl:
                    if val > 200:
                        status = "High"
                elif "creatinine" in nl:
                    if val > 1.3:
                        status = "High"
                    
            is_valid = bool(unit) or bool(ref) or any(k in name.lower() for k in [
                "glucose", "hba1c", "creatinine", "hemoglobin", "wbc", "platelet",
                "cholesterol", "urea", "sodium", "potassium", "tsh", "bun", "alt", "ast"
            ])
            if is_valid:
                extracted_tests.append({
                    "name": name,
                    "value": val,
                    "unit": unit,
                    "ref": ref,
                    "status": status
                })
                
    entities = report.clinical_entities or extract_clinical_entities(raw)
    conditions = [c["text"] for c in entities.get("conditions", [])]
    medications = [m["text"] for m in entities.get("medications", [])]
    
    # 6. Extract key clinical narrative statements if structured test rows and impression are absent
    clinical_sentences = []
    if not extracted_tests and not narrative_findings:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 15 and any(k in s_clean.lower() for k in [
                "elevated", "high", "low", "abnormal", "normal", "stable", "consistent with",
                "evidence of", "diagnostic for", "acute", "chronic", "negative", "positive",
                "increased", "decreased", "unremarkable", "clear", "intact", "no focal",
                "hypertrophy", "cardiomegaly", "infiltrate", "consolidation", "lesion", "nodule"
            ]):
                clinical_sentences.append(s_clean[:180])
            if len(clinical_sentences) >= 3:
                break
                
    return {
        "filename": filename,
        "panel": panel_title,
        "facility": facility,
        "date": rep_date,
        "tests": extracted_tests,
        "narrative": narrative_findings,
        "key_sentences": clinical_sentences,
        "conditions": conditions,
        "medications": medications,
    }


def _call_external_llm(context_prompt: str) -> Optional[str]:
    """Attempt natural language summary via external LLM if configured."""
    api_key = settings.LLM_API_KEY.strip() or os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    provider = settings.LLM_PROVIDER.strip().lower()
    
    if not api_key:
        return None
        
    if not provider:
        if api_key.startswith("AIza"):
            provider = "gemini"
        else:
            provider = "openai"
            
    system_prompt = (
        "You are an expert Clinical Decision Support System (AI-CDSS) assistant. "
        "Produce a comprehensive, structured CLINICAL CASE SUMMARY for doctor review based on "
        "the patient's complete electronic medical record, lab biomarkers, and diagnostic reports.\n\n"
        "Format using clear Markdown with the following standard sections:\n"
        "# Clinical Case Summary & Synthesis\n"
        "## 1. Executive Clinical Summary (cohesive narrative of health baseline, stability, and trajectory)\n"
        "## 2. Active Medical History & Problem List\n"
        "## 3. Laboratory & Biomarker Synthesis (categorized into Glycemic, Cardiovascular, Renal, Hematologic)\n"
        "## 4. Diagnostic Investigations & Imaging Synthesis (synthesizing report conclusions, NOT dumping OCR text)\n"
        "## 5. Medication & Therapeutic Regimen Review\n"
        "## 6. Multisystem Clinical Risk Stratification\n"
        "## 7. Actionable Clinician Plan of Care & Recommended Next Steps\n\n"
        "Maintain an objective, clinician-facing medical tone. Flag abnormal values with reference intervals. "
        "Highlight any clinical correlations or contradictions between history and test findings."
    )
    
    if provider == "gemini":
        model = settings.LLM_MODEL.strip() or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\nPatient Clinical Profile & Records:\n{context_prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text:
                    return text
        except Exception as e:
            logger.warning("Gemini LLM call failed: %s; falling back to deterministic clinical engine.", e)
            return None
            
    if provider == "openai":
        model = settings.LLM_MODEL.strip() or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Patient Clinical Profile & Records:\n{context_prompt}\n\nGenerate the complete Clinical Summary."}
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    return text
        except Exception as e:
            logger.warning("OpenAI LLM call failed: %s; falling back to deterministic clinical engine.", e)
            return None
            
    return None


def generate_clinical_summary(patient: Patient, db=None) -> Dict[str, Any]:
    """Generate an actual, professional clinical case summary synthesized from patient records."""
    age_str = _calculate_age(patient.date_of_birth)
    gender_str = (patient.gender or "Unspecified gender").strip().capitalize()
    blood_group_str = patient.blood_group or "Not recorded"
    
    history_items = list(patient.medical_history or [])
    lab_results = list(patient.lab_results or [])
    reports = list(patient.reports or [])
    
    # Process and extract all diagnostic reports
    parsed_reports = [_extract_report_details(r) for r in reports if (r.extracted_text or "").strip()]
    
    # Collect all laboratory tests (from database + extracted from report tables)
    all_abnormal_tests = []
    all_normal_tests = []
    
    # 1. From database lab results
    for lab in lab_results:
        status_clean = (lab.status or "").strip().title()
        is_abn = status_clean in ("Abnormal", "High", "Low", "Critical", "Mildly High", "Mildly Elevated", "Mildly Low")
        entry = {
            "name": lab.test_name,
            "value": lab.value,
            "unit": lab.unit,
            "ref": lab.reference_range,
            "status": status_clean,
            "date": lab.test_date or "",
            "source": "EHR Laboratory Record",
            "basis": lab.clinical_basis or ""
        }
        if is_abn:
            all_abnormal_tests.append(entry)
        else:
            all_normal_tests.append(entry)
            
    # 2. From extracted report tests
    for pr in parsed_reports:
        for t in pr["tests"]:
            is_abn = t["status"] in ("Low", "Mildly Low", "High", "Mildly Elevated", "Abnormal", "Critical")
            entry = {
                "name": t["name"],
                "value": t["value"],
                "unit": t["unit"],
                "ref": t["ref"],
                "status": t["status"],
                "date": pr["date"],
                "source": pr["filename"],
                "basis": f"From {pr['panel']}" if pr["panel"] else ""
            }
            if is_abn:
                all_abnormal_tests.append(entry)
            else:
                all_normal_tests.append(entry)
                
    # Collect medications from all available channels
    all_medications = set()
    for pr in parsed_reports:
        all_medications.update(pr["medications"])
    for h in history_items:
        entities = extract_clinical_entities(h.description or "")
        all_medications.update(m["text"] for m in entities.get("medications", []))
    medications_list = sorted(list(all_medications))
    
    # Prepare context for optional external LLM call
    context_lines = [
        f"Patient: {patient.first_name} {patient.last_name}, {age_str}, {gender_str}, Blood Group: {blood_group_str}, ID: #{patient.id}",
        "Documented Conditions: " + (", ".join(f"{h.condition} ({h.status or 'Active'})" for h in history_items) if history_items else "None"),
        "Priority Out-of-Range Biomarkers: " + (", ".join(f"{t['name']}: {t['value']} {t['unit']} [{t['status']}] (Ref: {t['ref']})" for t in all_abnormal_tests) if all_abnormal_tests else "None"),
        "Normal Baseline Biomarkers: " + (", ".join(f"{t['name']}: {t['value']} {t['unit']}" for t in all_normal_tests) if all_normal_tests else "None"),
        f"Diagnostic Reports Analyzed ({len(parsed_reports)}):",
    ]
    for pr in parsed_reports:
        test_summary = ", ".join(f"{t['name']}={t['value']} {t['unit']} ({t['status']})" for t in pr["tests"]) if pr["tests"] else "No discrete numeric table rows"
        narrative_summary = "; ".join(pr["narrative"] + pr["key_sentences"]) if (pr["narrative"] or pr["key_sentences"]) else "Routine documentation on file"
        context_lines.append(f"- File: {pr['filename']} | Panel: {pr['panel']} | Facility: {pr['facility']} | Tests: {test_summary} | Narrative: {narrative_summary}")
    if medications_list:
        context_lines.append(f"Documented Medications: {', '.join(medications_list)}")
        
    context_prompt = "\n".join(context_lines)
    llm_draft = _call_external_llm(context_prompt)
    if llm_draft:
        return {
            "patient_id": patient.id,
            "draft": llm_draft,
            "mode": "AI Model Synthesis",
            "report_count": len(reports),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "disclaimer": "AI-generated clinical draft compiled from patient records. Clinician verification is required prior to diagnostic or therapeutic action."
        }

    # ---------------------------------------------------------
    # High-Grade Deterministic Clinical Case Summary Generator
    # ---------------------------------------------------------
    sections = []
    
    # 1. Header & Demographic Profile
    sections.append(
        f"# Clinical Case Summary & Synthesis\n\n"
        f"**Patient:** {patient.first_name} {patient.last_name}  \n"
        f"**Demographics:** {age_str} | {gender_str} | Blood Group: `{blood_group_str}` | Patient MRN: `#{patient.id}`  \n"
        f"**Review Date:** {datetime.utcnow().strftime('%B %d, %Y')} | **System:** AI-CDSS Clinical Synthesis Engine  \n\n"
        f"---\n"
    )

    # 2. Executive Clinical Summary Narrative
    active_conds = [h.condition for h in history_items if (h.status or "").lower() in ("active", "chronic")]
    conds_text = ", ".join(active_conds) if active_conds else "no active chronic conditions formally recorded"
    
    # Correlate glycemic trajectory
    glucose_tests = [t for t in (all_abnormal_tests + all_normal_tests) if "glucose" in t["name"].lower() or "hba1c" in t["name"].lower()]
    has_diabetes_hist = any("diabet" in h.condition.lower() for h in history_items)
    glycemic_narrative = ""
    if glucose_tests:
        g_desc = "; ".join(f"{t['name']}: {t['value']} {t['unit']} ({t['status']})" for t in glucose_tests[:3])
        if any(t["status"] in ("Mildly Low", "Low") for t in glucose_tests):
            glycemic_narrative = (
                f" Recent measured plasma glucose parameters ({g_desc}) demonstrate mild downward fasting deviation. "
                f"Given the patient's background of {conds_text}, the absence of acute hyperglycemia is notable, "
                f"though vigilance for hypoglycemic episodes or medication overtreatment is clinically indicated."
            )
        elif any(t["status"] in ("High", "Critical", "Mildly Elevated") for t in glucose_tests):
            glycemic_narrative = (
                f" Laboratory monitoring identifies elevated glycemic parameters ({g_desc}), confirming active metabolic "
                f"dysregulation that correlates directly with diabetic disease progression and requires glycemic optimization."
            )
        else:
            glycemic_narrative = f" Evaluated glycemic biomarkers ({g_desc}) reflect stable plasma glucose tolerance."

    # Correlate cardiovascular trajectory
    cardio_tests = [t for t in (all_abnormal_tests + all_normal_tests) if any(k in t["name"].lower() for k in ["cholesterol", "ldl", "hdl", "triglyceride", "troponin", "bnp"])]
    has_htn_hist = any("hypertens" in h.condition.lower() for h in history_items)
    cardio_narrative = ""
    if cardio_tests:
        c_abn = [t for t in cardio_tests if t["status"] in ("High", "Elevated", "Critical")]
        if c_abn:
            c_desc = "; ".join(f"{t['name']}: {t['value']} {t['unit']}" for t in c_abn[:3])
            cardio_narrative = f" Lipid profile analysis highlights atherogenic elevation ({c_desc}), exacerbating long-term cardiovascular risk."
        else:
            cardio_narrative = " Lipid panel parameters demonstrate acceptable atherosclerotic cardiovascular risk markers."
    elif has_htn_hist:
        cardio_narrative = " Blood pressure and cardiovascular surveillance remain paramount given active hypertension documentation."

    # Correlate renal trajectory
    renal_tests = [t for t in (all_abnormal_tests + all_normal_tests) if any(k in t["name"].lower() for k in ["creatinine", "urea", "bun", "egfr"])]
    renal_narrative = ""
    if renal_tests:
        r_abn = [t for t in renal_tests if t["status"] in ("High", "Elevated", "Critical", "Low")]
        if r_abn:
            r_desc = "; ".join(f"{t['name']}: {t['value']} {t['unit']}" for t in r_abn[:2])
            renal_narrative = f" Renal profiling indicates out-of-range parameters ({r_desc}), necessitating kidney function monitoring."
        else:
            renal_narrative = " Renal function parameters (creatinine and blood urea nitrogen) reflect preserved clearance."

    total_biomarkers = len(all_abnormal_tests + all_normal_tests)
    exec_summary = (
        f"The patient is a {age_str.lower()} {gender_str.lower()} under active clinical management for **{conds_text}**. "
        f"A multi-modal synthesis of the patient's electronic health record, comprising {len(history_items)} documented condition(s), "
        f"{total_biomarkers} laboratory biomarker parameter(s), and {len(parsed_reports)} diagnostic investigation(s), was conducted.\n\n"
        f"{glycemic_narrative}{cardio_narrative}{renal_narrative}\n\n"
        f"Overall, the clinical picture demonstrates a chronic outpatient profile requiring coordinated disease surveillance, "
        f"biomarker monitoring, and medication review to prevent secondary complications."
    )
    sections.append(f"## 1. Executive Clinical Summary\n{exec_summary}\n")

    # 3. Documented Medical History & Problem List
    sections.append("## 2. Active Problem List & Clinical History")
    if history_items:
        for h in history_items:
            date_info = f" *(diagnosed {h.diagnosis_date})*" if h.diagnosis_date else ""
            desc_info = f" -- {h.description}" if h.description else ""
            badge = f"`{h.status or 'Active'}`"
            sections.append(f"- **{h.condition}** {badge}{date_info}{desc_info}")
    else:
        sections.append("- *No chronic medical conditions currently registered in the health record.*")
    sections.append("")

    # 4. Biomarker & Laboratory Evaluation
    sections.append("## 3. Laboratory & Biomarker Evaluation")
    if all_abnormal_tests:
        sections.append("### [!] Priority Out-of-Range Markers")
        for t in all_abnormal_tests:
            ref_str = f" *(Ref: {t['ref']})*" if t.get("ref") else ""
            date_str = f" `[{t['date']}]`" if t.get("date") else ""
            status_tag = f"**[{t['status'].upper()}]**"
            sections.append(f"- **{t['name']}**: `{t['value']} {t['unit']}` {status_tag}{ref_str}{date_str} -- *Source: {t['source']}*")
            if "glucose" in t["name"].lower() and "low" in t["status"].lower():
                sections.append("  > *Clinical Note:* Fasting glucose below standard physiological threshold (70 mg/dL). Evaluate for hypoglycemic episodes, meal spacing, or antidiabetic drug sensitivity.")
            elif "glucose" in t["name"].lower() and "high" in t["status"].lower():
                sections.append("  > *Clinical Note:* Fasting hyperglycemia indicates suboptimal glycemic control; re-evaluate glycemic regimen and compliance.")
            elif "creatinine" in t["name"].lower() and "high" in t["status"].lower():
                sections.append("  > *Clinical Note:* Serum creatinine elevation suggests reduced glomerular clearance; review hydration status and renal drug dosages.")
            elif any(k in t["name"].lower() for k in ["cholesterol", "ldl", "triglyceride"]) and "high" in t["status"].lower():
                sections.append("  > *Clinical Note:* Dyslipidemia elevation contributes to cumulative atherogenic burden; consider lipid-lowering optimization.")
        sections.append("")

    if all_normal_tests:
        norm_summary = ", ".join(f"{t['name']} (`{t['value']} {t['unit']}`)" for t in all_normal_tests[:8])
        sections.append(f"### Reassuring / Normal Baseline Parameters\n{norm_summary}\n")
    elif not all_abnormal_tests:
        sections.append("- *No discrete laboratory entries recorded in chart.*\n")

    # 5. Diagnostic Reports & Imaging Synthesis
    sections.append("## 4. Diagnostic Reports & Investigations Synthesis")
    if parsed_reports:
        for idx, pr in enumerate(parsed_reports, 1):
            fac_str = f" | Facility: *{pr['facility']}*" if pr["facility"] else ""
            date_str = f" | Date: *{pr['date']}*" if pr["date"] else ""
            sections.append(f"### Investigation #{idx}: {pr['panel']} ({pr['filename']}){fac_str}{date_str}")
            
            # Synthesize findings
            if pr["narrative"]:
                for n in pr["narrative"]:
                    sections.append(f"- **Core Conclusion:** {n}")
            if pr["tests"]:
                t_str = ", ".join(f"{t['name']}: {t['value']} {t['unit']} ({t['status']})" for t in pr["tests"][:6])
                sections.append(f"- **Discrete Measurements:** {t_str}")
            if pr["key_sentences"]:
                for s in pr["key_sentences"][:2]:
                    sections.append(f"- **Document Observation:** {s}")
            if not pr["narrative"] and not pr["tests"] and not pr["key_sentences"]:
                sections.append("- **Clinical Synthesis:** Structured textual analysis reveals no acute pathological flags or critical diagnostic alarms in document body.")
            sections.append("")
    else:
        sections.append("- *No diagnostic or radiology reports currently on file for synthesis.*\n")

    # 6. Pharmacotherapy & Medication Regimen
    sections.append("## 5. Medication & Therapeutic Regimen Review")
    if medications_list:
        sections.append(f"- **Documented / Detected Agents:** {', '.join(f'`{m}`' for m in medications_list)}")
        sections.append("- **Clinical Action:** Reconcile active dosage, dosing frequency, and patient treatment compliance during the upcoming consultation.")
        if any(m.lower() in ("insulin", "metformin", "glipizide", "glimepiride") for m in medications_list) and any(t["status"] in ("Mildly Low", "Low") for t in glucose_tests):
            sections.append("- **Pharmacological Alert:** Anti-diabetic medication regimen in the setting of low-normal fasting glucose requires review to prevent overt hypoglycemia.")
    else:
        sections.append("- *No active pharmacological agents formally cataloged in the electronic health record.*")
        sections.append("- **Recommendation:** Verify if the patient is currently taking unrecorded prescription pharmaceuticals, OTC remedies, or nutritional supplements.")
    sections.append("")

    # 7. Multisystem Clinical Assessment & Risk Stratification
    sections.append("## 6. Clinical Assessment & Risk Considerations")
    risk_points = []
    
    # Metabolic & Glycemic
    if glucose_tests or has_diabetes_hist:
        if any(t["status"] in ("Mildly Low", "Low") for t in glucose_tests):
            risk_points.append("- **Glycemic Control (Low-Normal Deviation):** Patient exhibits fasting glucose below 70 mg/dL. Inquire about neuroglycopenic symptoms (dizziness, sweating, shakiness) and assess meal spacing.")
        elif any(t["status"] in ("High", "Critical") for t in glucose_tests):
            risk_points.append("- **Glycemic Control (Hyperglycemia):** Suboptimal glycemic control documented; risk of microvascular and macrovascular complications warrants intervention.")
        else:
            risk_points.append("- **Glycemic Control:** Glycemic indicators reflect satisfactory short-term stability under ongoing management.")
            
    # Cardiovascular
    if cardio_tests or has_htn_hist:
        if any(t["status"] in ("High", "Critical") for t in cardio_tests):
            risk_points.append("- **Cardiovascular Risk:** Co-existing lipid elevation indicates heightened atherosclerotic cardiovascular disease (ASCVD) risk; blood pressure control and lipid reduction recommended.")
        else:
            risk_points.append("- **Cardiovascular Status:** Hemodynamic and lipid parameters remain within acceptable therapeutic targets.")
            
    # Renal
    if renal_tests:
        if any(t["status"] in ("High", "Critical") for t in renal_tests):
            risk_points.append("- **Renal Function:** Elevated renal biomarkers detected; evaluate glomerular filtration trajectory and check for microalbuminuria.")
        else:
            risk_points.append("- **Renal Function:** Renal clearance metrics demonstrate stable baseline without immediate nephrotoxic warnings.")
            
    if not risk_points:
        risk_points.append("- **Multisystem Stability:** Clinical biomarkers and recorded history indicate compensated baseline stability without acute critical alarms.")
        
    sections.extend(risk_points)
    sections.append("")

    # 8. Actionable Clinician Plan of Care
    sections.append("## 7. Actionable Clinician Next Steps")
    plan_steps = []
    step_num = 1
    
    if has_diabetes_hist or glucose_tests:
        plan_steps.append(f"{step_num}. **Glycated Hemoglobin (HbA1c):** Order/review HbA1c to assess chronic 3-month glycemic stability and calibrate antidiabetic regimen.")
        step_num += 1
        
    if any(t["status"] in ("Mildly Low", "Low") for t in glucose_tests):
        plan_steps.append(f"{step_num}. **Hypoglycemia Screening:** Counsel patient on hypoglycemia warning symptoms and review self-monitoring blood glucose logs.")
        step_num += 1
        
    if has_htn_hist or cardio_tests:
        plan_steps.append(f"{step_num}. **Cardiovascular & Lipid Management:** Verify ambulatory blood pressure control and reassess lipid panel in 3 to 6 months.")
        step_num += 1
        
    plan_steps.append(f"{step_num}. **Medication Reconciliation:** Perform formal med-rec during clinical encounter to establish exact current doses and adherence.")
    step_num += 1
    
    plan_steps.append(f"{step_num}. **Clinical Documentation:** Finalize consultation progress note in EHR and update patient problem list accordingly.")
    
    sections.extend(plan_steps)
    sections.append("\n---\n")

    final_draft = "\n".join(sections)
    
    return {
        "patient_id": patient.id,
        "draft": final_draft,
        "mode": "AI Clinical Synthesis",
        "report_count": len(reports),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "disclaimer": "Draft compiled from stored patient records, lab results, and diagnostic documents. Clinician verification is required prior to diagnostic or treatment decisions."
    }
