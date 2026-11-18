"""Small, review-only clinical text parser used for the Week 3 prototype."""
import re


CONDITIONS = (
    "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "hypertension",
    "asthma", "copd", "pneumonia", "migraine", "anemia", "hypothyroidism",
    "hyperthyroidism", "heart failure", "coronary artery disease", "stroke",
    "chronic kidney disease", "kidney disease", "depression", "arthritis",
)
MEDICATIONS = (
    "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
    "aspirin", "paracetamol", "acetaminophen", "ibuprofen", "omeprazole",
    "levothyroxine", "salbutamol", "albuterol", "amoxicillin", "azithromycin",
)
LAB_NAMES = "HbA1c|Hemoglobin|WBC|Platelets|Glucose|Creatinine|Urea|Sodium|Potassium|Cholesterol|LDL|HDL|Triglycerides|TSH"
EVENT_WORDS = "admitted|discharged|hospitalized|surgery|follow-up|follow up|referred|diagnosis|complaint"

LAB_EXPANDED_PATTERNS = [
    (r"\b(?:fasting\s+(?:blood\s+)?glucose|fasting\s+blood\s+sugar|blood\s+glucose|blood\s+sugar|serum\s+glucose|random\s+blood\s+sugar|glucose|fbs|rbs|fpg)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L)?", "Glucose", "mg/dL"),
    (r"\b(?:glycated\s+hemoglobin|glycosylated\s+hemoglobin|hba1c|ha1c|a1c)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(%)?", "HbA1c", "%"),
    (r"\b(?:hemoglobin|haemoglobin|hgb|hb)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(g/dL|g/L)?", "Hemoglobin", "g/dL"),
    (r"\b(?:white\s+blood\s+cells?|wbc|white\s+blood\s+count)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(/uL|cells/uL)?", "WBC", "/uL"),
    (r"\b(?:platelets?|plt)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(/uL|cells/uL)?", "Platelets", "/uL"),
    (r"\b(?:serum\s+creatinine|creatinine|creat)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "Creatinine", "mg/dL"),
    (r"\b(?:blood\s+urea\s+nitrogen|serum\s+urea|urea|bun)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "Urea", "mg/dL"),
    (r"\b(?:serum\s+sodium|sodium|na\+?)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mEq/L|mmol/L)?", "Sodium", "mEq/L"),
    (r"\b(?:serum\s+potassium|potassium|k\+?)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mEq/L|mmol/L)?", "Potassium", "mEq/L"),
    (r"\b(?:total\s+cholesterol|cholesterol)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "Total Cholesterol", "mg/dL"),
    (r"\b(?:ldl(?:\s*-\s*c)?|ldl\s+cholesterol)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "LDL", "mg/dL"),
    (r"\b(?:hdl(?:\s*-\s*c)?|hdl\s+cholesterol)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "HDL", "mg/dL"),
    (r"\b(?:triglycerides?|tg)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?", "Triglycerides", "mg/dL"),
    (r"\b(?:thyroid\s+stimulating\s+hormone|tsh)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mIU/L|uIU/mL)?", "TSH", "mIU/L"),
]


def _unique(items):
    seen = set()
    return [item for item in items if not (item["text"].lower() in seen or seen.add(item["text"].lower()))]


def extract_clinical_entities(text: str | None) -> dict:
    """Return conservative regex matches for clinician review; never a diagnosis."""
    text = text or ""
    conditions = [{"type": "condition", "text": match.group(0), "confidence": "possible"}
                  for name in CONDITIONS if (match := re.search(rf"\b{re.escape(name)}\b", text, re.I))]
    medications = [{"type": "medication", "text": match.group(0), "confidence": "possible"}
                   for name in MEDICATIONS if (match := re.search(rf"\b{re.escape(name)}\b", text, re.I))]
    
    labs = []
    seen_lab_keys = set()
    # 1. First run expanded patterns with canonical naming (e.g. Fasting Blood Sugar -> Glucose)
    for pattern, canonical_name, default_unit in LAB_EXPANDED_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            val = match.group(1)
            unit = match.group(2) if len(match.groups()) >= 2 and match.group(2) else default_unit
            dedup_key = f"{canonical_name.lower()}:{val}"
            if dedup_key not in seen_lab_keys:
                seen_lab_keys.add(dedup_key)
                labs.append({
                    "type": "lab_value",
                    "text": match.group(0).strip(),
                    "name": canonical_name,
                    "value": val,
                    "unit": unit,
                    "confidence": "possible",
                })

    # 2. Also run generic fallback for any other LAB_NAMES
    for match in re.finditer(rf"\b({LAB_NAMES})\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(%|mg/dL|mmol/L|g/dL|mEq/L|/uL|cells/uL)?", text, re.I):
        c_name = match.group(1).title()
        val = match.group(2)
        unit = match.group(3) or ""
        dedup_key = f"{c_name.lower()}:{val}"
        if dedup_key not in seen_lab_keys:
            seen_lab_keys.add(dedup_key)
            labs.append({
                "type": "lab_value",
                "text": match.group(0).strip(),
                "name": c_name,
                "value": val,
                "unit": unit,
                "confidence": "possible",
            })

    events = [{"type": "clinical_event", "text": match.group(0), "confidence": "possible"}
              for match in re.finditer(rf"\b(?:{EVENT_WORDS})\b[^.\n]*", text, re.I)]
    dates = [{"type": "date", "text": match.group(0), "confidence": "possible"}
             for match in re.finditer(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", text, re.I)]
    return {"conditions": _unique(conditions), "medications": _unique(medications), "lab_values": labs, "clinical_events": _unique(events), "dates": _unique(dates)}
