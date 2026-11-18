import os
import shutil
import logging
import subprocess
from datetime import datetime
from typing import Optional, Tuple
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.patient import Patient
from app.models.report import Report
from app.api.auth import get_current_user
from app.schemas.report import ReportOut
from app.schemas.report import ClinicalEntityApproval, ClinicalEntitiesUpdate
from app.models.patient import MedicalHistory
from app.services.clinical_nlp import extract_clinical_entities
from app.core.config import settings
from fastapi.responses import FileResponse
import re

logger = logging.getLogger("ai_cdss.reports")
# Ensure OCR failures are actually visible even if the app's root logger
# isn't configured elsewhere (this app has no central logging setup today).
# Scoped to this module only - doesn't touch the root/uvicorn logger config.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

router = APIRouter(prefix="/patients/{patient_id}/reports", tags=["Reports"])


def _find_tesseract() -> Optional[str]:
    """Return a configured or locally installed Tesseract executable."""
    candidates = [
        settings.TESSERACT_CMD,
        shutil.which("tesseract"),
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def _find_poppler_path() -> Optional[str]:
    """Return a configured or bundled Poppler directory for PDF conversion."""
    bundled = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools", "poppler"))
    candidates = [settings.POPPLER_PATH, bundled]
    return next(
        (path for path in candidates if path and os.path.isfile(os.path.join(path, "pdftoppm.exe"))),
        None,
    )


def _extract_embedded_pdf_text(saved_path: str) -> Optional[str]:
    """Use bundled Poppler to read a text-based PDF before invoking OCR.

    This is faster and more accurate for digitally generated reports.  A
    scanned PDF simply returns no useful text and continues to Tesseract.
    """
    poppler_path = _find_poppler_path()
    if not poppler_path:
        return None
    executable = os.path.join(poppler_path, "pdftotext.exe" if os.name == "nt" else "pdftotext")
    if not os.path.isfile(executable):
        return None
    try:
        result = subprocess.run(
            [executable, "-layout", saved_path, "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        text = _clean_text(result.stdout)
        return text if text.strip() else None
    except (OSError, subprocess.SubprocessError) as e:
        logger.info("Embedded PDF text extraction unavailable for %s: %s", saved_path, e)
        return None


def save_upload_file(upload_dir: str, file: UploadFile) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # sanitize filename
    base = os.path.basename(file.filename)
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe_name = f"{timestamp}_{base}"
    dest_path = os.path.join(upload_dir, safe_name)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest_path


def _clean_text(text: str) -> str:
    """Strip stray null bytes (seen with some Windows-authored .txt files)
    and trailing whitespace per line."""
    text = re.sub(r"\x00+", "", text)
    return "\n".join(line.rstrip() for line in text.splitlines())


def extract_text_from_file(saved_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract text from an uploaded report file.

    Returns (extracted_text, error_reason). extracted_text is None if
    nothing could be extracted; error_reason is a short, non-sensitive
    machine-readable code describing why (safe to log; NOT sent to the
    frontend as-is, since it may mention local server paths/config).
    Does not raise: extraction failures are a normal, expected outcome
    (unsupported/corrupt file, OCR tooling not installed, etc.) and must
    not fail the upload itself.
    """
    lower = saved_path.lower()

    if lower.endswith(('.txt', '.md', '.csv')):
        try:
            with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return _clean_text(text), None
        except OSError as e:
            logger.warning("Text-file read failed for %s: %s", saved_path, e)
            return None, "text_read_failed"

    if lower.endswith('.pdf'):
        embedded_text = _extract_embedded_pdf_text(saved_path)
        if embedded_text:
            return embedded_text, None

    # Everything else goes through the OCR path (images, PDFs).
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.error(
            "OCR dependencies not installed (pytesseract/Pillow missing): %s. "
            "Run `pip install -r backend/requirements.txt`.", e
        )
        return None, "ocr_dependencies_missing"

    tesseract_cmd = _find_tesseract()
    if not tesseract_cmd:
        logger.error("Tesseract executable was not found. Configure TESSERACT_CMD in backend/.env.")
        return None, "tesseract_not_found"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if lower.endswith('.pdf'):
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            logger.error("pdf2image not installed: %s", e)
            return None, "pdf2image_missing"
        try:
            poppler_path = _find_poppler_path()
            pages = convert_from_path(saved_path, poppler_path=poppler_path)
        except Exception as e:
            # Most common cause: poppler (pdftoppm) not on PATH and
            # POPPLER_PATH not set in backend/.env. Also covers corrupt PDFs.
            logger.error(
                "PDF-to-image conversion failed for %s: %s. "
                "Check that poppler is installed and POPPLER_PATH (backend/.env) "
                "points to it if it's not on system PATH.", saved_path, e
            )
            return None, "pdf_to_image_failed"
        try:
            text_parts = [pytesseract.image_to_string(p) for p in pages]
            return _clean_text('\n'.join(text_parts)), None
        except Exception as e:
            logger.error(
                "Tesseract OCR failed on converted PDF pages for %s: %s. "
                "Check that tesseract is installed and TESSERACT_CMD "
                "(backend/.env) is set correctly if it's not on system PATH.",
                saved_path, e
            )
            return None, "ocr_failed"

    # Image file
    try:
        img = Image.open(saved_path)
    except Exception as e:
        logger.warning("Could not open %s as an image: %s", saved_path, e)
        return None, "unsupported_or_corrupt_image"
    try:
        text = pytesseract.image_to_string(img)
        return _clean_text(text), None
    except Exception as e:
        logger.error(
            "Tesseract OCR failed for %s: %s. "
            "Check that tesseract is installed and TESSERACT_CMD "
            "(backend/.env) is set correctly if it's not on system PATH.",
            saved_path, e
        )
        return None, "ocr_failed"


@router.post("", response_model=ReportOut)
def upload_report(
    patient_id: int,
    report_file: UploadFile = File(...),
    report_type: str = Form(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # verify patient belongs to doctor
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Enforce upload size limit (read current file pointer to compute size).
    # NOTE: the size check and the 413 HTTPException it raises must NOT sit
    # inside the same try/except that catches seek/tell failures - a bare
    # `except Exception: pass` around both would silently swallow the
    # intentional HTTPException too. Only the seek/tell call itself is
    # guarded; the size check + raise happens outside that guard.
    size_bytes = None
    try:
        report_file.file.seek(0, os.SEEK_END)
        size_bytes = report_file.file.tell()
        report_file.file.seek(0)
    except OSError as e:
        # Genuinely can't determine size (e.g. non-seekable stream) - log
        # and fall through; save_upload_file() will surface any real
        # problem when it actually tries to read/write the file.
        logger.warning("Could not determine upload size for %s: %s", report_file.filename, e)
        size_bytes = None

    if size_bytes is not None:
        max_bytes = int(settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        if size_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB"
            )

    upload_dir = os.path.abspath(settings.UPLOAD_DIR)
    patient_dir = os.path.join(upload_dir, f"patient_{patient_id}")
    saved_path = save_upload_file(patient_dir, report_file)

    extracted_text, error_reason = extract_text_from_file(saved_path)
    if error_reason:
        # Logged server-side with full detail in extract_text_from_file().
        # Only a generic notice reaches the client - no local paths, package
        # names, or stack traces are exposed to the frontend/patient data.
        logger.info(
            "Report %s for patient %s stored without extracted text (reason=%s)",
            report_file.filename, patient_id, error_reason
        )

    report = Report(
        patient_id=patient_id,
        filename=report_file.filename,
        file_path=saved_path,
        report_type=report_type,
        extracted_text=extracted_text,
        clinical_entities=extract_clinical_entities(extracted_text) if extracted_text else None,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return report


@router.get("/{report_id}/download")
def download_report(patient_id: int, report_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    report = db.query(Report).join(Patient).filter(Report.id == report_id, Report.patient_id == patient_id, Patient.doctor_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    if not os.path.exists(report.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on server")
    return FileResponse(path=report.file_path, filename=report.filename, media_type='application/octet-stream')


@router.get("", response_model=list[ReportOut])
def list_reports(patient_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    reports = db.query(Report).filter(Report.patient_id == patient_id).order_by(Report.uploaded_at.desc()).all()
    return reports


@router.get("/{report_id}", response_model=ReportOut)
def get_report(patient_id: int, report_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    report = db.query(Report).join(Patient).filter(Report.id == report_id, Report.patient_id == patient_id, Patient.doctor_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.post("/{report_id}/ocr", response_model=ReportOut)
def rerun_report_ocr(patient_id: int, report_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """Retry text extraction for an already uploaded report."""
    report = _owned_report(patient_id, report_id, db, current_user.id)
    if not os.path.isfile(report.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report file is no longer available on the server")

    extracted_text, error_reason = extract_text_from_file(report.file_path)
    if error_reason or not extracted_text:
        logger.info("OCR retry failed for report %s (reason=%s)", report.id, error_reason or "no_text_detected")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted. For scanned PDFs/images, install and configure Tesseract on the backend server.",
        )

    report.extracted_text = extracted_text
    report.clinical_entities = extract_clinical_entities(extracted_text)
    db.commit()
    db.refresh(report)
    return report


def _owned_report(patient_id: int, report_id: int, db: Session, doctor_id: int) -> Report:
    report = db.query(Report).join(Patient).filter(
        Report.id == report_id, Report.patient_id == patient_id, Patient.doctor_id == doctor_id
    ).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.post("/{report_id}/extract", response_model=ReportOut)
def extract_report_entities(patient_id: int, report_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """Re-run the conservative Week 3 parser on a report's saved OCR text."""
    report = _owned_report(patient_id, report_id, db, current_user.id)
    if not report.extracted_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="This report has no extracted text to analyse")
    report.clinical_entities = extract_clinical_entities(report.extracted_text)
    db.commit()
    db.refresh(report)
    return report


@router.put("/{report_id}/entities", response_model=ReportOut)
def review_report_entities(patient_id: int, report_id: int, payload: ClinicalEntitiesUpdate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """Save clinician-reviewed extraction edits; this does not create a diagnosis."""
    report = _owned_report(patient_id, report_id, db, current_user.id)
    report.clinical_entities = payload.model_dump()
    db.commit()
    db.refresh(report)
    return report


@router.post("/{report_id}/conditions/approve", status_code=status.HTTP_201_CREATED)
def add_reviewed_condition(patient_id: int, report_id: int, payload: ClinicalEntityApproval, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """Copy one doctor-reviewed possible condition into the patient's medical history."""
    _owned_report(patient_id, report_id, db, current_user.id)
    condition = payload.condition.strip()
    if not condition:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Condition is required")
    history = MedicalHistory(
        patient_id=patient_id, condition=condition, diagnosis_date=payload.diagnosis_date,
        description=payload.description, status=payload.status or "Active"
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return {"message": "Doctor-reviewed condition added to medical history", "id": history.id}
