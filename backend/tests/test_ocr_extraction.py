"""
Minimal Week 2 OCR pipeline tests.

These call app.api.reports.extract_text_from_file() directly against real
files on disk. They exercise the actual tesseract/poppler binaries
installed on THIS machine - they do not mock OCR. A pass here means OCR
genuinely ran successfully in this environment; it does NOT by itself prove
the full upload -> API -> DB -> frontend flow works (see the manual test
checklist in the audit response).

Run with:  cd backend && pytest tests/test_ocr_extraction.py -v
"""
import os
import sys

import pytest

# Make `app` importable when running pytest from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.reports import extract_text_from_file  # noqa: E402


# --- Fixtures: generate real test files (not hardcoded to any machine) ---

@pytest.fixture
def sample_image_path(tmp_path):
    """A PNG containing real rendered text, for OCR to read."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (600, 150), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 50), "PATIENT REPORT OCR TEST 12345", fill="black")
    path = tmp_path / "sample_report.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_pdf_path(tmp_path):
    """A one-page PDF containing real text, for the pdf2image -> OCR path."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample_report.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 700, "LABORATORY REPORT PDF OCR TEST 67890")
    c.save()
    return str(path)


@pytest.fixture
def sample_txt_path(tmp_path):
    """Plain text with a UTF-16 BOM-less encoding + embedded nulls,
    reproducing the exact bug found during the audit (unclean output
    stored in the DB for record #2)."""
    path = tmp_path / "sample_report.txt"
    # Simulate a Windows-authored UTF-16LE text file, same shape as the
    # stale DB record found during the audit.
    with open(path, "w", encoding="utf-16") as f:
        f.write("Sample\n\nreport\n\n")
    return str(path)


# --- 1. Image -> OCR ---

def test_image_ocr_extracts_expected_text(sample_image_path):
    text, error_reason = extract_text_from_file(sample_image_path)
    assert error_reason is None, f"OCR reported failure: {error_reason}"
    assert text is not None
    # OCR output isn't pixel-perfect; check for the distinctive token.
    assert "12345" in text


# --- 2. PDF -> image -> OCR ---

def test_pdf_ocr_extracts_expected_text(sample_pdf_path):
    text, error_reason = extract_text_from_file(sample_pdf_path)
    assert error_reason is None, f"PDF OCR reported failure: {error_reason}"
    assert text is not None
    assert "67890" in text


# --- Regression test for the audit finding: unclean null-byte output ---

def test_txt_extraction_strips_null_bytes(sample_txt_path):
    text, error_reason = extract_text_from_file(sample_txt_path)
    assert error_reason is None
    assert "\x00" not in text
    assert "Sample" in text
    assert "report" in text


# --- 3. Failure / error handling ---

def test_corrupt_image_file_fails_gracefully(tmp_path):
    """A file with an image extension but garbage content must not crash
    the endpoint - it should return (None, reason) instead of raising."""
    bad_path = tmp_path / "not_really_an_image.png"
    bad_path.write_bytes(b"this is not a real png file")

    text, error_reason = extract_text_from_file(str(bad_path))

    assert text is None
    assert error_reason == "unsupported_or_corrupt_image"


def test_corrupt_pdf_file_fails_gracefully(tmp_path):
    """A file with a .pdf extension but garbage content must not crash
    the endpoint - it should return (None, reason) instead of raising."""
    bad_path = tmp_path / "not_really_a.pdf"
    bad_path.write_bytes(b"%PDF-1.4 this is not a valid pdf body")

    text, error_reason = extract_text_from_file(str(bad_path))

    assert text is None
    assert error_reason == "pdf_to_image_failed"


def test_unreadable_text_file_fails_gracefully(tmp_path):
    """A .txt path that doesn't actually exist must not crash the
    endpoint - it should return (None, reason)."""
    missing_path = tmp_path / "does_not_exist.txt"

    text, error_reason = extract_text_from_file(str(missing_path))

    assert text is None
    assert error_reason == "text_read_failed"
