"""
ECG Image Validation and Secure File Handling Service.

Validates:
- MIME types (JPEG, PNG)
- Magic bytes and PIL image integrity (detects corrupted files and renamed non-images)
- File size within configured settings.MAX_UPLOAD_SIZE_MB
- Safe filename sanitization preventing directory traversal
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

ALLOWED_MIME_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/jpg": [".jpg", ".jpeg"],
    "image/png": [".png"],
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_ecg_upload_dir() -> Path:
    """Return the dedicated ECG uploads directory and ensure it exists."""
    upload_path = Path(settings.UPLOAD_DIR) / "ecg"
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and shell injection."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"ecg_upload_{clean_name}"
    return clean_name


def validate_ecg_image(file_bytes: bytes, filename: str, content_type: str | None) -> Tuple[str, Tuple[int, int]]:
    """
    Validate that bytes represent a valid, uncorrupted PNG or JPEG image.
    
    Returns (detected_format, (width, height)).
    Raises HTTPException(413) if oversized, HTTPException(422) if invalid/corrupted.
    """
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Check extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: .png, .jpg, .jpeg",
        )

    # Check reported MIME type if provided
    if content_type:
        clean_ct = content_type.lower()
        if clean_ct not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported MIME type '{content_type}'. Allowed types are: image/png, image/jpeg",
            )

    # Inspect image bytes with PIL
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
            img_format = (img.format or "").upper()
            if img_format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"File contains format '{img_format}'. Only PNG and JPEG images are supported.",
                )
            
        # Re-open to read dimensions and verify complete decompression (catches truncated/corrupt streams)
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
            dimensions = img.size  # (width, height)
            
            # Minimum sanity check: an ECG image needs reasonable resolution
            if dimensions[0] < 50 or dimensions[1] < 50:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image dimensions {dimensions} are too small for an ECG scan.",
                )

        return img_format, dimensions

    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or corrupted image file: {str(exc)}",
        )


def save_ecg_file(file_bytes: bytes, original_filename: str) -> Tuple[str, str]:
    """
    Save validated image bytes to the dedicated ECG upload directory.
    Returns (stored_filename, absolute_file_path).
    """
    upload_dir = get_ecg_upload_dir()
    safe_name = sanitize_filename(original_filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    stored_filename = f"{timestamp}_{safe_name}"
    target_path = (upload_dir / stored_filename).resolve()

    # Safety check: prevent path traversal out of upload directory
    if not str(target_path).startswith(str(upload_dir.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path detected.",
        )

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return stored_filename, str(target_path)
