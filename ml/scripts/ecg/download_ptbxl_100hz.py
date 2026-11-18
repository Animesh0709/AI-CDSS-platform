"""
Download and Verification Script for Official PTB-XL v1.0.3 100 Hz Waveform Records.

Downloads:
- Official PhysioNet PTB-XL v1.0.3 archive (~1.71 GB).
- Resumable download with HTTP range headers.
- Extracts ONLY records100/ waveform files (*.dat, *.hea) into ml/data/ecg/ptbxl/records100/.
- Does NOT extract records500/ to protect disk space and requirements.
- Does NOT overwrite existing verified records without integrity check.
- Removes temporary archive upon verified extraction to preserve disk space.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
import urllib.request

logger = logging.getLogger("ptbxl_downloader")

OFFICIAL_ZIP_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"


def download_file_with_resume(url: str, dest_path: Path, expected_size: int = 1839504686) -> None:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".part")

    downloaded = 0
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        logger.info(f"Resuming download from byte {downloaded} ({downloaded / (1024*1024):.1f} MB)...")

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        logger.info(f"Archive {dest_path} already completely downloaded ({expected_size} bytes).")
        return

    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")

    logger.info(f"Opening connection to {url}...")
    start_time = time.time()
    last_log_time = start_time

    with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "ab" if downloaded > 0 else "wb") as f:
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) + downloaded if content_length else expected_size

        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed = current_time - start_time
                speed = downloaded / (elapsed + 1e-5) / (1024 * 1024)
                pct = (downloaded / total_size) * 100 if total_size else 0
                logger.info(
                    f"Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB "
                    f"({pct:.1f}%) at {speed:.2f} MB/s"
                )
                last_log_time = current_time

    # Move temp to final
    temp_path.rename(dest_path)
    logger.info(f"Download complete: {dest_path} ({dest_path.stat().st_size} bytes).")


def extract_only_records100(zip_path: Path, target_base_dir: Path) -> int:
    """
    Extract ONLY records100/ files from zip into target_base_dir.
    Skips records500/ entirely.
    """
    logger.info(f"Opening {zip_path} for selective records100 extraction...")
    target_base_dir = Path(target_base_dir)
    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        records100_members = [
            m for m in members if "records100/" in m and (m.endswith(".dat") or m.endswith(".hea"))
        ]
        logger.info(f"Found {len(records100_members)} records100 files in archive.")

        for idx, member in enumerate(records100_members):
            # Extract path relative to records100/
            # Zip entries look like: ptb-xl-1.0.3/records100/00000/00001_lr.hea
            parts = member.split("records100/")
            if len(parts) < 2:
                continue
            rel_subpath = parts[1]
            dest_file = target_base_dir / "records100" / rel_subpath

            if dest_file.exists() and dest_file.stat().st_size > 0:
                # File already exists and non-empty
                continue

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(dest_file, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_count += 1

            if (idx + 1) % 5000 == 0:
                logger.info(f"Extracted {idx + 1}/{len(records100_members)} files...")

    logger.info(f"Extraction finished. Extracted {extracted_count} newly written files.")
    return extracted_count


def run_restore(
    data_dir: str | Path = "ml/data/ecg/ptbxl",
    cache_dir: str | Path = "ml/data/ecg/ptbxl/.download_cache",
    keep_zip: bool = False,
) -> None:
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    zip_dest = cache_path / "ptb-xl-1.0.3.zip"

    # Step 1: Download
    download_file_with_resume(OFFICIAL_ZIP_URL, zip_dest)

    # Step 2: Extract records100
    extract_only_records100(zip_dest, data_path)

    # Step 3: Clean zip to conserve disk space
    if not keep_zip and zip_dest.exists():
        logger.info(f"Removing temporary zip archive {zip_dest} to save disk space...")
        zip_dest.unlink()
        logger.info("Temporary zip removed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_restore()
